"""显式持续时间推断的核对：短序列全枚举对照、决胜规则、数值与约定边界。"""
import contextlib
import io as sysio
import json
import math
import random
import sys
import unittest
from response_data.io import load
from response_data.logmath import (logsumexp, log_norm_cdf, log_gaussian_pdf,
                                   log_interval_prob, log_upper_tail)
from response_data import hsmm
from response_data import cli

NEG = -math.inf


def _config(states, initial, transition, observations, state_order):
    return {'states': states, 'initial': initial, 'transition': transition,
            'observations': observations, 'state_order': state_order,
            'probability_tolerance': 1e-10}


def _two_state(dur_a, dur_b, initial=(0.5, 0.5), order=('a', 'b')):
    """两状态对称转移的最小配置，观测另行指定。"""
    return lambda obs: _config(
        [{'id': 'a', 'mean': 0, 'sd': 1, 'duration': dur_a},
         {'id': 'b', 'mean': 0, 'sd': 1, 'duration': dur_b}],
        list(initial), [[0, 1], [1, 0]], obs, list(order))


class LogMathTests(unittest.TestCase):
    def test_logsumexp(self):
        self.assertEqual(logsumexp([]), NEG)
        self.assertEqual(logsumexp([NEG, NEG]), NEG)
        self.assertAlmostEqual(logsumexp([0.0, 0.0]), math.log(2), places=15)
        vals = [-3.0, -1.0, -2.0, 0.5]
        self.assertEqual(logsumexp(vals), logsumexp(list(reversed(vals))))

    def test_log_norm_cdf_known(self):
        self.assertAlmostEqual(log_norm_cdf(0.0), math.log(0.5), places=15)
        self.assertAlmostEqual(log_norm_cdf(1.96), math.log(0.9750021048517795), places=12)
        # 远左尾参照：Phi(-10) = 7.61985302416047e-24
        self.assertAlmostEqual(log_norm_cdf(-10.0), math.log(7.61985302416047e-24), places=12)
        # Phi(x) + Phi(-x) = 1
        self.assertAlmostEqual(math.exp(log_norm_cdf(1.5)) + math.exp(log_norm_cdf(-1.5)),
                               1.0, places=14)

    def test_log_norm_cdf_far_tail_continuous(self):
        # erfc 与渐近展开两条分支在切换处 -38 附近连续（导数约 38，微扰 2e-6 内一致）
        self.assertAlmostEqual(log_norm_cdf(-37.999999), log_norm_cdf(-38.000001), delta=1e-3)
        self.assertTrue(math.isfinite(log_norm_cdf(-100.0)))
        self.assertEqual(log_norm_cdf(-1e300), NEG)  # 真下溢，诚实零支持

    def test_interval_tiny_probability_no_underflow(self):
        # z≈7 的微小区间概率：线性空间 Phi 差会相消，erfc 尾部差仍精确，可作参照
        ref = 0.5 * math.erfc(7 / math.sqrt(2)) - 0.5 * math.erfc(7.5 / math.sqrt(2))
        got = log_interval_prob(7.0, 7.5, 0.0, 1.0)
        self.assertTrue(math.isfinite(got))
        self.assertAlmostEqual(got, math.log(ref), places=9)

    def test_interval_adjacency(self):
        a = log_interval_prob(0.0, 1.0, 0.3, 0.7)
        b = log_interval_prob(1.0, 2.0, 0.3, 0.7)
        ab = log_interval_prob(0.0, 2.0, 0.3, 0.7)
        self.assertAlmostEqual(logsumexp([a, b]), ab, places=12)

    def test_interval_honest_zero_support(self):
        # Phi(40) 与 Phi(41) 在浮点下都是 1，区间概率不可分辨 -> 零支持
        self.assertEqual(log_interval_prob(40.0, 41.0, 0.0, 1.0), NEG)

    def test_upper_tail(self):
        self.assertAlmostEqual(log_upper_tail(0.0, 0.0, 1.0), math.log(0.5), places=15)
        self.assertAlmostEqual(log_upper_tail(-8.0, 0.0, 1.0), 0.0, places=12)


class DataFileCrossCheckTests(unittest.TestCase):
    """三份演练资料：动态规划与全枚举必须一致。"""

    def _check(self, path):
        v, _ = load(path)
        r = hsmm.analyze(v)
        self.assertEqual(r['status'], 'ok')
        chk = r['checks']['enumeration']
        self.assertIsNotNone(chk)
        self.assertLess(chk['log_likelihood_abs_diff'], 1e-9)
        self.assertLess(chk['state_posterior_max_abs_diff'], 1e-9)
        self.assertLess(chk['transition_posterior_max_abs_diff'], 1e-9)
        self.assertTrue(chk['dp_path_matches_enumeration'])
        self.assertTrue(r['checks']['state_posterior_rows_sum_to_one'])
        return r

    def test_normal(self):
        r = self._check('data/normal.json')
        self.assertEqual(r['axis']['length'], 7)
        got = [(s['state'], s['length'], s['censored']) for s in r['best_path']['segments']]
        self.assertEqual(got, [('dry', 2, False), ('rising', 2, False),
                               ('holding', 2, False), ('falling', 1, True)])

    def test_saturated_gap(self):
        r = self._check('data/saturated-gap.json')
        # 缺 slot 2 显式补 missing，分析轴不压缩
        self.assertEqual(r['axis']['inserted_missing_slots'], [2])
        self.assertEqual(r['axis']['length'], 4)
        self.assertEqual(len(r['state_posterior']), 4)

    def test_right_censored_first_segment(self):
        r = self._check('data/right-censored-first-segment.json')
        # T=1：末段即首段，持续因子 P(D>=1)=1，似然就是正态密度
        self.assertAlmostEqual(r['log_likelihood'], log_gaussian_pdf(0.1, 0.0, 0.2), places=12)
        seg = r['best_path']['segments'][0]
        self.assertEqual((seg['state'], seg['length'], seg['censored']), ('dry', 1, True))


class RandomModelCrossCheckTests(unittest.TestCase):
    """随机冻结配置：短序列上 DP 与全枚举互相核对。"""

    def _random_case(self, rng):
        K = rng.choice([2, 3, 4])
        ids = ['s%d' % i for i in range(K)]
        states = []
        for i in range(K):
            D = rng.randint(1, 4)
            w = [rng.random() + 0.05 for _ in range(D)]
            if D > 1 and rng.random() < 0.2:
                w[rng.randrange(D)] = 0.0  # 持续分布中的零概率长度
            z = sum(w)
            states.append({'id': ids[i], 'mean': rng.uniform(-2, 5),
                           'sd': rng.uniform(0.2, 1.5), 'duration': [x / z for x in w]})
        initial = [rng.random() + 0.05 for _ in range(K)]
        z = sum(initial)
        transition = []
        for i in range(K):
            row = [0.0 if j == i else rng.random() + 0.05 for j in range(K)]
            if K > 2 and rng.random() < 0.3:
                row[rng.choice([j for j in range(K) if j != i])] = 0.0
            zr = sum(row)
            transition.append([x / zr for x in row])
        obs = []
        for t in range(rng.randint(1, 8)):
            if rng.random() < 0.2:
                continue  # 缺槽，应显式补 missing
            kind = rng.choice(['exact', 'interval', 'right', 'missing'])
            mu = states[rng.randrange(K)]['mean']
            if kind == 'exact':
                obs.append({'slot': t, 'kind': 'exact', 'value': mu + rng.gauss(0, 1)})
            elif kind == 'interval':
                left = mu + rng.uniform(-2, 1)
                obs.append({'slot': t, 'kind': 'interval',
                            'left': left, 'right': left + rng.uniform(0.1, 2)})
            elif kind == 'right':
                obs.append({'slot': t, 'kind': 'right', 'left': mu + rng.uniform(-1, 2)})
            else:
                obs.append({'slot': t, 'kind': 'missing'})
        if not obs:
            obs.append({'slot': 0, 'kind': 'missing'})
        order = ids[:]
        rng.shuffle(order)
        return _config(states, [x / z for x in initial], transition, obs, order)

    def test_random_models(self):
        for seed in range(40):
            rng = random.Random(seed)
            v = self._random_case(rng)
            with self.subTest(seed=seed):
                r = hsmm.analyze(v)
                self.assertEqual(r['status'], 'ok')
                chk = r['checks']['enumeration']
                self.assertIsNotNone(chk)
                self.assertLess(chk['log_likelihood_abs_diff'], 1e-8)
                self.assertLess(chk['state_posterior_max_abs_diff'], 1e-8)
                self.assertLess(chk['transition_posterior_max_abs_diff'], 1e-8)
                self.assertTrue(chk['dp_path_matches_enumeration'])
                self.assertTrue(r['checks']['state_posterior_rows_sum_to_one'])


class TieBreakTests(unittest.TestCase):
    """决胜：先逐段状态 state_order 名次序列字典序，再内部段末时隙序列字典序。"""

    def test_state_order_and_prefix_rule(self):
        # 四条路径得分完全相同：(a),(b),(a,b),(b,a)；最短且名次最小者胜
        for order, winner in ((('a', 'b'), 'a'), (('b', 'a'), 'b')):
            v = _two_state([0.5, 0.5], [0.5, 0.5], order=order)(
                [{'slot': 0, 'kind': 'missing'}, {'slot': 1, 'kind': 'missing'}])
            r = hsmm.analyze(v)
            got = r['best_path']['segments']
            self.assertEqual(len(got), 1)
            self.assertEqual((got[0]['state'], got[0]['length'], got[0]['censored']),
                             (winner, 2, True))

    def test_boundary_sequence_rule(self):
        # (a[0..0],b[1..1],a[2..2])、(a[0..0],b[1..2])、(a[0..1],b[2..2]) 三方同分：
        # 状态序列 (a,b) 以前缀胜 (a,b,a)；同为 (a,b) 时边界序列 (0) 胜 (1)
        v = _two_state([2 / 3, 1 / 3], [0.5, 0.5], initial=(1.0, 0.0))(
            [{'slot': t, 'kind': 'missing'} for t in range(3)])
        r = hsmm.analyze(v)
        got = [(s['state'], s['start'], s['length'], s['censored'])
               for s in r['best_path']['segments']]
        self.assertEqual(got, [('a', 0, 1, False), ('b', 1, 2, True)])
        # 三方确实同分（枚举前三名的 log 得分相等）
        tops = r['top_paths']
        self.assertAlmostEqual(tops[0]['log_score'], tops[1]['log_score'], places=15)
        self.assertAlmostEqual(tops[1]['log_score'], tops[2]['log_score'], places=15)


class MapVsPathTests(unittest.TestCase):
    def test_map_label_can_differ_from_best_path(self):
        # 构造：slot1 边际后验最大为 b，但最优整路径经过 a
        v = _config(
            [{'id': 'a', 'mean': 0, 'sd': 1, 'duration': [0.7, 0.3]},
             {'id': 'b', 'mean': 0, 'sd': 1.19, 'duration': [0.6, 0.4]}],
            [0.5, 0.5], [[0, 1], [1, 0]],
            [{'slot': 0, 'kind': 'missing'}, {'slot': 1, 'kind': 'exact', 'value': 0}],
            ['a', 'b'])
        r = hsmm.analyze(v)
        row = r['state_posterior'][1]
        self.assertEqual(row['map_state'], 'b')
        self.assertEqual(row['path_state'], 'a')
        self.assertTrue(row['map_differs_from_path'])
        self.assertEqual([(s['state'], s['length']) for s in r['best_path']['segments']],
                         [('b', 1), ('a', 1)])


class FinalSegmentTests(unittest.TestCase):
    def test_censored_tail_not_forced_end(self):
        # 末段持续因子必须是 P(D>=已见长度)=0.9，而不是 P(D=2)=0.3
        v = _config(
            [{'id': 'a', 'mean': 0, 'sd': 1, 'duration': [0.1, 0.3, 0.6]},
             {'id': 'b', 'mean': 0, 'sd': 1, 'duration': [1.0]}],
            [0, 1], [[0, 1], [1, 0]],
            [{'slot': t, 'kind': 'missing'} for t in range(3)], ['a', 'b'])
        r = hsmm.analyze(v)
        # 似然 = (b,a,b) 的 log0.1 与 (b,a[1..2]) 的 log0.9 之和
        self.assertAlmostEqual(r['log_likelihood'], math.log(0.1 + 0.9), places=12)
        seg = r['best_path']['segments'][-1]
        self.assertEqual((seg['state'], seg['length'], seg['censored']), ('a', 2, True))
        bd = r['best_path']['score_breakdown']['segments'][-1]
        self.assertEqual(bd['duration_term'], 'P(D>=2)')
        self.assertAlmostEqual(bd['log_duration'], math.log(0.9), places=15)

    def test_impossible_duration_length(self):
        v, _ = load('data/normal.json')
        # dry 最大支持 3，末段已见长度 4 超过一切配置支持
        out = hsmm.score_command(v, [{'state': 'dry', 'length': 4},
                                     {'state': 'rising', 'length': 3}])
        self.assertEqual(out['log_score'], NEG)
        self.assertIsNotNone(out['breakdown']['zero_support'])
        # 合法分段中不存在超过最大支持的段
        r = hsmm.analyze(v)
        for p in r['top_paths']:
            for s in p['segments']:
                dmax = len(v['states'][v['states'].index(
                    next(x for x in v['states'] if x['id'] == s['state']))]['duration'])
                self.assertLessEqual(s['length'], dmax)


class UninterpretableTests(unittest.TestCase):
    def test_zero_support_interval(self):
        # 区间 (41,42] 对全部四个状态都是浮点不可分辨的零支持
        v, _ = load('data/normal.json')
        v['observations'] = [{'slot': 0, 'kind': 'interval', 'left': 41, 'right': 42}]
        r = hsmm.analyze(v)
        self.assertEqual(r['status'], 'uninterpretable')
        self.assertIsNone(r['state_posterior'])
        self.assertIsNone(r['best_path'])
        self.assertEqual(r['zero_support_slots'], [0])

    def test_zero_support_extreme_censor(self):
        v, _ = load('data/normal.json')
        v['observations'] = [{'slot': 0, 'kind': 'right', 'left': 1e300}]
        r = hsmm.analyze(v)
        self.assertEqual(r['status'], 'uninterpretable')

    def test_invalid_domain_rejected(self):
        v, _ = load('data/invalid-domain.json')
        with self.assertRaises(ValueError):
            hsmm.Model(v)


class ScoreRecomputeTests(unittest.TestCase):
    def test_breakdown_recomputes_best_path(self):
        v, _ = load('data/normal.json')
        r = hsmm.analyze(v)
        spec = [{'state': s['state'], 'length': s['length']}
                for s in r['best_path']['segments']]
        out = hsmm.score_command(v, spec)
        self.assertAlmostEqual(out['log_score'], r['best_path']['log_score'], places=12)
        # 与全枚举最优得分严格一致（同为 fsum 规范求和）
        self.assertEqual(out['log_score'], r['best_path']['log_score'])
        # 逐项抽查首段：initial log1=0，P(D=2)=log0.5，逐观测项
        seg0 = out['breakdown']['segments'][0]
        self.assertEqual(seg0['log_in'], 0.0)
        self.assertAlmostEqual(seg0['log_duration'], math.log(0.5), places=15)
        self.assertAlmostEqual(seg0['emissions'][0]['log'],
                               log_gaussian_pdf(0.1, 0.0, 0.2), places=15)
        self.assertEqual(seg0['emissions'][1]['log'], 0.0)  # missing 因子 1
        # 逐项手算总和
        manual = math.fsum(x for seg in out['breakdown']['segments']
                           for x in [seg['log_in'], seg['log_duration']]
                           + [e['log'] for e in seg['emissions']])
        self.assertAlmostEqual(manual, out['log_score'], places=12)

    def test_enumeration_cap_fallback(self):
        v, _ = load('data/normal.json')
        r = hsmm.analyze(v, enumerate_cap=2)
        self.assertEqual(r['best_path']['selection'], 'dp')
        self.assertIsNone(r['checks'].get('enumeration'))
        self.assertTrue(any('枚举上限' in d for d in r['diagnostics']))


class CliTests(unittest.TestCase):
    def _run(self, argv):
        buf = sysio.StringIO()
        old = sys.argv
        try:
            sys.argv = argv
            with contextlib.redirect_stdout(buf):
                cli.main()
        finally:
            sys.argv = old
        return json.loads(buf.getvalue())

    def test_inspect_unchanged(self):
        r = self._run(['cli', 'inspect', 'data/normal.json'])
        self.assertEqual(r['summary']['states'], 4)
        self.assertEqual(len(r['source_sha256']), 64)

    def test_infer_cli(self):
        r = self._run(['cli', 'infer', 'data/normal.json'])
        self.assertEqual(r['status'], 'ok')
        self.assertEqual(len(r['source_sha256']), 64)

    def test_score_cli(self):
        spec = json.dumps([{'state': 'dry', 'length': 2}, {'state': 'rising', 'length': 2},
                           {'state': 'holding', 'length': 2}, {'state': 'falling', 'length': 1}])
        r = self._run(['cli', 'score', 'data/normal.json', spec])
        self.assertAlmostEqual(r['log_score'], -5.498327237534716, places=9)
        self.assertAlmostEqual(r['posterior_probability'], 0.8408765908418935, places=9)

    def test_uninterpretable_cli_has_null_posterior(self):
        v, _ = load('data/normal.json')
        v['observations'] = [{'slot': 0, 'kind': 'interval', 'left': 41, 'right': 42}]
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'zero.json'
            p.write_text(json.dumps(v), encoding='utf8')
            r = self._run(['cli', 'infer', str(p)])
        self.assertEqual(r['status'], 'uninterpretable')
        self.assertIsNone(r['state_posterior'])
        self.assertIsNone(r['log_likelihood'])


if __name__ == '__main__':
    unittest.main()
