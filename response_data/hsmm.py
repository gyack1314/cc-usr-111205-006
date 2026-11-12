"""显式持续时间隐半马尔可夫模型的前向/后向推断。

口径完全遵循 docs/data-contract.md（冻结约定）：
- 分析轴为最小 slot 到最大 slot 的连续整数时隙，缺 slot 显式补 missing，不压缩；
- 首时隙视为新段起点：初始分布 + 普通持续分布 P(D=d)，不是平稳过程随机抽到一段中间；
- 段内状态不变；转移矩阵对角为 0，持续由 duration 表达；
- 发射因子：exact=正态密度，interval=P(left<Y<=right)，right=P(Y>left)，missing=1；
  量程上限的饱和读数按右删失处理，不当精确测量；
- 观测窗末段是尚未结束的右删失段：持续因子 P(D>=已见长度)，既不强迫它恰在
  窗口末结束，也不允许超过配置最大支持的长度；
- 全部计算在 log 域；真正零支持时输出 uninterpretable，不交一张归一化失败的后验表。

误差条件独立只是本分析的设定，不推断成设备真实机理。
初始概率、冻结转移和观测参数不被本次资料重估。
"""
import math
import sys
from .io import number, positive, integer, unique
from .logmath import logsumexp, logadd, log_gaussian_pdf, log_interval_prob, log_upper_tail

NEG = -math.inf


class Model:
    """冻结配置：状态、初始、转移、持续与观测参数。"""

    __slots__ = ('ids', 'rank', 'log_initial', 'log_transition', 'log_duration',
                 'log_tail', 'mean', 'sd', 'tolerance')

    def __init__(self, v):
        self.tolerance = positive(v.get('probability_tolerance', 1e-9))
        rows = v['states']
        unique(rows, 'id')
        self.ids = [s['id'] for s in rows]
        K = len(self.ids)
        order = v['state_order']
        if not isinstance(order, list) or sorted(order) != sorted(self.ids):
            raise ValueError('state_order 必须是状态身份的排列')
        # 决胜用的预给状态顺序：rank[身份] = 名词，小者优先。
        self.rank = {sid: r for r, sid in enumerate(order)}
        self.mean = [number(s['mean']) for s in rows]
        self.sd = [positive(s['sd']) for s in rows]
        self.log_duration = []
        self.log_tail = []
        for s in rows:
            p = [number(x) for x in s['duration']]
            if not p or any(x < 0 for x in p) or abs(sum(p) - 1) > self.tolerance:
                raise ValueError('持续概率无效')
            lp = [math.log(x) if x > 0 else NEG for x in p]
            self.log_duration.append(lp)
            # log_tail[d] = log P(D >= d)，d=1..len(p)；超过最大支持为 -inf。
            lt = [NEG] * (len(p) + 2)
            for d in range(1, len(p) + 1):
                lt[d] = logsumexp(lp[d - 1:])
            self.log_tail.append(lt)
        init = [number(x) for x in v['initial']]
        if len(init) != K or any(x < 0 for x in init) or abs(sum(init) - 1) > self.tolerance:
            raise ValueError('初始概率无效')
        self.log_initial = [math.log(x) if x > 0 else NEG for x in init]
        tr = v['transition']
        if len(tr) != K or any(len(r) != K for r in tr):
            raise ValueError('转移形状无效')
        self.log_transition = []
        for i, row in enumerate(tr):
            pr = [number(x) for x in row]
            if pr[i] != 0 or any(x < 0 for x in pr) or abs(sum(pr) - 1) > self.tolerance:
                raise ValueError('转移概率无效')
            self.log_transition.append([math.log(x) if x > 0 else NEG for x in pr])

    def k(self):
        return len(self.ids)

    def dmax(self, k):
        return len(self.log_duration[k])

    def tail(self, k, d):
        """log P(D_k >= d)；d 超过配置最大支持时为 -inf。"""
        if d < 1 or d > self.dmax(k):
            return NEG
        return self.log_tail[k][d]


def build_axis(raw_obs):
    """解析观测记录，返回 (slots, obs, inserted)。

    slots 为最小 slot 到最大 slot 的连续整数绝对编号；obs[t] 为相对下标 t 的
    观测 dict；缺 slot 显式补 {'kind': 'missing'} 并记入 inserted，不压缩分析轴。
    """
    if not raw_obs:
        raise ValueError('观测为空')
    by_slot = {}
    for r in raw_obs:
        slot = integer(r['slot'])
        if slot in by_slot:
            raise ValueError('观测身份或类别无效')
        kind = r['kind']
        if kind == 'exact':
            obs = {'kind': 'exact', 'value': number(r['value'])}
        elif kind == 'interval':
            left, right = number(r['left']), number(r['right'])
            if not left < right:
                raise ValueError('区间观测需要 left < right')
            obs = {'kind': 'interval', 'left': left, 'right': right}
        elif kind == 'right':
            obs = {'kind': 'right', 'left': number(r['left'])}
        elif kind == 'missing':
            obs = {'kind': 'missing'}
        else:
            raise ValueError('观测身份或类别无效')
        by_slot[slot] = obs
    lo, hi = min(by_slot), max(by_slot)
    slots = list(range(lo, hi + 1))
    obs = []
    inserted = []
    for slot in slots:
        if slot in by_slot:
            obs.append(by_slot[slot])
        else:
            obs.append({'kind': 'missing'})
            inserted.append(slot)
    return slots, obs, inserted


def emission_logs(model, obs):
    """e[k][t] = 状态 k 在相对时隙 t 的 log 发射因子。"""
    K = model.k()
    e = [[0.0] * len(obs) for _ in range(K)]
    for t, o in enumerate(obs):
        for k in range(K):
            mu, sd = model.mean[k], model.sd[k]
            kind = o['kind']
            if kind == 'exact':
                w = log_gaussian_pdf(o['value'], mu, sd)
            elif kind == 'interval':
                w = log_interval_prob(o['left'], o['right'], mu, sd)
            elif kind == 'right':
                w = log_upper_tail(o['left'], mu, sd)
            else:
                w = 0.0
            e[k][t] = w
    return e


def make_segment_query(e):
    """返回 seg(k, s, t)：相对时隙 s..t（含）状态 k 的发射 log 和。

    任一项为 -inf 则整段为 -inf。短段用 math.fsum 精确求和（与求值顺序无关，
    供全枚举逐分核对）；长段用前缀差，前缀差只用于超过 64 格的段。
    """
    K = len(e)
    T = len(e[0])
    fpref = []
    nneg = []
    for k in range(K):
        acc = [0.0]
        for w in e[k]:
            acc.append(acc[-1] + (w if w != NEG else 0.0))
        fpref.append(acc)
        nn = [T + 1] * (T + 1)
        nxt = T + 1
        for t in range(T - 1, -1, -1):
            if e[k][t] == NEG:
                nxt = t
            nn[t] = nxt
        nneg.append(nn)

    def seg(k, s, t):
        if nneg[k][s] <= t:
            return NEG
        if t - s < 64:
            return math.fsum(e[k][s:t + 1])
        return fpref[k][t + 1] - fpref[k][s]

    return seg


def forward(model, seg, T):
    """前向。alpha[t][j]：段 j 恰好结束于相对时隙 t（0<=t<T-1）的 log 联合概率；
    into[t][j]：log sum_{i!=j} exp(alpha[t][i]) * A[i][j]，即边界 t 之后进入 j 的项。"""
    K = model.k()
    alpha = [[NEG] * K for _ in range(max(T - 1, 0))]
    into = [[NEG] * K for _ in range(max(T - 1, 0))]
    for t in range(T - 1):
        for j in range(K):
            cands = []
            for d in range(1, min(t + 1, model.dmax(j)) + 1):
                s = t - d + 1
                w = model.log_duration[j][d - 1] + seg(j, s, t)
                if w == NEG:
                    continue
                w += model.log_initial[j] if s == 0 else into[s - 1][j]
                cands.append(w)
            alpha[t][j] = logsumexp(cands)
        for j in range(K):
            into[t][j] = logsumexp(alpha[t][i] + model.log_transition[i][j]
                                   for i in range(K) if i != j)
    return alpha, into


def likelihood(model, into, seg, T):
    """资料似然。观测窗末段一律按右删失处理：持续因子 P(D>=已见长度)。"""
    cands = []
    for j in range(model.k()):
        for L in range(1, min(T, model.dmax(j)) + 1):
            s = T - L
            w = model.tail(j, L) + seg(j, s, T - 1)
            if w == NEG:
                continue
            w += model.log_initial[j] if s == 0 else into[s - 1][j]
            cands.append(w)
    return logsumexp(cands)


def backward(model, seg, T):
    """后向。beta[t][i]：段 i 恰好结束于 t 时，t+1..T-1 的 log 条件概率；
    rest[t][j]：从 t+1 起、状态 j 的一段（精确结束接 beta，或删失末段）的 log 概率。"""
    K = model.k()
    beta = [[NEG] * K for _ in range(max(T - 1, 0))]
    rest = [[NEG] * K for _ in range(max(T - 1, 0))]
    for t in range(T - 2, -1, -1):
        for j in range(K):
            cands = []
            for d in range(1, model.dmax(j) + 1):
                end = t + d
                if end > T - 2:
                    break
                cands.append(model.log_duration[j][d - 1] + seg(j, t + 1, end) + beta[end][j])
            L = T - 1 - t
            cands.append(model.tail(j, L) + seg(j, t + 1, T - 1))
            rest[t][j] = logsumexp(cands)
        for i in range(K):
            beta[t][i] = logsumexp(model.log_transition[i][j] + rest[t][j]
                                   for j in range(K) if j != i)
    return beta, rest


def posterior_state(model, into, beta, seg, T, logL):
    """gamma[t][k] = P(相对时隙 t 状态为 k | 资料)，每个时隙上和为 1。"""
    K = model.k()
    glog = [[NEG] * K for _ in range(T)]
    for k in range(K):
        for s in range(T):
            fwd = model.log_initial[k] if s == 0 else into[s - 1][k]
            if fwd == NEG:
                continue
            for d in range(1, model.dmax(k) + 1):
                end = s + d - 1
                if end >= T:
                    break
                if end < T - 1:
                    w = fwd + model.log_duration[k][d - 1] + seg(k, s, end) + beta[end][k]
                else:
                    w = fwd + model.tail(k, d) + seg(k, s, T - 1)
                if w == NEG:
                    continue
                for t in range(s, end + 1):
                    glog[t][k] = logadd(glog[t][k], w)
    return [[math.exp(glog[t][k] - logL) if glog[t][k] != NEG else 0.0
             for k in range(K)] for t in range(T)]


def posterior_transition(model, alpha, rest, T, logL):
    """xi[t][i][j] = P(段 i 恰好结束于相对时隙 t 且下一段为 j | 资料)。

    全部边界、全部配对的和是预期段间转移数，不为 1。
    """
    K = model.k()
    xi = []
    for t in range(T - 1):
        mat = [[0.0] * K for _ in range(K)]
        for i in range(K):
            if alpha[t][i] == NEG:
                continue
            for j in range(K):
                if i == j:
                    continue
                w = alpha[t][i] + model.log_transition[i][j] + rest[t][j]
                if w != NEG:
                    mat[i][j] = math.exp(w - logL)
        xi.append(mat)
    return xi


def viterbi(model, seg, T):
    """概率最大的分段路径（动态规划）。

    决胜按约定：逐段状态在 state_order 中名次的序列字典序，再按内部段末
    时隙序列字典序（边界序列不含观测窗末）。每个结点只保留最优签名；
    短序列应再用 enumerate_paths 的规范全枚举复核（analyze 自动做）。

    返回 (segments, log_score)；segments 为 [(状态索引, 相对起点, 长度, 是否删失末段)]；
    无任何合法分段时返回 None。
    """
    K = model.k()
    rank_of = [model.rank[sid] for sid in model.ids]

    def better(a, b):
        if a is None:
            return b
        if b is None:
            return a
        if a[0] != b[0]:
            return a if a[0] > b[0] else b
        return a if (a[1], a[2]) <= (b[1], b[2]) else b

    delta = [[None] * K for _ in range(max(T - 1, 0))]
    for t in range(T - 1):
        for j in range(K):
            best = None
            for d in range(1, min(t + 1, model.dmax(j)) + 1):
                s = t - d + 1
                base = model.log_duration[j][d - 1] + seg(j, s, t)
                if base == NEG:
                    continue
                if s == 0:
                    if model.log_initial[j] != NEG:
                        best = better(best, (base + model.log_initial[j], (rank_of[j],), ()))
                    continue
                for i in range(K):
                    prev = delta[s - 1][i]
                    if prev is None or i == j:
                        continue
                    la = model.log_transition[i][j]
                    if la == NEG:
                        continue
                    best = better(best, (base + la + prev[0],
                                         prev[1] + (rank_of[j],), prev[2] + (s - 1,)))
            delta[t][j] = best
    best = None
    for j in range(K):
        for L in range(1, min(T, model.dmax(j)) + 1):
            s = T - L
            base = model.tail(j, L) + seg(j, s, T - 1)
            if base == NEG:
                continue
            if s == 0:
                if model.log_initial[j] != NEG:
                    best = better(best, (base + model.log_initial[j], (rank_of[j],), ()))
                continue
            for i in range(K):
                prev = delta[s - 1][i]
                if prev is None or i == j:
                    continue
                la = model.log_transition[i][j]
                if la == NEG:
                    continue
                # 末段是删失段，窗口末不写入边界序列；但末段之前的边界（s-1）要写入。
                best = better(best, (base + la + prev[0], prev[1] + (rank_of[j],),
                                     prev[2] + (s - 1,)))
    if best is None:
        return None
    score, rtuple, btuple = best
    if len(rtuple) != len(btuple) + 1:
        raise AssertionError('路径签名内部不一致')
    inv_rank = [0] * K
    for k in range(K):
        inv_rank[rank_of[k]] = k
    starts = [0] + [b + 1 for b in btuple]
    ends = list(btuple) + [T - 1]
    segments = [(inv_rank[r], st, en - st + 1, en == T - 1)
                for r, st, en in zip(rtuple, starts, ends)]
    return segments, score


def map_labels(model, gamma):
    """每个时隙边际后验最大的状态；并列时取 state_order 名次小者。

    与最优整路径是两种口径，二者可以不同，输出必须区分。
    """
    K = model.k()
    labels = []
    for row in gamma:
        best_k = 0
        for k in range(1, K):
            if (row[k], -model.rank[model.ids[k]]) > (row[best_k], -model.rank[model.ids[best_k]]):
                best_k = k
        labels.append(best_k)
    return labels


def score_path(model, e, slots, segments):
    """复算一条路径的 log 联合得分：逐段转移、持续概率、逐观测项之和（math.fsum）。

    segments 为 [(状态身份, 长度)]，长度以时隙计，总和须等于窗长；
    末段按右删失处理（P(D>=已见长度)），其余段用 P(D=d)。
    返回 (log_score, breakdown)；任一项零支持则总分为 -inf，breakdown 标出位置。
    """
    T = len(slots)
    idx = {sid: k for k, sid in enumerate(model.ids)}
    terms = []
    parts = []
    pos = 0
    prev = None
    bad = None
    for r, (sid, length) in enumerate(segments):
        if sid not in idx:
            raise ValueError('未知状态身份: %r' % (sid,))
        k = idx[sid]
        length = integer(length, 1)
        if pos + length > T:
            raise ValueError('分段长度总和超过窗长 %d' % T)
        last = r == len(segments) - 1
        seg_terms = []
        if prev is None:
            t_in = model.log_initial[k]
            in_kind = 'initial'
        else:
            t_in = model.log_transition[prev][k]
            in_kind = 'transition'
        seg_terms.append(t_in)
        if last:
            t_dur = model.tail(k, length)
            dur_kind = 'P(D>=%d)' % length
        else:
            t_dur = model.log_duration[k][length - 1] if length <= model.dmax(k) else NEG
            dur_kind = 'P(D=%d)' % length
        seg_terms.append(t_dur)
        em = []
        for t in range(pos, pos + length):
            w = e[k][t]
            em.append({'slot': slots[t], 'log': w})
            seg_terms.append(w)
        terms.extend(seg_terms)
        parts.append({'index': r, 'state': sid, 'start': slots[pos], 'length': length,
                      'censored': last, 'in_kind': in_kind, 'log_in': t_in,
                      'duration_term': dur_kind, 'log_duration': t_dur,
                      'emissions': em, 'subtotal': math.fsum(seg_terms)})
        if bad is None:
            if t_in == NEG:
                bad = '段 %d 的 %s 项零支持' % (r, in_kind)
            elif t_dur == NEG:
                bad = '段 %d 的持续长度超出配置最大支持或概率为零' % r
            elif any(x['log'] == NEG for x in em):
                bad = '段 %d 含零支持观测' % r
        prev = k
        pos += length
    if pos != T:
        raise ValueError('分段长度总和 %d 不等于窗长 %d' % (pos, T))
    return math.fsum(terms), {'segments': parts, 'total': math.fsum(terms),
                              'zero_support': bad}


class _CapExceeded(Exception):
    pass


def enumerate_paths(model, e, T, cap):
    """枚举全部合法分段路径，返回 [(segments, log_score)]。

    segments 为 [(状态索引, 相对起点, 长度, 是否删失末段)]；得分用 math.fsum
    规范求和，与求值顺序无关，可与 score_path 逐分核对。零得分路径不收录。
    路径数超过 cap 时抛 _CapExceeded。
    """
    K = model.k()
    paths = []
    stack = []
    terms = []

    def rec(s, prev):
        if len(paths) > cap:
            raise _CapExceeded
        for k in range(K):
            if prev is None:
                t_in = model.log_initial[k]
            else:
                t_in = model.log_transition[prev][k]
            if t_in == NEG:
                continue
            for d in range(1, model.dmax(k) + 1):
                end = s + d - 1
                if end > T - 1:
                    break
                last = end == T - 1
                t_dur = model.tail(k, d) if last else model.log_duration[k][d - 1]
                if t_dur == NEG:
                    continue
                em = e[k][s:end + 1]
                terms.append(t_in)
                terms.append(t_dur)
                terms.extend(em)
                stack.append((k, s, d, last))
                if last:
                    sc = math.fsum(terms)
                    if sc != NEG:
                        paths.append((list(stack), sc))
                else:
                    rec(end + 1, k)
                stack.pop()
                del terms[len(terms) - (2 + len(em)):]

    rec(0, None)
    return paths


def enumeration_report(model, paths, T):
    """由全枚举计算似然、状态后验、转移后验与最优路径（规范决胜）。"""
    K = model.k()
    rank_of = [model.rank[sid] for sid in model.ids]
    cands = []
    for segs, sc in paths:
        rt = tuple(rank_of[k] for k, _, _, _ in segs)
        bt = tuple(s + d - 1 for _, s, d, last in segs if not last)
        cands.append((sc, rt, bt, segs))
    logL = logsumexp(c[0] for c in cands)
    glog = [[NEG] * K for _ in range(T)]
    xlog = [[[NEG] * K for _ in range(K)] for _ in range(max(T - 1, 0))]
    for sc, _, _, segs in cands:
        for k, s, d, _ in segs:
            for t in range(s, s + d):
                glog[t][k] = logadd(glog[t][k], sc)
        for a, b in zip(segs, segs[1:]):
            xlog[a[1] + a[2] - 1][a[0]][b[0]] = logadd(xlog[a[1] + a[2] - 1][a[0]][b[0]], sc)
    gamma = [[math.exp(glog[t][k] - logL) if glog[t][k] != NEG else 0.0
              for k in range(K)] for t in range(T)]
    xi = [[[math.exp(xlog[t][i][j] - logL) if xlog[t][i][j] != NEG else 0.0
            for j in range(K)] for i in range(K)] for t in range(T - 1)]
    best = None
    for c in cands:
        if best is None or c[0] > best[0] or (c[0] == best[0] and (c[1], c[2]) < (best[1], best[2])):
            best = c
    top = sorted(cands, key=lambda c: (-c[0], c[1], c[2]))[:5]
    return {'log_likelihood': logL, 'gamma': gamma, 'xi': xi, 'best': best,
            'top': top, 'count': len(cands)}


def _segments_out(model, slots, segs):
    out = []
    for k, s, d, last in segs:
        row = {'state': model.ids[k], 'start': slots[s], 'length': d, 'censored': last}
        if last:
            row['note'] = '末段为右删失段：长度是已见长度，不声称真实完整长度'
        out.append(row)
    return out


def analyze(v, enumerate_cap=200000):
    """完整推断：似然、状态后验、边界转移后验、最优分段路径与逐项核对。"""
    model = Model(v)
    slots, obs, inserted = build_axis(v['observations'])
    T = len(slots)
    K = model.k()
    e = emission_logs(model, obs)
    seg = make_segment_query(e)
    alpha, into = forward(model, seg, T)
    logL = likelihood(model, into, seg, T)
    result = {
        'states': list(model.ids),
        'state_order': [sid for sid, _ in sorted(model.rank.items(), key=lambda kv: kv[1])],
        'axis': {'min_slot': slots[0], 'max_slot': slots[-1], 'length': T,
                 'inserted_missing_slots': inserted},
        'observations': [dict({'slot': s}, **o) for s, o in zip(slots, obs)],
    }
    diagnostics = []
    if inserted:
        diagnostics.append('时隙 %s 缺测，已显式补 missing，分析轴未压缩' % inserted)
    for t in range(T):
        zero = [model.ids[k] for k in range(K) if e[k][t] == NEG]
        if zero:
            diagnostics.append('时隙 %d 状态 %s 发射因子零支持' % (slots[t], zero))
    if logL == NEG:
        zero_slots = [slots[t] for t in range(T) if all(e[k][t] == NEG for k in range(K))]
        result.update({
            'status': 'uninterpretable',
            'reason': '没有任何合法分段能以正概率解释资料（零支持）；不出具归一化失败的后验表',
            'zero_support_slots': zero_slots,
            'log_likelihood': None,
            'state_posterior': None,
            'transition_posterior': None,
            'expected_transitions': None,
            'best_path': None,
            'top_paths': None,
            'diagnostics': diagnostics,
        })
        return result
    beta, rest = backward(model, seg, T)
    gamma = posterior_state(model, into, beta, seg, T, logL)
    xi = posterior_transition(model, alpha, rest, T, logL)
    vit = viterbi(model, seg, T)
    labels = map_labels(model, gamma)

    enum = None
    if enumerate_cap:
        if T > 900:
            sys.setrecursionlimit(max(sys.getrecursionlimit(), T * 4 + 100))
        try:
            enum = enumeration_report(model, enumerate_paths(model, e, T, enumerate_cap), T)
        except _CapExceeded:
            diagnostics.append('合法分段数超过枚举上限 %d，未做全枚举核对' % enumerate_cap)

    dp_segments = vit[0] if vit else None
    if enum is not None and enum['best'] is not None:
        best_segments = enum['best'][3]
        selection = 'enumeration'
        dp_agrees = dp_segments == best_segments
    else:
        best_segments = dp_segments
        selection = 'dp'
        dp_agrees = None
    total, breakdown = score_path(model, e, slots,
                                  [(model.ids[k], d) for k, _, d, _ in best_segments])
    path_state = [0] * T
    for k, s, d, _ in best_segments:
        for t in range(s, s + d):
            path_state[t] = k

    state_posterior = []
    for t in range(T):
        probs = {model.ids[k]: gamma[t][k] for k in
                 sorted(range(K), key=lambda k: model.rank[model.ids[k]])}
        mk = labels[t]
        state_posterior.append({
            'slot': slots[t],
            'probability': probs,
            'map_state': model.ids[mk],
            'map_probability': gamma[t][mk],
            'path_state': model.ids[path_state[t]],
            'map_differs_from_path': mk != path_state[t],
        })
    if any(r['map_differs_from_path'] for r in state_posterior):
        diagnostics.append('存在边际后验最大标注与最优整路径不一致的时隙：两者口径不同，属正常输出')

    transition_posterior = []
    expected = 0.0
    for t in range(T - 1):
        pairs = {}
        any_p = 0.0
        for i in sorted(range(K), key=lambda k: model.rank[model.ids[k]]):
            for j in sorted(range(K), key=lambda k: model.rank[model.ids[k]]):
                if i == j:
                    continue
                p = xi[t][i][j]
                pairs['%s>%s' % (model.ids[i], model.ids[j])] = p
                any_p += p
        expected += any_p
        transition_posterior.append({'after_slot': slots[t], 'any_transition': any_p,
                                     'pair': pairs})

    top_paths = None
    if enum is not None:
        top_paths = [{'segments': _segments_out(model, slots, c[3]),
                      'log_score': c[0],
                      'posterior_probability': math.exp(c[0] - logL)} for c in enum['top']]

    checks = {'state_posterior_rows_sum_to_one':
              all(abs(sum(row) - 1) <= 1e-6 for row in gamma)}
    if enum is not None:
        gdiff = max(abs(enum['gamma'][t][k] - gamma[t][k])
                    for t in range(T) for k in range(K))
        xdiff = max([abs(enum['xi'][t][i][j] - xi[t][i][j])
                     for t in range(T - 1) for i in range(K) for j in range(K)] or [0.0])
        checks['enumeration'] = {
            'path_count': enum['count'],
            'log_likelihood_abs_diff': abs(enum['log_likelihood'] - logL),
            'state_posterior_max_abs_diff': gdiff,
            'transition_posterior_max_abs_diff': xdiff,
            'dp_path_matches_enumeration': dp_agrees,
        }

    result.update({
        'status': 'ok',
        'log_likelihood': logL,
        'state_posterior': state_posterior,
        'transition_posterior': transition_posterior,
        'expected_transitions': expected,
        'best_path': {
            'segments': _segments_out(model, slots, best_segments),
            'log_score': total,
            'posterior_probability': math.exp(total - logL),
            'tie_break': '逐段状态state_order名次序列字典序，再内部段末时隙序列字典序（不含观测窗末）',
            'selection': selection,
            'score_breakdown': breakdown,
        },
        'top_paths': top_paths,
        'checks': checks,
        'diagnostics': diagnostics,
        'notes': [
            '末段是尚未结束的右删失段，持续因子为 P(D>=已见长度)，不声称在窗口末结束',
            '边际后验最大标注与最优整路径是两种口径，可以不同，本输出分别给出',
            '误差条件独立只是本分析的设定，不推断成设备真实机理',
        ],
    })
    return result


def score_command(v, segments):
    """复算一条给定分段路径的得分，并给出相对资料似然的后验概率。"""
    model = Model(v)
    slots, obs, _ = build_axis(v['observations'])
    e = emission_logs(model, obs)
    seg = make_segment_query(e)
    alpha, into = forward(model, seg, len(slots))
    logL = likelihood(model, into, seg, len(slots))
    spec = []
    for s in segments:
        if not isinstance(s, dict) or 'state' not in s or 'length' not in s:
            raise ValueError('每段需要 {"state": 身份, "length": 正整数}')
        spec.append((s['state'], s['length']))
    total, breakdown = score_path(model, e, slots, spec)
    out = {'log_likelihood': logL, 'log_score': total, 'breakdown': breakdown}
    if logL != NEG and total != NEG:
        out['posterior_probability'] = math.exp(total - logL)
    else:
        out['posterior_probability'] = None
    return out
