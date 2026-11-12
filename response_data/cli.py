"""命令行：inspect 显示原始资料摘要；infer 输出显式持续时间推断；score 复算单条路径得分。"""
import argparse
import json
import math
from pathlib import Path
from .io import load
from .domain import summarize
from . import hsmm

def _jsonable(x):
    """log 域的 -inf（零支持）在 JSON 中表示为 null。"""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x

def _emit(obj):
    print(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False))

def main():
    p = argparse.ArgumentParser(prog='response_data.cli')
    sub = p.add_subparsers(dest='command', required=True)
    pi = sub.add_parser('inspect', help='原始资料概况')
    pi.add_argument('file')
    pf = sub.add_parser('infer', help='显式持续时间前向/后向推断')
    pf.add_argument('file')
    pf.add_argument('--enumerate-cap', type=int, default=200000, help='全枚举核对的路径数上限')
    pf.add_argument('--no-enumerate', action='store_true', help='不做全枚举核对')
    ps = sub.add_parser('score', help='复算一条分段路径的得分')
    ps.add_argument('file')
    ps.add_argument('path', help='分段 JSON，如 [{"state":"dry","length":2},...]，或 @文件')
    a = p.parse_args()
    v, digest = load(a.file)
    if a.command == 'inspect':
        _emit({'source_sha256': digest, 'summary': summarize(v)})
        return
    if a.command == 'infer':
        cap = 0 if a.no_enumerate else a.enumerate_cap
        r = hsmm.analyze(v, enumerate_cap=cap)
        r['source_sha256'] = digest
        _emit(r)
        return
    if a.command == 'score':
        spec = a.path
        if spec.startswith('@'):
            spec = Path(spec[1:]).read_text(encoding='utf8')
        r = hsmm.score_command(v, json.loads(spec))
        r['source_sha256'] = digest
        _emit(r)

if __name__ == '__main__':
    main()
