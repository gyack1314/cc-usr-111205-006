"""核对冻结状态参数与观测种类。"""
from collections import Counter
from .io import unique,number,positive,integer

def summarize(v):
    states=unique(v['states'],'id')
    for s in v['states']:
        number(s['mean']);positive(s['sd'])
        if not s['duration'] or any(number(p)<0 for p in s['duration']) or abs(sum(s['duration'])-1)>1e-12:raise ValueError('持续概率无效')
    if len(v['transition'])!=len(states) or any(len(r)!=len(states) for r in v['transition']):raise ValueError('转移形状无效')
    for i,row in enumerate(v['transition']):
        if row[i]!=0 or any(number(p)<0 for p in row) or abs(sum(row)-1)>1e-12:raise ValueError('转移概率无效')
    seen=set();counts=Counter()
    for r in v['observations']:
        slot=integer(r['slot'])
        if slot in seen or r['kind'] not in ('exact','interval','right','missing'):raise ValueError('观测身份或类别无效')
        seen.add(slot);counts[r['kind']]+=1
    return {'states':len(states),'observed_slots':len(seen),'kinds':dict(sorted(counts.items()))}
