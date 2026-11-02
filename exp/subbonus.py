"""小节加分的分级策略：为什么「平均加分」等于没加

第一版给三个选中小节同样的分，结果 8.3 那题完全没动：
subs = [8.3.4, 8.3.2, 8.3.1]，三个各 +0.8，而 8.3.1 的块本来精排分就最高，
一起抬轿子之后它还是最高。选 3 个小节＝章内大半块都加分＝等于没加。

所以只有「首位小节」带信息量，后两位是模型的保险丝（它自己也拿不准）。
策略全按这个前提试：
  none      不补池不加工（对照）
  pool      只补池、不加分
  first_x   只给首位小节 +x
  tier      按顺序分档 [首位, 次位, 三位]
  conf      首位 +x，但首位匹配度低于阈值就不加
    
conf 那档是给「模型也不确定」留退路：不确定就别插嘴，交回精排。
"""
import json
import sys

import numpy as np

import eval as E
from eval import chunks, rank_of
from metrics import mrr
from answer import ROUTE_BONUS, CONF_GATE

strats = [("none", "none"), ("pool", "pool"),
          ("first0.3", ("first", 0.3)), ("first0.5", ("first", 0.5)),
          ("first0.8", ("first", 0.8)), ("first1.2", ("first", 1.2)),
          ("tier.8/.2", ("tier", [0.8, 0.2, 0.0])),
          ("tier1.0/.3", ("tier", [1.0, 0.3, 0.0])),
          ("conf60", ("gate", 0.8, 60)), ("conf75", ("gate", 0.8, 75)),
          ("conf85", ("gate", 0.8, 85))]


def weights(sub, confs):
    """返回 {小节号: 加分}"""
    kind = sub[0]
    if kind == "none":
        return {}
    if kind == "pool":
        return {}
    if kind == "first":
        return {subs[0]: sub[1]} if subs else {}
    if kind == "tier":
        return {no: sub[1][i] for i, no in enumerate(subs) if i < len(sub[1])}
    if kind == "gate":
        _, x, th = sub
        if not subs:
            return {}
        # 首位小节自己的匹配度，拿不到就当没把握（宁可不加）
        return {subs[0]: x} if confs and confs[0] >= th else {}
    return {}


def rerank(row, sw, k=3):
    cand, raw = row["cand"], np.array(row["raw"], float)
    rs = raw.copy()
    wt = {}
    for pos, p in enumerate(row["picks"]):
        x = ROUTE_BONUS[pos] if pos < len(ROUTE_BONUS) else ROUTE_BONUS[-1]
        if CONF_GATE and (pos >= len(row["confs"]) or row["confs"][pos] < CONF_GATE):
            x = 0.0
        if x:
            wt[p.strip()] = x
    for j, i in enumerate(cand):
        rs[j] += wt.get(chunks[i][1].strip(), 0)
    if sw:
        for j, i in enumerate(cand):
            hit = E.SUB_MAP.get(i, set()) & set(sw)
            if hit:
                rs[j] += max(sw[x] for x in hit)
    out, seen = [], set()
    for j in np.argsort(-rs):
        t = chunks[cand[int(j)]][1]
        if t in seen:
            continue
        seen.add(t)
        out.append((int(cand[int(j)]), float(raw[int(j)])))
        if len(out) == k:
            break
    return out


def main(tag):
    rows = json.load(open("sub_pools_%s.json" % tag, encoding="utf-8"))
    N = len(rows)
    print("%d 题" % N)
    print("  %-12s %-7s %-7s %-7s" % ("策略", "小@1", "小@3", "小MRR"))
    for name, spec in strats:
        s1 = s3 = 0
        rr = []
        for r in rows:
            global subs
            subs = r["subs"]
            sw = weights(spec, r.get("sub_confs") or [])
            hits = rerank(r, sw)
            rk = rank_of(hits, [r["sec"]], r["subs_want"])
            s1 += rk == 1
            s3 += rk is not None
            rr.append(rk or 0)
        print("  %-12s %.3f   %.3f   %.3f" % (name, s1 / N, s3 / N, mrr(rr)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h3")
