"""全量小节直选之后：加分怎么打

picks 是从小节反推的，所以章节 bonus 和小节 bonus 现在来自同一个信号、
只是粒度不同。要点在于：章节 bonus 会把「同章所有块」一起抬，
小节 bonus 只抬那一段——如果只想要后者，前者就该关掉。
这里把三种组合摆在一起看：
  sec     只加章节（等价于「拿小节当章节用」，粗粒度）
  sub     只加小节（精确到段，不抬整章）
  both    两级都加
每档再试「只给首位」和「按顺序分档」。
"""
import json
import sys

import numpy as np

import eval as E
from eval import chunks, rank_of
from metrics import mrr

def build(row, sec_bonus, sub_w, k=3):
    cand, raw = row["cand"], np.array(row["raw"], float)
    rs = raw.copy()
    if sec_bonus:
        wt = {}
        for pos, p in enumerate(row["picks"]):
            x = sec_bonus[pos] if pos < len(sec_bonus) else sec_bonus[-1]
            if x:
                wt[p.strip()] = x
        for j, i in enumerate(cand):
            rs[j] += wt.get(chunks[i][1].strip(), 0)
    if sub_w:
        for j, i in enumerate(cand):
            hit = E.SUB_MAP.get(i, set()) & set(sub_w)
            if hit:
                rs[j] += max(sub_w[x] for x in hit)
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


STRATS = [
    ("none",        None,        lambda r: {}),
    ("sec.6/.2",    [0.6, 0.2],  lambda r: {}),
    ("sub.8first",  None,        lambda r: {r["subs"][0]: 0.8} if r["subs"] else {}),
    ("sec.6+sub.8", [0.6, 0.2],  lambda r: {r["subs"][0]: 0.8} if r["subs"] else {}),
    ("sec.6+sub.4", [0.6, 0.2],  lambda r: {r["subs"][0]: 0.4} if r["subs"] else {}),
    ("sec.4+sub.8", [0.4, 0.15], lambda r: {r["subs"][0]: 0.8} if r["subs"] else {}),
    ("sub.8/.2/0",  None,        lambda r: {n: [0.8, 0.2, 0.0][i]
                                            for i, n in enumerate(r["subs"][:3])}),
    ("sec.6 sub tier", [0.6, 0.2], lambda r: {n: [0.8, 0.2, 0.0][i]
                                              for i, n in enumerate(r["subs"][:3])}),
]


def main(tag):
    rows = json.load(open("allsub_pools_%s.json" % tag, encoding="utf-8"))
    N = len(rows)
    print("%d 题" % N)
    print("  %-16s %-7s %-7s %-7s" % ("策略", "小@1", "小@3", "小MRR"))
    for name, sb, fw in STRATS:
        s1 = s3 = 0
        rr = []
        for r in rows:
            hits = build(r, sb, fw(r))
            rk = rank_of(hits, [r["sec"]], r["subs_want"])
            s1 += rk == 1
            s3 += rk is not None
            rr.append(rk or 0)
        print("  %-16s %.3f   %.3f   %.3f" % (name, s1 / N, s3 / N, mrr(rr)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h3")
