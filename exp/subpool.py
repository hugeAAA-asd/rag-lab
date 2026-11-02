"""上限实验：假设「章节选对 + 小节选对」，候选池补齐后能救几题

问的是同一个问题：小节级选块值不值得做。
做法是给它开挂——直接告诉它正确小节的号，把那些块全塞进候选池，
再看精排能不能把它们顶进前 3。开挂都救不回来，就别指望真做能救。

三档：
  A 现状        原始池，不加分
  B 章节开挂    池子不变，给正确章节 0.6 加分（模拟路由选对且排第一）
  C 小节也开挂  B + 正确小节的所有块进池
B→C 的差，就是「小节选块」这个动作本身能贡献的上限量。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys
import numpy as np

import eval as E
from eval import chunks, rank_of, rerank_text, reranker

BONUS = 0.6

rows = json.load(open("eval_holdout3_pools.json", encoding="utf-8"))
res = {x["q"]: x for x in json.load(open("eval_holdout3_result.json", encoding="utf-8"))}


def order(cand, raw, w_title, k=3):
    rs = np.array(raw, dtype=float).copy()
    for j, i in enumerate(cand):
        x = w_title.get(chunks[i][1].strip())
        if x:
            rs[j] += x
    out, seen = [], set()
    for j in np.argsort(-rs):
        t = chunks[cand[int(j)]][1]
        if t in seen:
            continue
        seen.add(t)
        out.append(int(cand[int(j)]))
        if len(out) == k:
            break
    return out


def subblocks(sec, subs):
    """该小节的所有块（含续块），按全库顺序"""
    want = [i for i in range(len(chunks))
            if chunks[i][1].strip().startswith(sec.strip())
            and (E.SUB_MAP.get(i, set()) & set(subs))]
    return want


from route import parse_conf
from answer import ROUTE_BONUS, CONF_GATE

s0 = s1 = sa = sb = sc = 0
s3 = {"0": 0, "A": 0, "B": 0, "C": 0}
detail = []
for r in rows:
    q = r["q"]
    meta = res[q]
    sec, subs = meta["sec"], meta["subs"]
    cand, raw = list(r["cand"]), list(r["raw"])

    # 生产口径：按路由给出的 picks 顺序加分，置信度不足的不加
    confs = parse_conf(r.get("route_raw", ""), len(r["picks"]))
    w0 = {}
    for pos, t in enumerate(r["picks"]):
        x = ROUTE_BONUS[pos] if pos < len(ROUTE_BONUS) else ROUTE_BONUS[-1]
        if CONF_GATE and (pos >= len(confs) or confs[pos] < CONF_GATE):
            x = 0.0
        if x:
            w0[t.strip()] = x
    P0 = order(cand, raw, w0)
    A = order(cand, raw, {})
    B = order(cand, raw, {sec.strip(): BONUS})

    add = [i for i in subblocks(sec, subs) if i not in cand]
    cand2 = cand + add
    raw2 = list(raw) + [float(x) for x in
                        reranker.predict([(q, rerank_text(i)) for i in add], batch_size=16)] if add else list(raw)
    C = order(cand2, raw2, {sec.strip(): BONUS})

    def at1(x):
        return rank_of([(i, 0) for i in x], [sec], subs) == 1

    def at3(x):
        return rank_of([(i, 0) for i in x], [sec], subs) is not None

    o0, okA, okB, okC = at1(P0), at1(A), at1(B), at1(C)
    s0 += o0; s1 += okA; sa += okB; sb += okC
    s3["0"] += at3(P0); s3["A"] += at3(A); s3["B"] += at3(B); s3["C"] += at3(C)
    detail.append((q, sec, subs, o0, okA, okB, okC, len(add), C))
    print("%s%s%s%s 补%-2d块  %s %s" % ("√" if o0 else "×", "√" if okA else "×",
          "√" if okB else "×", "√" if okC else "×", len(add), q[:32], subs))

N = len(rows)
print("\n小节@1  生产配置 %.3f  无加分 %.3f  B章节开挂 %.3f  C小节也开挂 %.3f" %
      (s0 / N, s1 / N, sa / N, sb / N))
print("小节@3  生产配置 %.3f  无加分 %.3f  B章节开挂 %.3f  C小节也开挂 %.3f" %
      (s3["0"] / N, s3["A"] / N, s3["B"] / N, s3["C"] / N))
print("\nC 仍失败的题：")
for q, sec, subs, o0, okA, okB, okC, add, C in detail:
    if not okC:
        print("  %s %s  %s" % (sec.strip(), subs, q[:40]))
        print("     补了 %d 块，前3是：%s" % (add, [chunks[i][1][:22].strip() for i in C]))
