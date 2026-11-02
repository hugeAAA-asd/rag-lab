"""接着 exp_subpool.py：补齐候选后为什么还进不了第 1

C 档（正确小节全进池 + 章节 0.6）在 @3 涨到 0.850、@1 只有 0.650。
说明块已经在池里、也被精排看见了，只是同一章里别的块比它分高——
按节去重时，那一章的代表块由章内最高分决定，小节选得再准也没用。

所以这里测第二层：让「小节」也参与打分，而不只是让块进池。
  D 章节 0.6 + 正确小节 0.6
  E 只给正确小节 0.6（完全不信章节）
E 若比 D 好，说明该把加分的粒度从小节开始，章节只是用来圈定范围。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import numpy as np

import eval as E
from eval import chunks, rank_of, rerank_text, reranker
from route import parse_conf
from answer import ROUTE_BONUS, CONF_GATE

B = 0.6
rows = json.load(open("eval_holdout3_pools.json", encoding="utf-8"))
res = {x["q"]: x for x in json.load(open("eval_holdout3_result.json", encoding="utf-8"))}


def order(cand, raw, w_idx, w_title, k=3):
    rs = np.array(raw, dtype=float).copy()
    for j, i in enumerate(cand):
        rs[j] += w_idx.get(i, 0) + w_title.get(chunks[i][1].strip(), 0)
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


def prod_w(r):
    confs = parse_conf(r.get("route_raw", ""), len(r["picks"]))
    w = {}
    for pos, t in enumerate(r["picks"]):
        x = ROUTE_BONUS[pos] if pos < len(ROUTE_BONUS) else ROUTE_BONUS[-1]
        if CONF_GATE and (pos >= len(confs) or confs[pos] < CONF_GATE):
            x = 0.0
        if x:
            w[t.strip()] = x
    return w


score = {k: 0 for k in ("P0", "C", "D", "E")}
score3 = {k: 0 for k in ("P0", "C", "D", "E")}
for r in rows:
    q = r["q"]
    meta = res[q]
    sec, subs = meta["sec"], meta["subs"]
    cand, raw = list(r["cand"]), list(r["raw"])

    tgt = [i for i in range(len(chunks))
           if chunks[i][1].strip().startswith(sec.strip())
           and (E.SUB_MAP.get(i, set()) & set(subs))]
    add = [i for i in tgt if i not in cand]
    cand2 = cand + add
    raw2 = list(raw) + ([float(x) for x in reranker.predict(
        [(q, rerank_text(i)) for i in add], batch_size=16)] if add else [])

    P0 = order(cand, raw, {}, prod_w(r))
    Cc = order(cand2, raw2, {}, {sec.strip(): B})
    Dd = order(cand2, raw2, {i: B for i in tgt}, {sec.strip(): B})
    Ee = order(cand2, raw2, {i: B for i in tgt}, {})

    def rk(x):
        return rank_of([(i, 0) for i in x], [sec], subs)

    for k, x in (("P0", P0), ("C", Cc), ("D", Dd), ("E", Ee)):
        score[k] += rk(x) == 1
        score3[k] += rk(x) is not None
    print("P0=%s C=%s D=%s E=%s  %s %s" % (rk(P0), rk(Cc), rk(Dd), rk(Ee), subs, q[:30]))

N = len(rows)
print("\n小节@1  " + "  ".join("%s %.3f" % (k, score[k] / N) for k in score))
print("小节@3  " + "  ".join("%s %.3f" % (k, score3[k] / N) for k in score3))
