"""单题耗时拆解：先量，再砍

跑一次完整链路，把每一段单独计时。不量就优化 = 凭直觉改参数，
这几天的教训已经够多了。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import time

import numpy as np

from embed import embed_query
from eval import vecs, reranker, chunks
from llm import load_key
from rewrite import rewrite, _collect
from route import route

Q = sys.argv[1] if len(sys.argv) > 1 else "边想边干是啥意思"


def t(label, fn):
    t0 = time.time()
    out = fn()
    print("  %-22s %6.2fs" % (label, time.time() - t0))
    return out


key = load_key()
print("问题：%s\n" % Q)

rews = t("① 改写（LLM，已缓存）", lambda: rewrite(Q, key))
picks = t("② 路由（LLM，已缓存）", lambda: route(Q, key)[0])

qv = t("③ 原问 embed", lambda: embed_query(Q))
t("④ 改写 ×3 embed", lambda: [embed_query(r) for r in rews])

scores = vecs @ qv
main = _collect(scores, 15, 2)
seen = set(main)
extra = []
for r in rews:
    got = _collect(vecs @ embed_query(r), 5, 1, skip_idx=seen)
    extra += got
    seen |= set(got)
for title in picks:
    idxs = [i for i, c in enumerate(chunks) if c[1].strip() == title.strip()]
    if not idxs:
        continue
    chosen = idxs[:2]
    best = max(idxs, key=lambda i: scores[i])
    if best not in chosen:
        chosen.append(best)
    for i in chosen:
        if i not in seen:
            seen.add(i)
            extra.append(i)

cand = main + extra
print("\n  候选块数：%d（主 %d + 补 %d）" % (len(cand), len(main), len(extra)))

pairs = [(Q, chunks[i][1] + "\n" + chunks[i][2]) for i in cand]
print("  候选平均长度：%d 字，最长 %d 字" % (
    sum(len(p[1]) for p in pairs) // len(pairs), max(len(p[1]) for p in pairs)))

t("⑤ 精排 %d 对" % len(pairs), lambda: reranker.predict(pairs, batch_size=16))

# 对照：只精排 30 对要多久
t("⑥ 精排 30 对（对照）", lambda: reranker.predict(pairs[:30], batch_size=16))
t("⑦ 精排 15 对（对照）", lambda: reranker.predict(pairs[:15], batch_size=16))
