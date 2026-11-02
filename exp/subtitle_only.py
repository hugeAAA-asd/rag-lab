"""最小改动对照：生产链路原样 + 精排文本加小节标题

前面几组都在动「候选池」（head / sub / sub1 / sub2），结果两边偏好相反：
  43 题书面问法：head + bonus 0.907，去掉 bonus 掉到 0.767
  20 题含口语：  head + bonus 0.350，sub1 无 bonus 0.650

这组只动一处——喂给精排的文本多一行三级小节标题，候选池和 bonus 全不变。
如果两边都不掉，这就是今天唯一稳赚的改动。

用法：python exp_subtitle_only.py [43|holdout] [limit]
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys
import time

from eval import chunks, SUB_MAP, rank_of, RERANK_WITH_SUB
from rewrite import search_multi, rewrite, load_cache, save_cache
from route import route
from llm import load_key
from metrics import mrr
from answer import ROUTE_BONUS


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "43"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if which == "43":
        from eval import CASES, SUBS
        items = [(n, q[:12], q, a[0], s)
                 for n, ((q, a, k), s) in enumerate(zip(CASES, SUBS), 1)]
    else:
        from eval_holdout import CASES
        items = [(n, c[0], c[2], c[3], c[4]) for n, c in enumerate(CASES, 1)]
    if limit:
        items = items[:limit]

    key = load_key()
    cache = load_cache()
    rows = []
    t0 = time.time()
    print("RERANK_WITH_SUB =", RERANK_WITH_SUB, "| 题数", len(items))

    for n, name, q, sec, subs in items:
        picks = route(q, key)[0]
        if q in cache:
            rews = cache[q]
        else:
            rews = rewrite(q, key)
            cache[q] = rews
            save_cache(cache)
        hits = search_multi(q, rews, k=3, picks=picks,
                            bonus=(ROUTE_BONUS if picks else 0.0), route_pick="head")
        r1 = rank_of(hits, [sec])
        r2 = rank_of(hits, [sec], subs)
        rows.append({"n": n, "name": name, "q": q, "sec": r1, "sub": r2})
        print("%2d %-14s 节%-4s 小%-4s" % (
            n, name, "第%d" % r1 if r1 else "✗", "第%d" % r2 if r2 else "✗"))

    N = len(rows)
    secs = [r["sec"] for r in rows]
    subs_ = [r["sub"] for r in rows]
    print("\n=%d 题 %.0fs" % (N, time.time() - t0))
    print("节@1 %.3f 节@3 %.3f | 小节@1 %.3f 小节@3 %.3f 小节MRR %.3f" % (
        sum(1 for x in secs if x == 1) / N, sum(1 for x in secs if x) / N,
        sum(1 for x in subs_ if x == 1) / N, sum(1 for x in subs_ if x) / N,
        mrr(subs_)))

    tag = which + ("_subtitle" if RERANK_WITH_SUB else "_plain")
    json.dump(rows, open("exp_headbonus_%s.json" % tag, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("明细存 exp_headbonus_%s.json" % tag)


if __name__ == "__main__":
    main()
