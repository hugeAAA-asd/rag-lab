"""路由加分扫描：bonus 给多少合适

上一轮结论：路由在极短口语档命中率约 12/13，但 @1 只有 0.308——
因为排序权全在 cross-encoder 手里，而它在这档是瞎的
（「边想边干是啥意思」被判给 (开头) 块 0.271，路由明明选中 4.2 ReAct）。

现在让路由参与排序：命中章节的块精排分 +bonus。
bonus 太大 → 路由独裁，精排失去「同一章节内挑哪块」的能力；
bonus 太小 → 压不住精排的误判。扫一遍找平衡点。

路由结果缓存到 routes.json，避免重复调 LLM。
"""
import json
import os
import time

from exp.router_accuracy import route, search_routed, rank_of
from eval_oral import CASES
from llm import load_key

CACHE = "routes.json"


def load():
    if os.path.exists(CACHE):
        return json.load(open(CACHE, encoding="utf-8"))
    return {}


def save(d):
    json.dump(d, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main(bonuses):
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    cache = load()
    routes = {}
    for name, want, q in CASES:
        if q in cache:
            routes[q] = cache[q]
        else:
            picks, raw = route(q, key)
            cache[q] = picks
            save(cache)
            print("  路由 %s → %s" % (q, picks))
        routes[q] = cache[q]
        print("路由 %-16s → %s" % (q, " / ".join(p.replace("## ", "")[:18] for p in routes[q])))

    # 先算路由自身的准确率：正确章节有没有出现在 LLM 挑的 3 个里
    hit_route = 0
    hit_route1 = 0
    for name, want, q in CASES:
        picks = routes[q]
        if any(p.startswith(want.strip()) for p in picks):
            hit_route += 1
        if picks and picks[0].startswith(want.strip()):
            hit_route1 += 1
    N = len(CASES)
    print("\n路由自身准确率：@3 = %2d/%d = %.3f   第1名即正确 = %2d/%d = %.3f"
          % (hit_route, N, hit_route / N, hit_route1, N, hit_route1 / N))
    print("（这是上限：再好的排序也超不过它）\n")

    print("-" * 70)
    print("%-8s %14s %14s" % ("bonus", "@1", "@3"))
    for b in bonuses:
        h1 = h3 = 0
        t0 = time.time()
        for name, want, q in CASES:
            hits = search_routed(q, routes[q], bonus=b)
            r = rank_of(hits, want)
            if r == 1:
                h1 += 1
            if r:
                h3 += 1
        print("%-8.2f %6d/%-2d %.3f %6d/%-2d %.3f   (%.0fs)"
              % (b, h1, N, h1 / N, h3, N, h3 / N, time.time() - t0))
    print("-" * 70)
    print("对照：纯原问 0.231/0.538   查询改写 0.385/0.615   路由无加分 0.308/0.692")


if __name__ == "__main__":
    import sys
    args = [float(x) for x in sys.argv[1:]] or [0.3, 0.6, 1.0]
    main(args)
