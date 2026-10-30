"""组合方案全量验证：路由加分会不会把已经对的题弄坏

昨天两次教训：单题现象改全局规则、一档有效不等于全局有效。
路由在极短口语档 +53.8 个点，但必须在绕弯档 43 题上确认不掉分才能接进主链路。

两组对照：
  A 现有方案：原问 + 改写补漏，bonus=0
  B 组合方案：原问 + 改写补漏 + 路由补漏，路由章节 bonus=1.0

改写结果用 rewrites.json 缓存，路由结果用 routes.json 缓存，避免重复烧 token。
"""
import json
import os
import time

from eval import CASES, chunks
from rewrite import search_multi, rewrite, load_cache, save_cache
from exp_route import route
from llm import load_key

ROUTE_CACHE = "routes.json"


def load_routes():
    if os.path.exists(ROUTE_CACHE):
        return json.load(open(ROUTE_CACHE, encoding="utf-8"))
    return {}


def save_routes(d):
    json.dump(d, open(ROUTE_CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def rank_of(hits, answers):
    for r, (idx, s) in enumerate(hits, 1):
        if any(chunks[idx][1].startswith(a) for a in answers):
            return r
    return None


# 默认两组对照；命令行传数字则只跑指定 bonus 的那一组
GROUPS = [("A 改写", False, 0.0), ("B 改写+路由", True, 1.0)]


def main():
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    # ① 路由缓存
    rc = load_routes()
    routes = {}
    for q, _a, _k in CASES:
        if q not in rc:
            picks, raw = route(q, key)
            rc[q] = picks
            save_routes(rc)
            print("  路由 %s → %s" % (q[:30], picks[:2]))
        routes[q] = rc[q]

    # ② 改写缓存
    wc = load_cache()
    rews = {}
    for q, _a, _k in CASES:
        if q not in wc:
            wc[q] = rewrite(q, key)
            save_cache(wc)
        rews[q] = wc[q]

    N = len(CASES)
    print("\n两组各 %d 题，开始\n" % N)

    print("%-16s %14s %14s %8s" % ("方案", "@1", "@3", "耗时"))
    print("-" * 58)
    detail = {}
    for tag, use_picks, bonus in GROUPS:
        h1 = h3 = 0
        ranks = []
        t0 = time.time()
        for q, answers, kind in CASES:
            hits = search_multi(q, rews[q],
                                picks=routes[q] if use_picks else None,
                                bonus=bonus)
            r = rank_of(hits, answers)
            ranks.append((q, r))
            if r == 1:
                h1 += 1
            if r:
                h3 += 1
        dt = time.time() - t0
        detail[tag] = ranks
        print("%-16s %6d/%-2d %.3f %6d/%-2d %.3f %7.0fs"
              % (tag, h1, N, h1 / N, h3, N, h3 / N, dt))
    print("-" * 58)
    print("基线（无改写无路由）@1 0.651  @3 0.907")

    # ③ 谁被弄坏了（只在 A/B 两组都跑过时才有得比）
    if "A 改写" not in detail or "B 改写+路由" not in detail:
        return
    a, b = detail["A 改写"], detail["B 改写+路由"]
    print("\nB 相对 A 的变化：")
    for (q, ra), (_, rb) in zip(a, b):
        if ra and not rb:
            print("  ✗ 弄坏  %s    A=%s → B=%s" % (q[:40], "第%d名" % ra, "未进前3"))
        elif rb and not ra:
            print("  ✓ 救回  %s    A=%s → B=%s" % (q[:40], "未进前3", "第%d名" % rb))
        elif ra != rb:
            print("  ~ 改善  %s    A=%s → B=%s" % (q[:40], "第%d名" % ra, "第%d名" % rb))


if __name__ == "__main__":
    import sys
    # 默认跑 A/B 两组对照；传一个数字就只跑「路由 + 该 bonus」那一组
    args = [float(x) for x in sys.argv[1:]]
    if args:
        GROUPS = [("路由 bonus=%.1f" % b, True, b) for b in args]
    main()
