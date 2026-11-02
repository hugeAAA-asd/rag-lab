"""路由注入方式对照：head（现状）vs sub（按小节覆盖）

背景
----
20 道新题把问题照出来了：节级 @1 从 0.50 涨到 0.60，小节级 @1 从 0.45 *跌*到 0.35。
7 道失败题是同一个病——路由选对了章节，但塞进候选池的 3 个块碰不到答案所在的小节：
    10.2 有 5 小节 31 块，旧规则只碰 10.2.1 那一片，而答案在 10.2.3
    11.5 问评估指标，塞进来的是 11.5.3/11.5.4，而 11.5.1 才是答案

这是候选缺失，不是排序错。所以先改「塞哪些块」，再谈权重。

三组：
  A head + 分级 bonus   现状（基线数字从 eval_holdout_result.json 读，不重跑）
  B sub  + 无 bonus     只看「按小节覆盖」本身够不够
  C sub  + 分级 bonus   两个一起上

用法：python exp_route_pick.py [holdout|43] [limit]
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys
import time

from eval import chunks, SUB_MAP, rank_of
from rewrite import search_multi, rewrite, load_cache, save_cache
from route import route
from llm import load_key
from metrics import mrr
from answer import ROUTE_BONUS

# 注入方式与精排文本开关都从环境变量来，方便一次跑一种组合
PICK = os.environ.get("ROUTE_PICK", "sub")
TAG = "%s%s" % (PICK, "_subtitle" if os.environ.get("RERANK_WITH_SUB") == "1" else "")


def items_holdout():
    from eval_holdout import CASES
    return [(n, c[0], c[1], c[2], c[3], c[4]) for n, c in enumerate(CASES, 1)]


def items_43():
    from eval import CASES, SUBS
    return [(n, q[:12], kind, q, ans[0], subs)
            for n, ((q, ans, kind), subs) in enumerate(zip(CASES, SUBS), 1)]


def base_numbers(rows, which):
    """A 组（head + 分级 bonus）从上次结果里读，不重跑精排"""
    if which != "holdout":
        return None
    try:
        old = {r["n"]: r for r in json.load(open("eval_holdout_result.json",
                                                 encoding="utf-8"))}
    except FileNotFoundError:
        return None
    out = {}
    for r in rows:
        o = old.get(r["n"])
        if o:
            out[r["n"]] = (o["all_sec"], o["all_sub"])
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "holdout"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    only_b = len(sys.argv) > 3 and sys.argv[3] == "B"   # 只跑 B 组，省一半精排时间

    key = load_key()
    items = items_holdout() if which == "holdout" else items_43()
    if limit:
        items = items[:limit]
    cache = load_cache()
    rows = []
    t0 = time.time()

    print("=" * 76)
    print("注入方式对照：%d 题（%s）" % (len(items), which))
    print("=" * 76)

    for n, name, style, q, sec, subs in items:
        picks = route(q, key)[0]
        routed = any(p.strip().startswith(sec.strip()) for p in picks)
        if q in cache:
            rews = cache[q]
        else:
            rews = rewrite(q, key)
            cache[q] = rews
            save_cache(cache)

        b = ROUTE_BONUS if picks else 0.0
        h_sub0 = search_multi(q, rews, k=3, picks=picks, bonus=0.0, route_pick=PICK)
        h_subT = None if only_b else search_multi(
            q, rews, k=3, picks=picks, bonus=b, route_pick=PICK)

        rows.append({
            "n": n, "name": name, "q": q, "sec": sec, "subs": subs, "picks": picks,
            "routed": routed,
            "B_sec": rank_of(h_sub0, [sec]), "B_sub": rank_of(h_sub0, [sec], subs),
            "C_sec": rank_of(h_subT, [sec]) if h_subT else None,
            "C_sub": rank_of(h_subT, [sec], subs) if h_subT else None,
        })
        def f(x):
            return "第%d" % x if x else "✗"
        print("%2d %-12s %-24s 路由%s  sub无bonus 节%s/小%s   sub+bonus 节%s/小%s" % (
            n, name, q[:24], "中" if routed else "偏",
            f(rows[-1]["B_sec"]), f(rows[-1]["B_sub"]),
            f(rows[-1]["C_sec"]), f(rows[-1]["C_sub"])))

    base = base_numbers(rows, which)
    N = len(rows)
    print("\n" + "=" * 76)
    print("汇总（%d 题，%.0fs）" % (N, time.time() - t0))
    print("=" * 76)

    def show(tag, secs, subs):
        print("%-22s 节@1 %.3f 节@3 %.3f | 小节@1 %.3f 小节@3 %.3f 小节MRR %.3f" % (
            tag,
            sum(1 for x in secs if x == 1) / N, sum(1 for x in secs if x) / N,
            sum(1 for x in subs if x == 1) / N, sum(1 for x in subs if x) / N,
            mrr(subs)))

    if base:
        show("A head + 分级bonus（旧）",
             [base[r["n"]][0] for r in rows], [base[r["n"]][1] for r in rows])
    show("B sub  + 无bonus",
         [r["B_sec"] for r in rows], [r["B_sub"] for r in rows])
    if not only_b:
        show("C sub  + 分级bonus",
             [r["C_sec"] for r in rows], [r["C_sub"] for r in rows])

    if base and not only_b:
        print("\n逐题变化（小节级 A → C）：")
        for r in rows:
            a, c = base[r["n"]][1], r["C_sub"]
            if a == c:
                continue
            tag = "改善" if (c and (not a or c < a)) else "弄坏"
            print("  %-12s %-24s %-4s → %-4s %s" % (
                r["name"], r["q"][:24],
                "第%d" % a if a else "✗", "第%d" % c if c else "✗", tag))

    col = "B_sub" if only_b else "C_sub"
    print("\n%s 组仍失败：" % ("B" if only_b else "C"))
    for r in rows:
        if r[col] == 1:
            continue
        print("  %-12s %-24s %s（路由%s）" % (
            r["name"], r["q"][:24],
            "第%d名" % r[col] if r[col] else "没进前3",
            "中" if r["routed"] else "偏"))

    with open("exp_route_pick_%s_%s.json" % (which, TAG), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1, default=str)
    print("\n明细存 exp_route_pick_%s_%s.json" % (which, TAG))


if __name__ == "__main__":
    main()
