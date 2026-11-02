"""块级判定：把「章节对了」和「块对了」拆开看

昨天所有指标都是「章节命中」——8.3 一节有 22 块，命中讲 HyDE 的那块也算对。
今天给每题标了三级小节（eval.SUBS），同一批检索结果按两套口径各算一遍，
两者的差值就是「章节对了、块错了」的比例。

用法：
    python eval_level.py                 # 当前主方案（改写+路由）
    python eval_level.py --nosub         # 只跑节级（和历史数字对齐）
"""
import time
import sys

from eval import CASES, SUBS, chunks, rank_of
from metrics import header, row, mrr
from rewrite import search_multi, rewrite
from route import route
from llm import load_key
from answer import ROUTE_BONUS


def main():
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    # --limit 3 只跑前 3 题：验证脚本本身用，别让它跑完 9 分钟才发现报错
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    items = list(zip(CASES, SUBS))[:limit] if limit else list(zip(CASES, SUBS))

    n = len(items)
    print("\n%d 题，方案：改写 + 路由（bonus=%s）\n" % (n, ROUTE_BONUS))

    rows = []
    t0 = time.time()
    for i, ((q, answers, kind), subs) in enumerate(items, 1):
        rews = rewrite(q, key)
        picks, _raw = route(q, key)
        hits = search_multi(q, rews, k=3, picks=picks, bonus=ROUTE_BONUS)

        r_sec = rank_of(hits, answers)                       # 旧口径：节
        r_sub = rank_of(hits, answers, subs) if subs else r_sec   # 新口径：小节

        rows.append({"q": q, "kind": kind, "sec": r_sec, "sub": r_sub,
                     "tag": "节" if subs is None else "小节",
                     "top1": (chunks[hits[0][0]][1].strip(), round(float(hits[0][1]), 3))})
        flag = ""
        if r_sec == 1 and r_sub != 1:
            flag = "   ← 章节对、块错"
        print("%2d/%d [%s] 节:%-4s 小节:%-4s %s%s" % (
            i, n, rows[-1]["tag"],
            r_sec or "✗", r_sub or "✗", q[:30], flag))

    dt = time.time() - t0

    print("\n" + "=" * 58)
    header()
    print(row("节级判定", [r["sec"] for r in rows]))
    print(row("小节级判定", [r["sub"] for r in rows]))
    print("-" * 58)
    print("耗时 %.0fs（每题 %.1fs）" % (dt, dt / n))

    # 只统计「标了小节」的题，看纯块级质量（4 题退回节级的混在一起会虚高）
    labeled = [r for r in rows if r["tag"] == "小节"]
    print("\n只看标了小节的 %d 题：" % len(labeled))
    header()
    print(row("节级", [r["sec"] for r in labeled]))
    print(row("小节级", [r["sub"] for r in labeled]))

    bad = [r for r in rows if r["sec"] == 1 and r["sub"] != 1]
    print("\n章节对了但块错了（%d 题）：" % len(bad))
    for r in bad:
        print("  top1=%.3f %s" % (r["top1"][1], r["top1"][0][:44]))
        print("       %s" % r["q"][:44])
    if not bad:
        print("  （没有）")


if __name__ == "__main__":
    main()
