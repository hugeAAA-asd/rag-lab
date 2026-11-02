"""口语 13 题的块级判定：把「章节对了」和「块对了」拆开

口语档最典型的一次失败就藏在这：「怎么让它查资料」路由到 8.3 是对的，
但章节内挑到讲 HyDE 的块（8.3.5），模型答成「正式名称是 HyDE」。
节级判定看不见这种错——所以 0.769 这个数只有节级意义。

每题的小节标注都在 x.y.1（定义/介绍节）：口语问的都是「这是啥」。
"""
import time

from eval_oral import CASES                    # (考点, 期望节前缀, 口语短问)
from eval import chunks, rank_of
from metrics import header, row
from rewrite import search_multi, rewrite
from route import route
from llm import load_key
from answer import ROUTE_BONUS

SUBS = [
    ["1.1.1"],      # 智能体定义
    ["3.3.1"],      # 缩放法则
    ["4.2.1"],      # ReAct
    ["4.3.1"],      # Plan-and-Solve
    ["4.4.1"],      # Reflection
    ["5.3.1"],      # Dify
    ["6.5.1"],      # LangGraph
    ["8.3.1"],      # RAG
    ["10.2.1"],     # MCP
    ["10.3.1"],     # A2A
    ["11.4.1"],     # GRPO
    ["12.3.1"],     # GAIA
    ["12.2.1"],     # BFCL
]
assert len(SUBS) == len(CASES)


def main():
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    rows = []
    t0 = time.time()
    for i, ((name, want, q), subs) in enumerate(zip(CASES, SUBS), 1):
        hits = search_multi(q, rewrite(q, key), k=3,
                            picks=route(q, key)[0], bonus=ROUTE_BONUS)
        r_sec = rank_of(hits, [want])
        r_sub = rank_of(hits, [want], subs)
        rows.append((name, q, r_sec, r_sub, hits))
        flag = "   ← 章节对、块错" if r_sec == 1 and r_sub != 1 else ""
        print("%2d/%d %-14s %-16s 节:%-3s 小节:%-3s%s" % (
            i, len(CASES), name, q, r_sec or "✗", r_sub or "✗", flag))

    dt = time.time() - t0
    print("\n" + "=" * 62)
    header()
    print(row("节级判定", [r[2] for r in rows]))
    print(row("小节级判定", [r[3] for r in rows]))
    print("-" * 62)
    print("耗时 %.0fs" % dt)
    print("\n对照（节级）：纯原问 0.231 / +改写 0.385 / +改写+路由 0.769")

    bad = [r for r in rows if r[2] == 1 and r[3] != 1]
    print("\n章节对了但块错了（%d 题）：" % len(bad))
    for name, q, _a, _b, hits in bad:
        idx = hits[0][0]
        print("  %-14s %-16s top1=%.3f  %s" % (
            name, q, hits[0][1], chunks[idx][2].strip().replace("\n", " ")[:52]))


if __name__ == "__main__":
    main()
