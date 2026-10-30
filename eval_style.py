"""问法分布实验：同一批考点，三种问法

要回答的问题不是「哪个方案更好」，而是「同一个考点换个问法，系统会不会崩」。
现有 43 题全是「绕弯口语」——刻意回避术语，测的是翻译能力。
但真实用户里有一大批直接说术语（"RAG 是什么？"），还有一批只丢一个词（"MCP"）。

三种问法：
  绕弯  能感知环境并自主采取行动的系统，指的是什么？   ← 现有 43 题的风格
  术语  智能体是什么？                              ← 用户已经知道术语
  极短  智能体                                      ← 用户只丢关键词

预期与陷阱：
  术语档看起来最简单（标题直接命中），但书里「提到过某术语」的块很多，
  真正的风险从「找不到」变成「找错块」——这正是 1.1 那类标题饱和。
  极短档没有谓词，embedding 抓不住意图，可能反而是最差的一档。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import time
from rewrite import search_multi, rewrite, load_cache, save_cache
from eval import chunks
from llm import load_key

# (考点名, 期望章节前缀, 绕弯问法, 术语问法, 极短问法)
CASES = [
    ("智能体定义", "## 1.1 ",
     "能感知环境并自主采取行动的系统，指的是什么？", "智能体是什么？", "智能体"),
    ("缩放法则", "## 3.3 ",
     "模型越大能力越强，这个规律叫什么？它有什么代价？", "缩放法则是什么？", "缩放法则"),
    ("ReAct", "## 4.2 ",
     "一边推理一边调用工具的范式是怎么运作的？", "ReAct 是什么？", "ReAct"),
    ("Plan-and-Solve", "## 4.3 ",
     "先把任务拆成完整计划、再一步步执行，这种范式叫什么？", "Plan-and-Solve 是什么？", "Plan and Solve"),
    ("Reflection", "## 4.4 ",
     "智能体生成答案之后，怎么自我批评并改进？", "Reflection 反思机制怎么工作？", "Reflection"),
    ("Dify", "## 5.3 ",
     "哪个平台是开源的，还融合了后端即服务和 LLMOps 理念？", "Dify 是什么？", "Dify"),
    ("LangGraph", "## 6.5 ",
     "用状态图来编排智能体流程的框架是哪个？", "LangGraph 是什么？", "LangGraph"),
    ("RAG", "## 8.3 ",
     "先检索外部资料再让模型生成答案，这套做法叫什么？", "RAG 是什么？", "RAG"),
    ("MCP", "## 10.2 ",
     "智能体怎么像插 USB-C 一样接入各种各样的外部工具？", "MCP 协议是什么？", "MCP"),
    ("A2A", "## 10.3 ",
     "智能体和智能体之间直接通信协作，靠的是什么协议？", "A2A 协议是什么？", "A2A"),
    ("GRPO", "## 11.4 ",
     "不训练价值网络、靠组内相对比较的强化学习方法叫什么？", "GRPO 是什么？", "GRPO"),
    ("GAIA", "## 12.3 ",
     "评估通用 AI 助手解决真实世界问题的基准是哪个？", "GAIA 基准是什么？", "GAIA"),
    ("BFCL", "## 12.2 ",
     "怎么衡量一个模型的函数调用能力？", "BFCL 怎么评估工具调用能力？", "BFCL"),
]

STYLES = ("绕弯", "术语", "极短")


def run_one(q, want, key):
    rews = rewrite(q, key)
    hits = search_multi(q, rews, k=3)
    rank, top1 = None, None
    for r, (idx, s) in enumerate(hits, 1):
        if top1 is None:
            top1 = (chunks[idx][1].strip(), s)
        if chunks[idx][1].startswith(want):
            rank = r
            break
    return rank, top1, rews


def main():
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    cache = load_cache()
    rows = []          # (考点, 档位, 排名, top1标题, top1分, 问法)
    t0 = time.time()
    total = len(CASES) * len(STYLES)
    done = 0

    for name, want, *qs in CASES:
        for style, q in zip(STYLES, qs):
            if q in cache:
                rews = cache[q]
            else:
                rews = rewrite(q, key)
                cache[q] = rews
                save_cache(cache)
            hits = search_multi(q, rews, k=3)

            rank = None
            idx0, s0 = hits[0]
            top1 = (chunks[idx0][1].strip(), float(s0),
                    len(chunks[idx0][2]), chunks[idx0][2].strip()[:34].replace("\n", " "))
            for r, (idx, s) in enumerate(hits, 1):
                if chunks[idx][1].startswith(want):
                    rank = r
                    break

            rows.append((name, style, rank, top1, q))
            done += 1
            print("  %2d/%d  [%s/%s] %s → %s" % (
                done, total, name, style, q[:30],
                "第%d名" % rank if rank else "未进前3"))

    dt = time.time() - t0

    # ── 汇总 ──
    print("\n" + "=" * 72)
    print("问法分布实验   耗时 %.0fs（%d 题，%.1fs/题）" % (dt, total, dt / total))
    print("=" * 72)
    print("%-6s %8s %8s   %s" % ("档位", "@1", "@3", "说明"))
    for style in STYLES:
        sub = [r for r in rows if r[1] == style]
        h1 = sum(1 for r in sub if r[2] == 1)
        h3 = sum(1 for r in sub if r[2] is not None)
        print("%-6s %4d/%-3d %4d/%-3d" % (style, h1, len(sub), h3, len(sub)))

    # ── 明细：每个考点三档对比 ──
    print("\n" + "-" * 72)
    print("%-16s %-10s %-10s %-10s" % ("考点", "绕弯", "术语", "极短"))
    for name, want, *_ in CASES:
        cells = []
        for style in STYLES:
            r = next(x for x in rows if x[0] == name and x[1] == style)
            cells.append("第%d名" % r[2] if r[2] else "✗")
        marks = "  ←注意力" if len(set(cells)) > 1 else ""
        print("%-16s %-10s %-10s %-10s%s" % (name, cells[0], cells[1], cells[2], marks))

    # ── 术语档排第 1 但块错的题 ──
    print("\n" + "-" * 72)
    print("术语档的 top1 是什么（看有没有「章节对了但块错」）：")
    print("  %-14s %-6s %-6s %-5s %s" % ("考点", "排名", "分数", "块长", "块开头"))
    for name, style, rank, top1, q in rows:
        if style != "术语":
            continue
        print("  %-14s %-6s %.3f  %-5d %s" % (name, "第%d名" % rank if rank else "✗",
                                              top1[1], top1[2], top1[3][:32]))


if __name__ == "__main__":
    main()
