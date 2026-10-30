"""极短口语档：真实用户怎么打字，系统就怎么崩

用户原话：「用户不知道这些专业词汇，大多数都只是口语询问查询」

这一档的特征是三个「没有」：
  没有术语（不会说 RAG、MCP）
  没有谓词（不说「是什么」「怎么运作」）
  没有上下文（4-12 个字，丢一句就走）

和已有三档的区别：
  术语   RAG 是什么？        ← 有术语 + 书面
  极短   RAG                 ← 有术语 + 单词
  绕弯   先检索外部资料再让模型生成答案叫什么？ ← 无术语 + 完整书面句
  口语   怎么让它查资料      ← 无术语 + 碎片化口语

同时跑两组：带改写（search_multi）和不带改写（纯原问），
用来回答一个单独的问题——查询改写在最难的这一档上到底救了多少。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import time
from rewrite import search_multi, rewrite, load_cache, save_cache
from eval import chunks
from llm import load_key

# (考点, 期望章节前缀, 口语短问)
CASES = [
    ("智能体定义", "## 1.1 ", "智能体是啥"),
    ("缩放法则", "## 3.3 ", "模型越大越聪明吗"),
    ("ReAct", "## 4.2 ", "边想边干是啥意思"),
    ("Plan-and-Solve", "## 4.3 ", "先规划再动手"),
    ("Reflection", "## 4.4 ", "让它自己挑错"),
    ("Dify", "## 5.3 ", "开源的低代码平台"),
    ("LangGraph", "## 6.5 ", "画流程图搭智能体"),
    ("RAG", "## 8.3 ", "怎么让它查资料"),
    ("MCP", "## 10.2 ", "怎么接外部工具"),
    ("A2A", "## 10.3 ", "两个智能体咋对话"),
    ("GRPO", "## 11.4 ", "不用价值网络的强化学习"),
    ("GAIA", "## 12.3 ", "测AI助手水平的榜单"),
    ("BFCL", "## 12.2 ", "怎么测调工具的能力"),
]


def rank_of(hits, want):
    for r, (idx, s) in enumerate(hits, 1):
        if chunks[idx][1].startswith(want):
            return r
    return None


def top1_info(hits):
    idx, s = hits[0]
    return (chunks[idx][1].strip(), float(s), len(chunks[idx][2]),
            chunks[idx][2].strip()[:32].replace("\n", " "))


def main():
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    cache = load_cache()
    rows = []
    t0 = time.time()

    for n, (name, want, q) in enumerate(CASES, 1):
        # ① 纯原问（改写不用）
        raw_hits = search_multi(q, None, k=3)
        r_raw = rank_of(raw_hits, want)

        # ② 带改写
        if q in cache:
            rews = cache[q]
        else:
            rews = rewrite(q, key)
            cache[q] = rews
            save_cache(cache)
        rw_hits = search_multi(q, rews, k=3)
        r_rw = rank_of(rw_hits, want)

        rows.append({
            "name": name, "q": q, "raw": r_raw, "rw": r_rw,
            "raw_top1": top1_info(raw_hits), "rw_top1": top1_info(rw_hits),
            "rews": rews,
        })
        print("%2d/%d %-14s %-16s 原问:%-6s 改写:%s" % (
            n, len(CASES), name, q,
            "第%d名" % r_raw if r_raw else "✗",
            "第%d名" % r_rw if r_rw else "✗"))

    dt = time.time() - t0
    N = len(CASES)

    print("\n" + "=" * 74)
    print("极短口语档   耗时 %.0fs（%d 题，两组检索）" % (dt, N))
    print("=" * 74)
    for tag, field in (("纯原问", "raw"), ("+ 查询改写", "rw")):
        h1 = sum(1 for r in rows if r[field] == 1)
        h3 = sum(1 for r in rows if r[field] is not None)
        print("%-10s @1 = %2d/%d = %.3f    @3 = %2d/%d = %.3f" % (
            tag, h1, N, h1 / N, h3, N, h3 / N))

    print("\n" + "-" * 74)
    print("逐题对照（原问排名 / 改写后排名）：")
    for r in rows:
        a = "第%d名" % r["raw"] if r["raw"] else "✗"
        b = "第%d名" % r["rw"] if r["rw"] else "✗"
        flag = "  ← 改写救回" if r["raw"] is None and r["rw"] else ""
        flag += "  ← 改写弄坏" if r["raw"] and r["rw"] is None else ""
        print("  %-14s %-16s %-6s → %-6s%s" % (r["name"], r["q"], a, b, flag))

    print("\n" + "-" * 74)
    print("改写后 top1 是什么：")
    for r in rows:
        t = r["rw_top1"]
        print("  %-14s %.3f  %-4d %s" % (r["name"], t[1], t[2], t[3]))

    print("\n" + "-" * 74)
    print("LLM 把这些口语改写成了什么：")
    for r in rows:
        print("  %s" % r["q"])
        for x in r["rews"]:
            print("     → %s" % x)


if __name__ == "__main__":
    main()
