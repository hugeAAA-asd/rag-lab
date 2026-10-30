"""章节路由：把目录交给 LLM，让任务从「猜术语」变成「挑章节」

为什么需要它（2026-10-30 实测）：
  极短口语查询（「怎么让它查资料」「边想边干是啥意思」）上，
  查询改写几乎无效——@1 只有 0.385，改写输出全是同义替换：
      「怎么让它查资料」→「如何实现资料检索功能」   ← 没到 RAG
      「边想边干是啥意思」→「边想边干是什么意思」   ← 没到 ReAct
  根因不是模型不行，是它不知道这本书里有什么。改写 prompt 里的
  「技术资料库里可能出现的说法」是泛指，LLM 手里没有词典。

  把 79 个二级标题塞进 prompt（约 1200 token），它立刻就选对了：
      「边想边干是啥意思」→ 4.2 ReAct
      「怎么让它查资料」  → 8.3 RAG系统
      「两个智能体咋对话」→ 10.3 A2A 协议实战
  路由自身准确率 12/13 = 0.923。

成本：一次 LLM 调用（temperature=0，输出就几个数字）+ 每次约 1200 token 的目录。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import re

from eval import chunks
from llm import ask_llm, load_key

# ── 目录：79 个二级标题 ──
TITLES = []
for c in chunks:
    t = c[1].strip()
    if t.startswith("## ") and t not in TITLES:
        TITLES.append(t)

MENU = "\n".join("No.%-2d %s" % (i + 1, t.lstrip("# ").strip()) for i, t in enumerate(TITLES))

ROUTE_SYSTEM = "你是图书检索员。用户会问一个问题，你要从图书目录里挑出最可能回答它的章节。"

ROUTE_TEMPLATE = (
    "这是一本关于智能体（Agent）的技术书，目录如下：\n\n%s\n\n"
    "读者的问题：%s\n\n"
    "从上面的目录里选出最可能回答这个问题的 3 个章节，"
    "只输出章节号，用逗号分隔，不要解释，不要输出章节名。例如：4.2,8.3,10.2\n"
    "如果目录里确实没有任何章节能回答这个问题，只输出 NONE。"
    "这一条很重要：读者经常会问书里没有的东西（比如推荐火锅店），"
    "这时候硬选 3 个章节反而会误导检索，必须输出 NONE。"
)


def _parse(txt, n):
    """兼容三种回答格式——实测 LLM 三种都用过

    ① 章节号（最常见）：8.3,7.5 → 直接匹配 ## 8.3 开头的标题
    ② 编号：No.52 → 取第 52 条
    ③ 裸整数：4,52,53 → 只有整段没出现小数点时才可信

    踩过的坑：第一版用 re.findall(r"\\d+") 解析，LLM 输出 "8.3,14.4,9.5" 时
    被拆成 8、3、14、4、9、5，选出来的章节全错，而且不报错。
    """
    # 模型明确说「书里没有」——这时候 picks 必须是空，不能硬选
    if "NONE" in txt.upper():
        return []

    picks = []

    for m in re.findall(r"\d+\.\d+", txt):
        for t in TITLES:
            if t.startswith("## " + m):
                if t not in picks:
                    picks.append(t)
                break

    if not picks:
        for s in re.findall(r"[Nn]o\.?\s*(\d+)", txt):
            i = int(s) - 1
            if 0 <= i < len(TITLES) and TITLES[i] not in picks:
                picks.append(TITLES[i])

    if not picks and not re.search(r"\d+\.\d+", txt):
        for s in re.findall(r"\d+", txt):
            i = int(s) - 1
            if 0 <= i < len(TITLES) and TITLES[i] not in picks:
                picks.append(TITLES[i])

    return picks[:n]


def route(question, key=None, n=3):
    """返回最可能回答这个问题的 n 个章节标题。失败返回 []（降级路径）。"""
    try:
        txt, _ = ask_llm(ROUTE_TEMPLATE % (MENU, question), key or load_key(),
                         ROUTE_SYSTEM, temperature=0.0)
    except Exception as e:
        print("   ! 路由失败，退回纯检索：%s" % e)
        return [], ""
    return _parse(txt, n), txt.strip()
