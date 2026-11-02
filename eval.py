import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 模型已在本地缓存，不再联网（下新模型时临时设 0）

import re

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from embed import embed_query
from build_chunks import build_all

chunks = build_all()
vecs = np.load("chunks_vec.npy")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# ── 精排的两个提速开关（bench_speed.py 量的，同进程横向对比）──────────────
#   A 全文 + 14 线程          15.22s / 38 对   1.00x
#   B 截断 300 字 + 14 线程    9.65s           1.58x
#   D 截断 300 字 + 4 线程     5.86s           2.60x
#   G 截断 + 20 块 + 8 线程    4.37s           3.48x   （候选减半，会掉分，未采用）
#
# 线程越少越快这件事反直觉：14 个线程在 20 核上互相抢，调度开销大过并行收益。
# 截断只影响送进 cross-encoder 的文本长度，不影响喂给大模型的材料（那边用全文）。
#
# 截断到多少是拿 43 题扫出来的，不是拍的：
#   CUT=600   @1 0.953  @3 0.977  MRR 0.961   541s   ← 采用，和全文同分
#   CUT=300   @1 0.930  @3 0.977  MRR 0.950   276s   掉 1 题
#   全文       @1 0.953  @3 0.977  MRR 0.965   955s
# 也就是说 300~600 之间那 300 个字值 1 道题，600 以后再加字纯浪费。
RERANK_CUT = int(os.environ.get("RERANK_CUT", "600"))
RERANK_THREADS = int(os.environ.get("RERANK_THREADS", "4"))
torch.set_num_threads(RERANK_THREADS)


# 精排文本里要不要带上「三级小节标题」。
# 起因：9.5 的第 6 块、13.2 的第 3 块都是「小节续块」，正文里没有标题，
# cross-encoder 只能拿一段孤立的代码去猜这一段在讲什么。
# 而「10.2.3 MCP 传输方式详解」「11.5.1 评估指标体系」「13.2.3 Pydantic 的核心概念」
# 这些小节标题，恰恰是词面上最能对上用户问题的东西——不喂给精排等于把最有用的线索丢了。
# 默认关，用环境变量 RERANK_WITH_SUB=1 打开，方便和旧口径对照。
RERANK_WITH_SUB = bool(int(os.environ.get("RERANK_WITH_SUB", "0")))


def rerank_text(idx):
    """拼给 cross-encoder 的文本：标题完整保留，正文截断

    标题不截——41/43 题的标题本身就是答案，它是信息密度最高的部分。
    打开 RERANK_WITH_SUB 后，二级标题下面再补一行三级小节标题，
    让「续块」也知道自己在讲哪一小节。
    """
    sub = SUB_TITLE.get(idx) if RERANK_WITH_SUB else None
    head = chunks[idx][1]
    # 小节开头那块正文里已经含标题了，再补一遍等于重复占字数
    if sub and sub not in chunks[idx][2][:150]:
        head += "\n" + sub
    return head + "\n" + chunks[idx][2][:RERANK_CUT]

CASES = [
    # ── 1 章 ──────────────────────────────────────────────
    ("能感知环境并自主采取行动的系统，指的是什么？", ["## 1.1 "], "定义"),
    ("用什么框架来描述一个智能体的任务环境？", ["## 1.2 "], "定义"),
    ("任务环境里的性能度量、环境、执行器、传感器这四个东西合起来叫什么？", ["## 1.2 "], "细节"),
    ("智能体内部「先想、再动手、再看结果」这个循环怎么称呼？", ["## 1.3 "], "定义"),
    ("实际应用里，智能体跟人的关系可以分成哪两种角色？", ["## 1.4 "], "分类"),
    # ── 2 章 ──────────────────────────────────────────────
    ("早期认为智能可以用符号和逻辑形式化表达的流派叫什么？", ["## 2.1 "], "定义"),
    ("麻省理工那个模拟心理治疗师说话的早期聊天程序叫什么？", ["## 2.2 "], "细节"),
    ("谁提出智能来自许多简单个体协作，而不是一个中心推理引擎？", ["## 2.3 "], "细节"),
    ("智能从「人工设计规则」转向「从数据里学」的转折是怎么发生的？", ["## 2.4 "], "原因"),
    # ── 3 章 ──────────────────────────────────────────────
    ("最早靠前面几个词来预测下一个词的模型是怎么做的？", ["## 3.1 "], "定义"),
    ("和模型沟通时，怎么设计输入措辞才拿到想要的输出？", ["## 3.2 "], "应用"),
    ("模型越大能力越强，这个规律叫什么？它有什么代价？", ["## 3.3 "], "定义"),
    # ── 4 章 ──────────────────────────────────────────────
    ("一边推理一边调用工具的范式是怎么运作的？", ["## 4.2 "], "原因"),
    ("先把任务拆成完整计划、再一步步执行，这种范式叫什么？", ["## 4.3 "], "定义"),
    ("智能体生成答案之后，怎么自我批评并改进？", ["## 4.4 "], "原因"),
    # ── 5 章 ──────────────────────────────────────────────
    ("书里介绍了哪几个可视化拖拽搭建智能体的低代码平台？", ["## 5.2 ", "## 5.3 ", "## 5.4 ", "## 5.5 "], "分类"),
    ("两个低代码智能体平台之间该怎么选？", ["## 5.2 ", "## 5.3 ", "## 5.4 ", "## 5.5 "], "对比"),
    ("哪个平台是开源的，还融合了后端即服务和 LLMOps 理念？", ["## 5.3 "], "细节"),
    ("哪个平台核心定位是知识库问答和 RAG？", ["## 5.4 "], "细节"),
    ("哪个平台本质上是通用工作流自动化、大模型只是其中一环？", ["## 5.5 "], "细节"),
    # ── 6 章 ──────────────────────────────────────────────
    ("用状态图来编排智能体流程的框架是哪个？", ["## 6.5 "], "定义"),
    ("哪个框架的设计哲学是「用对话驱动多角色协作」？", ["## 6.2 "], "细节"),
    ("阿里达摩院做的、强调工程化和高可靠的多智能体平台是哪个？", ["## 6.3 "], "细节"),
    ("只靠两个智能体角色扮演、几乎不要人干预就能解决问题的框架？", ["## 6.4 "], "细节"),
    # ── 7 章 ──────────────────────────────────────────────
    ("市面上已经有那么多框架了，为什么还要自己造一个？", ["## 7.1 "], "原因"),
    ("框架里怎么统一管理一大堆工具，让模型知道有哪些可用？", ["## 7.5 "], "应用"),
    # ── 8 章 ──────────────────────────────────────────────
    ("智能体的长期记忆借鉴了哪些认知科学的结论？", ["## 8.1 "], "原因"),
    ("记忆系统整体上是按什么流程运转的？", ["## 8.2 "], "原因"),
    ("先检索外部资料再让模型生成答案，这套做法叫什么？", ["## 8.3 "], "定义"),
    # ── 9 章 ──────────────────────────────────────────────
    ("上下文工程和以前说的提示工程，差别在哪？", ["## 9.1 "], "对比"),
    ("上下文塞得太长，模型反而变糊涂，这个现象叫什么？", ["## 9.2 "], "定义"),
    ("长时程任务里，用什么工具以 Markdown + YAML 记录进度和结论？", ["## 9.4 "], "细节"),
    # ── 10 章 ─────────────────────────────────────────────
    ("智能体怎么像插 USB-C 一样接入各种各样的外部工具？", ["## 10.2 "], "原因"),
    ("智能体和智能体之间直接通信协作，靠的是什么协议？", ["## 10.3 "], "定义"),
    ("面向大规模开放网络环境做智能体管理的协议是哪个？", ["## 10.4 "], "细节"),
    # ── 11 章 ─────────────────────────────────────────────
    ("不训练价值网络、靠组内相对比较的强化学习方法叫什么？", ["## 11.4 "], "定义"),
    ("正式强化学习之前，那个让模型先学会输出格式的步骤是什么？", ["## 11.3 "], "定义"),
    # ── 12 章 ─────────────────────────────────────────────
    ("怎么衡量一个模型的函数调用能力？", ["## 12.2 "], "应用"),
    ("评估通用 AI 助手解决真实世界问题的基准是哪个？", ["## 12.3 "], "定义"),
    # ── 13/14 章 ──────────────────────────────────────────
    ("旅行助手为什么不直接调地图 API，而要包一层协议？", ["## 13.4 "], "原因"),
    ("深度研究任务怎么拆成一堆子任务逐个完成？", ["## 14.2 "], "原因"),
    # ── 15/16 章 ──────────────────────────────────────────
    ("NPC 对玩家的态度怎么用数值分级？", ["## 15.3 "], "细节"),
    ("想做一个拿得出手的项目，选题有什么原则？", ["## 16.2 "], "应用"),
]


# ── 块级判定：答对「哪一节」不算对，得答对「哪一小节」──────────────────
# 原来只看 top-k 块的标题前缀（"## 8.3 "）。可 8.3 有 22 块，
# 命中讲 HyDE 的那块也算对——「章节对了、块错了」被系统性算成成功。
#
# 这里给每题标一个三级小节号，判定改成「命中的块属不属于这一小节」。
# None = 答案跨小节 / 该节没有三级标题，退回节级判定。
# 顺序与 CASES 一一对应，改 CASES 必须同步改这里（下面有断言兜着）。
SUBS = [
    ["1.1.1"],                  # 1  智能体定义
    ["1.2.1"],                  # 2  PEAS 框架
    ["1.2.1"],                  # 3  四个要素合称
    ["1.3.3"],                  # 4  行动循环
    ["1.4.1", "1.4.2"],         # 5  两种角色（各占一小节）
    ["2.1.1"],                  # 6  物理符号系统假说
    ["2.2.1"],                  # 7  ELIZA
    ["2.3.1", "2.3.2"],         # 8  协作体智能
    ["2.4.1"],                  # 9  从符号到联结
    ["3.1.1"],                  # 10 N-gram
    ["3.2.1"],                  # 11 提示工程
    ["3.3.1"],                  # 12 缩放法则
    ["4.2.1"],                  # 13 ReAct 工作流程
    ["4.3.1"],                  # 14 Plan-and-Solve
    ["4.4.1"],                  # 15 Reflection
    None,                       # 16 跨 5.2~5.5，答哪个平台都算对
    None,                       # 17 平台对比，同上
    ["5.3.1"],                  # 18 Dify
    ["5.4.1"],                  # 19 FastGPT
    ["5.5.1"],                  # 20 n8n
    ["6.5.1"],                  # 21 LangGraph
    ["6.2.1"],                  # 22 AutoGen
    ["6.3.1"],                  # 23 AgentScope
    ["6.4.1"],                  # 24 CAMEL
    ["7.1.1"],                  # 25 为何自建框架
    ["7.5.1"],                  # 26 工具注册机制
    ["8.1.1"],                  # 27 认知科学启发
    ["8.2.1"],                  # 28 记忆工作流程
    ["8.3.1"],                  # 29 RAG
    None,                       # 30 9.1 没有三级标题
    ["9.2.0"],                  # 31 答案在 9.2 的导语块（讲「会走神或混乱」那段）
    ["9.4.1"],                  # 32 NoteTool 本身在 9.4.1，不是 9.4.2 存储格式
    ["10.2.1"],                 # 33 MCP 概念
    ["10.3.1"],                 # 34 A2A 动机/定义
    ["10.4.1"],                 # 35 ANP
    ["11.4.1"],                 # 36 GRPO
    ["11.3.1"],                 # 37 SFT
    ["12.2.1"],                 # 38 BFCL
    ["12.3.1"],                 # 39 GAIA
    ["13.4.1"],                 # 40 为什么不直接调 API
    ["14.2.1"],                 # 41 TODO 驱动
    ["15.3.1"],                 # 42 好感度分级
    ["16.2.1"],                 # 43 选题原则
]
assert len(SUBS) == len(CASES), "SUBS(%d) 和 CASES(%d) 长度不一致" % (len(SUBS), len(CASES))

_SUB_RE = re.compile(r"^###+ (\d+\.\d+\.\d+)", re.M)
_SEC_RE = re.compile(r"^## (\d+\.\d+) ")


def _sec_no(title):
    m = _SEC_RE.match(title)
    return m.group(1) if m else ""


def _build_sub_map():
    """块下标 → 它涉及的小节号集合

    块和小节不是一对一，三个真实情况都得处理：
      · 一个小节只有第一块带 ### 标题，后面几块没有 → 往下继承
      · 有的块正文里出现两个 ###（按 800 字切，不看标题边界）→ 两个都记
      · 小节开始前的导语块（`### 9.2.1` 之前那部分）不属于任何小节 →
        记成 "9.2.0"。不这么记的话它们归属为空，答案落在导语里的题
        （9.2 讲「上下文太长会走神」就在导语）永远判错。
    """
    m, cur, title = {}, None, None
    for i, c in enumerate(chunks):
        if c[1] != title:            # 换节了，小节跟着重置
            title, cur = c[1], None
        hits = _SUB_RE.findall(c[2])
        if hits:
            s = set(hits)
            if cur:
                s.add(cur)           # 跨小节块：上一小节也算它身上
            m[i] = s
            cur = hits[-1]
        elif cur:
            m[i] = {cur}
        else:
            sec = _sec_no(title)     # 导语块
            m[i] = {sec + ".0"} if sec else set()
    return m


SUB_MAP = _build_sub_map()


_SUB_TITLE_RE = re.compile(r"^###+ (\d+\.\d+\.\d+)\s*(.*)$", re.M)


def _build_sub_title():
    """块下标 → 它所属小节的标题文本（如 "10.2.3 MCP 传输方式详解"）

    和 SUB_MAP 的区别：那份记的是小节**号**（用来判定命中），
    这份记的是小节**标题原文**（用来喂精排）。续块继承上一个标题。
    """
    m, cur, title = {}, None, None
    for i, c in enumerate(chunks):
        if c[1] != title:                 # 换二级章节，小节跟着重置
            title, cur = c[1], None
        hits = _SUB_TITLE_RE.findall(c[2])
        if hits:
            # 块里可能同时出现两个标题（800 字硬切造成），正文主体属于第一个
            if cur is None:
                cur = "%s %s" % hits[0]
            m[i] = "%s %s" % hits[0]
            cur = "%s %s" % hits[-1]
        else:
            m[i] = cur
    return m


SUB_TITLE = _build_sub_title()


def rank_of(hits, answers, subs=None):
    """命中排名。给了 subs 就按小节判（严），否则按节前缀判（旧口径）

    两套口径必须都能跑：小节级是新的真话，节级留着跟历史数字对齐。
    """
    for r, (idx, s) in enumerate(hits, 1):
        if subs:
            if SUB_MAP[idx] & set(subs):
                return r
        elif any(chunks[idx][1].startswith(a) for a in answers):
            return r
    return None


N_SECTIONS = 15      # 粗筛取多少节
PER_SECTION = 2      # 每节放几块进精排（>1 是为了让节内真正相关的那块有入场券）


def search(question, k=3):
    # 第一段：双塔粗筛，只要「不漏」。按节限量，而不是一刀切前 N 块
    qv = embed_query(question)
    scores = vecs @ qv
    order = np.argsort(-scores)

    cand, count, nsec = [], {}, 0
    for i in order:
        title = chunks[int(i)][1]
        c = count.get(title, 0)
        if c >= PER_SECTION:
            continue
        if c == 0:
            if nsec >= N_SECTIONS:
                break
            nsec += 1
        count[title] = c + 1
        cand.append(int(i))

    # 第二段：交叉精排，负责「排对」
    pairs = [(question, rerank_text(i)) for i in cand]
    rs = reranker.predict(pairs, batch_size=16)

    # 第三段：精排之后才去重，同一节最多留一块
    picked, seen = [], set()
    for j in np.argsort(-rs):
        title = chunks[cand[int(j)]][1]
        if title in seen:
            continue
        seen.add(title)
        picked.append((cand[int(j)], float(rs[int(j)])))
        if len(picked) == k:
            break
    return picked


def evaluate():
    """一轮检索同时读出 @1 和 @3 —— 重排只跑一次"""
    rows = []
    for q, answers, kind in CASES:
        res = search(q)
        rank = None
        for r, (idx, s) in enumerate(res, 1):
            if any(chunks[idx][1].startswith(a) for a in answers):
                rank = r
                break
        rows.append((q, rank, res, kind))
    return rows


if __name__ == "__main__":
    import time

    t0 = time.time()
    rows = evaluate()
    dt = time.time() - t0
    n = len(CASES)
    print("%d 题全部精排完成，耗时 %.1fs（每题 %.1fs）" % (n, dt, dt / n))
    print()

    for k in [1, 3]:
        hit = sum(1 for _, r, _, _ in rows if r is not None and r <= k)
        print("Recall@%d = %2d/%d = %.3f" % (k, hit, n, hit / n))
    print()

    print("按题型拆开（看弱在哪类问题上）：")
    kinds = []
    for _, _, _, kind in rows:
        if kind not in kinds:
            kinds.append(kind)
    for kind in kinds:
        sub = [(r) for _, r, _, kd in rows if kd == kind]
        h1 = sum(1 for r in sub if r == 1)
        h3 = sum(1 for r in sub if r is not None and r <= 3)
        print("  %-4s %2d 题 | @1 %2d/%2d = %.2f | @3 %2d/%2d = %.2f"
              % (kind, len(sub), h1, len(sub), h1 / len(sub), h3, len(sub), h3 / len(sub)))
    print()

    print("每题命中排名（- 表示前3没中）：")
    for q, r, _, kind in rows:
        print("  %-4s %s  %s" % ("-" if r is None else str(r), kind, q))

    print()
    print("没进前 3 的题，看它检索到了什么：")
    for q, r, res, kind in rows:
        if r is not None and r <= 3:      # 排第 4 名也是没中，不能漏
            continue
        print("  [%s] %s" % ("-" if r is None else "第%d名" % r, q))
        for idx, s in res[:3]:
            print("      %.3f  %s" % (s, chunks[idx][1][:36]))
