"""第 ⑤ 站：检索 → 拼 prompt → 调模型 → 带引用回答

前四站（切块 / 向量化 / 检索 / 精排）解决的是「找到对的材料」，
这一站解决的是「把材料变成一句人话，并且说得出来源」。

三道闸门，缺一不可：
  ① 检索     —— 找不到材料，后面全是编
  ② 分数阈值 —— top1 太不像，直接拒答，不浪费一次 API 调用
  ③ 系统提示 —— 逼模型只准用材料回答，材料不够就说不知道

用法：
    python answer.py                    # 跑内置的几个演示问题
    python answer.py "什么是智能体？"    # 问一个问题
    python answer.py --dry "..."        # 只看检索和 prompt，不调模型
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import re
import sys
import time
import urllib.error

from eval import search, chunks
from llm import ask_llm, load_key
from rewrite import rewrite, search_multi
from route import route

TOP_K = 3             # 喂给模型几段材料
MIN_SCORE = 0.60      # 精排分低于这个就拒答（abstain.py 扫出来的最优阈值）
USE_REWRITE = True    # 关掉就退回纯原问题检索（对照组）
USE_ROUTE = True      # 章节路由：把目录交给 LLM 挑章节，管改写管不了的口语查询

# 路由加分。0.5 是扫出来的：43 题上 1.0 会让 @1 涨到 0.884 但 @3 掉到 0.930
# （路由独裁，精排没有否决权，路由 3 个候选全错时必然失败）；
# 0.5 时 @1 保持 0.884、@3 回到 0.953——降 bonus 不损失收益，还换回被挤掉的题。
ROUTE_BONUS = 0.5

# 这段话是整个系统的「宪法」。没有它，模型见到材料就会开始自由发挥。
SYSTEM_PROMPT = (
    "你是一个只根据给定材料回答问题的助手，必须遵守以下规则：\n"
    "1. 只能使用【材料】里的内容回答，不许用自己的知识补充，也不许推测。\n"
    "2. 每句结论后面用 [1] [2] [3] 标出它出自哪段材料；用到几段就标几个。\n"
    "3. 如果材料里没有足够信息回答问题，直接回答「材料里没有相关内容」，不要硬凑。\n"
    "4. 回答控制在 3 句话以内，直接给结论，不要复述问题。\n"
    "5. 用户常常用口语提问（比如「边想边干」「怎么让它查资料」），"
    "而材料里用的是术语。如果材料讲的内容确实就是用户问的那件事，"
    "只是叫法不同，就正常回答，并在第一句说清它在材料里的正式名称。"
    "这不算推测——推测是指材料里根本没有、你凭自己的知识补出来的内容。"
)


# ── ① 把检索结果拼成材料 ────────────────────────────────────────────

def build_material(hits):
    """每段材料带编号，编号就是后面引用的锚点"""
    parts = []
    for n, (idx, score) in enumerate(hits, 1):
        title, body = chunks[idx][1], chunks[idx][2]
        title = title.strip().lstrip("#").strip()      # "## 1.1 什么是智能体" → "1.1 什么是智能体"
        parts.append("[%d] 出自《%s》\n%s" % (n, title, body.strip()))
    return "\n\n".join(parts)


def build_prompt(question, hits):
    return "【材料】\n%s\n\n【问题】\n%s" % (build_material(hits), question)


# ── ② 调用大模型（已抽到 llm.py，这里只剩业务） ──────────────────────


# ── ③ 主流程 ────────────────────────────────────────────────────────

def answer(question, key=None, dry=False):
    """返回一个 dict：hits 是证据，answer 是结论，refused 表示兜没兜住"""
    k = key or load_key()

    # 查询改写：把口语问题翻译成资料库里的说法，补上词汇鸿沟。
    # 没有 key（比如只想 --dry 预演）就跳过，保证不花钱也能跑通。
    rews = rewrite(question, k) if (USE_REWRITE and k) else []

    # 章节路由：改写是从「说法」推出术语，靠推断；路由是直接从目录里挑，靠匹配。
    # 口语碎片（「怎么让它查资料」）信息量太少推不出术语，这时候只有路由管用。
    picks, raw_route = [], ""
    if USE_ROUTE and k:
        picks, raw_route = route(question, k)

    # 路由说 NONE = 目录里根本没有能回答这个问题的章节。
    # 这比分数阈值可靠得多：阈值是拿绝对分数卡，而极短口语查询下精排对所有块的
    # 分数都塌到 0.00x，阈值分不出「不沾边」和「相关但模型看不懂」。
    none_answer = bool(raw_route) and "NONE" in raw_route.upper()
    if none_answer:
        return {"question": question, "hits": [], "answer": None, "rews": rews,
                "picks": [], "refused": True, "reason": "路由判定：目录里没有能回答这个问题的章节"}

    # 改写只负责补漏，排序权留给原问题；路由则参与打分（ROUTE_BONUS）。
    # 详见 rewrite.search_multi 的注释
    if rews or picks:
        hits = search_multi(question, rews, k=TOP_K, picks=picks, bonus=ROUTE_BONUS)
    else:
        hits = search(question, k=TOP_K)
    top = hits[0][1]

    # top1 是路由亲自选中的章节时，不看分数阈值——
    # 路由已经是比精排分数更可靠的信号了（12/13 vs 4/13）。
    # 阈值只在路由没参与（无 key / 路由失败）时才生效。
    routed_top1 = picks and any(
        chunks[hits[0][0]][1].strip() == p.strip() for p in picks)

    if top < MIN_SCORE and not routed_top1:
        return {
            "question": question, "hits": hits, "answer": None, "rews": rews,
            "picks": picks,
            "refused": True, "reason": "top1 精排 %.3f < 阈值 %.2f" % (top, MIN_SCORE),
        }

    prompt = build_prompt(question, hits)
    if dry:
        return {"question": question, "hits": hits, "answer": None, "rews": rews,
                "picks": picks, "refused": False, "prompt": prompt}

    try:
        text, usage = ask_llm(prompt, k, SYSTEM_PROMPT)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:200]
        return {"question": question, "hits": hits, "answer": None, "refused": False,
                "rews": rews, "picks": picks, "prompt": prompt,
                "error": "接口返回 %d：%s" % (e.code, detail)}
    except urllib.error.URLError as e:
        return {"question": question, "hits": hits, "answer": None, "refused": False,
                "rews": rews, "picks": picks, "prompt": prompt,
                "error": "网络不通：%s" % e.reason}

    return {"question": question, "hits": hits, "answer": text.strip(), "rews": rews,
            "picks": picks, "refused": False, "prompt": prompt, "usage": usage}


def cited(text):
    """从答案里抽出 [1] [2] 这些引用号，用来检查模型有没有乱标"""
    return sorted({int(x) for x in re.findall(r"\[(\d+)\]", text or "")})


def show(r):
    line = "─" * 56
    print(line)
    print("问：", r["question"])
    for x in r.get("rews") or []:
        print("   ↳ 改写:", x)
    for x in r.get("picks") or []:
        print("   ↳ 路由:", x.strip().lstrip("# ").strip())
    for n, (idx, s) in enumerate(r["hits"], 1):
        print("  [%d] %.3f  %s" % (n, s, chunks[idx][1].strip()[:40]))
    print()
    if r["refused"]:
        print("拒答：%s" % r["reason"])
        print("回答：材料里没有相关内容。")
    elif r.get("error"):
        print("出错：%s" % r["error"])
    elif r["answer"] is None:
        print("【dry 模式，未调用模型】prompt 共 %d 字：" % len(r["prompt"]))
        print(r["prompt"][:600] + ("\n...(省略)" if len(r["prompt"]) > 600 else ""))
    else:
        print("答：", r["answer"])
        c = cited(r["answer"])
        bad = [n for n in c if n < 1 or n > len(r["hits"])]
        print()
        print("引用：%s%s" % (c or "（未标引用）", "  ⚠ 越界 %s" % bad if bad else ""))
        u = r.get("usage", {})
        if u:
            print("token：prompt %d + 输出 %d" % (u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))
    print()


DEMO = [
    "什么是智能体？",
    "边想边干是啥意思",                      # 极短口语：改写推不出术语，只有路由能救
    "怎么让它查资料",                        # 极短口语：改写会输出「如何实现资料检索功能」
    "智能体怎么像插 USB-C 一样接入各种各样的外部工具？",
    "CrewAI 和 AutoGen 该怎么选？",          # T3 阴性样本：书里只提了名字，看它会不会硬编
    "推荐几家北京好吃的火锅店",               # T1 完全离题：应该被阈值挡在门外
]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry"]
    dry = "--dry" in sys.argv[1:]

    key = load_key()
    if not dry and not key:
        print("没找到 DEEPSEEK_API_KEY。二选一：")
        print("  1. 在项目根目录建 .env，写一行 DEEPSEEK_API_KEY=sk-xxx")
        print("  2. 临时设置环境变量 DEEPSEEK_API_KEY")
        print("或者先加 --dry 只看检索和 prompt。\n")
        sys.exit(1)

    questions = [" ".join(args)] if args else DEMO

    t_all = time.time()
    for q in questions:
        t0 = time.time()
        r = answer(q, key, dry=dry)
        show(r)
        print("本次耗时 %.1fs\n" % (time.time() - t0))
    print("合计 %.1fs" % (time.time() - t_all))
