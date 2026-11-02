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
from route import route, parse_conf, route_allsubs, secs_of_subs, sub_conf

TOP_K = 3             # 喂给模型几段材料
MIN_SCORE = 0.60      # 精排分低于这个就拒答（abstain.py 扫出来的最优阈值）
USE_REWRITE = True    # 关掉就退回纯原问题检索（对照组）
USE_ROUTE = True      # 路由：把清单交给 LLM 挑范围，管改写管不了的口语查询

# ── v1.0 冻结配置 ────────────────────────────────────────────
#
# 这一版的所有数字都挂在这个标签下。改任何一项都要重跑
# `python reproduce.py` 并更新 README ⑧ 的表，否则文档立刻失效。
#
# 六项配置（值就在下面这个文件里，逐条可 grep）：
#   ROUTE_MODE = "subs"          287 个小节一次选 3 个，章节由小节号反推
#   ROUTE_BONUS = [0.6, 0.2, 0]  章节加分，按选中的顺序递减
#   SUB_BONUS = [0.4, 0, 0]      只给首位小节加分（三个都加=等于没加）
#   CONF_GATE = 60               章节匹配度低于 60 不加分（只对两级模式生效）
#   ROUTE_PICK = "sub2"          候选池注入方式：小节首块 ∪ 章首块
#   MIN_SCORE = 0.60             低于此分拒答
#
# 冻结基线（小节级 @1 / @3，分母＝题目总数）：
#   43 开发集 .925/.950   h1 .800/.850   h2 .700/.800   h3 .950/1.000
# 数字由 `python reproduce.py` 从候选池复算得出，不要手改。
VERSION = "v1.0-subs"

# 路由用哪一级清单
#   subs = 287 个小节一次选完，章节由小节号反推（默认）
#   sec  = 79 个章节选 3 个，再在 3 章里挑小节（旧，两级）
#
# 为什么默认换成 subs：两级方案在 h3 上的首选只有 0.750，直选是 0.800。
# 差的不是「选得细不细」，是两级在大方向上就漏了——正确章节没进 3 个候选，
# 小节再准也只能在错的章里挑。分级检索的前提是上一级不漏，漏了就是放大器。
#
# 小节@1 / 小节@3（同上四批题，同一套加分 sec.6+sub.4）：
#            小节@1            小节@3
#   43      .900 → .925      .900 → .950
#   h1      .650 → .800      .700 → .850
#   h2      .350 → .700      .650 → .800
#   h3      .700 → .950      .700 → 1.000
# 代价是 prompt 从 ~1200 token 涨到 ~5000，但省掉一次调用，净成本差不多。
# 更真实的一笔账：每题多补 9~10 个块进精排（小节整段捞进来），约 +35% 耗时。
ROUTE_MODE = os.getenv("ROUTE_MODE", "subs")

# 小节加分（只给首位）。为什么不给三个都加：选中 3 个小节往往覆盖章内
# 大半的块，一起抬轿子等于没抬——8.3 那题选的 [8.3.4, 8.3.2, 8.3.1] 把
# 整章都加了分，8.3.1 的块精排分本来最高，结果一个都没动。
# 0.4 是和章节 0.6 配着试出来的：43 上再加 0.6 反而掉 1 题。
SUB_BONUS = [0.4, 0.0, 0.0]

# 路由加分，按路由给出的顺序分档。
#
# 为什么不是三节同权 0.5：路由自己排了序，第 1 个候选和第 3 个的把握差很多。
# 「能感知环境并自主行动的系统指什么」路由给 1.1/1.2/1.4，同权 +0.5 之后
# 1.4.3（raw 0.885）压过 1.1.1（raw 0.860）——第 3 个候选赢了第 1 个。
# 反过来 A2A 那题，路由 10.3 是对的，但块 raw 分太低，+0.5 顶不进前三。
# 首位抬高、后位压低，两个问题一起解决。
#
# 口语 13 题实测（exp_tiers.py，同一批精排分只重排）：
#   0.5 同权      节 @1 0.769  小节 @1 0.615
#   0.8/0.3/0.1   节 @1 0.923  小节 @1 0.692
#   1.0/0.3/0.0   节 @1 1.000  小节 @1 0.769
# 注意：这三组是在这 13 题上挑的，过拟合风险在那摆着，绕弯 43 题必须复验。
#
# 复验结论（pools_sub2 候选池，加分强度扫描，43/h1/h2 三列一起看）：
#   [1.0,0.3,0.0]  43 .900/.900/.977   h1 .600/.750/.800   h2 .350/.450/.450
#   [0.6,0.2,0.0]  43 .900/.900/.977   h1 .650/.750/.850   h2 .350/.450/.450
# 缩到 0.6 之后 43 一格没掉，h1 多 1 题、节级多 1 题，h2 不动——三列全 ≥。
# 为什么会这样：加分只要够把路由章节顶进前三就行，顶进去之后再加就是
# 让路由替精排做决定，而路由只知道章节、不知道块。
ROUTE_BONUS = [0.6, 0.2, 0.0]

# 路由章节里怎么挑块塞进候选池
#   head = 前 2 块 + 余弦最高 1 块（旧）
#   sub1 = 首位章节按「每个小节第一块」展开，后两位仍用 head
#   sub2 = sub1 再并上 head 那三块（并集）
#
# 为什么要有 sub2：一个二级章节常有 5 个小节、30 多个块，只碰开头那 3 块时，
# 答案落在第 4 个小节的题永远救不回来——那是候选缺失，加多少 bonus 都没用。
# 但 sub1 有个缺陷：它**不是 head 的超集**。4.2 那小节的开头块被
# _sub_blocks 认成「小节首块」，紧跟其后的正文块却没被选中，
# 于是「一边推理一边调用工具」那题的正确块被自己挤出了池子。
# 并集之后「不加任何加分」的三列（43/h1/h2）全部 ≥ head，
# 所以它是个可以无条件替换 head 的版本：提名的活儿交给规则，裁决的活儿仍归精排。
#
# 为什么只在首位章节展开：最终按章节去重、一章只出一个代表，弱章节多塞块
# 只会让「错块当代表」的机会变多（43 题上全展开实测掉 8 题）。
ROUTE_PICK = "sub2"

# 路由置信度低于这个值，就不给它的章节加分。
# 模型现在会输出「8.3:92」这种带匹配度的结果（见 route.py 的 prompt），
# 60 这个阈值在 40 道留出题上有平台（50/60/70 同分），不是掐出来的一个点。
CONF_GATE = 60

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
    subs, raw_sub = [], ""
    if USE_ROUTE and k:
        if ROUTE_MODE == "subs":
            subs, raw_sub = route_allsubs(question, k)
            picks = secs_of_subs(subs)
        else:
            picks, raw_route = route(question, k)

    # 路由说 NONE = 它认为目录里没有能回答这个问题的章节。
    #
    # 但 NONE 不能当终审判决，只能当「路由这次不参评」：
    # 「问法和原文用词不一样就搜不到」这题路由给 NONE，而纯检索把
    # 8.3.5 高级检索策略排在第 1、精排 0.662——真答案一直在，是路由把它判死了。
    # 判死比排错严重得多：排错还有 3 个候选可以兜，判死等于直接回「没有相关内容」。
    #
    # 那为什么还敢让阈值接手：它分得开「沾边」和「不沾边」。同一批里
    # 「推荐几家北京好吃的火锅店」top1 只有 0.144，离 0.60 差得远，照样拦得住。
    # 真正需要 NONE 豁免阈值的场景是反过来的——路由**选中**了章节但精排分数塌了
    # （极短口语），那种情况走下面 routed_top1 的分支，不受这里影响。
    # NONE 有两个来源：两级方案里章节路由的判断、直选方案里小节清单的判断。
    # 后者更容易漏——章节层面对「问天气」这种问题会老实说 NONE，
    # 但小节的清单长 287 行，模型总能在里面找到一句沾边的凑数。
    route_none = (bool(raw_route) and "NONE" in raw_route.upper()) or \
                 (bool(raw_sub) and "NONE" in raw_sub.upper())
    if route_none:
        picks = []

    # 改写只负责补漏，排序权留给原问题；路由则参与打分（ROUTE_BONUS）。
    # 详见 rewrite.search_multi 的注释
    if rews or picks:
        hits = search_multi(question, rews, k=TOP_K, picks=picks, bonus=ROUTE_BONUS,
                            route_pick=ROUTE_PICK,
                            # 章节置信度只在「两级路由」下存在：那时候原始输出里
                            # 每个章节带一个匹配度。subs 模式下 raw_route 是空串，
                            # parse_conf("") 返回 []，而 search_multi 用
                            # `confs is not None` 判断——空列表也算「有」，于是
                            # `pos >= len(confs)` 成立，章节加分被全量清零。
                            # 后果：@1 看不出来，@3 悄悄掉了（h1 .850→.800 / h2 .800→.750）。
                            # 正确语义是「没有置信度信息」＝不过门控，全额加分。
                            confs=parse_conf(raw_route, len(picks))
                            if (picks and raw_route) else None,
                            gate=CONF_GATE,
                            subs=subs, sub_bonus=SUB_BONUS if subs else 0.0)
    else:
        hits = search(question, k=TOP_K)
    top = hits[0][1]

    # top1 是路由亲自选中的章节时，不看分数阈值——
    # 路由已经是比精排分数更可靠的信号了（12/13 vs 4/13）。
    # 阈值只在路由没参与（无 key / 路由失败）时才生效。
    routed_top1 = picks and any(
        chunks[hits[0][0]][1].strip() == p.strip() for p in picks)

    if top < MIN_SCORE and not routed_top1:
        reason = "top1 精排 %.3f < 阈值 %.2f" % (top, MIN_SCORE)
        if route_none:
            reason = "路由判定目录里没有相关章节，" + reason
        return {
            "question": question, "hits": hits, "answer": None, "rews": rews,
            "picks": picks, "subs": subs, "refused": True, "reason": reason,
        }

    prompt = build_prompt(question, hits)
    if dry:
        return {"question": question, "hits": hits, "answer": None, "rews": rews,
                "picks": picks, "subs": subs, "refused": False, "prompt": prompt}

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
            "picks": picks, "subs": subs, "refused": False,
            "prompt": prompt, "usage": usage}


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
