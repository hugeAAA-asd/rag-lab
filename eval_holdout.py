"""过拟合验证：20 道从未参与调参的新题

为什么要跑这个
--------------
分级 bonus [1.0, 0.3, 0.0] 是在口语 13 题上挑出来的，绕弯 43 题只是「没掉」。
但 43 题里的路由缓存、改写缓存、三档标注，全都在过去几轮被反复看过。
也就是说：**所有已知数字都在同一批题上来回优化过**，涨分可能是在背答案。

所以这里造一批新题，规则三条：
  ① 考点必须是已有 56 题**没覆盖过的小节**（逐条比对 SUB_MAP 写的）
  ② 题目写在跑之前，标注只按书本内容定，不看检索结果，跑完不改
  ③ 口语 / 绕弯各占一半——口语考「不知道术语怎么问」，绕弯考「知道但不说术语」

跑完只看一个东西：这 20 题上的分，和那 56 题差多少。
差得多 = 过拟合；差不多 = 这套权重是真的。

标注
----
want_sec  二级章节前缀（节级判定）
want_sub  三级小节号（小节级判定，真话口径）
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys
import time

from eval import chunks, SUB_MAP, rank_of, rerank_text
from rewrite import search_multi, rewrite, load_cache, save_cache
from route import route
from llm import load_key
from metrics import mrr
from answer import ROUTE_BONUS

# (考点, 风格, 问法, 期望章节, 期望小节)
# 风格 oral = 极短口语（无术语、无谓词、丢一句就走）
# 风格 indirect = 绕弯书面（有完整句子，但故意不说术语）
CASES = [
    # ── 口语 11 道 ────────────────────────────────────────────
    ("智能体类型",  "oral", "智能体还能分成哪些类",            "## 1.1 ", ["1.1.3"]),
    ("模型幻觉",    "oral", "它一本正经地瞎编，这毛病叫啥",    "## 3.3 ", ["3.3.2"]),
    ("四种记忆",    "oral", "记忆还分几种",                    "## 8.2 ", ["8.2.5"]),
    ("高级检索策略", "oral", "问法和原文用词不一样就搜不到",   "## 8.3 ", ["8.3.5"]),
    ("TerminalTool", "oral", "让它自己敲命令去看文件",         "## 9.5 ", ["9.5.1"]),
    ("三协议分工",  "oral", "几种通信协议各自管什么",          "## 10.1 ", ["10.1.2"]),
    ("MCP 传输",    "oral", "接工具那层数据怎么传过去",        "## 10.2 ", ["10.2.3"]),
    ("LLM Judge",   "oral", "拿另一个模型当裁判打分",          "## 12.4 ", ["12.4.4"]),
    ("ToolRegistry", "oral", "工具一大堆怎么统一登记和调用",   "## 14.4 ", ["14.4.3"]),
    ("NPC 记忆",    "oral", "游戏里NPC怎么记住聊过啥",         "## 15.2 ", ["15.2.3"]),
    ("README",      "oral", "项目写完要交个说明文档",          "## 16.4 ", ["16.4.1"]),

    # ── 绕弯 9 道 ─────────────────────────────────────────────
    ("Workflow对比", "indirect", "让AI照固定流程走，和放手让它自己想办法，本质区别在哪", "## 1.4 ", ["1.4.3"]),
    ("专家系统",    "indirect", "把人类专家的经验写成规则存进机器、让它照着推，早年这套东西叫什么", "## 2.1 ", ["2.1.2"]),
    ("Decoder-Only", "indirect", "只保留解码器那一半、天生只能往后预测的结构是什么", "## 3.1 ", ["3.1.3"]),
    ("框架选型",    "indirect", "选智能体框架的时候该看哪些维度", "## 6.1 ", ["6.1.2"]),
    ("GSSC",        "indirect", "上下文不是随便拼字符串，得按哪几步组装", "## 9.3 ", ["9.3.3"]),
    ("奖励函数",    "indirect", "强化学习里奖励那一项该怎么设计", "## 11.2 ", ["11.2.2"]),
    ("评估指标",    "indirect", "怎么判断训出来的模型到底行不行", "## 11.5 ", ["11.5.1"]),
    ("Pydantic",    "indirect", "别到处传字典、把数据结构定死，用什么办法", "## 13.2 ", ["13.2.3"]),
    ("好感度计算",  "indirect", "NPC对玩家的好感是怎么算出来的", "## 15.3 ", ["15.3.2"]),
]

assert len(CASES) == 20, "说好 20 题"


def stats(rows, field, level):
    """level: 'sec' 走 want_sec 前缀，'sub' 走 want_sub 交集"""
    key = field + "_" + level
    ranks = [r[key] for r in rows]
    n = len(ranks)
    h1 = sum(1 for x in ranks if x == 1)
    h3 = sum(1 for x in ranks if x)
    return h1, h3, n, mrr([x for x in ranks if x]), mrr(ranks)


def main():
    limit = None
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])

    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    items = CASES[:limit] if limit else CASES
    cache = load_cache()
    rows = []
    t0 = time.time()

    print("=" * 78)
    print("新题验证：%d 道，全部考点均为 56 道旧题未覆盖的小节" % len(items))
    print("=" * 78)

    for n, (name, style, q, sec, subs) in enumerate(items, 1):
        picks, raw = route(q, key)
        # 比较用 startswith：picks 里存的是完整标题「## 1.1 什么是智能体？」，
        # 而标注只写前缀「## 1.1 」。用 == 会全部判成「没命中」，白丢一路信号。
        routed_hit = any(p.strip().startswith(sec.strip()) for p in picks)

        if q in cache:
            rews = cache[q]
        else:
            rews = rewrite(q, key)
            cache[q] = rews
            save_cache(cache)
        bonus = ROUTE_BONUS if picks else 0.0

        # 三臂对照，精排三遍（候选池由「带不带改写」和「加不加路由补块」决定）：
        #   纯原问   = 完全不加任何辅助信号，就是最朴素的向量召回 + 精排
        #   只改写   = 加改写补漏，但不给路由加分
        #   全开     = 生产链路
        # 注：路由会把章节里的块加进候选池，所以「全开」的候选池比另外两臂大，
        # 这本身就是路由的贡献之一，不能拿同一个池子重排来假装公平对比。
        hits_raw = search_multi(q, None, k=3)
        hits_rw = search_multi(q, rews, k=3)
        hits_all = search_multi(q, rews, k=3, picks=picks, bonus=bonus)
        arms = {"raw": hits_raw, "rw": hits_rw, "all": hits_all}

        row = {"n": n, "name": name, "style": style, "q": q,
               "sec": sec, "subs": subs, "picks": picks, "routed": routed_hit,
               "raw_scores": raw}
        for a, hits in arms.items():
            row[a + "_sec"] = rank_of(hits, [sec])
            row[a + "_sub"] = rank_of(hits, [sec], subs)
            if a == "all":
                row["hits"] = [(i, float(s)) for i, s in hits]

        rows.append(row)

        def fmt(x):
            return "第%d名" % x if x else "✗"

        print("\n%2d [%s] %s「%s」" % (n, "口语" if style == "oral" else "绕弯", name, q))
        print("    期望 %s %s   路由命中:%s  %s" % (
            sec.strip(), subs, "是" if routed_hit else "否",
            [p.replace("## ", "")[:10] for p in picks]))
        print("    原问 %-5s | 只改写 %-5s | 全开 %-5s   （小节级）" % (
            fmt(row["raw_sub"]), fmt(row["rw_sub"]), fmt(row["all_sub"])))
        for r, (i, s) in enumerate(row["hits"], 1):
            mark = "✓" if (SUB_MAP[i] & set(subs)) else " "
            print("      %s#%d %.3f %-9s %s" % (
                mark, r, s, sorted(SUB_MAP[i]),
                chunks[i][2].strip().replace("\n", " ")[:46]))

    dt = time.time() - t0
    N = len(rows)
    print("\n" + "=" * 78)
    print("汇总（%d 题，耗时 %.0fs，平均 %.1fs/题）" % (N, dt, dt / N))
    print("=" * 78)
    print("%-14s %-6s %-6s %-6s %-6s" % ("方案", "节@1", "节@3", "小节@1", "小节@3"))
    for a, tag in (("raw", "纯原问"), ("rw", "只改写"), ("all", "改写+路由")):
        s1, s3, n, _, _ = stats(rows, a, "sec")
        b1, b3, n2, _, _ = stats(rows, a, "sub")
        print("%-14s %.3f  %.3f  %.3f  %.3f" % (
            tag, s1 / n, s3 / n, b1 / n2, b3 / n2))

    print("\n%-14s %-6s %-6s  %s" % ("方案", "小节MRR", "节MRR", "（MRR 命中才算，未命中记 0）"))
    for a, tag in (("raw", "纯原问"), ("rw", "只改写"), ("all", "改写+路由")):
        _, _, _, msub, msec = stats(rows, a, "sub")
        print("%-14s %.3f    %.3f" % (tag, msub, msec))

    # 按风格拆开看——口语和绕弯哪一档在裸奔，一眼就看出来
    print("\n按风格拆（小节级）：")
    for st, tag in (("oral", "口语"), ("indirect", "绕弯")):
        sub = [r for r in rows if r["style"] == st]
        if not sub:
            continue
        print("  %s %d 题：" % (tag, len(sub)), end="")
        for a, at in (("raw", "原问"), ("all", "全开")):
            h1 = sum(1 for r in sub if r[a + "_sub"] == 1)
            h3 = sum(1 for r in sub if r[a + "_sub"])
            print("  %s 小@1 %.3f 小@3 %.3f" % (at, h1 / len(sub), h3 / len(sub)), end="")
        print()

    routed = sum(1 for r in rows if r["routed"])
    print("\n路由准确率 %d/%d = %.3f" % (routed, N, routed / N))

    print("\n失败清单（小节级，全开仍错）：")
    for r in rows:
        if r["all_sub"] == 1:
            continue
        rank = "第%d名" % r["all_sub"] if r["all_sub"] else "没进前3"
        print("  %-14s %-22s %s（路由%s）" % (
            r["name"], r["q"], rank, "中" if r["routed"] else "偏"))

    with open("eval_holdout_result.json", "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items()} for r in rows],
                  f, ensure_ascii=False, indent=1)
    print("\n明细已存 eval_holdout_result.json")


if __name__ == "__main__":
    main()
