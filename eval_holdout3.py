"""第三留出集：反查词改完之后的干净验收

为什么还要第三批
----------------
前两批（h1/h2）已经不能用来证明「反查词有效」了：
  · h2 的 5 道路由错误，我是看着它们去补的词 —— 它涨是必然的，不能当成绩；
  · h1 虽然没照它的题补词，但它早被用来挑过 sub1 注入方式，也算开发集。
所以再造 20 道，用它给「反查词」这个改动盖最后一次章。

规矩和前两批一样（唯一的价值就在这三条上）
  ① 考点全部来自前 83 题一个都没碰过的小节（183 个可选，逐条比对）
  ② 只按书本正文定答案，写完不看检索结果，跑完不改标注
  ③ 口语 / 绕弯各 10 道

和 h2 唯一的区别：这批题的「反查词」是从三级标题（`### x.y.z`）里抄的实词，
不是从这 20 道题里抄的。但也要承认：写题的时候我脑子里已经有那份词表了，
所以这批数仍然偏乐观，只是比 h2 好得多。
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import time

import numpy as np

from eval import chunks, SUB_MAP, rank_of
from rewrite import search_multi, rewrite, load_cache, save_cache
from route import route
from llm import load_key
from metrics import mrr
from answer import ROUTE_BONUS

# (考点, 风格, 问法, 期望章节, 期望小节)
CASES = [
    # ── 口语 10 道：不知道术语，只描述需求 ──────────────────────
    ("ELIZA匹配",   "oral", "早期那种聊天程序，是怎么从我说的句子里挑出关键词再接一句的", "## 2.2 ", ["2.2.2"]),
    ("分词",        "oral", "模型看不懂字，得先切成它认识的小块，这步怎么做的",     "## 3.2 ", ["3.2.2"]),
    ("执行器",      "oral", "计划列好了，谁去一步步做、上一步的结果怎么带给下一步", "## 4.3 ", ["4.3.3"]),
    ("反思成本",    "oral", "让它自己反复返工，划算吗，代价在哪儿",               "## 4.4 ", ["4.4.5"]),
    ("邮件助手",    "oral", "收邮件、看明白啥意思、再回一封，这一整套怎么自动搭起来", "## 5.5 ", ["5.5.2"]),
    ("SimpleAgent", "oral", "框架里最基础那个能对话的智能体，想写一个自己的",     "## 7.4 ", ["7.4.1"]),
    ("问答助手类",  "oral", "把文档问答做成一个类，那个类里都装了些什么",         "## 8.4 ", ["8.4.2"]),
    ("工具协同",    "oral", "敲命令查出来的东西，怎么顺手存进记忆和笔记里",       "## 9.5 ", ["9.5.4"]),
    ("GRPO配置",    "oral", "组内比较那套训练，实际跑起来要配哪些参数",           "## 11.4 ", ["11.4.2"]),
    ("对话UI",      "oral", "游戏里跟NPC打字聊天的那个框怎么搭",                  "## 15.6 ", ["15.6.2"]),

    # ── 绕弯 10 道：知道概念，但故意不说术语 ────────────────────
    ("智能体循环",  "indirect", "它不是一个动作就完事，而是跟环境来回转圈，这个圈分哪几个阶段", "## 1.2 ", ["1.2.2"]),
    ("心智社会影响", "indirect", "几十年前那个「智能是一群小角色凑出来」的说法，后来在哪个方向上被真正用上了", "## 2.3 ", ["2.3.3"]),
    ("开发团队",    "indirect", "让几个AI分别扮产品、开发、测试，凑成一个小组把一个网站做出来", "## 6.2 ", ["6.2.2"]),
    ("三步流程",    "indirect", "用节点和边串一个先听懂、再查、最后答的小流程",     "## 6.5 ", ["6.5.2"]),
    ("本地推理",    "indirect", "不想走云端，让模型跑在自己机器上，还得扛住一批并发", "## 7.2 ", ["7.2.2"]),
    ("RAG分层",     "indirect", "把外挂知识库那套拆开，从上到下分成几层",          "## 8.3 ", ["8.3.4"]),
    ("MCP生态",     "indirect", "现成的工具服务别人早就写好了，上哪儿去找",        "## 10.2 ", ["10.2.5"]),
    ("评估方法",    "indirect", "判断一批机器造出来的题目靠不靠谱，有哪几种打法",   "## 12.4 ", ["12.4.1"]),
    ("请求封装",    "indirect", "前端调后端的那些请求，怎么统一包一层、超时设多长", "## 13.5 ", ["13.5.3"]),
    ("Fork仓库",    "indirect", "把老师那个仓库复制一份到自己账号下，之后还要能同步更新", "## 16.3 ", ["16.3.2"]),
]

assert len(CASES) == 20, "说好 20 题"


def rerank(cand, raw, picks=None, bonus=0.0, k=3):
    """本地重排：和 rewrite.search_multi 末尾那段逐行对齐

    精排分已经算好了，这里只改排序。按节去重、每个节只出一个代表，
    和线上口径完全一致——不一致的话后面所有数字都不可信。
    """
    rs = raw.copy()
    if picks and bonus:
        w = {}
        for pos, p in enumerate(picks):
            if isinstance(bonus, (list, tuple)):
                w[p.strip()] = bonus[pos] if pos < len(bonus) else bonus[-1]
            else:
                w[p.strip()] = bonus
        for j, i in enumerate(cand):
            x = w.get(chunks[i][1].strip())
            if x:
                rs[j] += x
    out, seen = [], set()
    for j in np.argsort(-rs):
        t = chunks[cand[int(j)]][1]
        if t in seen:
            continue
        seen.add(t)
        out.append((cand[int(j)], float(raw[int(j)])))
        if len(out) == k:
            break
    return out


def stats(rows, field, level):
    key = field + "_" + level
    ranks = [r[key] for r in rows]
    n = len(ranks)
    return (sum(1 for x in ranks if x == 1) / n,
            sum(1 for x in ranks if x) / n,
            mrr(ranks))


def main():
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    cache = load_cache()
    rows, pools = [], []
    t0 = time.time()

    print("=" * 78)
    print("第三留出集：%d 道干净题（考点均未被前 83 题覆盖）" % len(CASES))
    print("=" * 78)

    for n, (name, style, q, sec, subs) in enumerate(CASES, 1):
        picks, rtxt = route(q, key)
        routed_hit = any(p.strip().startswith(sec.strip()) for p in picks)

        if q in cache:
            rews = cache[q]
        else:
            rews = rewrite(q, key)
            cache[q] = rews
            save_cache(cache)

        # 三遍精排，对应三种候选池：
        #   ① 纯原问     ② 原问+改写     ③ 原问+改写+路由补块
        hits_raw = search_multi(q, None, k=3)
        hits_rw = search_multi(q, rews, k=3)
        cand_c, raw_c, _ = search_multi(q, rews, k=3, picks=picks,
                                        bonus=0.0, return_raw=True)
        hits_all = rerank(cand_c, raw_c, picks, ROUTE_BONUS)

        # 对拍：本地重排必须和生产函数一致，否则后面所有数字作废
        if n == 1:
            prod = search_multi(q, rews, k=3, picks=picks, bonus=ROUTE_BONUS)
            assert [(i, round(s, 6)) for i, s in prod] == \
                   [(i, round(s, 6)) for i, s in hits_all], "本地重排与生产链路不一致"

        row = {"n": n, "name": name, "style": style, "q": q,
               "sec": sec, "subs": subs, "picks": picks, "routed": routed_hit}
        for a, hits in (("raw", hits_raw), ("rw", hits_rw), ("all", hits_all)):
            row[a + "_sec"] = rank_of(hits, [sec])
            row[a + "_sub"] = rank_of(hits, [sec], subs)

        rows.append(row)
        # 候选池留档：以后换任何排序权重都不用再精排
        pools.append({"n": n, "q": q, "picks": picks, "route_raw": rtxt,
                      "cand": [int(i) for i in cand_c],
                      "raw": [float(x) for x in raw_c]})

        def fmt(x):
            return "第%d名" % x if x else "✗"

        print("\n%2d [%s] %s「%s」" % (n, "口语" if style == "oral" else "绕弯", name, q))
        print("    期望 %s %s  路由命中:%s %s" % (
            sec.strip(), subs, "是" if routed_hit else "否",
            [p.replace("## ", "")[:10] for p in picks]))
        print("    原问 %-5s | 只改写 %-5s | 全开 %-5s  （小节级）" % (
            fmt(row["raw_sub"]), fmt(row["rw_sub"]), fmt(row["all_sub"])))

    dt = time.time() - t0
    N = len(rows)
    print("\n" + "=" * 78)
    print("汇总（%d 题，耗时 %.0fs，平均 %.1fs/题）" % (N, dt, dt / N))
    print("=" * 78)
    print("%-12s %-8s %-8s %-8s %-8s" % ("方案", "节@1", "节@3", "小@1", "小@3"))
    for a, tag in (("raw", "纯原问"), ("rw", "只改写"), ("all", "改写+路由")):
        s1, s3, _ = stats(rows, a, "sec")
        b1, b3, _ = stats(rows, a, "sub")
        print("%-12s %.3f    %.3f    %.3f    %.3f" % (tag, s1, s3, b1, b3))

    print("\n%-12s %-8s %-8s" % ("方案", "小节MRR", "节MRR"))
    for a, tag in (("raw", "纯原问"), ("rw", "只改写"), ("all", "改写+路由")):
        _, _, msub = stats(rows, a, "sub")
        _, _, msec = stats(rows, a, "sec")
        print("%-12s %.3f    %.3f" % (tag, msub, msec))

    print("\n按风格拆（小节级 @1）：")
    for st, tag in (("oral", "口语"), ("indirect", "绕弯")):
        sub = [r for r in rows if r["style"] == st]
        if not sub:
            continue
        print("  %s %d 题：" % (tag, len(sub)), end="")
        for a, at in (("raw", "原问"), ("rw", "只改写"), ("all", "全开")):
            print("  %s %.3f" % (at, sum(1 for r in sub if r[a + "_sub"] == 1) / len(sub)), end="")
        print()

    routed = sum(1 for r in rows if r["routed"])
    print("\n路由准确率 %d/%d = %.3f" % (routed, N, routed / N))

    print("\n失败清单（小节级，全开仍错）：")
    for r in rows:
        if r["all_sub"] == 1:
            continue
        print("  %-12s %-24s %s（路由%s）" % (
            r["name"], r["q"],
            "第%d名" % r["all_sub"] if r["all_sub"] else "没进前3",
            "中" if r["routed"] else "偏"))

    json.dump(rows, open("eval_holdout3_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(pools, open("eval_holdout3_pools.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n明细 eval_holdout3_result.json ／ 候选池 eval_holdout3_pools.json")


if __name__ == "__main__":
    main()
