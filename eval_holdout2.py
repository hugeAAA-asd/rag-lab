"""第二留出集：20 道终裁题

为什么还要第二批
----------------
第一批 20 题（eval_holdout.py）已经用来做了三件事：
  ① 证明过拟合  ② 发现候选缺失  ③ 挑出 sub1 注入方式
也就是说它现在也是开发集了。再拿它挑「置信度缩放」的参数，
就是在同一批题上第二轮磨参数，磨出来的数一样不可信。

所以再造 20 道，考点严格来自 287 个小节里**前 76 题一个都没碰过**的部分
（227 个可选，逐条比对 SUB_MAP）。这批题的用途只有一个：
最后用一次，判断「置信度方案」到底有没有比现线上更好。

三条规矩（和第一批相同）
  ① 考点全部未覆盖，题目之间也不重复
  ② 标注只按书本正文定，写完不看检索结果，跑完不改
  ③ 口语 / 绕弯各 10 道

额外做一件事：把每题的候选池和精排分一起存下来
------------------------------------------------
精排是唯一瓶颈（每题 ~10s）。存下 (候选块下标, 原始精排分) 之后，
任何只改排序的方案（bonus 权重、置信度缩放、top-k）都能离线重排，
不再花一秒精排。后续扫参数的成本从「每题 10s」降到「0」。
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
    ("ReAct局限",   "oral", "边想边干这招有啥毛病",                    "## 4.2 ", ["4.2.4"]),
    ("RAG原理",     "oral", "它先去翻完资料再开口，中间怎么串的",      "## 8.3 ", ["8.3.2"]),
    ("Terminal功能", "oral", "让它敲命令那个工具到底能干几件事",        "## 9.5 ", ["9.5.2"]),
    ("自定义MCP",   "oral", "公司内部的数据想让它也能调，咋办",        "## 10.5 ", ["10.5.1"]),
    ("评估基准",    "oral", "比智能体能耐，现在都有哪些公认的考卷",    "## 12.1 ", ["12.1.2"]),
    ("任务规划",    "oral", "谁负责把一个大课题拆成几件小事",          "## 14.5 ", ["14.5.1"]),
    ("工具链",      "oral", "干一件事要连着用好几个工具，怎么串起来",  "## 7.5 ", ["7.5.4"]),
    ("NPC行为",     "oral", "游戏里NPC怎么自己晃悠、人凑近就搭话",     "## 15.5 ", ["15.5.3"]),
    ("模型选择",    "oral", "模型这么多，挑哪个合适",                  "## 3.2 ", ["3.2.4"]),
    ("记忆分层",    "oral", "记忆那块为啥要拆成两层来管",              "## 8.2 ", ["8.2.4"]),

    # ── 绕弯 10 道：知道概念，但故意不说术语 ────────────────────
    ("符号主义挑战", "indirect", "光靠人一条条写规则那条路，后来撞上了什么过不去的东西", "## 2.1 ", ["2.1.4"]),
    ("LLM智能体",   "indirect", "今天这一代智能体，是拿什么当脑子的",   "## 2.4 ", ["2.4.4"]),
    ("Transformer", "indirect", "扔掉循环结构、全靠注意力并行算的那个网络", "## 3.1 ", ["3.1.2"]),
    ("LangGraph优劣", "indirect", "把流程画成节点和边的那种框架，好在哪、差在哪", "## 6.5 ", ["6.5.3"]),
    ("Agent基类",   "indirect", "框架里那个所有智能体都得继承、只管定接口不管实现的模板", "## 7.3 ", ["7.3.3"]),
    ("JIT上下文",   "indirect", "资料太多塞不下，改成要用的时候再去取，这叫啥思路", "## 9.2 ", ["9.2.2"]),
    ("自动检测",    "indirect", "没告诉它你用的哪家服务，它自己怎么就认出来了", "## 7.2 ", ["7.2.3"]),
    ("自底向上建模", "indirect", "先把最小的零件定死，再一层层拼成复杂结构", "## 13.2 ", ["13.2.4"]),
    ("多智能体角色", "indirect", "做个旅行助手为啥要拆成好几个各管一段的角色", "## 13.3 ", ["13.3.2"]),
    ("ANP服务发现", "indirect", "服务一多，怎么知道哪个能干什么、上哪儿找它", "## 10.4 ", ["10.4.2"]),
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
    print("第二留出集：%d 道终裁题（考点均未被前 76 题覆盖）" % len(CASES))
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

    json.dump(rows, open("eval_holdout2_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(pools, open("eval_holdout2_pools.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n明细 eval_holdout2_result.json ／ 候选池 eval_holdout2_pools.json")


if __name__ == "__main__":
    main()
