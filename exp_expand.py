"""对比三种融合方式：改写到底该在哪一步介入

A 基线     只问原问题                    —— 昨天的最好成绩 @1 0.651 / @3 0.907
B max 融合 4 个 query 每块取最高分        —— 全量跑完 @1 0.605 / @3 0.907（救 4 坏 4）
C 候选扩充 原问选主候选，改写只补漏       —— 本实验要验的

B 的病因：改写句非常准（StateGraph / BaaS LLMOps），但越准越容易撞上「别处提到过
这个词」的块。max 取最高分等于让改写和原问题平起平坐，于是「提及型」块把「专讲型」
那一节挤掉了。

C 的思路：排序权只给原问题，改写只负责把可能漏掉的节捞进候选池。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import numpy as np

from embed import embed_query
from eval import vecs, reranker, chunks

CACHE = "rewrites.json"


def collect(scores, nsec, per, skip_titles=()):
    """按分数从高到低扫，按节发配额，返回块下标"""
    order = np.argsort(-scores)
    out, count, nsec_seen = [], {}, 0
    for i in order:
        title = chunks[int(i)][1]
        if title in skip_titles:
            continue
        c = count.get(title, 0)
        if c >= per:
            continue
        if c == 0:
            if nsec_seen >= nsec:
                break
            nsec_seen += 1
        count[title] = c + 1
        out.append(int(i))
    return out


def rerank(question, cand, k=3):
    pairs = [(question, chunks[i][1] + "\n" + chunks[i][2]) for i in cand]
    rs = reranker.predict(pairs, batch_size=16)
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


def run_base(q, rews):
    qv = embed_query(q)
    return rerank(q, collect(vecs @ qv, 15, 2))


def run_max(q, rews):
    qvs = [embed_query(x) for x in [q] + rews]
    s = np.max([vecs @ v for v in qvs], axis=0)
    return rerank(q, collect(s, 15, 2))


def run_expand(q, rews, extra_sec=5, extra_per=1):
    """主候选来自原问题，改写只补原问题没覆盖到的节"""
    qv = embed_query(q)
    main = collect(vecs @ qv, 15, 2)
    used = {chunks[i][1] for i in main}

    extra = []
    for r in rews:
        extra += collect(vecs @ embed_query(r), extra_sec, extra_per, skip_titles=used)
        used |= {chunks[i][1] for i in extra}

    return rerank(q, main + [i for i in extra if i not in main])


CASES = [
    ("用状态图来编排智能体流程的框架是哪个？", ["## 6.5 "], "弄坏"),
    ("哪个平台是开源的，还融合了后端即服务和 LLMOps 理念？", ["## 5.3 "], "弄坏"),
    ("哪个平台本质上是通用工作流自动化、大模型只是其中一环？", ["## 5.5 "], "弄坏"),
    ("智能体内部「先想、再动手、再看结果」这个循环怎么称呼？", ["## 1.3 "], "弄坏"),
    ("先检索外部资料再让模型生成答案，这套做法叫什么？", ["## 8.3 "], "救回"),
    ("一边推理一边调用工具的范式是怎么运作的？", ["## 4.2 "], "救回"),
    ("怎么衡量一个模型的函数调用能力？", ["## 12.2 "], "救回"),
    ("框架里怎么统一管理一大堆工具，让模型知道有哪些可用？", ["## 7.5 "], "救回"),
]

if __name__ == "__main__":
    cache = json.load(open(CACHE, encoding="utf-8"))
    tally = {"A 基线": 0, "B max": 0, "C 扩充": 0}

    for q, answers, tag in CASES:
        rews = cache.get(q, [])
        line = {}
        for name, fn in [("A 基线", run_base), ("B max", run_max), ("C 扩充", run_expand)]:
            res = fn(q, rews)
            rank = next((r for r, (i, s) in enumerate(res, 1)
                         if any(chunks[i][1].startswith(a) for a in answers)), None)
            line[name] = rank
            if rank is not None and rank <= 3:
                tally[name] += 1
        print("[%s] %s" % (tag, q[:36]))
        print("     基线 %-4s | max %-4s | 扩充 %-4s"
              % (line["A 基线"] or "-", line["B max"] or "-", line["C 扩充"] or "-"))
        print()

    print("=" * 52)
    for k, v in tally.items():
        print("  %-8s 进前3 %d/%d" % (k, v, len(CASES)))
