"""查询改写：把用户的口语问题，翻译成资料库里的说法

要解决的病：词汇鸿沟。
  用户：先检索外部资料再让模型生成答案，这套做法叫什么？
  书里：## 8.3 RAG系统：知识检索增强
  两边意思一样，一个字都不重合，bi-encoder 给到 0.790，前 3 都进不去。

做法：让 LLM 生成 3 个改写，连同原问题一共 4 个 query 去粗筛，
     每个块取 4 个 query 里的最高分（max 融合，不是平均），最后用原问题精排。

为什么是 max 而不是平均：
  平均会把「ReAct」这种强信号和两个口语变体搅在一起，稀释掉；
  max 只要有一个 query 咬住了就算数，不会互相拖累。

为什么精排必须用原问题：
  粗筛要的是「召回」——只要相关就行，怎么改写都可以；
  精排要的是「这段材料能不能回答用户真正问的那个问题」，所以必须拿原话去比。
  用改写句做精排，等于在回答一个用户没问的问题。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import numpy as np

from embed import embed_query
from eval import vecs, reranker, chunks
from llm import ask_llm, load_key

N_REWRITE = 3
CACHE = "rewrites.json"

REWRITE_SYSTEM = (
    "你是检索查询改写器。把用户的口语问题改写成技术资料库里可能出现的说法，"
    "补上这个领域的专业术语、英文名、缩写。"
    "只做改写，不回答问题。"
)

REWRITE_TEMPLATE = (
    "把下面的问题改写成 %d 个不同的检索查询，每行一个，不要编号，不要解释：\n\n问题：%s"
)


def rewrite(question, key=None, n=N_REWRITE):
    """返回 n 条改写。任何失败都退回空列表——没有改写也要能正常检索。"""
    try:
        txt, _ = ask_llm(REWRITE_TEMPLATE % (n, question), key or load_key(),
                         REWRITE_SYSTEM, temperature=0.3)
    except Exception as e:
        print("   ! 改写失败，退回原问题：%s" % e)
        return []

    lines = []
    for raw in txt.strip().splitlines():
        s = raw.strip().lstrip("0123456789.、)-•· ").strip()
        if s and s not in lines:
            lines.append(s)
    return lines[:n]


def _collect(scores, nsec, per, skip_idx=()):
    """按分数从高到低扫，按节发配额，返回块下标

    skip_idx 记的是块下标，不是节标题。
    这里踩过一次：原来按节标题跳过，结果 1.1 节有 12 块、主候选只进了 2 块，
    改写补漏时因为「1.1 这一节已经覆盖了」整节被跳过，真定义永远进不来。
    节被覆盖 ≠ 块被覆盖。
    """
    order = np.argsort(-scores)
    out, count, nsec_seen = [], {}, 0
    for i in order:
        idx = int(i)
        if idx in skip_idx:
            continue
        title = chunks[idx][1]
        c = count.get(title, 0)
        if c >= per:
            continue
        if c == 0:
            if nsec_seen >= nsec:
                break
            nsec_seen += 1
        count[title] = c + 1
        out.append(idx)
    return out


def search_multi(question, rews=None, k=3, per=2, nsec=15, extra_sec=5):
    """原问题选主候选，改写只补漏，精排只用原问题

    注意这里不是 max 融合。试过：让改写和原问题平起平坐地投票（每块取最高分），
    改写句越准越坏事——「StateGraph」「BaaS LLMOps」这些词会撞上别处「提到过」
    它们的块，把真正讲这一节的挤掉。8 题实测：基线 4/8、max 4/8、本方案 6/8，
    而且本方案是基线的严格超集（救回 2 题，0 题被弄坏）。
    """
    main = _collect(vecs @ embed_query(question), nsec, per)
    seen = set(main)

    extra = []
    for r in (rews or []):
        got = _collect(vecs @ embed_query(r), extra_sec, 1, skip_idx=seen)
        extra += got
        seen |= set(got)

    cand = main + extra

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


def load_cache():
    if os.path.exists(CACHE):
        return json.load(open(CACHE, encoding="utf-8"))
    return {}


def save_cache(data):
    json.dump(data, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    import time
    from eval import CASES

    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    cache = load_cache()
    rows = []
    t0 = time.time()

    for n, (q, answers, kind) in enumerate(CASES, 1):
        if q in cache:
            rews = cache[q]
        else:
            rews = rewrite(q, key)
            cache[q] = rews
            save_cache(cache)
            print("  改写 %d/%d: %s" % (n, len(CASES), q[:36]))

        res = search_multi(q, rews)
        rank = None
        for r, (idx, s) in enumerate(res, 1):
            if any(chunks[idx][1].startswith(a) for a in answers):
                rank = r
                break
        rows.append((q, rank, rews, kind))

    dt = time.time() - t0
    N = len(CASES)
    h1 = sum(1 for _, r, _, _ in rows if r == 1)
    h3 = sum(1 for _, r, _, _ in rows if r is not None and r <= 3)

    print("\n" + "=" * 60)
    print("查询改写 + max 融合   耗时 %.0fs" % dt)
    print("  @1 = %2d/%d = %.3f   （基线 0.651，%+.3f）" % (h1, N, h1 / N, h1 / N - 0.651))
    print("  @3 = %2d/%d = %.3f   （基线 0.907，%+.3f）" % (h3, N, h3 / N, h3 / N - 0.907))
    print("=" * 60)

    print("\n还没进的：")
    for q, r, _, kind in rows:
        if r is not None and r <= 3:
            continue
        print("  [%s] %s" % ("-" if r is None else "第%d名" % r, q[:46]))
