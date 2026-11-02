"""章节路由：把 79 个标题交给 LLM 做选择题，而不是让它凭空猜术语

上一轮发现：极短口语档 @1 只有 0.385，改写只救回 1 题。
看改写内容就明白了——「边想边干是啥意思」被改写成「边想边干是什么意思」，
LLM 从头到尾没认出这是 ReAct。它不是不会，是**手里没有词典**。

改写 prompt 让 LLM 补「技术资料库里可能出现的说法」，这是泛指的。
它不知道这本书有 79 个章节，更不知道 `## 4.2 ReAct` 就在里面。

现在把标题列给它，任务从「造句」变成「选择」。

这也是对「LLM 不知道资料库里有什么」这个一直悬着的风险，第一次正面回应。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from embed import embed_query
from eval import vecs, reranker, chunks
from llm import load_key
from route import route, TITLES, MENU      # 路由逻辑已固化到 route.py


def search_routed(question, picks, k=3, per=2, nsec=15, bonus=0.0):
    """原问题选主候选 + 路由到的章节补候选，精排仍用原问题

    bonus：给路由选中的章节加分，让路由参与排序而不只是递候选。
    为什么需要它：极短口语档上 cross-encoder 本身是瞎的——
    「边想边干是啥意思」被精排打给 (开头) 块 0.271，而路由明明选中了 4.2 ReAct。
    信号比裁判准的时候，不能让裁判独裁。
    """
    qv = embed_query(question)
    scores = vecs @ qv

    # ① 主候选：原问题余弦，按节发配额
    order = np.argsort(-scores)
    main, count, nsec_seen = [], {}, 0
    for i in order:
        idx = int(i)
        title = chunks[idx][1]
        c = count.get(title, 0)
        if c >= per:
            continue
        if c == 0:
            if nsec_seen >= nsec:
                break
            nsec_seen += 1
        count[title] = c + 1
        main.append(idx)

    # ② 路由补充：被选中的章节里，取余弦最高的 per 块
    seen = set(main)
    extra = []
    for title in picks:
        idxs = [i for i in range(len(chunks)) if chunks[i][1].strip() == title]
        idxs.sort(key=lambda i: -scores[i])
        for i in idxs[:per]:
            if i not in seen:
                seen.add(i)
                extra.append(i)

    cand = main + extra
    pairs = [(question, chunks[i][1] + "\n" + chunks[i][2]) for i in cand]
    rs = np.array(reranker.predict(pairs, batch_size=16), dtype=float)

    if bonus > 0:
        pick_set = {p.strip() for p in picks}
        for j, i in enumerate(cand):
            if chunks[i][1].strip() in pick_set:
                rs[j] += bonus

    picked, seen_t = [], set()
    for j in np.argsort(-rs):
        title = chunks[cand[int(j)]][1]
        if title in seen_t:
            continue
        seen_t.add(title)
        picked.append((cand[int(j)], float(rs[int(j)])))
        if len(picked) == k:
            break
    return picked


def rank_of(hits, want):
    for r, (idx, s) in enumerate(hits, 1):
        if chunks[idx][1].startswith(want):
            return r
    return None


if __name__ == "__main__":
    import time
    from eval_oral import CASES

    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    print("目录共 %d 条，MENU 长度 %d 字\n" % (len(TITLES), len(MENU)))

    t0 = time.time()
    rows = []
    for n, (name, want, q) in enumerate(CASES, 1):
        picks, raw = route(q, key)
        hits = search_routed(q, picks)
        r = rank_of(hits, want)
        rows.append((name, q, r, picks, raw, hits))
        print("%2d/%d %-14s %-16s → %-6s  路由到: %s" % (
            n, len(CASES), name, q, "第%d名" % r if r else "✗",
            " / ".join(p.replace("## ", "")[:18] for p in picks)))

    dt = time.time() - t0
    N = len(CASES)
    h1 = sum(1 for x in rows if x[2] == 1)
    h3 = sum(1 for x in rows if x[2] is not None)

    print("\n" + "=" * 74)
    print("章节路由（极短口语档）  耗时 %.0fs" % dt)
    print("  @1 = %2d/%d = %.3f" % (h1, N, h1 / N))
    print("  @3 = %2d/%d = %.3f" % (h3, N, h3 / N))
    print("  对照：纯原问 0.231 / 0.538，原改写方案 0.385 / 0.615")
    print("=" * 74)

    print("\n路由错在哪（判断章节还算对了才谈块）：")
    for name, q, r, picks, raw, hits in rows:
        if r:
            continue
        print("  %-14s %s" % (name, q))
        print("       LLM 原始输出: %r" % raw[:60])
        print("       top1: %.3f %s" % (hits[0][1], chunks[hits[0][0]][1].strip()[:40]))
