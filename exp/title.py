"""实验：修「标题饱和」——精排阶段不给标题 + 放宽每节进精排的块数

起因：问「什么是智能体？」，top1 是 0.999 的 191 字尾巴块，真定义（663 字）没进前 3。
根因：
  1.1 节被切成 12 块，12 块标题全一样（## 1.1 什么是智能体？）。
  粗筛和精排都把标题拼进正文 → cross-encoder 看到问题和标题字面雷同，
  12 块分数全挤在 0.974~0.999，区分度归零；191 字的短块标题占比最大，反而赢。
  同时粗筛余弦也是短块占便宜，真定义只排第 3，被 PER_SECTION=2 挡在精排门外。

对照基线（43 题）：@1 = 0.651  @3 = 0.907
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import time

from embed import embed_query
from eval import vecs, reranker, chunks, CASES

N_SECTIONS = 15


def search(question, per_section, use_title, k=3):
    qv = embed_query(question)
    scores = vecs @ qv
    order = np.argsort(-scores)

    cand, count, nsec = [], {}, 0
    for i in order:
        title = chunks[int(i)][1]
        c = count.get(title, 0)
        if c >= per_section:
            continue
        if c == 0:
            if nsec >= N_SECTIONS:
                break
            nsec += 1
        count[title] = c + 1
        cand.append(int(i))

    if use_title:
        pairs = [(question, chunks[i][1] + "\n" + chunks[i][2]) for i in cand]
    else:
        pairs = [(question, chunks[i][2]) for i in cand]
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


def evaluate(per_section, use_title):
    rows = []
    for q, answers, kind in CASES:
        res = search(q, per_section, use_title)
        rank = None
        for r, (idx, s) in enumerate(res, 1):
            if any(chunks[idx][1].startswith(a) for a in answers):
                rank = r
                break
        rows.append((q, rank, res, kind))
    return rows


CONFIGS = [
    ("B  精排去标题  PER=2", 2, False),
    ("C  精排去标题  PER=3", 3, False),
]

if __name__ == "__main__":
    n = len(CASES)
    print("基线（精排带标题 PER=2）：@1 0.651  @3 0.907\n")

    for name, per, ut in CONFIGS:
        t0 = time.time()
        rows = evaluate(per, ut)
        dt = time.time() - t0

        h1 = sum(1 for _, r, _, _ in rows if r == 1)
        h3 = sum(1 for _, r, _, _ in rows if r is not None and r <= 3)
        print("=" * 62)
        print("%s  —— %.0fs（%.1fs/题）" % (name, dt, dt / n))
        print("  @1 = %2d/%d = %.3f   （基线 0.651，%+.3f）" % (h1, n, h1 / n, h1 / n - 0.651))
        print("  @3 = %2d/%d = %.3f   （基线 0.907，%+.3f）" % (h3, n, h3 / n, h3 / n - 0.907))
        print()

        kinds = []
        for _, _, _, kd in rows:
            if kd not in kinds:
                kinds.append(kd)
        for kind in kinds:
            sub = [r for _, r, _, kd in rows if kd == kind]
            s1 = sum(1 for r in sub if r == 1)
            s3 = sum(1 for r in sub if r is not None and r <= 3)
            print("  %-4s %2d 题 | @1 %.2f | @3 %.2f" % (kind, len(sub), s1 / len(sub), s3 / len(sub)))
        print()

        print("  没进前 3 的题：")
        for q, r, res, kind in rows:
            if r is not None and r <= 3:
                continue
            print("    [%s] %s" % ("-" if r is None else "第%d名" % r, q[:46]))
        print()
