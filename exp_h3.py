"""3 题快速验证：按 ### 切（块标题带子节名）能不能救回「标题饱和」

对照基线：level=2，PER_SECTION=2，精排带标题 → 43 题 @1 0.651 / @3 0.907
本实验只改切块层级，其他一切不动。

三题怎么选的：
  1. 「什么是智能体？」                     —— 原始 bug，top1 拿到 191 字尾巴
  2. 「先检索外部资料再让模型生成答案…」    —— exp_title 的 C 方案里丢掉的题
  3. 「一边推理一边调用工具的范式…」        —— 同上
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from build_chunks import build_all
from embed import embed_chunks, embed_query

PER_SECTION = 2
N_SECTIONS = 15
K = 3

CASES = [
    ("什么是智能体？", ["## 1.1 "], "1.1 的定义"),
    ("先检索外部资料再让模型生成答案，这套做法叫什么？", ["## 8.3 "], "8.3 RAG"),
    ("一边推理一边调用工具的范式是怎么运作的？", ["## 4.2 "], "4.2 ReAct"),
]


def search(q, chunks, vecs, reranker):
    qv = embed_query(q)
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

    pairs = [(q, chunks[i][1] + "\n" + chunks[i][2]) for i in cand]
    rs = reranker.predict(pairs, batch_size=16)

    picked, seen = [], set()
    for j in np.argsort(-rs):
        title = chunks[cand[int(j)]][1]
        if title in seen:
            continue
        seen.add(title)
        picked.append((cand[int(j)], float(rs[int(j)])))
        if len(picked) == K:
            break
    return picked


def run(tag, chunks, vecs, reranker):
    print("=" * 66)
    print(tag, "｜块数", len(chunks))
    print("=" * 66)
    hit_all = 0
    for q, answers, name in CASES:
        res = search(q, chunks, vecs, reranker)
        rank, hit_idx = None, None
        for r, (idx, s) in enumerate(res, 1):
            if any(chunks[idx][1].startswith(a) for a in answers):
                rank = r
                hit_idx = idx
                break
        if rank:
            hit_all += 1
        print("\n[%s] %s" % (name, q))
        print("  命中章节排名：%s" % ("-" if rank is None else "第 %d 名" % rank))
        for n, (idx, s) in enumerate(res, 1):
            mark = " <<<" if idx == hit_idx else ""
            print("   [%d] %.3f len=%-4d %s%s"
                  % (n, s, len(chunks[idx][2]), chunks[idx][1][:52], mark))
        if hit_idx is not None:
            print("   命中块正文开头：%s" % chunks[hit_idx][2].strip()[:70].replace("\n", " "))
    print("\n三题命中 %d/%d\n" % (hit_all, len(CASES)))
    return hit_all


if __name__ == "__main__":
    from eval import reranker

    # ── 基线：level=2，用已存好的向量 ──
    base_chunks = build_all(level=2)
    base_vecs = np.load("chunks_vec.npy")
    run("A 基线 level=2（现有）", base_chunks, base_vecs, reranker)

    # ── 实验：level=3，重新向量化 ──
    h3_chunks = build_all(level=3)
    cache = "chunks_vec_h3.npy"
    if os.path.exists(cache) and np.load(cache).shape[0] == len(h3_chunks):
        h3_vecs = np.load(cache)
        print("复用缓存的 h3 向量\n")
    else:
        print("开始向量化 %d 块…" % len(h3_chunks))
        h3_vecs = embed_chunks(h3_chunks)
        np.save(cache, h3_vecs)
        print("已保存 %s\n" % cache)
    run("B 实验 level=3（按 ### 切）", h3_chunks, h3_vecs, reranker)
