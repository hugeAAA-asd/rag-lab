"""实验：把「按节去重」从粗筛之后挪到精排之后
配置 A（现状）：粗筛每节留 1 块，20 节 → 精排 20 块 → 去重已在粗筛完成
配置 B（本次）：粗筛每节留 2 块，15 节 → 精排 30 块 → 精排后再去重取 3
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import time
import numpy as np
from embed import embed_query
from build_chunks import build_all
import eval as E

chunks, vecs = E.chunks, E.vecs
reranker = E.reranker          # 复用 eval 里已加载的，不再加载第二份
CASES = E.CASES


def candidates(question, n_sections, per_section):
    """粗筛：按节限量，每节最多 per_section 块，最多取 n_sections 节"""
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
            if nsec >= n_sections:
                break
            nsec += 1
        count[title] = c + 1
        cand.append(int(i))
    return cand


def search(question, n_sections, per_section, k=3):
    cand = candidates(question, n_sections, per_section)
    pairs = [(question, chunks[i][1] + "\n" + chunks[i][2]) for i in cand]
    rs = reranker.predict(pairs, batch_size=16)

    picked, seen = [], set()
    for j in np.argsort(-rs):                    # 精排之后才去重
        title = chunks[cand[int(j)]][1]
        if title in seen:
            continue
        seen.add(title)
        picked.append((cand[int(j)], float(rs[int(j)])))
        if len(picked) == k:
            break
    return picked


def run(label, n_sections, per_section):
    t0 = time.time()
    ranks, details = [], []
    for q, answers, _kind in CASES:
        res = search(q, n_sections, per_section)
        rank = None
        for r, (idx, s) in enumerate(res, 1):
            if any(chunks[idx][1].startswith(a) for a in answers):
                rank = r
                break
        ranks.append(rank)
        details.append((q, rank, res))
    dt = time.time() - t0

    hit1 = sum(1 for r in ranks if r == 1)
    hit3 = sum(1 for r in ranks if r is not None and r <= 3)
    n = len(CASES)
    print("=" * 70)
    print("%s（%d 节 x 每节 %d 块，精排 %d 对）"
          % (label, n_sections, per_section, n_sections * per_section))
    print("  Recall@1 = %2d/%d = %.3f" % (hit1, n, hit1 / n))
    print("  Recall@3 = %2d/%d = %.3f" % (hit3, n, hit3 / n))
    print("  耗时 %.1fs（每题 %.1fs）" % (dt, dt / n))
    print("  每题排名：", " ".join("-" if r is None else str(r) for r in ranks))
    print()
    print("  没进前 3 的题：")
    for q, r, res in details:
        if r is not None and r <= 3:
            continue
        print("    [%s] %s" % ("-" if r is None else "第%d名" % r, q))
        for idx, s in res[:3]:
            print("         %.3f  %s" % (s, chunks[idx][1][:34]))
    print()
    return ranks


if __name__ == "__main__":
    print("块数:", len(chunks))
    print()
    run("A 现状：粗筛每节 1 块（去重在粗筛）", 20, 1)
    run("B 新法：粗筛每节 2 块（去重在精排后）", 15, 2)
