"""路由加分该怎么分档：三个候选同权，还是按顺序递减

起因（同一批数据里冒出来的两个反向需求）：
  · 第 1 题「能感知环境并自主行动的系统指什么」路由给 1.1/1.2/1.4，
    三节同权 +0.5 之后，1.4.3（raw 0.885）压过了 1.1.1（raw 0.860）。
    路由自己都说 1.1 最像，却让第 3 个候选和它平起平坐。
  · 口语档 A2A「两个智能体咋对话」路由 10.3 是对的，但那个块的 raw 分太低，
    +0.5 都顶不进前三。

一个要压后位，一个要抬首位——分级 bonus 正好同时满足。

精排只跑一次（search_multi 的 return_raw 出口），后面各组只重排，
所以扫 3 组和扫 1 组的耗时一样。
"""
import time

import numpy as np

from eval import chunks, rank_of
from eval_oral import CASES
from eval_oral_level import SUBS
from rewrite import search_multi, rewrite
from route import route
from llm import load_key
from metrics import header, row

BONUS_GROUPS = [
    ("平权 0.5", 0.5),
    ("分级 0.8/0.3/0.1", [0.8, 0.3, 0.1]),
    ("分级 1.0/0.3/0.0", [1.0, 0.3, 0.0]),
]


def rerank(cand, raw, picks, bonus, k=3):
    """给定候选和原始分，用指定 bonus 重排（和 search_multi 里那段保持一致）"""
    rs = raw.copy()
    if picks and bonus:
        w = {}
        for pos, p in enumerate(picks):
            if isinstance(bonus, (list, tuple)):
                w[p.strip()] = bonus[pos] if pos < len(bonus) else bonus[-1]
            else:
                w[p.strip()] = bonus
        for j, i in enumerate(cand):
            b = w.get(chunks[i][1].strip())
            if b:
                rs[j] += b

    picked, seen = [], set()
    for j in np.argsort(-rs):
        t = chunks[cand[int(j)]][1]
        if t in seen:
            continue
        seen.add(t)
        picked.append((cand[int(j)], float(raw[int(j)])))
        if len(picked) == k:
            break
    return picked


def main():
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    t0 = time.time()
    data = []
    for i, ((name, want, q), subs) in enumerate(zip(CASES, SUBS), 1):
        rews = rewrite(q, key)
        picks, _raw = route(q, key)
        cand, raw, _ = search_multi(q, rews, picks=picks, bonus=0,
                                    return_raw=True)
        data.append((name, want, subs, cand, raw, picks))
        print("精排 %2d/%d %s" % (i, len(CASES), name))

    print("\n精排完成 %.0fs，下面各组只是重排\n" % (time.time() - t0))

    header()
    detail = {}
    for tag, bonus in BONUS_GROUPS:
        ranks_sec, ranks_sub = [], []
        rows = []
        for name, want, subs, cand, raw, picks in data:
            hits = rerank(cand, raw, picks, bonus)
            rs = rank_of(hits, [want])
            rb = rank_of(hits, [want], subs)
            ranks_sec.append(rs)
            ranks_sub.append(rb)
            rows.append((name, rs, rb))
        detail[tag] = rows
        print(row(tag + " 节", ranks_sec))
        print(row(tag + " 小节", ranks_sub))
    print("-" * 58)

    a = detail[BONUS_GROUPS[0][0]]
    for tag, _b in BONUS_GROUPS[1:]:
        b = detail[tag]
        diff = [(x[0], x[1], y[1], x[2], y[2]) for x, y in zip(a, b)
                if x[1:] != y[1:]]
        print("\n%s 相对平权 0.5 的变化：%d 题" % (tag, len(diff)))
        for name, s1, s2, b1, b2 in diff:
            print("  %-14s 节 %s→%s   小节 %s→%s" % (
                name, s1 or "✗", s2 or "✗", b1 or "✗", b2 or "✗"))


if __name__ == "__main__":
    main()
