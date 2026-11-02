"""两个变量一次看全：注入方式（head / sub1）× 加分方案（无 / 分档 / 分档+门控）

为什么要放在一张表里
--------------------
之前的问题是每次只改一个变量、换一个数据集就翻转，然后各说各话：
43 题说 bonus 有用，h1 说 bonus 有害，h2 说都没差别。
这里的做法是把四种组合排成矩阵，一次看三列是否同向——
只有「三列都不掉」的组合才值得上线，其余都算在挪分。

所有数字都来自 pools_*.json，零精排成本。
"""
import json
import sys

import numpy as np

from eval import chunks, SUB_MAP, rank_of
from metrics import mrr

BASE = [1.0, 0.3, 0.0]
PICKS = ("head", "sub2", "sub2_subtitle")


def load(ds, pick="head"):
    suf = "" if pick == "head" else "_" + pick
    try:
        return json.load(open("pools_%s%s.json" % (ds, suf), encoding="utf-8"))
    except FileNotFoundError:
        return None


def rerank(cand, raw, picks, bonus, k=3):
    """和 search_multi 末尾完全一致：加分排序、按章节去重、返回原始精排分"""
    rs = np.array(raw, dtype=float).copy()
    if picks and bonus:
        w = {}
        for pos, p in enumerate(picks):
            w[p.strip()] = bonus[pos] if pos < len(bonus) else bonus[-1]
        for j, i in enumerate(cand):
            if w.get(chunks[i][1].strip()):
                rs[j] += w[chunks[i][1].strip()]
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


def gated(confs, base=BASE, gate=60):
    """置信度当开关：低于 gate 的候选不给分（不给分=不动它的排序）"""
    return [b if (c >= gate) else 0.0 for b, c in zip(base, confs)]


SCHEMES = {
    "无加分": lambda r: [],
    "分档[1,.3,0]": lambda r: BASE,
    "分档+门控60": lambda r: gated(r.get("confs") or [100] * 3),
}


def eval_ds(rows, scheme):
    secs, subs = [], []
    for r in rows:
        hits = rerank(r["cand"], r["raw"], r["picks"], SCHEMES[scheme](r))
        secs.append(rank_of(hits, [r["sec"]]))
        if r.get("subs"):
            subs.append(rank_of(hits, [r["sec"]], r["subs"]))
    def rate(v, n=1):
        if not v:            # 口语档没有小节标注，整块返回 nan
            return float("nan")
        return sum(1 for x in v if x == n or (n == 3 and x)) / len(v)
    return dict(sec1=rate(secs, 1), sec3=rate(secs, 3),
                sub1=rate(subs, 1), sub3=rate(subs, 3),
                mrr=mrr(subs) if subs else float("nan"))


def main():
    dss = sys.argv[1:] or ["43", "oral", "h1", "h2"]
    table = {}
    for ds in dss:
        for pick in PICKS:
            rows = load(ds, pick)
            if not rows:
                continue
            for sc in SCHEMES:
                table[(ds, pick, sc)] = eval_ds(rows, sc)

    for field, name in (("sub1", "小节级 @1"), ("sub3", "小节级 @3"), ("mrr", "小节级 MRR")):
        print("\n" + "=" * 84)
        print("【%s】" % name)
        print("=" * 84)
        print("%-8s %-6s " % ("数据集", "注入") +
              " ".join("%10s" % sc for sc in SCHEMES))
        for ds in dss:
            for pick in PICKS:
                if (ds, pick, "无加分") not in table:
                    continue
                cells = ["%10.3f" % table[(ds, pick, sc)][field] for sc in SCHEMES]
                print("%-8s %-6s %s" % (ds, pick, " ".join(cells)))
    # 节级单独看一眼——bonus 的本质是抬章节，这里最能说明它到底在做什么
    print("\n" + "=" * 84)
    print("【节级 @1】bonus 是在抬章节，所以看这一行才知道它在做好事还是坏事")
    print("=" * 84)
    print("%-8s %-6s " % ("数据集", "注入") +
          " ".join("%10s" % sc for sc in SCHEMES))
    for ds in dss:
        for pick in PICKS:
            if (ds, pick, "无加分") not in table:
                continue
            cells = ["%10.3f" % table[(ds, pick, sc)]["sec1"] for sc in SCHEMES]
            print("%-8s %-6s %s" % (ds, pick, " ".join(cells)))


if __name__ == "__main__":
    main()
