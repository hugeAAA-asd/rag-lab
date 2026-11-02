"""门控阈值扫描：确认「置信度门控」不是刚好卡在某一个数上才好看的

为什么必须扫而不是挑一个数
--------------------------
如果只报 gate=60 的结果，读者没法区分「这个阈值有物理意义」和「我在 40 道题上
试了 10 次挑到的最好那个」。扫描后看**平台**：一段连续的阈值给出同样的分数，
说明结果对这个参数不敏感，那才是可信的；如果只有一个点高、旁边就掉，那是噪声。

三个数据集一起看，因为门控的目的正是「43 题不受伤、留出集捞回来」。
"""
import json
import sys

import numpy as np

from eval import chunks, SUB_MAP, rank_of
from metrics import mrr

BASE = [1.0, 0.3, 0.0]


def load(ds, pick):
    suf = "" if pick == "head" else "_" + pick
    return json.load(open("pools_%s%s.json" % (ds, suf), encoding="utf-8"))


def rerank(r, bonus, k=3):
    rs = np.array(r["raw"], dtype=float).copy()
    if bonus:
        w = {}
        for pos, p in enumerate(r["picks"]):
            w[p.strip()] = bonus[pos] if pos < len(bonus) else bonus[-1]
        for j, i in enumerate(r["cand"]):
            v = w.get(chunks[i][1].strip())
            if v:
                rs[j] += v
    out, seen = [], set()
    for j in np.argsort(-rs):
        t = chunks[r["cand"][int(j)]][1]
        if t in seen:
            continue
        seen.add(t)
        out.append((r["cand"][int(j)], float(r["raw"][int(j)])))
        if len(out) == k:
            break
    return out


def score(rows, fn):
    subs, secs = [], []
    for r in rows:
        hits = rerank(r, fn(r))
        secs.append(rank_of(hits, [r["sec"]]))
        if r.get("subs"):
            subs.append(rank_of(hits, [r["sec"]], r["subs"]))
    return (sum(1 for x in subs if x == 1) / len(subs) if subs else float("nan"),
            sum(1 for x in subs if x) / len(subs) if subs else float("nan"),
            mrr(subs) if subs else float("nan"),
            sum(1 for x in secs if x == 1) / len(secs))


def main():
    pick = sys.argv[1] if len(sys.argv) > 1 else "sub1"
    ds_list = ["43", "h1", "h2"]
    data = {ds: load(ds, pick) for ds in ds_list}

    print("注入方式 %s。门控：置信度 < T 的候选不给加分\n" % pick)
    print("%-6s %s" % ("T", "  ".join("%-22s" % ds for ds in ds_list)))
    print("%-6s %s" % ("", "  ".join("%-22s" % "小@1  小@3  MRR  节@1" for _ in ds_list)))

    rows = []
    for T in [0, 20, 30, 40, 50, 60, 70, 80, 90]:
        def fn(r, T=T):
            cs = r.get("confs") or [100] * 3
            return [b if (c >= T) else 0.0 for b, c in zip(BASE, cs)]

        cells = []
        vals = []
        for ds in ds_list:
            v = score(data[ds], fn)
            vals.append(v)
            cells.append("%.3f %.3f %.3f %.3f" % v)
        rows.append((T, vals))
        print("%-6d %s" % (T, "  ".join(cells)))

    # 留出集（h1+h2）合计——唯一没参与过调参的口径
    print("\n留出集合计（h1+h2，40 题）：")
    print("%-6s %8s %8s %8s" % ("T", "小@1", "小@3", "MRR"))
    for T, vals in rows:
        h = vals[1], vals[2]
        m = [(h[0][i] + h[1][i]) / 2 for i in range(3)]
        print("%-6d %8.3f %8.3f %8.3f" % (T, *m))


if __name__ == "__main__":
    main()
