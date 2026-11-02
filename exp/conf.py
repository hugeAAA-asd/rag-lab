"""离线扫 bonus 方案：读 pools_*.json，零精排成本

为什么可以离线
--------------
所有待比较的方案（分档权重、置信度缩放、门控阈值）都只改**排序**，
不改候选池。候选池和精排分已经在 dump_pools.py 里存死了，
所以这里每一轮都是毫秒级——扫 20 个方案和扫 1 个方案一样快。

为什么必须一次看三个数据集
--------------------------
43 题（书面）、h1（第一批留出集）、h2（第二批留出集）偏好相反：
43 题要「路由章节整体抬上去」，口语题要「按小节展开」。
只在其中一个上赢的方案，都是在挪分，不是改进。
判定标准：**三列都不掉**，且 h2（唯一没参与过调参的）进前三。

方案
----
flat    [1.0, 0.3, 0.0]                    现线上，不看置信度
conf    [1.0, 0.3, 0.0] × (conf/100)       按位置分档，再乘置信度
equal   conf/100 统一基准，不看位置        只信置信度，不信排序
gate    conf ≥ T 才给分，否则归零          置信度当开关用
pow     [1.0, 0.3, 0.0] × (conf/100)^k     调整置信度的影响强度
"""
import json
import sys

import numpy as np

from eval import chunks, SUB_MAP, rank_of
from metrics import mrr

BASE = [1.0, 0.3, 0.0]


def bonus_vec(picks, confs, mode, base=BASE, power=1.0, gate=50, scale=1.0):
    """按方案算出每个候选章节的加分"""
    out = []
    for i in range(len(picks)):
        b = base[i] if i < len(base) else base[-1]
        c = confs[i] if i < len(confs) else 100
        if mode == "flat":
            out.append(b * scale)
        elif mode == "conf":
            out.append(b * (c / 100.0) ** power * scale)
        elif mode == "equal":
            out.append((c / 100.0) * scale)
        elif mode == "gate":
            out.append(b * scale if c >= gate else 0.0)
        else:
            raise ValueError(mode)
    return out


def rerank(cand, raw, picks, bonus, k=3):
    """和 rewrite.search_multi 末尾一致：按节去重，每节只出一个代表"""
    rs = np.array(raw, dtype=float)
    w = {p.strip(): bonus[i] for i, p in enumerate(picks) if i < len(bonus)}
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


def evaluate(rows, mode, **kw):
    """返回 (节@1, 节@3, 小@1, 小@3, 小MRR)；没有小节标注的数据集小*列为 None"""
    r_sec, r_sub = [], []
    has_sub = any(r.get("subs") for r in rows)
    for r in rows:
        b = bonus_vec(r["picks"], r["confs"], mode, **kw) if r["picks"] else []
        hits = rerank(r["cand"], r["raw"], r["picks"], b)
        r_sec.append(rank_of(hits, [r["sec"]]))
        if has_sub and r.get("subs"):
            r_sub.append(rank_of(hits, [r["sec"]], r["subs"]))
    n = len(r_sec)
    sec1 = sum(1 for x in r_sec if x == 1) / n
    sec3 = sum(1 for x in r_sec if x) / n
    if not r_sub:
        return sec1, sec3, None, None, None
    m = len(r_sub)
    sub1 = sum(1 for x in r_sub if x == 1) / m
    sub3 = sum(1 for x in r_sub if x) / m
    return sec1, sec3, sub1, sub3, mrr(r_sub)


def paired(rows, kw_a, kw_b):
    """逐题比较两个方案，看每道题是变好、变坏还是没动

    为什么必须有这个：20 题的粒度是 0.05/题，两个方案差 1 道题就是 5 个点，
    但 1 道题的差异完全可能是噪声。逐题配对看「几个好、几个坏」，
    比看总分可靠得多——3 好 3 坏和 5 好 0 坏，总分可能一样，含义完全不同。
    """
    win = lose = tie = 0
    detail = []
    for r in rows:
        if not r.get("subs"):
            continue
        pa = rank_of(rerank(r["cand"], r["raw"], r["picks"],
                            bonus_vec(r["picks"], r["confs"], **kw_a)), [r["sec"]], r["subs"]) or 9
        pb = rank_of(rerank(r["cand"], r["raw"], r["picks"],
                            bonus_vec(r["picks"], r["confs"], **kw_b)), [r["sec"]], r["subs"]) or 9
        if pa == pb:
            tie += 1
        elif pb < pa:
            win += 1
            detail.append("  好 %-26s 第%d→第%d" % (r["q"][:26], pa, pb))
        else:
            lose += 1
            detail.append("  坏 %-26s 第%d→第%d" % (r["q"][:26], pa, pb))
    return win, lose, tie, detail


def load(ds):
    try:
        return json.load(open("pools_%s.json" % ds, encoding="utf-8"))
    except FileNotFoundError:
        return None


def main():
    data = {ds: load(ds) for ds in ("43", "oral", "h1", "h2")}
    data = {k: v for k, v in data.items() if v}
    if not data:
        print("没有 pools_*.json，先跑 dump_pools.py")
        return

    schemes = [("flat 现线上", dict(mode="flat"))]
    for p in (0.5, 1.0, 1.5, 2.0):
        schemes.append(("conf p=%.1f" % p, dict(mode="conf", power=p)))
    for g in (40, 60, 75):
        schemes.append(("gate>=%d" % g, dict(mode="gate", gate=g)))
    schemes.append(("equal 只看置信度", dict(mode="equal")))

    for ds, rows in data.items():
        print("\n" + "=" * 82)
        print("数据集 %-4s  %d 题   置信度样本 %s" % (
            ds, len(rows), [r["confs"] for r in rows[:3]]))
        print("=" * 82)
        print("%-16s %-6s %-6s %-6s %-6s %-6s" % ("方案", "节@1", "节@3", "小@1", "小@3", "小MRR"))
        best = None
        for name, kw in schemes:
            s1, s3, b1, b3, bm = evaluate(rows, **kw)
            def f(x):
                return "  -  " if x is None else "%.3f" % x
            print("%-16s %s  %s  %s  %s  %s" % (name, f(s1), f(s3), f(b1), f(b3), f(bm)))
            if b1 is not None:
                score = b1 * 2 + b3
                if best is None or score > best[0]:
                    best = (score, name)
        if best:
            print("  → 该数据集最优：%s" % best[1])

    # 跨数据集汇总：只列「每列都 ≥ flat」的方案
    print("\n" + "=" * 82)
    print("跨数据集判定（相对 flat 的差值，小@1）")
    print("=" * 82)
    flat = {ds: evaluate(rows, mode="flat")[2] for ds, rows in data.items()}
    print("%-16s %s" % ("方案", "  ".join("%-8s" % ds for ds in data)))
    for name, kw in schemes[1:]:
        cells = []
        for ds, rows in data.items():
            v = evaluate(rows, **kw)[2]
            if v is None:
                cells.append("   -    ")
            else:
                cells.append("%+.3f  " % (v - flat[ds]))
        print("%-16s %s" % (name, "  ".join(cells)))

    print("\n逐题配对（相对 flat，小节级排名；好/坏/不变）")
    for ds, rows in data.items():
        if not any(r.get("subs") for r in rows):
            continue
        print("  数据集 %s（%d 题有标注）" % (ds, sum(1 for r in rows if r.get("subs"))))
        for name, kw in schemes[1:]:
            w, l, t, _ = paired(rows, dict(mode="flat"), kw)
            print("    %-16s 好 %d　坏 %d　不变 %d" % (name, w, l, t))


if __name__ == "__main__":
    main()
