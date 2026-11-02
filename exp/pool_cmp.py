"""head 注入 vs sub1 注入，同一批题、同一份路由缓存下的候选池对比

为什么现在才做这件事
--------------------
上次测 sub1（每小节第一块）用的是**旧路由缓存**（prompt 改之前），
结论是「43 题 0.767 / h1 0.650」，于是被判为「两边偏好相反，不上线」。
但路由 prompt 换了之后首位章节变了 15/96，候选池也跟着变了，
那个结论已经过期。这里用新缓存重测，只改注入方式一个变量。

怎么比才公平
------------
两份池子（pools_X.json / pools_X_sub1.json）都是 bonus=0 存下来的，
也就是说里面只有「候选 + 精排原始分」，没有任何加分。
这里只用精排分排序，不比 bonus——因为 bonus 属于下一层的问题，
一层只测一个变量。

除了排名，还要看**覆盖率**：目标小节在不在池子里。
排名差可以靠 bonus 调，覆盖不到就彻底没救，这是两种不同的病。
"""
import json
import sys

import numpy as np

from eval import chunks, SUB_MAP, rank_of
from metrics import mrr


def load(ds, pick="head"):
    suf = "" if pick == "head" else "_" + pick
    try:
        return json.load(open("pools_%s%s.json" % (ds, suf), encoding="utf-8"))
    except FileNotFoundError:
        return None


def order(cand, raw, k=3):
    """按精排分排序，按章节去重取前 k（和 search_multi 末尾的逻辑一致）"""
    out, seen = [], set()
    for j in np.argsort(-np.array(raw)):
        i = cand[int(j)]
        t = chunks[i][1]
        if t in seen:
            continue
        seen.add(t)
        out.append((i, float(raw[int(j)])))
        if len(out) == k:
            break
    return out


def covered(row, level):
    """目标小节/目标节有没有块进候选池——这是排名的天花板"""
    if level == "sub":
        if not row.get("subs"):
            return None
        return any(SUB_MAP[i] & set(row["subs"]) for i in row["cand"])
    sec = row["sec"]
    return any(chunks[i][1].startswith(sec) for i in row["cand"])


def stat(rows, pick):
    res = {"sec": [], "sub": [], "cov_sec": [], "cov_sub": [], "size": []}
    for r in rows:
        hits = order(r["cand"], r["raw"])
        res["sec"].append(rank_of(hits, [r["sec"]]))
        if r.get("subs"):
            res["sub"].append(rank_of(hits, [r["sec"]], r["subs"]))
        res["cov_sec"].append(covered(r, "sec"))
        c = covered(r, "sub")
        if c is not None:
            res["cov_sub"].append(c)
        res["size"].append(len(r["cand"]))
    return res


def pct(v, hit):
    """命中率。**失败要留在分母里**——这里曾经把 None 滤掉算比率，
    结果 20 题里 8 道没进前 3 被静默剔除，@1 变成「命中里的命中率」，
    h1 显示 0.917（11/12）而真实是 11/20。分母必须是题目总数。"""
    if not v:
        return float("nan")
    return sum(1 for x in v if x is not None and hit(x)) / len(v)


def show(ds, head, sub1):
    print("\n" + "=" * 78)
    print("数据集 %s（%d 题）" % (ds, len(head)))
    print("=" * 78)
    a, b = stat(head, "head"), stat(sub1, "sub1")

    print("%-22s %10s %10s %8s" % ("指标", "head", "sub1", "差值"))
    rows = [
        ("节级 @1", pct(a["sec"], lambda x: x == 1), pct(b["sec"], lambda x: x == 1)),
        ("节级 @3", pct(a["sec"], lambda x: x), pct(b["sec"], lambda x: x)),
        ("节覆盖(池内)", pct(a["cov_sec"], lambda x: x), pct(b["cov_sec"], lambda x: x)),
    ]
    # 口语档没有小节标注，小节级整块跳过（否则 mrr 除以 0）
    if a["sub"]:
        rows += [
            ("小节级 @1", pct(a["sub"], lambda x: x == 1), pct(b["sub"], lambda x: x == 1)),
            ("小节级 @3", pct(a["sub"], lambda x: x), pct(b["sub"], lambda x: x)),
            ("小节级 MRR", mrr(a["sub"]), mrr(b["sub"])),
            ("小节覆盖(池内)", pct(a["cov_sub"], lambda x: x), pct(b["cov_sub"], lambda x: x)),
        ]
    for name, x, y in rows:
        d = y - x
        mark = "  <<<" if abs(d) > 0.05 else ""
        print("%-22s %10.3f %10.3f %+8.3f%s" % (name, x, y, d, mark))
    print("%-22s %10.1f %10.1f %+8.1f" % ("平均候选池大小",
          np.mean(a["size"]), np.mean(b["size"]), np.mean(b["size"]) - np.mean(a["size"])))

    # 逐题：只有小节级排名变化才值得看
    diffs = []
    for ra, rb in zip(head, sub1):
        if not ra.get("subs"):
            continue
        ha, hb = order(ra["cand"], ra["raw"]), order(rb["cand"], rb["raw"])
        x = rank_of(ha, [ra["sec"]], ra["subs"])
        y = rank_of(hb, [ra["sec"]], ra["subs"])
        if x != y:
            diffs.append((ra, x, y, covered(ra, "sub"), covered(rb, "sub")))
    if diffs:
        print("\n逐题变化（小节级，%d 题）：" % len(diffs))
        for r, x, y, ca, cb in diffs:
            tag = "好" if (y or 99) < (x or 99) else "坏"
            cov = ""
            if not ca and cb:
                cov = "  ← 池子补上了"
            elif ca and not cb:
                cov = "  ← 池子反而丢了"
            print("  %-26s %-14s 第%s → 第%s  %s%s" % (
                r["q"][:26], str(r["confs"])[:14],
                x if x else "✗", y if y else "✗", tag, cov))
    print("\n小结：好 %d / 坏 %d" % (
        sum(1 for _, x, y, _, _ in diffs if (y or 99) < (x or 99)),
        sum(1 for _, x, y, _, _ in diffs if (y or 99) > (x or 99))))


def main():
    for ds in (sys.argv[1:] or ["43", "oral", "h1", "h2"]):
        head, sub1 = load(ds, "head"), load(ds, "sub1")
        if not head or not sub1:
            print("跳过 %s：缺池子文件" % ds)
            continue
        show(ds, head, sub1)


if __name__ == "__main__":
    main()
