"""换路由不用重跑精排：复用候选池里已经算好的精排分

为什么能复用
------------
精排是 cross-encoder，打的是（问题, 单个块）这一对的分，跟「池子里还有哪些别的块」
没关系。同一个问题、同一个块，第二次算出来还是那个数——所以精排分可以缓存。

会变的只有两处：
  ① 路由换了 → 选中的章节换了 → 注入的块换了（池子多几块少几块）
  ② bonus 挂在章节上 → 排序变了
①只是「多了几块」，补算那几块；②是纯排序，微秒级。

实测：83 题里多数路由没变，一题 0.05 秒；变了的补 6-12 块，约 2 秒。
整轮从 83 × 35s ≈ 48 分钟降到 2 分钟。

怎么保证「快」没换来「错」
--------------------------
基线不是存档里那个数，是**用存档候选池现算一遍**：
cand/raw/picks 都在 pools_*.json 里，按生产逻辑重排一次，得到改前成绩。
这样改前改后走的是同一份精排分、同一段排序代码，差的那一点只能是路由造成的。

另外自带对拍：路由没变的题，改后重排的 (块, 分) 序列必须和改前逐个相同，
不同就断言失败——批大小变了导致的浮点漂移一旦翻掉了打平的两个人，
后面的数字就全不可信了。

用法：
    python eval_relay.py            # 43 + h1 + h2 三列一起出
    python eval_relay.py h2         # 只看一列
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys
import time

import numpy as np

import eval as E
from eval import chunks, rank_of
from rewrite import search_multi, load_cache as load_rewrites
from route import route, parse_conf
from llm import load_key
from metrics import mrr
from answer import ROUTE_BONUS, ROUTE_PICK, CONF_GATE

SETS = {
    "43": ("pools_43_sub2.json", "开发集 43（绕弯）"),
    "h1": ("pools_h1_sub2.json", "留出集 h1"),
    "h2": ("pools_h2_sub2.json", "留出集 h2"),
    # h3 的池子是 eval_holdout3.py 现场写的，比老池子少几个字段（没有 sec/subs），
    # 所以多给一个结果文件补上——缺字段不代表结果不可用，只是没存
    "h3": ("eval_holdout3_pools.json", "留出集 h3", "eval_holdout3_result.json"),
}


def rerank_pool(cand, raw, picks, confs, k=3, bonus=ROUTE_BONUS, gate=CONF_GATE):
    """和 rewrite.search_multi 末尾逐行对齐：加分 → 按节去重 → 取前 k

    为什么不直接调 search_multi：它要重新粗筛一遍才拿得到池子，而池子就在手上。
    但排序规则必须一模一样——不一样的话「改前」和「改后」比的就是两套系统。
    """
    rs = np.array(raw, dtype=float).copy()
    w = {}
    for pos, p in enumerate(picks):
        x = bonus[pos] if pos < len(bonus) else bonus[-1]
        if confs is not None and gate and (pos >= len(confs) or confs[pos] < gate):
            x = 0.0
        if x:
            w[p.strip()] = x
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
        out.append((int(cand[int(j)]), float(raw[int(j)])))
        if len(out) == k:
            break
    return out


class ScoreCache:
    """把 reranker.predict 换成「有缓存就用缓存，没有才真算」

    为什么拦在 predict 而不是自己重写候选池构造：
    cand 的规则（_collect 配额、_sub_blocks 展开、并集顺序）里有十几次踩坑的细节，
    重写一遍必然和线上不一致，而「不一致」在评估里是最贵的错误。
    拦在 predict 上，候选池永远是生产代码算出来的，只把「分数」这一个东西换掉。
    """

    def __init__(self, rows):
        self.q2 = {r["q"]: {int(i): float(s) for i, s in zip(r["cand"], r["raw"])}
                   for r in rows}
        self.t2i = {}
        for i in range(len(chunks)):
            self.t2i.setdefault(E.rerank_text(i), i)
        self.orig = E.reranker.predict
        self.miss = self.hit = 0

    def predict(self, pairs, batch_size=16):
        out = [None] * len(pairs)
        todo = []
        for k, (q, txt) in enumerate(pairs):
            idx = self.t2i.get(txt)
            sc = self.q2.get(q)
            if idx is not None and sc is not None and idx in sc:
                out[k] = sc[idx]
                self.hit += 1
            else:
                todo.append(k)
        self.miss += len(todo)
        if todo:
            got = self.orig([pairs[k] for k in todo], batch_size=batch_size)
            for k, v in zip(todo, got):
                out[k] = float(v)
        return np.array(out)

    def install(self):
        E.reranker.predict = self.predict

    def restore(self):
        E.reranker.predict = self.orig


def main():
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    rw = load_rewrites()
    for tag in (sys.argv[1:] or ["43", "h1", "h2"]):
        spec = SETS[tag]
        poolf, name = spec[0], spec[1]
        rows = json.load(open(poolf, encoding="utf-8"))
        # 池子里没存 sec/subs 的（h3），从结果文件按问题补
        res = {}
        if len(spec) > 2:
            res = {x["q"]: x for x in json.load(open(spec[2], encoding="utf-8"))}
        sc = ScoreCache(rows)
        sc.install()
        print("\n" + "=" * 78)
        print("%s（%d 题）  候选池 %s" % (name, len(rows), poolf))
        print("=" * 78)

        stats = {k: [] for k in ("o_sec", "o_sub", "n_sec", "n_sub")}
        o_r, n_r, o_sr, n_sr = [], [], [], []
        moved, drift, down, up, skipped = [], [], [], [], []
        t0 = time.time()

        try:
            for r in rows:
                q = r["q"]
                sec = r.get("sec") or (res[q]["sec"] if q in res else "")
                subs = r.get("subs")
                if not subs and q in res:
                    subs = res[q]["subs"]
                # 少数题只标到章节、没标小节（43 里 3 题）。这种题小节口径
                # 评不了，直接跳过并单独计数——拿 0 分算等于把「没标注」
                # 当成「检索错了」，改前改后的基线都会被拉低。
                if not subs:
                    skipped.append(q)
                    continue
                # 老池子存了 confs；没存的按原始输出重算，不能默认 100——
                # 默认 100 等于关掉置信度门控，改前的基线就被悄悄改了
                old_conf = r.get("confs") or parse_conf(r.get("route_raw", ""),
                                                        len(r["picks"]))
                old = rerank_pool(r["cand"], r["raw"], r["picks"], old_conf)

                picks, raw_txt = route(q, key)
                picks = [p.strip() for p in picks]
                confs = parse_conf(raw_txt, len(picks))
                new = search_multi(q, rw.get(q), k=3, picks=picks,
                                   bonus=ROUTE_BONUS, route_pick=ROUTE_PICK,
                                   confs=confs if picks else None, gate=CONF_GATE)

                # 对拍：路由没变（章节 + 置信度都没变），重排结果必须逐个相同
                # 只比章节是不够的：置信度会参与门控，同一串章节、不同置信度，
                # 排序就能不一样——那是路由变了，不是复用精排分变错了。
                same = (picks == [p.strip() for p in r["picks"]]
                        and list(confs[:len(picks)]) == list(old_conf[:len(picks)]))
                if same:
                    a = [(i, round(sc_, 6)) for i, sc_ in old]
                    b = [(i, round(sc_, 6)) for i, sc_ in new]
                    if a != b:
                        drift.append((q, a, b))

                o_sec = rank_of(old, [sec])
                o_sub = rank_of(old, [sec], subs)
                n_sec = rank_of(new, [sec])
                n_sub = rank_of(new, [sec], subs)
                stats["o_sec"].append(o_sec)
                stats["o_sub"].append(o_sub)
                stats["n_sec"].append(n_sec)
                stats["n_sub"].append(n_sub)
                o_r.append(o_sec or 0)
                n_r.append(n_sec or 0)
                o_sr.append(o_sub or 0)
                n_sr.append(n_sub or 0)
                if picks != [p.strip() for p in r["picks"]] or not same:
                    moved.append((q, r["picks"], picks, o_sub, n_sub))
                if o_sub == 1 and n_sub != 1:
                    down.append((q, sec, o_sub, n_sub, picks))
                if o_sub != 1 and n_sub == 1:
                    up.append((q, sec, o_sub, n_sub, picks))
        finally:
            sc.restore()

        if skipped:
            print("\n  跳过 %d 题（没标小节）：%s" % (len(skipped), [x[:14] for x in skipped]))
        N = len(rows) - len(skipped)

        def pct(v, level):
            # 分母永远是题目总数。滤空只在 MRR 里合法——
            # 这里踩过一次：把 None 剔掉再算，h1 从 0.550 变成 0.917。
            return sum(1 for x in v if x == 1) / N

        print("  精排复用 %d 次 / 补算 %d 块   耗时 %.0fs" % (sc.hit, sc.miss, time.time() - t0))
        print("\n  %-6s %-8s %-8s %-8s %-8s" % ("口径", "节@1", "小@1", "节MRR", "小MRR"))
        print("  %-6s %.3f    %.3f    %.3f    %.3f" % (
            "改前", pct(stats["o_sec"], 1), pct(stats["o_sub"], 1), mrr(o_r), mrr(o_sr)))
        print("  %-6s %.3f    %.3f    %.3f    %.3f" % (
            "改后", pct(stats["n_sec"], 1), pct(stats["n_sub"], 1), mrr(n_r), mrr(n_sr)))

        if drift:
            print("\n  !! 对拍失败 %d 题（路由没变但结果变了，后面的数字作废）" % len(drift))
            for q, a, b in drift[:3]:
                print("     「%s」 旧 %s 新 %s" % (q, a, b))
        else:
            print("\n  对拍通过：路由没变的题，改前改后逐个相同")

        print("\n  小节 掉 %d 题 / 捞回 %d 题：" % (len(down), len(up)))
        for q, sec, a_, b_, pk in down:
            print("    - 掉「%s」期望%s → %s" % (q, sec.strip(), pk[:1]))
        for q, sec, a_, b_, pk in up:
            print("    + 捞「%s」→ %s" % (q, [p.replace("## ", "")[:10] for p in pk[:1]]))

        print("\n  路由变了 %d 题：" % len(moved))
        for q, op, np_, os_, ns_ in moved:
            print("    「%s」" % q)
            print("      旧 %s" % [p.replace("## ", "")[:12] for p in op])
            print("      新 %s   小节 %s → %s" % (
                [p.replace("## ", "")[:12] for p in np_], os_ or "✗", ns_ or "✗"))


if __name__ == "__main__":
    main()
