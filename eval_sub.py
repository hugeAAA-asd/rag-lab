"""小节路由的验收：真跑一遍「章节 → 小节 → 补池 → 加分」

为什么不能再用 eval_relay.py：那个脚本的前提是「池子不变，只换排序」。
小节这一层会改池子（把小节的块捞进来），前提没了。所以这里调真的
search_multi，但分数仍然走 ScoreCache——老池子里的块一个都不用重算，
只有新捞进来那十来块要算，20 题约一分钟。

口径说明：base 是指「同一个 picks、不加小节」，new 是「加了小节」。
两条路径共用同一个 picks，所以差异只能是这一层造成的。
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
from route import route, route_sub, parse_conf, sub_conf
from llm import load_key
from answer import ROUTE_BONUS, ROUTE_PICK, CONF_GATE
from eval_relay import ScoreCache
from metrics import mrr

SCAN = [0.0, 0.2, 0.4, 0.6, 0.8]
SETS = {
    "h3": ("eval_holdout3_pools.json", "eval_holdout3_result.json", "留出集 h3"),
    "h1": ("pools_h1_sub2.json", None, "留出集 h1"),
    "h2": ("pools_h2_sub2.json", None, "留出集 h2"),
    "43": ("pools_43_sub2.json", None, "开发集 43"),
}


def rerank(cand, raw, picks, confs, subs, sb, k=3):
    rs = np.array(raw, dtype=float).copy()
    wt = {}
    for pos, p in enumerate(picks):
        x = ROUTE_BONUS[pos] if pos < len(ROUTE_BONUS) else ROUTE_BONUS[-1]
        if confs is not None and CONF_GATE and (pos >= len(confs) or confs[pos] < CONF_GATE):
            x = 0.0
        if x:
            wt[p.strip()] = x
    for j, i in enumerate(cand):
        rs[j] += wt.get(chunks[i][1].strip(), 0)
    if subs and sb:
        sw = {no: sb for no in subs}
        for j, i in enumerate(cand):
            hit = E.SUB_MAP.get(i, set()) & set(sw)
            if hit:
                rs[j] += max(sw[x] for x in hit)
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


def main(tag):
    poolf, resf, name = SETS[tag]
    rows = json.load(open(poolf, encoding="utf-8"))
    if resf:
        res = {x["q"]: x for x in json.load(open(resf, encoding="utf-8"))}
    else:
        res = {x["q"]: x for x in rows}

    key = load_key()
    rw = load_rewrites()
    sc = ScoreCache(rows)
    sc.install()

    agg = {b: {"s": [], "s3": [], "r": []} for b in SCAN}
    dump = []
    skipped = []
    sel1 = sel3 = 0
    t0 = time.time()
    try:
        for r in rows:
            q = r["q"]
            meta = res[q]
            sec, subs_want = meta["sec"].strip(), (meta["subs"] or [])
            if not subs_want:
                # 少数题只标到章节、没标小节，小节口径评不了。跳过并计数，
                # 不能拿 0 分糊过去——那等于把「没标注」当成「检索错了」。
                skipped.append(q)
                continue

            picks, raw_txt = route(q, key)
            picks = [p.strip() for p in picks]
            confs = parse_conf(raw_txt, len(picks))
            subs, sub_txt = route_sub(q, picks, key)

            want = set(subs_want)
            ok1 = bool(subs) and subs[0] in want
            ok3 = bool(want & set(subs))
            sel1 += ok1
            sel3 += ok3

            # 一次精排拿到含小节块的池子；各档 bonus 只重排，不再算分
            cand, raw, _ = search_multi(q, rw.get(q), picks=picks, bonus=0.0,
                                        return_raw=True, route_pick=ROUTE_PICK,
                                        subs=subs, sub_bonus=0.0)
            dump.append({"q": q, "sec": sec, "subs_want": subs_want, "picks": picks,
                         "subs": subs, "confs": confs,
                         # 小节的匹配度必须单独存：拿章节的分数去给小节做门控，
                         # 两件事的把握根本不是一回事（8.3 很确定，8.3.4 却很虚）
                         "sub_confs": sub_conf(sub_txt),
                         "cand": [int(x) for x in cand],
                         "raw": [float(x) for x in raw]})
            rks = {}
            for b in SCAN:
                hits = rerank(cand, raw, picks, confs, subs, b)
                rk = rank_of(hits, [sec], subs_want)
                rks[b] = rk
                agg[b]["s"].append(rk == 1)
                agg[b]["s3"].append(rk is not None)
                agg[b]["r"].append(rk or 0)
            print("  %s p%s 小节%s 期望%s rk=%s | %s" % (
                "√" if ok1 else ("~" if ok3 else "×"),
                [x.replace("## ", "")[:5] for x in picks], subs, sorted(want),
                [rks[b] for b in SCAN], q[:26]))
    finally:
        sc.restore()

    json.dump(dump, open("sub_pools_%s.json" % tag, "w", encoding="utf-8"),
              ensure_ascii=False)
    N = len(rows)
    if skipped:
        print("\n   跳过 %d 题（没标小节）：%s" % (len(skipped), [x[:16] for x in skipped]))
    N = N - len(skipped)
    print("\n%s（%d 题）小节选择：首选 %.3f  前三 %.3f   精排复用 %d / 补算 %d   %.0fs" % (
        name, N, sel1 / N, sel3 / N, sc.hit, sc.miss, time.time() - t0))
    print("\n  %-8s %-8s %-8s %-8s" % ("小节加分", "小@1", "小@3", "小MRR"))
    for b in SCAN:
        a = agg[b]
        print("  %-8.1f %.3f    %.3f    %.3f" % (
            b, sum(a["s"]) / N, sum(a["s3"]) / N, mrr(a["r"])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h3")
