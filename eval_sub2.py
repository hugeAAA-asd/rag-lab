"""越级方案验收：全量小节直选 → 反推章节 → 补池 → 加分

和 eval_sub.py 的区别只有一处：picks 的来源。
  eval_sub.py   章节路由给 3 章，小节只在 3 章里挑（上一级漏了就没救）
  eval_sub2.py  287 小节直选，章节是小节号反推出来的（没有上一级可漏）

其余完全一致：同一套 search_multi、同一套加分、同一份精排缓存。
所以两边的差只能是「谁来定范围」造成的。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys
import time

from rewrite import search_multi, load_cache as load_rewrites
from route import route_allsubs, secs_of_subs, sub_conf
from llm import load_key
from answer import ROUTE_PICK
from eval_relay import ScoreCache

SETS = {
    "h3": ("eval_holdout3_pools.json", "eval_holdout3_result.json", "留出集 h3"),
    "43": ("pools_43_sub2.json", None, "开发集 43"),
    "h1": ("pools_h1_sub2.json", None, "留出集 h1"),
    "h2": ("pools_h2_sub2.json", None, "留出集 h2"),
}


def main(tag):
    poolf, resf, name = SETS[tag]
    rows = json.load(open(poolf, encoding="utf-8"))
    res = {x["q"]: x for x in (json.load(open(resf, encoding="utf-8")) if resf else rows)}
    key = load_key()
    rw = load_rewrites()
    sc = ScoreCache(rows)
    sc.install()

    dump, skipped = [], []
    sel1 = sel3 = 0
    t0 = time.time()
    try:
        for r in rows:
            q = r["q"]
            meta = res[q]
            sec, want = meta["sec"].strip(), list(meta["subs"] or [])
            if not want:
                skipped.append(q)
                continue

            subs, txt = route_allsubs(q, key)
            picks = secs_of_subs(subs)
            ok1 = bool(subs) and subs[0] in set(want)
            ok3 = bool(set(want) & set(subs))
            sel1 += ok1
            sel3 += ok3

            cand, raw, _ = search_multi(q, rw.get(q), picks=picks, bonus=0.0,
                                        return_raw=True, route_pick=ROUTE_PICK,
                                        subs=subs, sub_bonus=0.0)
            dump.append({"q": q, "sec": sec, "subs_want": want,
                         "picks": [p.strip() for p in picks], "subs": subs,
                         "sub_confs": sub_conf(txt),
                         "cand": [int(x) for x in cand],
                         "raw": [float(x) for x in raw]})
            print("  %s%s 选%-34s 章%-22s %s" % (
                "√" if ok1 else ("~" if ok3 else "×"), "", str(subs),
                str([p.replace("## ", "")[:5] for p in picks]), q[:26]))
    finally:
        sc.restore()

    json.dump(dump, open("allsub_pools_%s.json" % tag, "w", encoding="utf-8"), ensure_ascii=False)
    N = len(rows) - len(skipped)
    if skipped:
        print("\n   跳过 %d 题（没标小节）" % len(skipped))
    print("\n%s（%d 题）小节直选：首选 %.3f  前三 %.3f   精排复用 %d / 补算 %d   %.0fs" % (
        name, N, sel1 / N, sel3 / N, sc.hit, sc.miss, time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h3")
