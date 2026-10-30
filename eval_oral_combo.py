"""极短口语档：用正式链路（rewrite.search_multi + route.route）复验

eval_oral.py 只测了「原问 / 改写」两条路，没带路由。
exp_route_bonus.py 带路由，但它用的是 exp_route.search_routed（老实现），
而正式代码里路由章节挑块已经改成「前 2 块 + 余弦最高 1 块」。
所以这里用正式代码重测一遍，才是当前系统的真实成绩。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys
import time

from eval import chunks
from eval_oral import CASES, rank_of, top1_info
from llm import load_key
from rewrite import search_multi, rewrite
from route import route

CACHE = "route_cache_oral.json"


def load():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save(d):
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def main(bonus):
    key = load_key()
    cache = load()

    routes, rews_all = {}, {}
    for name, want, q in CASES:
        if q in cache:
            routes[q] = cache[q]["picks"]
            rews_all[q] = cache[q].get("rews", [])
        else:
            picks, raw = route(q, key)
            rw = rewrite(q, key)
            routes[q], rews_all[q] = picks, rw
            cache[q] = {"raw": raw, "picks": picks, "rews": rw}
            save(cache)
            print("  %-14s 路由 %-16r → %s" % (q, raw[:16],
                  " / ".join(p.replace("## ", "")[:18] for p in picks)))

    print()
    print("%-16s %14s %14s" % ("方案", "@1", "@3"))
    print("-" * 46)
    for tag, use_r, use_p in (("原问", False, False),
                              ("+改写", True, False),
                              ("+改写+路由", True, True)):
        t0 = time.time()
        n1 = n3 = 0
        for name, want, q in CASES:
            hits = search_multi(q, rews_all[q] if use_r else None, k=3,
                                picks=routes[q] if use_p else [],
                                bonus=bonus)
            r = rank_of(hits, want)
            if r == 1:
                n1 += 1
            if r:
                n3 += 1
        print("%-16s %5d/%d %.3f %5d/%d %.3f  (%.0fs)" % (
            tag, n1, len(CASES), n1 / len(CASES), n3, len(CASES), n3 / len(CASES),
            time.time() - t0))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.5)
