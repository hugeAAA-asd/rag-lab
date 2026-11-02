"""极短口语档：用正式链路（rewrite.search_multi + route.route）复验

eval_oral.py 只测了「原问 / 改写」两条路，没带路由。
exp_route_bonus.py 带路由，但它用的是 exp_route.search_routed（老实现），
而正式代码里路由章节挑块已经改成「前 2 块 + 余弦最高 1 块」。
所以这里用正式代码重测一遍，才是当前系统的真实成绩。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import time

from eval_oral import CASES, rank_of
from metrics import header, row
from llm import load_key
from rewrite import search_multi, rewrite
from route import route


def main(bonus):
    key = load_key()

    # route() / rewrite() 自带缓存，重复跑结果稳定
    routes = {q: route(q, key)[0] for _n, _w, q in CASES}
    rews_all = {q: rewrite(q, key) for _n, _w, q in CASES}

    print()
    header()
    for tag, use_r, use_p in (("原问", False, False),
                              ("+改写", True, False),
                              ("+改写+路由", True, True)):
        t0 = time.time()
        ranks = []
        for name, want, q in CASES:
            hits = search_multi(q, rews_all[q] if use_r else None, k=3,
                                picks=routes[q] if use_p else [],
                                bonus=bonus)
            ranks.append(rank_of(hits, want))
        print(row(tag, ranks, time.time() - t0))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.5)
