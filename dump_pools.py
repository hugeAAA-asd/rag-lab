"""把各数据集的候选池 + 精排分一次性留档

为什么要做这件事
----------------
精排（cross-encoder）是整条链路唯一的瓶颈，每题约 10s。
而所有「排序类」改动——bonus 权重、置信度缩放、top-k、去重粒度——
都只改排序，不改候选池。

所以花一次时间（四个数据集约 15 分钟）把候选池存成 json，
之后每换一个方案都是毫秒级重排，不用再碰精排。
今天之前每次扫参数都要重跑 43 题 955s，就是没做这一步。

产物：pools_<数据集>.json
  每条 = {n, q, sec, subs, style, picks, confs, route_raw, cand, raw}
  cand/raw 是「带路由注入」的那一份候选池（线上用的就是它）

用法：
  python dump_pools.py           # 全部
  python dump_pools.py h2        # 只做第二留出集
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys
import time

from rewrite import search_multi, rewrite, load_cache, save_cache
from route import route, parse_conf
from llm import load_key


def items_43():
    from eval import CASES, SUBS
    return [{"n": n, "q": q, "sec": ans[0], "subs": subs, "style": "indirect"}
            for n, ((q, ans, kind), subs) in enumerate(zip(CASES, SUBS), 1)]


def items_oral():
    from eval_oral import CASES
    return [{"n": n, "q": q, "sec": want, "subs": None, "style": "oral"}
            for n, (name, want, q) in enumerate(CASES, 1)]


def items_h1():
    import eval_holdout as H
    return [{"n": n, "q": q, "sec": sec, "subs": subs, "style": style}
            for n, (name, style, q, sec, subs) in enumerate(H.CASES, 1)]


def items_h2():
    import eval_holdout2 as H
    return [{"n": n, "q": q, "sec": sec, "subs": subs, "style": style}
            for n, (name, style, q, sec, subs) in enumerate(H.CASES, 1)]


DATASETS = {"43": items_43, "oral": items_oral, "h1": items_h1, "h2": items_h2}


# 注入方式从环境变量来：head=线上现状（前2块+余弦最高），sub1=首位章节按小节展开
PICK = os.environ.get("ROUTE_PICK", "head")
# 精排文本是否带三级小节标题——也是要单独对照的一维
SUB = os.environ.get("RERANK_WITH_SUB") == "1"
SUFFIX = ("" if PICK == "head" else "_" + PICK) + ("_subtitle" if SUB else "")


def main():
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    want = sys.argv[1:] or list(DATASETS)
    cache = load_cache()

    for ds in want:
        items = DATASETS[ds]()
        rows = []
        t0 = time.time()
        print("\n" + "=" * 70)
        print("数据集 %s：%d 题（注入 %s）" % (ds, len(items), PICK))
        print("=" * 70)

        for it in items:
            q = it["q"]
            picks, rtxt = route(q, key)
            confs = parse_conf(rtxt, len(picks))

            if q in cache:
                rews = cache[q]
            else:
                rews = rewrite(q, key)
                cache[q] = rews
                save_cache(cache)

            # bonus=0 只影响加分、不影响注入，所以拿到的就是线上那份候选池
            cand, raw, _ = search_multi(q, rews, k=3, picks=picks,
                                        bonus=0.0, return_raw=True,
                                        route_pick=PICK)

            rows.append(dict(it, picks=picks, confs=confs, route_raw=rtxt,
                             cand=[int(i) for i in cand],
                             raw=[float(x) for x in raw]))
            print("  %2d 候选%3d 路由%s 分%s  %s" % (
                it["n"], len(cand),
                [p.replace("## ", "")[:8] for p in picks], confs, q[:30]))

        out = "pools_%s%s.json" % (ds, SUFFIX)
        json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("  → %s（%.0fs）" % (out, time.time() - t0))


if __name__ == "__main__":
    main()
