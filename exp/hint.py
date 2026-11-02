"""只跑路由，不跑检索：反查词到底让路由多选对几题

为什么先只跑路由
----------------
全链路每道题要精排，35s/题，20 题就是 12 分钟。但反查词改的只有一件事：
「目标章节有没有被选进候选」。这件事不依赖精排——路由选对了，章节才可能进池子；
选错了，后面精排再准也白搭。所以先花 1.5 分钟只看路由，
数字不好看就直接停，不浪费那 12 分钟。

三个指标
--------
章节命中：目标章节出现在 3 个候选里（这是老口径，会高估，因为选对章节不等于
          小节进前 3；但改的正是这一步，所以要看）
小节可达：目标小节所在的那一节被选中了（才是真正有用的信号）
NONE    ：模型说「这问题书里没有」——最坏的一种，bonus 全给 0

对比基线是 route_cache.json 里存的老 pick（旧目录问出来的），不是我想象的。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys
import time

from route import route, MENU
from llm import load_key

from eval_holdout import CASES as H1
from eval_holdout2 import CASES as H2

# 43 题开发集在 eval.py 里，CASES 是 (问题, [期望章节], 风格)，SUBS 是小节标注
from eval import CASES as DEV_Q, SUBS as DEV_SUB


def load_old():
    """旧目录问出来的 pick：按问题号存"""
    if not os.path.exists("archive/route_cache.json"):
        return {}
    return json.load(open("archive/route_cache.json", encoding="utf-8"))


def run(name, cases, key, old):
    print("\n" + "=" * 78)
    print("%s：%d 题" % (name, len(cases)))
    print("=" * 78)
    hit_old = hit_new = sub_old = sub_new = none_new = 0
    changed = []
    for n, (tag, style, q, sec, subs) in enumerate(cases, 1):
        if isinstance(sec, str):
            sec = [sec]
        sec = [s.strip() for s in sec]
        picks, _ = route(q, key)
        picks = [p.strip() for p in picks]
        op = [p.strip() for p in old.get(q, [])]

        def has(ps, ss):
            return any(p.startswith(s) for p in ps for s in ss)

        a, b = has(op, sec), has(picks, sec)
        hit_old += a
        hit_new += b
        sub_old += bool(op)
        sub_new += bool(picks)
        none_new += not picks
        if a != b:
            changed.append((n, tag, q, "/".join(sec), a, b,
                            [p.replace("## ", "")[:14] for p in picks]))

    N = len(cases)
    print("  章节命中  旧 %d/%d = %.3f  →  新 %d/%d = %.3f" % (
        hit_old, N, hit_old / N, hit_new, N, hit_new / N))
    print("  非NONE    旧 %d/%d = %.3f  →  新 %d/%d = %.3f" % (
        sub_old, N, sub_old / N, sub_new, N, sub_new / N))
    if changed:
        print("\n  变了的 %d 题（+ 是新选对，- 是新选错）：" % len(changed))
        for n, tag, q, sec, a, b, picks in changed:
            print("  %s %2d %-10s 期望%s「%s」" % ("+" if b else "-", n, tag, sec, q))
            print("        新候选 %s" % picks)
    return hit_old / N, hit_new / N


def main():
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)
    print("目录字数 %d（无 HINT 时 3840）" % len(MENU))
    old = load_old()
    print("旧缓存 %d 条" % len(old))

    t0 = time.time()
    dev = [("dev%d" % (i + 1), st, q, secs, sub)
           for i, ((q, secs, st), sub) in enumerate(zip(DEV_Q, DEV_SUB))]
    run("开发集 43（绕弯）", dev, key, old)
    run("留出集 h2", H2, key, old)
    run("留出集 h1", H1, key, old)
    print("\n耗时 %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
