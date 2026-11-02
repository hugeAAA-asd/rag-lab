"""用新目录（带口语别名）重算路由缓存

route_cache.json 里存的是「加别名之前」的结果。route() 默认吃缓存，
所以加完别名直接跑评估，跑出来的还是旧目录的路由——等于白加。

这个脚本做三件事：
  1. 备份旧缓存（route_cache_old.json）
  2. 对 43 道绕弯题 + 13 道口语题全部用新目录重算，写回缓存
  3. 逐题对比「选择变了没有」，变了的标出来（可能变好也可能变坏）

跑一次约 3 分钟（56 次调用），之后所有评估都用新缓存。
"""
import json
import os
import shutil
import time

from eval import CASES
from eval_oral import CASES as ORAL
from eval_holdout import CASES as H1
from eval_holdout2 import CASES as H2
from llm import load_key
from route import route, MENU

CACHE = "route_cache.json"


def main():
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    old = {}
    if os.path.exists(CACHE):
        shutil.copy(CACHE, "route_cache_old.json")
        old = json.load(open(CACHE, encoding="utf-8"))
        print("旧缓存 %d 条，已备份到 route_cache_old.json" % (len(old) // 2))

    # 43 + 13 + 20 + 20 去重（有些题干可能重复）
    questions = []
    for q, _a, _k in CASES:
        if q not in questions:
            questions.append(q)
    for _name, _want, q in ORAL:
        if q not in questions:
            questions.append(q)
    for _name, _style, q, _sec, _subs in list(H1) + list(H2):
        if q not in questions:
            questions.append(q)

    print("共 %d 题，MENU %d 字，开始重算\n" % (len(questions), len(MENU)))

    new, changed = {}, []
    t0 = time.time()
    for i, q in enumerate(questions, 1):
        picks, raw = route(q, key, use_cache=False)
        new[q] = picks
        new[q + "|raw"] = raw
        o = old.get(q)
        if o is not None and o != picks:
            changed.append((q, o, picks))
        print("%2d/%d %-34s → %s" % (
            i, len(questions), q[:34],
            " / ".join(p.replace("## ", "")[:12] for p in picks) or raw[:16]))

    json.dump(new, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n写入 %d 题，耗时 %.0fs" % (len(questions), time.time() - t0))

    print("\n选择发生变化的 %d 题：" % len(changed))
    for q, o, p in changed:
        print("  %s" % q[:44])
        print("     旧: %s" % " / ".join(x.replace("## ", "")[:14] for x in o))
        print("     新: %s" % " / ".join(x.replace("## ", "")[:14] for x in p))


if __name__ == "__main__":
    main()
