"""越级实验：不给章节目录，直接给全部 287 个小节

动机来自 h3 的失败清单：5 个失败里 3 个是「正确章节根本不在路由的 3 个候选里」
（3.2、5.5、7.4）。小节层再准也救不了——它只在候选章里挑。
那不如把选择粒度一次降到底：287 个小节，直接选 3 个，不经章节。

代价是 prompt 从 ~1200 token 涨到 ~5000，而且 287 选 3 比 79 选 3 难。
这个脚本就是量「难多少、值不值」。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import hashlib
import json
import re
import sys

from llm import ask_llm, load_key
from route import SUBS, parse_sub, sub_conf

TPL = (
    "这是一本关于智能体（Agent）的技术书，全部小节清单如下（每行是「小节号 小节标题」）：\n\n%s\n\n"
    "读者的问题：%s\n\n"
    "从上面的清单里选出最可能回答这个问题的小节，按把握从大到小排，最多 3 个。\n"
    "每个写成「小节号:匹配度」，匹配度是 0-100 的整数。\n"
    "只输出这个列表，用逗号分隔，不要输出小节名，不要解释。例如：8.3.2:88,8.3.4:40\n"
    "匹配度要拉开差距，不要因为排在第一个就写 90。宁可少选：只有一个真的对，就只输出一个。"
)
SYSTEM = "你是图书检索员。用户会问一个问题，你要从图书的小节清单里挑出最可能回答它的小节。只输出小节号。"

ALL = [(no, name) for sec in SUBS.values() for no, name in sec]
print("清单 %d 个小节，%d 字" % (len(ALL), sum(len(a) + len(b) + 2 for a, b in ALL)))
BODY = "\n".join("%s %s" % (a, b) for a, b in ALL)
CACHE = "allsub_cache_%s.json" % hashlib.md5(TPL.encode("utf-8")).hexdigest()[:8]


def main(tag="h3"):
    rows = json.load(open("eval_holdout3_pools.json", encoding="utf-8"))
    res = {x["q"]: x for x in json.load(open("eval_holdout3_result.json", encoding="utf-8"))}
    key = load_key()
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    valid = {a for a, _ in ALL}

    h1 = h3 = 0
    for n, r in enumerate(rows, 1):
        q = r["q"]
        want = set(res[q]["subs"])
        if q in cache:
            got = cache[q]["got"]
        else:
            txt, _ = ask_llm(TPL % (BODY, q), key, SYSTEM, temperature=0.0, max_tokens=80)
            got = parse_sub(txt, valid)[:3]
            cache[q] = {"got": got, "raw": txt.strip(), "conf": sub_conf(txt)}
            json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        ok1 = bool(got) and got[0] in want
        ok3 = bool(want & set(got))
        h1 += ok1
        h3 += ok3
        print("  %2d %s 选%-36s 期望%-12s %s" % (
            n, "√" if ok1 else ("~" if ok3 else "×"), str(got), str(sorted(want)), q[:26]))

    print("\n全部小节直选（%d 题）  首选 %.3f  前三 %.3f" % (len(rows), h1 / len(rows), h3 / len(rows)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h3")
