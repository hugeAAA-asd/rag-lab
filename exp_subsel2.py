"""小节选择：把路由的 3 个候选章合并成一张清单，会不会互相干扰

exp_subsel.py 是「已知正确章节」的理想条件（首选 0.950），它掩盖了一个问题：
真实系统里章节一旦选错，小节再好也没用（h3 有 2 题章节就错了）。
所以更实际的架构是——路由给 3 章，把小节清单合成一张，一次选完。
好处是 top1 选错还有救；风险是 3 章的清单混在一起，模型分不清上下文。

这个脚本就是量这个风险。答案集不变，还是 h3 那 20 题。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import re
import sys

from eval import chunks
from llm import ask_llm, load_key
from exp_subsel import subs_of, SYSTEM

TPL = (
    "这是一本关于智能体（Agent）的技术书，下面是几个章节各自的小节清单。\n\n%s\n\n"
    "读者的问题：%s\n\n"
    "从上面的清单里选出最可能回答这个问题的小节，按把握从大到小排，最多 3 个。\n"
    "每个写成「小节号:匹配度」，匹配度是 0-100 的整数。\n"
    "只输出这个列表，用逗号分隔，不要输出小节名，不要解释。例如：8.3.2:88,8.3.4:40\n"
    "注意：不同章节下的小节号开头不同，不要把两个号拼在一起。\n"
    "匹配度要拉开差距，不要因为排在第一个就写 90。宁可少选。"
)


def parse(txt, valid):
    out = []
    for m in re.finditer(r"(\d+\.\d+\.\d+)\s*[:：]?\s*(\d{1,3})?", txt):
        no = m.group(1)
        if no in valid and no not in out:
            out.append(no)
    return out


def main(tag="h3"):
    rows = json.load(open("eval_holdout3_pools.json", encoding="utf-8"))
    res = {x["q"]: x for x in json.load(open("eval_holdout3_result.json", encoding="utf-8"))}
    key = load_key()
    path = "sub_route_multi_%s.json" % tag
    cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}

    h1 = h3 = 0
    for n, r in enumerate(rows, 1):
        q = r["q"]
        sec, subs = res[q]["sec"].strip(), res[q]["subs"]
        if q in cache:
            got = cache[q]["got"]
        else:
            blocks, valid = [], set()
            for p in r["picks"]:
                lst = subs_of(p.strip())
                if not lst:
                    continue
                blocks.append("【%s】\n%s" % (p.replace("## ", "").strip(),
                                             "\n".join("%s %s" % (a, b) for a, b in lst)))
                valid |= {a for a, _ in lst}
            txt, _ = ask_llm(TPL % ("\n\n".join(blocks), q), key, SYSTEM,
                             temperature=0.0, max_tokens=80)
            got = parse(txt, valid)
            cache[q] = {"got": got, "raw": txt.strip(), "nsub": len(valid)}
            json.dump(cache, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        want = set(subs)
        ok1 = bool(got) and got[0] in want
        ok3 = bool(want & set(got))
        h1 += ok1
        h3 += ok3
        print("  %2d %s 选%-34s 期望%-12s %s" % (
            n, "√" if ok1 else ("~" if ok3 else "×"), str(got), str(sorted(want)), q[:26]))

    print("\n合并清单（%d 题）  首选命中 %.3f   前三命中 %.3f" % (len(rows), h1 / len(rows), h3 / len(rows)))
    print("平均清单长度 %.1f 个小节" % (sum(v["nsub"] for v in cache.values()) / len(cache)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h3")
