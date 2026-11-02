"""小节选择原型：章节定了，能不能在章内挑对那个 x.y.z

为什么先只测「选得准不准」，不直接测检索分：
上限实验（exp_subpool2.py）已经证明——把小节开挂补池并给小节加分，
@1 就是 1.000，排序这层没有别的问题。剩下的全部风险都在「选小节」这一步。
选不准，后面所有工程都是白做；所以先单独量它的准确率，别的先不管。

输入只用两样：该章的小节清单（编号 + 标题）+ 用户原问。
不给小节正文——给了就等于把精排的活提前做了，既不省 token 也不好评估。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import re
import sys

import eval as E
from eval import chunks
from llm import ask_llm, load_key
from route import _sec

SUB_RE = re.compile(r"^###+ (\d+\.\d+\.\d+)\s*(.*)$", re.M)


def subs_of(sec):
    """某一章下的小节清单，按书中顺序去重"""
    out, seen = [], set()
    for i, c in enumerate(chunks):
        if not c[1].strip().startswith(sec):
            continue
        for m in SUB_RE.finditer(c[2]):
            no, name = m.group(1), m.group(2).strip()
            if no in seen:
                continue
            seen.add(no)
            out.append((no, name))
    return out


SYSTEM = ("你是图书检索员。用户会问一个问题，你要在一章的小节清单里挑出最可能回答它的小节。"
          "只输出小节号，不要解释。")

TPL = (
    "这是一本关于智能体（Agent）的技术书。%s\n"
    "它下面的小节清单：\n\n%s\n\n"
    "读者的问题：%s\n\n"
    "从上面的清单里选出最可能回答这个问题的小节，按把握从大到小排，最多 3 个。\n"
    "每个写成「小节号:匹配度」，匹配度是 0-100 的整数。\n"
    "只输出这个列表，用逗号分隔，不要输出小节名，不要解释。例如：8.3.2:88,8.3.4:40\n"
    "匹配度要拉开差距，不要因为排在第一个就写 90。\n"
    "宁可少选：只有一个真的对，就只输出一个。"
)


def parse(txt, valid):
    """只认清单里存在的小节号，防止模型把章号和节号揉一起"""
    out = []
    for m in re.finditer(r"(\d+\.\d+\.\d+)\s*[:：]?\s*(\d{1,3})?", txt):
        no = m.group(1)
        if no in valid and no not in out:
            out.append(no)
    return out


def main(tag):
    rows = json.load(open({"h3": "eval_holdout3_pools.json",
                           "43": "pools_43_sub2.json"}[tag], encoding="utf-8"))
    res = {}
    if tag == "h3":
        res = {x["q"]: x for x in json.load(open("eval_holdout3_result.json", encoding="utf-8"))}
    else:
        res = {x["q"]: x for x in json.load(open("pools_43_sub2.json", encoding="utf-8"))}

    key = load_key()
    cf, path = "sub_route_%s.json" % tag, "sub_route_%s.json" % tag
    cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}

    hit1 = hit3 = 0
    for n, r in enumerate(rows, 1):
        q = r["q"]
        sec = res[q]["sec"]
        subs = res[q]["subs"]
        sec = sec.strip()
        lst = subs_of(sec)
        valid = {a for a, _ in lst}
        if not valid:
            print("  %2d 章节无小节 %s %s" % (n, sec, q[:26]))
            continue
        if q in cache:
            got = cache[q]
        else:
            body = "\n".join("%s %s" % (a, b) for a, b in lst)
            txt, _ = ask_llm(TPL % ("第 %s 章下" % sec.replace("## ", "").strip(), body, q),
                             key, SYSTEM, temperature=0.0, max_tokens=80)
            got = parse(txt, valid)
            cache[q] = got
            json.dump(cache, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        want = set(subs)
        ok1 = bool(got) and got[0] in want
        ok3 = bool(want & set(got))
        hit1 += ok1
        hit3 += ok3
        print("  %2d %s%s 选%s 期望%s  %s" % (n, "√" if ok1 else ("~" if ok3 else "×"),
              "", got, sorted(want), q[:30]))

    N = len(rows)
    print("\n%s（%d 题）  首选命中 %.3f   前三命中 %.3f" % (tag, N, hit1 / N, hit3 / N))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h3")
