"""第 ④ 站：端到端答案正确率（READM 局限 #3 的那一条）

为什么必须有这个
----------------
README 里所有数字都是**检索**指标。小节 @3 = 0.950 的意思是「正确的那一节
进了前三」，**不等于答对**——模型完全可能读错材料、漏掉关键句、或者把
两段话拼成一句错的。这条局限一直写在 README 里，今天补上实测值。

三个指标，一个客观两个主观，分开报
---------------------------------
  ① 引用命中率（客观）  回答引用的块，是否落在正确小节里。
     不需要模型判分，用 SUB_MAP 直接查——**这条最硬**。
  ② 答案正确率（主观）  拿 answers_43.json 里的要点逐条对照。
     判分也是模型，所以它只能当参考，不能当结论。
  ③ 拒答正确率（客观）  目录外的问题该不该拒。沿用 abstain.py 的三档阴性。

为什么用「要点制」而不是打分
---------------------------
打分（0-1）在 20 题上没法解释：0.8 分和 0.85 分的差别是什么？换成
「预先写好 2-4 条必须出现的事实点，答中算过」，判分标准是人能复核的，
而且换个人换次跑，结果稳定。40 题里 1 题 = 2.5 个点，所以只看整体比例。

成本与缓存
----------
每题一次生成 + 一次判分，20 题约 4 分钟。结果按问题缓存在
answer_eval_cache.json，改判分标准要加 --fresh，否则会读到旧结果。

用法
----
    python eval_answer.py              # 20 题 + 12 道阴性
    python eval_answer.py --limit 5    # 先试 5 题，验证管线
    python eval_answer.py --no-neg     # 只跑正样本
    python eval_answer.py --fresh      # 忽略缓存，全部重跑
"""
import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import eval as E
from eval import chunks
from llm import ask_llm, load_key
from answer import answer
from abstain import NEG

CACHE = "answer_eval_cache.json"

JUDGE_SYSTEM = (
    "你是严格的阅卷员。用户会给你：一道问题、若干条「必须出现的事实点」、"
    "以及一份回答。你要逐条判断每个事实点是否被这份回答**明确表达出来**。"
    "同义改写算命中；含糊、暗示、只说了一半不算；回答里出现了错误信息则该点算未命中。"
    "只输出 JSON，不要解释。"
)

JUDGE_TPL = (
    "问题：%s\n\n"
    "必须出现的事实点：\n%s\n\n"
    "回答：\n%s\n\n"
    "逐条判断，输出严格如下的 JSON（hit 的长度必须等于事实点数量，顺序一致）：\n"
    '{"hit": [true, false, ...], "bad": "如果回答里有明显错误，写一句；没有就写空字符串"}'
)


def with_retry(fn, tries=3, wait=5, tag=""):
    """网络抖动不该毁掉一整轮评测

    第一次跑全量时死在第 5 题的判分上：llm.ask_llm 里 urlopen 读超时，
    整个进程退出，前面 4 题白跑。评测脚本必须假设「调用会失败」——
    失败就重试，重试仍失败就记一笔继续，而不是让一次网络抖动清空整轮。
    """
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                print("     ! %s 连续失败 %d 次：%s" % (tag, tries, e))
                return None
            time.sleep(wait * (i + 1))
    return None


def judge(q, points, ans, key):
    body = "\n".join("%d. %s" % (i + 1, p) for i, p in enumerate(points))
    txt, _ = ask_llm(JUDGE_TPL % (q, body, ans[:2000]), key, JUDGE_SYSTEM,
                     temperature=0.0, max_tokens=200)
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return [False] * len(points), "判分输出解析失败：%s" % txt[:120]
    try:
        d = json.loads(m.group(0))
        hit = [bool(x) for x in d.get("hit", [])]
        hit = (hit + [False] * len(points))[:len(points)]
        return hit, str(d.get("bad", "") or "")
    except Exception as e:
        return [False] * len(points), "判分 JSON 解析失败：%s" % e


def cite_hit(hits, sec, subs):
    """引用的块是否落在正确小节（客观，不调模型）"""
    for idx, _ in hits:
        t = chunks[idx][1].strip()
        if not t.startswith(sec.strip()):
            continue
        if not subs:
            return True
        if E.SUB_MAP.get(idx, set()) & set(subs):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-neg", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--repair", action="store_true",
                    help="只重跑「答案为空」和「有答案但没判分」的题")
    args = ap.parse_args()

    cases = json.load(open("answers_43.json", encoding="utf-8"))
    if args.limit:
        cases = cases[:args.limit]
    key = load_key()
    if not key:
        print("没有 API key，这条链路跑不了：它要真调模型生成答案。")
        return 1

    cache = {}
    if os.path.exists(CACHE) and not args.fresh:
        cache = json.load(open(CACHE, encoding="utf-8"))

    if args.repair:
        # 把两类「不是答错」的记录清掉，让主循环重新生成：
        #   ① 没拒答但答案是空的 —— 调用失败，不是模型答错，不能算 0 分
        #   ② 有答案但 hit 是 None —— 判分那次调用挂了
        # 不区分这两类，最后算出来的「正确率」会把基础设施故障算成模型错误。
        bad_keys = [q for q, r in cache.items()
                    if (not r.get("refused") and not (r.get("answer") or "").strip())
                    or ((r.get("answer") or "").strip() and r.get("hit") is None)]
        for q in bad_keys:
            cache.pop(q, None)
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("修复模式：清除 %d 条需要重跑的记录\n" % len(bad_keys))

    t0 = time.time()
    rows = []
    for n, c in enumerate(cases, 1):
        q = c["q"]
        if q in cache and not args.fresh:
            r = cache[q]
        else:
            # 模型偶尔会返回空串——不是报错，是「什么都没说」。
            # 这种也算可重试的抖动，不能直接记 0 分（GRPO 那题就这么被误判过）。
            out = None
            for _ in range(3):
                out = with_retry(lambda: answer(q, key), tag="生成")
                if out and (out.get("refused") or (out.get("answer") or "").strip()):
                    break
                out = None
            if out is None:
                r = {"q": q, "refused": False, "error": "调用失败", "hits": [],
                     "answer": ""}
                cache[q] = r
                json.dump(cache, open(CACHE, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
                rows.append(dict(c, refused=False, cite=False, hit=[False] * len(c["key_points"]),
                                 bad="调用失败", answer=""))
                print("  %2d 调用失败  %s" % (n, q[:26]))
                continue
            ans = out.get("answer") or ""
            r = {
                "q": q, "refused": bool(out.get("refused")),
                "reason": out.get("reason", ""),
                "hits": [int(i) for i, _ in out.get("hits", [])],
                "answer": ans,
            }
            # 先落盘生成的答案，再去判分：判分挂了不该把答案一起丢
            if not r["refused"] and ans:
                cache[q] = r
                json.dump(cache, open(CACHE, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
                got = with_retry(lambda: judge(q, c["key_points"], ans, key), tag="判分")
                if got is None:
                    r["hit"] = [False] * len(c["key_points"])
                    r["bad"] = "判分调用失败"
                else:
                    r["hit"], r["bad"] = got
            cache[q] = r
            json.dump(cache, open(CACHE, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)

        cite = cite_hit([(i, 0) for i in r["hits"]], c["sec"], c["subs"]) \
            if r["hits"] else False
        hit = r.get("hit") or [False] * len(c["key_points"])
        rows.append(dict(c, refused=r["refused"], cite=cite, hit=hit,
                         bad=r.get("bad", ""), answer=r.get("answer", "")))
        print("  %2d %s%s %s/%d  %s" % (
            n, "拒" if r["refused"] else ("引√" if cite else "引×"),
            " 要√" if all(hit) else " 要×", sum(hit), len(hit), q[:26]))

    # ── 拒答：目录外的问题该不该拒 ──
    neg_rows = []
    if not args.no_neg:
        print("\n阴性样本（该拒答）：")
        for q, tier in NEG:
            if q in cache and not args.fresh:
                r = cache[q]
            else:
                out = with_retry(lambda: answer(q, key), tag="阴性生成")
                if out is None:
                    print("     ! 阴性题调用失败：%s" % q[:24])
                    continue
                r = {"q": q, "refused": bool(out.get("refused")),
                     "reason": out.get("reason", ""),
                     "hits": [int(i) for i, _ in out.get("hits", [])],
                     "answer": out.get("answer") or ""}
                cache[q] = r
                json.dump(cache, open(CACHE, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            neg_rows.append({"q": q, "tier": tier, **r})
            print("  %s %s %s" % ("√拒" if r["refused"] else "×答", tier, q[:30]))

    # ── 汇总 ──
    N = len(rows)
    answered = [r for r in rows if not r["refused"]]
    strict = sum(1 for r in answered if all(r["hit"]))
    points_hit = sum(sum(r["hit"]) for r in rows)
    points_all = sum(len(r["hit"]) for r in rows)
    cite_ok = sum(1 for r in answered if r["cite"])

    print("\n" + "=" * 62)
    print("正样本 %d 题（拒答 %d 题）" % (N, N - len(answered)))
    print("  答案正确率（要点全中 / 已作答）  %d/%d = %.3f" % (
        strict, len(answered), strict / max(1, len(answered))))
    print("  要点覆盖率（命中的点 / 全部点）  %d/%d = %.3f" % (
        points_hit, points_all, points_hit / max(1, points_all)))
    print("  引用命中率（引用块落在正确小节）  %d/%d = %.3f" % (
        cite_ok, len(answered), cite_ok / max(1, len(answered))))
    if neg_rows:
        for tier in ("T1", "T2", "T3"):
            g = [r for r in neg_rows if r["tier"] == tier]
            if g:
                print("  拒答正确率 %s                %d/%d = %.3f" % (
                    tier, sum(r["refused"] for r in g), len(g),
                    sum(r["refused"] for r in g) / len(g)))
        print("  拒答正确率 合计                 %d/%d = %.3f" % (
            sum(r["refused"] for r in neg_rows), len(neg_rows),
            sum(r["refused"] for r in neg_rows) / len(neg_rows)))

    # 失败的题：给下一个人看「错在哪一环」
    bad = [r for r in answered if not all(r["hit"])]
    if bad:
        print("\n没全中的题（检索错 / 读错材料 / 要点没覆盖）：")
        for r in bad:
            print("  [%s] %s" % ("引√" if r["cite"] else "引×", r["q"][:34]))
            for p, h in zip(r["key_points"], r["hit"]):
                if not h:
                    print("      缺：%s" % p[:56])
            if r["bad"]:
                print("      错：%s" % r["bad"][:70])

    print("\n耗时 %.0fs   明细缓存 answer_eval_cache.json" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
