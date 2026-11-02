"""目录加口语别名，路由准确率能不能从 12/13 往上走

昨天的错法很具体：「画流程图搭智能体」该落 6.5 LangGraph，
但目录那行只写了「6.5 框架四：LangGraph」——
「状态图」「StateGraph」「流程图」一个词都没有，模型没得对。

今天给 79 个标题各补了 2-4 个口语说法（route.ALIAS），
这个脚本只测一件事：**路由候选里有没有正确章节**（不管排第几）。

必须绕过缓存：route_cache.json 里存的是旧目录的结果，
不绕的话跑出来一模一样，会误判成「加别名没用」。
"""
import json
import os

from eval_oral import CASES
from llm import load_key
from route import route, MENU


def old_hit(q, want):
    """旧目录（缓存里的结果）有没有选中正确章节"""
    path = "archive/route_cache.json"
    if not os.path.exists(path):
        return None
    cache = json.load(open(path, encoding="utf-8"))
    picks = cache.get(q)
    if picks is None:
        return None
    return any(p.strip().startswith(want.strip()) for p in picks)


def main():
    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    print("MENU 现在 %d 字（含口语别名）\n" % len(MENU))

    new_ok = old_ok = 0
    rows = []
    for n, (name, want, q) in enumerate(CASES, 1):
        picks, raw = route(q, key, use_cache=False)      # 关键：不吃旧缓存
        hit = any(p.strip().startswith(want.strip()) for p in picks)
        o = old_hit(q, want)
        new_ok += hit
        old_ok += bool(o)
        rows.append((name, q, want, picks, raw, hit, o))
        print("%2d/%d %-14s %-18s %s  旧:%s 新:%s" % (
            n, len(CASES), name, q,
            "✓" if hit else "✗",
            "✓" if o else ("✗" if o is False else "?"),
            " / ".join(p.replace("## ", "")[:16] for p in picks) or raw[:20]))

    N = len(CASES)
    print("\n" + "=" * 70)
    print("路由命中（正确章节在 3 个候选里）")
    print("  旧目录（无别名）  %2d/%d = %.3f" % (old_ok, N, old_ok / N))
    print("  新目录（带别名）  %2d/%d = %.3f" % (new_ok, N, new_ok / N))
    print("=" * 70)

    bad = [r for r in rows if not r[5]]
    if bad:
        print("\n新目录仍然没选中的：")
        for name, q, want, picks, raw, _h, _o in bad:
            print("  %-14s %-18s 期望 %s" % (name, q, want.strip()))
            print("       原始输出: %r" % raw[:70])
            print("       解析结果: %s" % [p.replace("## ", "")[:20] for p in picks])


if __name__ == "__main__":
    main()
