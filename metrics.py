"""评估指标：@1 / @3 / MRR

为什么需要 MRR：
  今天撞到过一次「@1 涨 16 点、@3 跌 2 点」，两个数方向相反，没法判断哪个更好。
  @1 只看第一名，@3 只看有没有进前 3，都对「第 2 名和第 3 名的差别」不敏感。

  MRR = 平均(1/排名)：第1名记 1，第2名 0.5，第3名 0.33，没进记 0。
  一个数字把整个排序质量都吃进去，两方案直接比大小就行。

  再往上是 NDCG（支持多级相关度），但那需要人工给每块标
  「相关 / 部分相关 / 不相关」，现阶段的成本不划算。
"""


def mrr(ranks):
    """ranks 是每题的排名（1/2/3）或 None（没进前 k）"""
    return sum(1.0 / r for r in ranks if r) / len(ranks)


def summary(ranks):
    n = len(ranks)
    h1 = sum(1 for r in ranks if r == 1)
    h3 = sum(1 for r in ranks if r)
    return h1, h3, h1 / n, h3 / n, mrr(ranks)


def row(tag, ranks, dt=None):
    h1, h3, a1, a3, m = summary(ranks)
    n = len(ranks)
    s = "%-16s %5d/%-2d %.3f %5d/%-2d %.3f %6.3f" % (tag, h1, n, a1, h3, n, a3, m)
    return s + ("  %6.0fs" % dt if dt is not None else "")


def header():
    print("%-16s %10s %10s %8s" % ("方案", "@1", "@3", "MRR"))
    print("-" * 52)
