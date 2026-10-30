"""阴性样本：测「库里没有答案时，系统会不会硬凑」

三档难度，一档比一档难：
  T1 完全离题      —— 检索应该彻底捞不到东西，分数极低
  T2 同领域无内容  —— 属于 AI/Agent 话题，但这本书没写，语义上会捞到相近章节
  T3 只提了名字    —— 书里出现过这个词但没有答案，最难，检索一定命中看似相关的块
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import eval as E
import time

chunks = E.chunks
search = E.search
CASES = E.CASES

NEG = [
    # T1 完全离题
    ("特斯拉股票明天会涨吗？",                                    "T1"),
    ("推荐几家北京好吃的火锅店",                                  "T1"),
    ("Excel 的数据透视表怎么做？",                                "T1"),
    # T2 同领域，书里没有
    ("智能体在量化交易里怎么落地？",                              "T2"),
    ("怎么用大模型自动生成公司的财务报表？",                      "T2"),
    ("智能体能不能替代招聘初筛？",                                "T2"),
    ("智能体用在直播带货的客服上，效果怎么样？",                  "T2"),
    ("合同审查该选哪个智能体框架？",                              "T2"),
    # T3 书里只提了名字
    ("CrewAI 和 AutoGen 该怎么选？",                              "T3"),
    ("Claude 和 GPT 在智能体任务上谁更强？",                      "T3"),
    ("Cursor 这类 AI 编辑器内部是怎么实现的？",                   "T3"),
    ("MCP 协议的版本号和向后兼容策略是怎样的？",                  "T3"),
]


def probe(q):
    """返回 top1/top2 精排分数 + top1 是哪一节"""
    res = search(q, k=3)
    s = [x[1] for x in res]
    return s[0], (s[1] if len(s) > 1 else 0.0), chunks[res[0][0]][1]


if __name__ == "__main__":
    t0 = time.time()

    print("=== 正样本（库里确实有答案，43 题）===")
    pos = []
    for q, ans, kind in CASES:
        s1, s2, title = probe(q)
        pos.append(s1)
        print("  %.3f | %.3f | %s" % (s1, s2, q[:44]))
    print()

    print("=== 阴性样本（库里没有答案，12 题）===")
    neg = []
    for q, tier in NEG:
        s1, s2, title = probe(q)
        neg.append((tier, s1, s2, q, title))
        print("  %s | %.3f | %.3f | %s" % (tier, s1, s2, q[:38]))
        print("          命中  %s" % title[:44])
    print()

    dt = time.time() - t0
    print("耗时 %.0fs" % dt)
    print()

    P = np.array(pos)
    print("=== 正样本 top1 分数分布 ===")
    print("  最低 %.3f | 5%% %.3f | 中位 %.3f | 95%% %.3f | 最高 %.3f"
          % (P.min(), np.percentile(P, 5), np.median(P), np.percentile(P, 95), P.max()))
    print()

    print("=== 各档阴性样本 top1 ===")
    for tier in ["T1", "T2", "T3"]:
        v = [s1 for t, s1, _, _, _ in neg if t == tier]
        print("  %s  n=%d  最高 %.3f  最低 %.3f  均值 %.3f"
              % (tier, len(v), max(v), min(v), sum(v) / len(v)))
    print()

    print("=== 阈值扫描：top1 低于阈值就拒答 ===")
    print("  阈值   正确拒答(T)  误拒(P)   总正确率")
    best = None
    for th in np.arange(0.10, 0.96, 0.05):
        rej = sum(1 for _, s1, _, _, _ in neg if s1 < th)      # 该拒的拒了
        wrong = sum(1 for s in pos if s < th)                  # 不该拒的拒了
        acc = (rej + (len(pos) - wrong)) / (len(pos) + len(neg))
        flag = ""
        if best is None or acc > best[1]:
            best = (th, acc)
        print("  %.2f      %2d/%2d       %2d/%2d      %.3f%s"
              % (th, rej, len(neg), wrong, len(pos), acc, flag))
    print()
    print("  最优阈值 %.2f，总正确率 %.3f" % best)
