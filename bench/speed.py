"""精排提速：同一个进程里横向对比，只比进程内相对值

为什么非要放进一个进程：第一次量出来「截断 800 字比截断 1200 字还慢」，
物理上讲不通。原因是两次运行之间机器负载差了 2 倍——跨进程的数字不能互相比较。
这一版把各配置放在同一进程、交替测量，消除机器负载这个变量。

试的方向：
  A 现状：全文 + 14 线程
  B 截断：标题保留，正文只送前 300 字
  C/D 线程数：CPU 核多不等于快，线程太多会互相抢
  E 批量：batch_size 调大，BLAS 利用率更高
  F 候选数：38 → 20（少算一半）
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import statistics
import time

import numpy as np
import torch

from eval import reranker, chunks

CUT = 300
pool = chunks[:38]
Q = "边想边干是啥意思"

full = [(Q, c[1] + "\n" + c[2]) for c in pool]
cut = [(Q, c[1] + "\n" + c[2][:CUT]) for c in pool]

print("候选 %d 对 | 全文平均 %d 字 | 截断后平均 %d 字"
      % (len(full), sum(len(t) for _, t in full) // len(full),
         sum(len(t) for _, t in cut) // len(cut)))


def timed(ps, bs=16, threads=None, reps=3):
    if threads:
        torch.set_num_threads(threads)
    reranker.predict(ps[:2], batch_size=2)
    ts = []
    for _ in range(reps):
        t0 = time.time()
        reranker.predict(ps, batch_size=bs)
        ts.append(time.time() - t0)
    return statistics.median(ts)


CONFIGS = [
    ("A 全文 14线程", dict(ps=full, threads=14)),
    ("B 截断 14线程", dict(ps=cut, threads=14)),
    ("C 截断 8线程", dict(ps=cut, threads=8)),
    ("D 截断 4线程", dict(ps=cut, threads=4)),
    ("E 截断 bs=32", dict(ps=cut, threads=14, bs=32)),
    ("F 截断 20块", dict(ps=cut[:20], threads=14)),
    ("G 截断 20块 8线程", dict(ps=cut[:20], threads=8)),
]

res = {k: [] for k, _ in CONFIGS}
for _round in range(2):                      # 交替两轮，摊平偶发负载
    for k, kw in CONFIGS:
        res[k].append(timed(**kw))
torch.set_num_threads(14)

base = statistics.median(res["A 全文 14线程"])
print("\n%-18s %8s %10s %8s" % ("配置", "中位", "对/秒", "提速"))
print("-" * 48)
for k, _ in CONFIGS:
    m = statistics.median(res[k])
    print("%-18s %7.2fs %9.1f %7.2fx" % (k, m, len(full) / m, base / m))

# 精度：截断到底改变了多少排序
sa = np.asarray(reranker.predict(full, batch_size=16))
sb = np.asarray(reranker.predict(cut, batch_size=16))
rk = lambda s: np.argsort(np.argsort(-s))
rho = np.corrcoef(rk(sa), rk(sb))[0, 1]
ta, tb = set(np.argsort(-sa)[:3]), set(np.argsort(-sb)[:3])
print("\n截断 vs 全文：Spearman %.4f   top3 重合 %d/3   分数平均偏差 %.4f"
      % (rho, len(ta & tb), np.abs(sa - sb).mean()))
