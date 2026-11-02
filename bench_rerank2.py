"""精排耗时复测：取多次的最小值

上一版 bench 出来的数不单调（截断 1200 字 9.29s，截断 800 字反而 17.41s），
物理上讲不通，说明单次测量被机器上的其他负载污染了。
同一配置跑 5 次取最小值——最小值最接近「没有干扰时的真实耗时」。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import statistics
import time

from eval import reranker, chunks

pool = chunks[:38]
Q = "边想边干是啥意思"
pairs = [(Q, c[1] + "\n" + c[2]) for c in pool]


def bench(ps, bs=16, reps=5):
    reranker.predict(ps[:2], batch_size=2)
    ts = []
    for _ in range(reps):
        t0 = time.time()
        reranker.predict(ps, batch_size=bs)
        ts.append(time.time() - t0)
    return min(ts), statistics.median(ts)


print("候选 %d 对，平均 %d 字，最长 %d 字" % (
    len(pairs), sum(len(p[1]) for p in pairs) // len(pairs),
    max(len(p[1]) for p in pairs)))
print()
print("%-26s %8s %8s %10s" % ("配置", "最快", "中位", "对/秒"))
print("-" * 56)

configs = [("不截断（现状）", pairs)]
for n in (512, 384, 256, 192, 128):
    configs.append(("截断 %d 字" % n, [(q, t[:n]) for q, t in pairs]))

for tag, ps in configs:
    lo, med = bench(ps)
    print("%-26s %7.2fs %7.2fs %9.1f" % (tag, lo, med, len(ps) / lo))
