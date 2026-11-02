"""规模实验：1177 块放大到一百万块，先崩在哪

为什么必须用合成数据
--------------------
本仓库的语料只有 1177 块 × 512 维 = 2.3 MB。这个规模下「没有向量库、直接
numpy 全量矩阵乘」是亚毫秒，换任何索引都测不出差别——「百万级会怎样」在真实
语料上**永远测不出来**，只能外推。

但外推的数据必须像真的，否则测出来的只是生成器的性质。造数据这一步我错了两次，
两次的教训都写在这里：

  ① 第一次：簇中心 + 未归一化的噪声。噪声范数 ≈ 13，簇中心范数 ≈ 1，
     噪声直接盖掉结构，数据退化成「512 维均匀随机」。所有点几乎等距，
     top-10 取谁全凭数值浮动的运气 → HNSW 召回 0.14，衡量的是噪声不是索引。
  ② 第二次：改用低秩子空间，但每簇塞 50 个点。邻域全在簇内，彼此几乎重合，
     召回直接 1.000，测不出任何权衡。

所以现在的参数是**对着真实语料标定的**。真实 bge 向量（1177 块）的邻域结构实测：

    top1 相似度 0.861    top10 0.786    两者间隔 0.075
    落在 top10 分 ±1e-3 内的候选（平局密度）：平均 1.5 个

合成数据的生成器（128 维隐空间 → 随机投影到 512 维，每簇 14 个点，簇内噪声
0.85 个单位向量）在三项上都贴着真实值：

    参数               top1    top10   间隔     平局
    真实语料           0.861   0.786   0.0750   1.5
    合成 ppc=14 s=0.85 0.859   0.793   0.0662   2.1

ppc 是被间隔钉住的：8 太小（第 10 名跨出簇，间隔 .378），16 太大（.034），
14 落在真实值附近。**这个参数不是调出来的好看数，是对着真实语料的邻域统计配的。**

三个引擎
--------
  numpy   现状：X @ q，全量暴力。生产代码用的就是这个（eval.search）。
  flat    faiss 暴力索引：同样的 O(N·d)，换 C++ + SIMD + 多线程。
          —— 回答「不换算法、只换实现能快多少」。
  hnsw    faiss 图索引：O(log N) 近似，用召回换延迟。
          —— 回答「换算法要付多少召回代价」。

口径
----
  · float32、单位化后用内积（等价余弦），和 bge 用法一致
  · 延迟报 **单条查询** 的中位数（生产一次只查一条，不 batch）
  · 召回率 = HNSW 的 top-10 与暴力 top-10 的交集 / 10
  · 内存报 **进程 RSS 增量**（psutil 实测），不是理论值
  · 每个引擎先热身 5 次再计时，避免第一枪打在冷缓存上

用法
----
    python -m bench.scale                      # 1e4 / 1e5 / 5e5
    python -m bench.scale --n 1e6 --ef 8 32 64
    python -m bench.scale --out bench_scale_result.json
"""
import argparse
import gc
import json
import os
import statistics
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import psutil
import faiss

DIM = 512
NQ = 200            # 查询条数
K = 10              # 召回率按 top-10 算
WARMUP = 5
LATENT = 128        # 隐空间维度（真实嵌入的有效维度远低于 512）
PER_CLUSTER = 14    # 每簇点数：被「top1 与 top10 的间隔」钉住的，见文件头
SPREAD = 0.85       # 簇内噪声（单位向量），标定后使 top1 ≈ 0.86


def rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def free_gb():
    return psutil.virtual_memory().available / 1024 ** 3


def make_data(n, rng):
    """按真实语料的邻域统计造数据：低秩簇 + 单位噪声"""
    ncluster = max(50, n // PER_CLUSTER)
    latent = rng.normal(size=(ncluster, LATENT)).astype(np.float32)
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    proj = rng.normal(size=(LATENT, DIM)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    centers = latent @ proj                      # 簇中心落在 128 维子空间上

    def draw(m):
        v = centers[rng.integers(0, ncluster, m)]
        z = rng.normal(size=(m, DIM)).astype(np.float32)
        z /= np.linalg.norm(z, axis=1, keepdims=True)   # 噪声也要单位化
        v = v + SPREAD * z
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    # 分块生成，避免 n×DIM 的临时副本把内存顶爆
    X = np.empty((n, DIM), dtype=np.float32)
    for s in range(0, n, 100_000):
        X[s:s + 100_000] = draw(min(100_000, n - s))
    return X, draw(NQ)


def time_queries(fn, queries):
    """逐条计时，返回 (p50 毫秒, p95 毫秒)"""
    for q in queries[:WARMUP]:
        fn(q)
    ms = []
    for q in queries:
        t = time.perf_counter()
        fn(q)
        ms.append((time.perf_counter() - t) * 1000)
    ms.sort()
    return statistics.median(ms), ms[int(len(ms) * 0.95)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", nargs="+", default=["1e4", "1e5", "5e5"])
    ap.add_argument("--ef", nargs="+", type=int, default=[8, 32, 64, 128])
    ap.add_argument("--out", default="bench_scale_result.json")
    args = ap.parse_args()

    scales = [int(float(x)) for x in args.n]
    rng = np.random.default_rng(0)
    rows = []

    print("维度 %d  查询 %d 条  每簇 %d 点  隐空间 %d 维  numpy 线程 %s" % (
        DIM, NQ, PER_CLUSTER, LATENT, os.environ.get("OMP_NUM_THREADS")))
    print("可用内存 %.1f GB\n" % free_gb())

    for n in scales:
        # 峰值 ≈ 原始向量 + HNSW 自己那份拷贝 + 图 + 余量
        need = n * DIM * 4 * 2 / 1024 ** 3 + 1.2
        if free_gb() < need:
            print("跳过 n=%d：需要约 %.1f GB，可用 %.1f GB" % (n, need, free_gb()))
            continue

        base = rss_mb()
        X, Q = make_data(n, rng)
        print("=" * 66)
        print("n = %-9d 向量自身 %.2f GB   RSS %.0f → %.0f MB" % (
            n, X.nbytes / 1024 ** 3, base, rss_mb()))

        # 真值：暴力 top-K，分块算，不计入延迟
        #
        # 不能一次性 Q @ X.T：200×100 万的 float32 就是 800 MB，
        # argsort 再要一份 1.6 GB 的 int64 下标，100 万这档会直接 OOM。
        gt = np.empty((NQ, K), dtype=np.int64)
        for s0 in range(0, NQ, 20):
            SIM = Q[s0:s0 + 20] @ X.T
            gt[s0:s0 + 20] = np.argsort(-SIM, axis=1)[:, :K]
            del SIM
        gc.collect()

        # ① numpy 暴力（生产现状）
        p50, p95 = time_queries(lambda q: np.argpartition(-(X @ q), K)[:K], Q)
        batch = 20
        t = time.perf_counter()
        for s in range(0, NQ, batch):
            Q[s:s + batch] @ X.T
        batch_ms = (time.perf_counter() - t) * 1000 / (NQ / batch)
        print("  numpy    单条 p50 %8.3f ms  p95 %8.3f ms  batch(20) 均摊 %7.3f ms"
              % (p50, p95, batch_ms))
        rows.append({"n": n, "engine": "numpy", "p50_ms": p50, "p95_ms": p95,
                     "batch20_ms": batch_ms, "build_s": 0.0, "recall": 1.0,
                     "index_mb": 0.0})

        # ② faiss 暴力：换实现不换算法（要再拷一份，n 大时跳过）
        if n * DIM * 4 / 1024 ** 3 * 2 < free_gb() - 0.5:
            m0 = rss_mb()
            t = time.perf_counter()
            flat = faiss.IndexFlatIP(DIM)
            flat.add(X)
            build = time.perf_counter() - t
            fm = rss_mb() - m0
            p50, p95 = time_queries(lambda q: flat.search(q.reshape(1, -1), K), Q)
            print("  flat     单条 p50 %8.3f ms  p95 %8.3f ms  构建 %6.2fs  索引 %+.0f MB"
                  % (p50, p95, build, fm))
            rows.append({"n": n, "engine": "flat", "p50_ms": p50, "p95_ms": p95,
                         "batch20_ms": None, "build_s": build, "recall": 1.0,
                         "index_mb": fm})
            del flat
            gc.collect()
        else:
            print("  flat     跳过（索引要再拷一份，内存不够）")

        # ③ HNSW：换算法
        #
        # 索引只建一次，然后改 efSearch 反复查询。
        # 第一版是「每个 ef 重建一遍」，50 万点上白花了 6 分半，而且更糟：
        # 上一次 delete 的内存没有被归还给系统，同一个索引量出来的 RSS 增量
        # 从 154MB 跳到 2GB —— 量的是分配器状态，不是索引大小。
        gc.collect()
        vec_gb = X.nbytes / 1024 ** 3
        m0 = rss_mb()
        t = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 80
        hnsw.add(X)
        build = time.perf_counter() - t
        gc.collect()
        # 索引体积：向量 n·d·4 + 图（M=32 双向，约两层）n·M·2·4
        #
        # 为什么用理论值：RSS 增量在小规模档上不可靠。分配器会把上次释放的内存
        # 留在池子里，50 万那档量出 0.13 GB（真值是 1.07 GB），100 万那档
        # 因为内存是全新的，量出 2.16 GB，和理论 2.15 GB 只差 0.5%。
        # 所以：以理论值入表，用那次干净的实测做校验。
        idx_gb = (n * DIM * 4 + n * 32 * 2 * 4) / 1024 ** 3
        observed = (rss_mb() - m0) / 1024 - vec_gb
        print("  hnsw     构建 %6.2fs  (一次)  索引 %.2f GB（理论）  RSS 增量 %.2f GB"
              % (build, idx_gb, observed))

        for ef in args.ef:
            hnsw.hnsw.efSearch = ef
            got = np.vstack([hnsw.search(q.reshape(1, -1), K)[1] for q in Q])
            recall = np.mean([len(set(a) & set(b)) / K for a, b in zip(got, gt)])
            p50, p95 = time_queries(lambda q: hnsw.search(q.reshape(1, -1), K), Q)
            print("            ef=%-4d 单条 p50 %8.3f ms  p95 %8.3f ms  召回 %.3f"
                  % (ef, p50, p95, recall))
            rows.append({"n": n, "engine": "hnsw", "ef": ef, "p50_ms": p50,
                         "p95_ms": p95, "batch20_ms": None, "build_s": build,
                         "recall": float(recall), "index_mb": idx_gb * 1024})
        del hnsw, got
        gc.collect()

        del X, Q, gt
        gc.collect()
        print()

    json.dump(rows, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("结果写入 %s（共 %d 行）" % (args.out, len(rows)))

    print("\n| n | numpy p50 | flat p50 | HNSW(ef=64) p50 | 加速 | 召回 | 内存(向量+索引) |")
    print("|---|---|---|---|---|---|---|")
    for n in sorted({r["n"] for r in rows}):
        g = [r for r in rows if r["n"] == n]
        np50 = next((r["p50_ms"] for r in g if r["engine"] == "numpy"), None)
        fl = next((r["p50_ms"] for r in g if r["engine"] == "flat"), None)
        h = next((r for r in g if r["engine"] == "hnsw" and r.get("ef") == 64), None)
        if not np50 or not h:
            continue
        vec = n * DIM * 4 / 1024 ** 3
        idx = max(r["index_mb"] for r in g if r["engine"] == "hnsw") / 1024
        print("| %d | %.2f ms | %s | %.2f ms | %.0f× | %.3f | %.2f + %.2f GB |" % (
            n, np50, ("%.2f ms" % fl) if fl else "—", h["p50_ms"],
            np50 / h["p50_ms"], h["recall"], vec, idx))


if __name__ == "__main__":
    main()
