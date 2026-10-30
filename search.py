import numpy as np
from embed import embed_query
from build_chunks import build_all

chunks = build_all()
vecs = np.load("chunks_vec.npy")          # (1278, 512)，每行是一个块


def search(question, k=3):
    qv = embed_query(question)            # (512,)  问题的向量
    scores = vecs @ qv              # ① 一次算出这 1278 个块各有多像
    order = np.argsort(-scores)[:k]             # ② 分数从大到小排，取前 k 个的下标
    return [(int(i), float(scores[i])) for i in order]


if __name__ == "__main__":
    for q in ["什么是智能体？", "怎么评估智能体的性能？"]:
        print("问:", q)
        for idx, s in search(q):
            print(f"   {s:.3f}  {chunks[idx][1][:34]}")
        print()
