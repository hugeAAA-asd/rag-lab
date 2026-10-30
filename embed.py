import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 模型已在本地缓存，不再联网（下新模型时临时设 0）

import numpy as np
from sentence_transformers import SentenceTransformer
from build_chunks import build_all

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

model = SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks):
    texts = []
    for c in chunks:
        texts.append(c[1] + "\n" + c[2])   # 标题 + 正文，标题是最强的语义信号
    return model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,          # ② 要不要归一化成单位长度？
        show_progress_bar=True,
    )


def embed_query(question):
    return model.encode("" + question,    # ③ 问题侧要加什么？
                        normalize_embeddings=True)


if __name__ == "__main__":
    chunks = build_all()
    vecs = embed_chunks(chunks)
    print("块数:", len(chunks))
    print("矩阵形状:", vecs.shape)
    np.save("chunks_vec.npy", vecs)
    print("已保存 chunks_vec.npy")
