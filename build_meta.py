"""把复现要用的块元数据固化成 JSON（标题 + 所属小节）

为什么需要这个文件
------------------
reproduce.py 要能在一台只装了 numpy 的机器上跑出 README 的数字，所以
不能 import eval.py —— 那个模块一 import 就要加载 embedding 和 cross-encoder
（几百 MB、十几秒），等于「复现」变成「先下载 500MB 模型」。

但排序必须知道两件事：块的标题（章节去重按标题比）、块属于哪个小节
（小节加分按 SUB_MAP 查）。这两样都是静态的，抽出来存盘即可。

代价：书变了要重跑一次（和 subs_index.json 一样）。
"""
import json
import sys

import eval as E
from eval import chunks

SUB_RE = __import__("re").compile(r"^###+ (\d+\.\d+\.\d+)\s*(.*)$", __import__("re").M)

out = []
for i, c in enumerate(chunks):
    title, body = c[1].strip(), c[2]
    out.append({
        "i": i,
        "title": title,
        # 章节标题（"## 8.3 RAG系统..."）单独存一份，省得每次正则
        "sec": title,
        "subs": sorted(E.SUB_MAP.get(i, set())),
        "chars": len(body),
    })

json.dump(out, open("chunks_meta.json", "w", encoding="utf-8"),
          ensure_ascii=False, separators=(",", ":"))
print("块 %d  有归属的 %d  文件 %.1f KB" % (
    len(out), sum(1 for x in out if x["subs"]),
    __import__("os").path.getsize("chunks_meta.json") / 1024))
