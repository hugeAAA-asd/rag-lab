"""把「章节 → 小节清单」固化成 JSON，给 route.py 用

为什么不直接在 route.py 里从 chunks 提取：chunks 来自 eval.py，
而 eval.py 一 import 就要加载 embedding + cross-encoder（十几秒、几百 MB）。
路由是小节选择的上游，为了读一张静态清单把精排模型拖进来不划算。
清单只在书变了的时候需要重建，生成一次存盘就行。
"""
import json

import eval as E
from eval import chunks

SUB_RE = __import__("re").compile(r"^###+ (\d+\.\d+\.\d+)\s*(.*)$", __import__("re").M)

idx = {}
for i, c in enumerate(chunks):
    t = c[1].strip()
    if not t.startswith("## "):
        continue
    for m in SUB_RE.finditer(c[2]):
        no, name = m.group(1), m.group(2).strip()
        lst = idx.setdefault(t, [])
        if all(x[0] != no for x in lst):
            lst.append([no, name])

json.dump(idx, open("subs_index.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("章节 %d   小节 %d" % (len(idx), sum(len(v) for v in idx.values())))
print("最多小节的章：", max(idx.items(), key=lambda kv: len(kv[1]))[0][:30],
      len(max(idx.values(), key=len)))
print("空清单的章：", [k[:20] for k, v in idx.items() if not v])
