"""复现 README 的评测表：零 API、零模型下载

为什么单独写这个
----------------
README 的数字如果能手抄，就一定会抄错；代码一改，抄下来的数字立刻作废。
所以数字只能有一个来源：一个能在干净机器上跑出同一张表的脚本。

它凭什么不加载模型
------------------
评测的重活（向量粗筛 → cross-encoder 精排）已经在生成候选池时算完了，
结果存在 allsub_pools_*.json 里：每道题的候选块 id、精排分、模型选中的
章节和小节。复现只需要「在已有分数上加权、去重、取前三」——纯算术，
读两个 json 就够了。

依赖：只用标准库。想验证这点可以 `python -c "import sys; print(sys.modules)"`
之外单独跑一遍，或者在没装 torch 的环境里跑。

和生产的耦合（重要）
--------------------
这里的加分/去重逻辑必须和 rewrite.search_multi 完全一致，否则复现出来的
不是线上行为。search_multi 改了就得同步改这里——这是这份脚本唯一的维护负担，
换来的是「任何人 clone 下来 30 秒出数」。

用法
----
    python reproduce.py            # 四套题全跑
    python reproduce.py h3         # 只跑一套
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


# ── 从 answer.py 读配置（不 import：import 会连带加载 eval 的模型）─────
def load_config():
    src = open(os.path.join(HERE, "answer.py"), encoding="utf-8").read()

    def num_list(name):
        m = re.search(r"^%s\s*=\s*(\[[^\]]*\])" % name, src, re.M)
        return json.loads(m.group(1)) if m else None

    def num(name):
        m = re.search(r"^%s\s*=\s*(-?\d+(?:\.\d+)?)" % name, src, re.M)
        return float(m.group(1)) if m else None

    return {
        "VERSION": (re.search(r'^VERSION\s*=\s*["\']([^"\']+)', src, re.M) or [None, "?"])[1],
        "ROUTE_MODE": (re.search(r'^ROUTE_MODE\s*=\s*os\.getenv\([^,]+,\s*["\']([^"\']+)',
                                 src, re.M) or [None, "?"])[1],
        "ROUTE_BONUS": num_list("ROUTE_BONUS"),
        "SUB_BONUS": num_list("SUB_BONUS"),
        "CONF_GATE": num("CONF_GATE"),
    }


def load_meta():
    """块 id → (标题, 所属小节集合)"""
    raw = json.load(open(os.path.join(HERE, "chunks_meta.json"), encoding="utf-8"))
    return {x["i"]: (x["title"], set(x["subs"])) for x in raw}


def rank(row, meta, cfg, gate_missing=False):
    """从候选池复算 top-3，返回 [(块id, 精排分), ...]

    三步和生产一致：
      ① 章节加分：按 picks 的顺序分档（ROUTE_BONUS），有置信度且偏低才不加
      ② 小节加分：按 subs 的顺序分档（SUB_BONUS），只取 max 不相加
      ③ 按标题去重（同一节最多出一个代表块），取前三
    """
    cand, raw, picks, subs = row["cand"], row["raw"], row["picks"], row["subs"]
    confs = row.get("confs") or []
    rs = [float(x) for x in raw]

    rb = cfg["ROUTE_BONUS"] or []
    for j, i in enumerate(cand):
        title = meta[i][0]
        for pos, p in enumerate(picks):
            if p.strip() != title:
                continue
            w = rb[pos] if pos < len(rb) else rb[-1]
            # 门控的语义：有置信度、而且偏低 → 不加分。
            # 「没有置信度」不等于「置信度低」——subs 模式压根没有章节原始输出，
            # 那时候的正确答案是全额加分（和 route.parse_conf 的文档一致）。
            # gate_missing=True 是封盘前那个 bug 的行为，只留作对照。
            if cfg["CONF_GATE"]:
                low = (confs[pos] < cfg["CONF_GATE"]) if pos < len(confs) \
                    else gate_missing
                if low:
                    w = 0.0
            rs[j] += w
            break

    sb = cfg["SUB_BONUS"] or []
    if subs and sb:
        w = {no: (sb[pos] if pos < len(sb) else sb[-1]) for pos, no in enumerate(subs)}
        for j, i in enumerate(cand):
            hit = meta[i][1] & set(w)
            if hit:
                rs[j] += max(w[x] for x in hit)

    order = sorted(range(len(cand)), key=lambda j: -rs[j])
    out, seen = [], set()
    for j in order:
        title = meta[cand[j]][0]
        if title in seen:
            continue
        seen.add(title)
        out.append((cand[j], float(raw[j])))
        if len(out) == 3:
            break
    return out


def rank_of(hits, meta, sec, subs):
    """正确小节第一次出现在第几位；前三都没有返回 None"""
    want = set(subs)
    for n, (i, _) in enumerate(hits, 1):
        t, s = meta[i]
        if t.strip().startswith(sec.strip()) and (s & want):
            return n
    return None


SETS = [
    ("43", "allsub_pools_43.json", "43 开发集", "主证据"),
    ("h1", "allsub_pools_h1.json", "h1 留出", "参与过选型，偏乐观"),
    ("h2", "allsub_pools_h2.json", "h2 留出", "照失败补过词，偏乐观"),
    ("h3", "allsub_pools_h3.json", "h3 干净集", "唯一未污染，主证据"),
]


def run(tags, gate_missing=False):
    cfg = load_config()
    meta = load_meta()
    print("配置  VERSION=%s  ROUTE_MODE=%s  ROUTE_BONUS=%s  SUB_BONUS=%s  CONF_GATE=%s" % (
        cfg["VERSION"], cfg["ROUTE_MODE"], cfg["ROUTE_BONUS"], cfg["SUB_BONUS"],
        int(cfg["CONF_GATE"] or 0)))
    print("章节加分：%s\n" % ("缺置信度时被清零（封盘前的 bug）" if gate_missing
                              else "缺置信度时全额生效"))
    print("| 数据集 | 题量 | 小节@1 | 小节@3 | 说明 |")
    print("|---|---|---|---|---|")

    rows_out = []
    for tag, f, name, note in SETS:
        if tags and tag not in tags:
            continue
        path = os.path.join(HERE, f)
        if not os.path.exists(path):
            print("| %s | — | 缺 %s | — | %s |" % (name, f, note))
            continue
        rows = json.load(open(path, encoding="utf-8"))
        s1 = s3 = 0
        for r in rows:
            hits = rank(r, meta, cfg, gate_missing)
            rk = rank_of(hits, meta, r["sec"], r["subs_want"])
            s1 += rk == 1
            s3 += rk is not None
        n = len(rows)
        print("| %s | %d | **%.3f** | %.3f | %s |" % (name, n, s1 / n, s3 / n, note))
        rows_out.append((tag, n, s1 / n, s3 / n))

    print("\n小节@1 的分母是题目总数；答不上来的题算错，不从中剔除。")
    return rows_out


if __name__ == "__main__":
    args = sys.argv[1:]
    tags = {x for x in args if not x.startswith("--")}
    if "--both" in args:
        print("=== 生产口径 ===")
        run(tags, False)
        print("\n=== 对照：封盘前（缺置信度即清零）===")
        run(tags, True)
    else:
        run(tags, "--gate-missing" in args)
