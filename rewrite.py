"""查询改写：把用户的口语问题，翻译成资料库里的说法

要解决的病：词汇鸿沟。
  用户：先检索外部资料再让模型生成答案，这套做法叫什么？
  书里：## 8.3 RAG系统：知识检索增强
  两边意思一样，一个字都不重合，bi-encoder 给到 0.790，前 3 都进不去。

做法：让 LLM 生成 3 个改写，连同原问题一共 4 个 query 去粗筛，
     每个块取 4 个 query 里的最高分（max 融合，不是平均），最后用原问题精排。

为什么是 max 而不是平均：
  平均会把「ReAct」这种强信号和两个口语变体搅在一起，稀释掉；
  max 只要有一个 query 咬住了就算数，不会互相拖累。

为什么精排必须用原问题：
  粗筛要的是「召回」——只要相关就行，怎么改写都可以；
  精排要的是「这段材料能不能回答用户真正问的那个问题」，所以必须拿原话去比。
  用改写句做精排，等于在回答一个用户没问的问题。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import re

import numpy as np

from embed import embed_query
from eval import vecs, reranker, chunks, rerank_text, SUB_MAP
from llm import ask_llm, load_key

N_REWRITE = 3
CACHE = "rewrites.json"

REWRITE_SYSTEM = (
    "你是检索查询改写器。把用户的口语问题改写成技术资料库里可能出现的说法，"
    "补上这个领域的专业术语、英文名、缩写。"
    "只做改写，不回答问题。"
)

REWRITE_TEMPLATE = (
    "把下面的问题改写成 %d 个不同的检索查询，每行一个，不要编号，不要解释：\n\n问题：%s"
)


def rewrite(question, key=None, n=N_REWRITE, use_cache=True):
    """返回 n 条改写。任何失败都退回空列表——没有改写也要能正常检索。

    use_cache 默认是开着的，这不只是为了省 token，更是为了「可复现」：
    改写用了 temperature=0.3，同一个问题每次调出来的句子都不一样。
    实测 13 题的小评估集上，同一次代码两次跑能差出 1 题（7.7 个点）——
    也就是说你以为在对比两个方案，其实在对比两次随机数。
    把结果写进 rewrites.json 之后，第二次跑的输入和第一次完全相同，
    剩下的差异才真的是方案差异。
    """
    if use_cache:
        cache = load_cache()
        if question in cache:
            return cache[question]

    try:
        txt, _ = ask_llm(REWRITE_TEMPLATE % (n, question), key or load_key(),
                         REWRITE_SYSTEM, temperature=0.3)
    except Exception as e:
        print("   ! 改写失败，退回原问题：%s" % e)
        return []

    lines = []
    for raw in txt.strip().splitlines():
        s = raw.strip().lstrip("0123456789.、)-•· ").strip()
        if s and s not in lines:
            lines.append(s)
    lines = lines[:n]

    if use_cache:
        cache = load_cache()          # 重新读一次，别覆盖别的问题刚写进去的
        cache[question] = lines
        save_cache(cache)
    return lines


def _collect(scores, nsec, per, skip_idx=()):
    """按分数从高到低扫，按节发配额，返回块下标

    skip_idx 记的是块下标，不是节标题。
    这里踩过一次：原来按节标题跳过，结果 1.1 节有 12 块、主候选只进了 2 块，
    改写补漏时因为「1.1 这一节已经覆盖了」整节被跳过，真定义永远进不来。
    节被覆盖 ≠ 块被覆盖。
    """
    order = np.argsort(-scores)
    out, count, nsec_seen = [], {}, 0
    for i in order:
        idx = int(i)
        if idx in skip_idx:
            continue
        title = chunks[idx][1]
        c = count.get(title, 0)
        if c >= per:
            continue
        if c == 0:
            if nsec_seen >= nsec:
                break
            nsec_seen += 1
        count[title] = c + 1
        out.append(idx)
    return out



_SUB_HEAD = re.compile(r"^###+ (\d+\.\d+\.\d+)\s*(.*)$", re.M)


def _sub_blocks(idxs, min_body=200):
    """该节每个小节取「第一块」，外加章节开头那块

    为什么题型从「前 2 块 + 余弦最高」换成「按小节覆盖」：
    一个二级章节就是若干小节，用户问的必然是其中某一个。旧规则在 10.2（5 小节 31 块）
    里永远只碰 10.2.1 那一片，问「数据怎么传过去」（答案在 10.2.3）时，
    正确块压根没进候选池——这种失败加多少 bonus 都救不回来，是候选缺失不是排序错。

    为什么必须补「标题后正文不足 200 字」的下一块：
    切块是 800 字硬切、不看标题边界，所以「### 10.2.3 MCP 传输方式详解」这一行
    经常落在某个块的**末尾**，正文在下一个块里（实测 10.2.3 标题后 0 字、
    8.2.5 标题后 0 字、16.4.1 标题后 35 字）。只把那个光有标题的块塞进去，
    等于给精排一句没有内容的话。

    为什么还留着 idxs[0]：
    章节开头那块常是「导语 + 第一个小节的引子」，有时连三级标题都没有，
    只按小节取会把它漏掉，而定义往往就在那儿。
    """
    out, seen_sub = [], set()
    for k, i in enumerate(idxs):
        body = chunks[i][2]
        fresh = [m for m in _SUB_HEAD.finditer(body) if m.group(1) not in seen_sub]
        if not fresh:
            continue
        for m in fresh:
            seen_sub.add(m.group(1))
        out.append(i)
        tail = body[fresh[-1].end():].strip()
        if len(tail) < min_body and k + 1 < len(idxs):
            out.append(idxs[k + 1])
    return list(dict.fromkeys(out))


def search_multi(question, rews=None, k=3, per=2, nsec=15, extra_sec=5,
                 picks=None, bonus=0.0, return_raw=False, route_pick="head",
                 confs=None, gate=0, subs=None, sub_bonus=0.0):
    """原问题选主候选，改写只补漏，精排只用原问题

    注意这里不是 max 融合。试过：让改写和原问题平起平坐地投票（每块取最高分），
    改写句越准越坏事——「StateGraph」「BaaS LLMOps」这些词会撞上别处「提到过」
    它们的块，把真正讲这一节的挤掉。8 题实测：基线 4/8、max 4/8、本方案 6/8，
    而且本方案是基线的严格超集（救回 2 题，0 题被弄坏）。

    picks / bonus 是后来加的：极短口语场景下 cross-encoder 本身是瞎的
    （「边想边干是啥意思」会被判给 (开头) 块 0.271，而路由明明选中 4.2 ReAct），
    这时候必须让路由这个更可靠的信号参与打分，否则它 12/13 的准确率会被
    精排压到 4/13。bonus=0 时本函数与加这两个参数之前完全等价。
    """
    qv = embed_query(question)
    scores = vecs @ qv

    main = _collect(scores, nsec, per)
    seen = set(main)

    extra = []
    for r in (rews or []):
        got = _collect(vecs @ embed_query(r), extra_sec, 1, skip_idx=seen)
        extra += got
        seen |= set(got)

    # 路由补漏：命中的章节挑几块进来，不经过改写的模糊匹配
    #
    # 挑块的规则是踩过坑之后定的：一开始只取「该节余弦最高」的块，结果
    #   8.3 拿到的是讲 HyDE 的那块（余弦 0.515），而正确答案在第 1 块
    #       「8.3.1 RAG 的基础知识」（余弦 0.439）；
    #   1.1 拿到的是讲旅行规划的例子，而真定义在 idx=1「智能体被定义为…」。
    # 余弦天然偏好短块和具体例子（信息密度高、词重复多），而章节开头
    # 恰恰是定义和导语，才是最该给模型的材料。所以改成前 2 块 + 余弦最高 1 块。
    if picks:
        for rank, title in enumerate(picks):
            idxs = [i for i, c in enumerate(chunks) if c[1].strip() == title.strip()]
            if not idxs:
                continue
            # sub1 = 只在首位章节做深度覆盖（每小节第一块），后位仍用浅覆盖。
            # 为什么：候选不是越多越好。实测把 3 个章节都深度展开后，43 题掉了 8 题——
            # 因为最终按章节去重、一个章节只出一个代表，弱章节多塞几块，
            # 反而让「错块当上代表」的机会变多。首位章节是路由最可信的判断，
            # 深度展开的钱只花在那儿。
            deep = route_pick == "sub" or (route_pick in ("sub1", "sub2") and rank == 0)
            if deep:
                chosen = [idxs[0]] + _sub_blocks(idxs)
                if route_pick == "sub2":
                    # 并上旧的「前 2 块 + 余弦最高」，两套先验各管一段：
                    #   · 章节开头两块：书里爱把定义、导语放在这儿，书面问题多半命中
                    #   · 每小节第一块：口语问题常指向小节中段，只靠开头够不着
                    #   · 余弦最高：补具体细节块
                    # 只取并集不重排，谁进池子由精排裁决——提名的活儿归规则，裁决的活儿归模型。
                    chosen = idxs[:2] + [max(idxs, key=lambda i: scores[i])] + chosen
            else:
                chosen = idxs[:2]
                best = max(idxs, key=lambda i: scores[i])
                if best not in chosen:
                    chosen.append(best)
            for i in chosen:
                if i not in seen:
                    seen.add(i)
                    extra.append(i)

    # 小节补漏：选中的小节，块全进池
    #
    # 为什么章节那句「前 2 块 + 每小节第一块」还不够：那套规则是在章节里
    # 撒一层网，网眼很大——8.3 有 13 个块，撒完只进来 2 个，而答案在
    # 8.3.4（分层架构）。实测 h3 的 6 个失败题里，期望小节的块在池中的
    # 数量是 0~2 个，不是分数低，是根本没进来。
    # 一旦知道是哪一小节，就不必撒网了，直接把它整段捞进来。
    if subs:
        for no in subs:
            for i in range(len(chunks)):
                if i in seen:
                    continue
                if no in SUB_MAP.get(i, set()):
                    seen.add(i)
                    extra.append(i)

    cand = main + extra

    pairs = [(question, rerank_text(i)) for i in cand]
    raw = np.array(reranker.predict(pairs, batch_size=16), dtype=float)
    rs = raw.copy()          # rs 只用来排序，raw 才是要交给下游的分数

    # 实验出口：把「候选 + 原始分」交出去，让调用方用不同 bonus 反复重排。
    # 精排是唯一瓶颈（每题 10s），bonus 只影响排序——重排一次是微秒级。
    # 不这么拆的话，扫 3 组 bonus 就要把同一批题精排 3 遍。
    if return_raw:
        return cand, raw, picks

    if picks and bonus:
        # bonus 可以是单个数字（所有路由章节同权），也可以是一个列表
        # （按路由给出的顺序分档：第 1 个候选最可信，往后递减）
        #
        # 为什么需要分档：三个候选同权时，第 3 个候选和第 1 个平起平坐。
        # 「能感知环境并自主行动的系统指什么」路由给 1.1/1.2/1.4，
        # 1.1.1（正式定义，0.860）被 1.4.3（「与工作流不同…」，0.885）压过——
        # 两个都 +0.5，而后者原始分本来就高 0.025。
        # 路由自己都说了 1.1 最像，就不该让第 3 个候选和它同权。
        weights = {}
        for pos, p in enumerate(picks):
            if isinstance(bonus, (list, tuple)):
                w = bonus[pos] if pos < len(bonus) else bonus[-1]
            else:
                w = bonus
            # 置信度门控：路由自己也说「没什么把握」的候选，不给加分。
            #
            # 为什么不是「按置信度打折」而是「开关」：打折只是把 +1.0 缩成 +0.35，
            # 能翻盘的照样翻、不能翻盘的照样不能——实测 43 题上 0 好 0 坏，
            # 等于白改。开关不一样，它让低置信度的候选彻底退回精排排序，
            # 也就是「不懂就别插嘴」。
            # 阈值 60 有平台：扫 50/60/70 结果完全一样，80 才开始掉
            #（见 exp_gate_scan.py），所以不是恰好卡在某个数上。
            if confs is not None and gate and (
                    pos >= len(confs) or confs[pos] < gate):
                w = 0.0
            if w:
                weights[p.strip()] = w
        for j, i in enumerate(cand):
            w = weights.get(chunks[i][1].strip())
            if w:
                rs[j] += w

    # 小节级加分：选中小节的块，在章内优先
    #
    # 这一层解决的是「章选对了、块挑错了」。前面只给章节加分，一个章节
    # 十几个块平起平坐，最后按章去重时，代表块由精排分决定——而精排
    # 在极短口语问题上是瞎的（「从上到下分成几层」给了开头块 0.03）。
    # 小节信号比章节细一级：它知道答案在第几段，那就让那几段先上。
    #
    # 取 max 而不是相加：一个块可能同时属于两个小节（800 字硬切造成的
    # 跨节块，见 eval._build_sub_map），相加会让跨节块的分数虚高，
    # 而这些块恰恰是最模糊的那一批。
    if subs and sub_bonus:
        sw = {}
        for pos, no in enumerate(subs):
            if isinstance(sub_bonus, (list, tuple)):
                w = sub_bonus[pos] if pos < len(sub_bonus) else sub_bonus[-1]
            else:
                w = sub_bonus
            if w:
                sw[no] = w
        for j, i in enumerate(cand):
            hit = SUB_MAP.get(i, set()) & set(sw)
            if hit:
                rs[j] += max(sw[x] for x in hit)

    # 为什么返回 raw 而不是 rs：answer.py 拿 top1 分数和 MIN_SCORE=0.60 比大小来决定拒答。
    # bonus 把分数推到 1.xx 之后这个阈值就永远不生效——一个被路由选中但内容完全
    # 不沾边的块，会带着高分闯进生成环节，还不会被拦。
    # 同一个分数在两个环节含义不能分裂：排序时它是综合信号，拒答时它必须是精排的原始判断。
    picked, seen = [], set()
    for j in np.argsort(-rs):
        title = chunks[cand[int(j)]][1]
        if title in seen:
            continue
        seen.add(title)
        picked.append((cand[int(j)], float(raw[int(j)])))
        if len(picked) == k:
            break
    return picked


def load_cache():
    if os.path.exists(CACHE):
        return json.load(open(CACHE, encoding="utf-8"))
    return {}


def save_cache(data):
    json.dump(data, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    import time
    from eval import CASES

    key = load_key()
    if not key:
        print("没找到 DEEPSEEK_API_KEY")
        raise SystemExit(1)

    cache = load_cache()
    rows = []
    t0 = time.time()

    for n, (q, answers, kind) in enumerate(CASES, 1):
        if q in cache:
            rews = cache[q]
        else:
            rews = rewrite(q, key)
            cache[q] = rews
            save_cache(cache)
            print("  改写 %d/%d: %s" % (n, len(CASES), q[:36]))

        res = search_multi(q, rews)
        rank = None
        for r, (idx, s) in enumerate(res, 1):
            if any(chunks[idx][1].startswith(a) for a in answers):
                rank = r
                break
        rows.append((q, rank, rews, kind))

    dt = time.time() - t0
    N = len(CASES)
    h1 = sum(1 for _, r, _, _ in rows if r == 1)
    h3 = sum(1 for _, r, _, _ in rows if r is not None and r <= 3)

    print("\n" + "=" * 60)
    print("查询改写 + max 融合   耗时 %.0fs" % dt)
    print("  @1 = %2d/%d = %.3f   （基线 0.651，%+.3f）" % (h1, N, h1 / N, h1 / N - 0.651))
    print("  @3 = %2d/%d = %.3f   （基线 0.907，%+.3f）" % (h3, N, h3 / N, h3 / N - 0.907))
    print("=" * 60)

    print("\n还没进的：")
    for q, r, _, kind in rows:
        if r is not None and r <= 3:
            continue
        print("  [%s] %s" % ("-" if r is None else "第%d名" % r, q[:46]))
