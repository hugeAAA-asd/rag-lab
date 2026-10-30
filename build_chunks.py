import glob
import re
DEF_RE = re.compile(r"^\s{0,4}(def |class |async def |func )")

def load_docs(folder):
    docs = []
    for p in glob.glob(folder + "/**/*.md", recursive=True):
        text = open(p, encoding="utf-8").read()
        docs.append((p,text))
    return docs

def join_title(h2, h3):
    """二级标题 + 三级标题拼成一个块标题
    保留 "## " 开头，是为了让「按答案前缀匹配」的评估代码不用改。"""
    return h2 if not h3 else h2 + " > " + h3


def chunk_by_heading(path, text, level=2):
    """level=2：只按 ## 切（现状）
       level=3：## 和 ### 都当边界，块标题带上子节名

    1.1 节有 3 个三级标题，只按 ## 切会让 12 块共用同一个标题，
    精排时标题字面全一样，分数挤成一团，区分度归零。
    """
    chunks = []
    cur_title = "(开头)"
    cur_sub = ""
    cur_text = ""
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code

        is_h2 = line.startswith("## ") and not in_code
        is_h3 = level >= 3 and line.startswith("### ") and not in_code

        if is_h2:
            if cur_text:
                chunks.append((path, join_title(cur_title, cur_sub), cur_text))
            cur_title = line
            cur_sub = ""
            cur_text = ""
        elif is_h3:
            if cur_text:
                chunks.append((path, join_title(cur_title, cur_sub), cur_text))
            cur_sub = line
            cur_text = ""
        else:
            cur_text += line + "\n"
    if cur_text:
        chunks.append((path, join_title(cur_title, cur_sub), cur_text))
    return chunks

def split_paras(text):
    paras = []
    cur = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if line.strip() == "" and not in_code:
            if cur:
                paras.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        paras.append("\n".join(cur))
    return paras


def split_long(path,title,text,max_len=800):
    parts = split_paras(text)
    chunks = []
    cur = ""
    for p in parts:
        p = p.strip()
        if not p :
            continue
        if len(cur) + len(p) > max_len and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = cur + "\n\n" + p if cur else p
    if cur:
        chunks.append(cur)
    return [(path,title,c) for c in chunks]


def split_code_block(text, max_len=800):
    segs = text.split("```")
    if len(segs) < 3:
        return [text]

    body_lines = segs[1].splitlines()
    fence = "```" + body_lines[0].strip()
    code = "\n".join(body_lines[1:])

    pieces, cur = [], []
    for line in code.splitlines():
        if DEF_RE.match(line) and cur:
            pieces.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        pieces.append("\n".join(cur))

    out, buf = [], ""
    for p in pieces:
        if len(buf) + len(p) > max_len and buf:
            out.append(buf)
            buf = p
        else:
            buf = buf + "\n\n" + p if buf else p
    if buf:
        out.append(buf)
    return [fence + "\n" + o +"\n```" for o in out]

def is_pure_code(text):
    t = text.strip()
    return t.startswith("```") and t.endswith("```") and t.count("```") == 2


NOISE_TITLES = ("习题", "参考文献", "讨论与交流", "本章小结", "本章总结", "结语", "总结与展望")


def is_noise(title):
    """标题命中无信息小节：题目列表、链接列表、空泛总结，检索时只会抢位置"""
    return any(k in title for k in NOISE_TITLES)

def build_all(folder="corpus", level=2):
    """读语料 → 按标题切 → 段落合并 → 代码块按函数切，返回 [(path, title, text), ...]"""
    docs = load_docs(folder)
    out = []
    for path, text in docs:
        for c in chunk_by_heading(path, text, level):
            for piece in split_long(c[0], c[1], c[2]):
                if len(piece[2]) > 800 and is_pure_code(piece[2]):
                    for t in split_code_block(piece[2]):
                        out.append((piece[0], piece[1], t))
                else:
                    out.append(piece)
    return [c for c in out if not is_noise(c[1])]


if __name__ == "__main__":
    chunks = build_all()
    print("二次切后块数:", len(chunks))
    print("平均每块字数:", sum(len(c[2]) for c in chunks) // len(chunks))