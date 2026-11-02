"""把检索链路包成 HTTP 服务：给别人一个能点开的入口

为什么是「只检索、不生成」
--------------------------
简历上那句 claim 是「小节直选把 Recall@1 从 .700 提到 .950」——它讲的是**找材料**，
不是写答案。所以 demo 只暴露检索这一段：

  · 不需要把 DeepSeek key 放进公开实例（生成才要 key，检索不要）
  · 不存在「模型编了一段话」这种和 claim 无关的变量
  · 响应快（2~3 秒），不会因为冷启动 + 生成让人等 60 秒

唯一的模型调用是「小节直选」（要 LLM 判目录）。没配 key 时它会降级成纯检索，
接口里用 mode 字段如实标出来，不假装。

为什么用标准库的 http.server
----------------------------
依赖越少，别人 clone 下来越可能跑得起来。这个仓库已经有 numpy +
sentence-transformers 两个重依赖，再为一个 demo 装 FastAPI + uvicorn 不值得。
代价是没有 SSE 和自动文档——但「只检索」根本不需要流式。

端点
----
    GET  /healthz          存活 + 配置 + 语料规模
    POST /search           {"q": "...", "k": 3}  → 检索结果
    GET  /                 案例对照页（demo.html）
    GET  /api/cases        离线案例数据（不加载模型，直接读 json）

启动
----
    python api.py                 # 默认 127.0.0.1:8000
    python api.py --port 8000 --host 0.0.0.0
    python api.py --no-route      # 强制只用检索，不调 LLM（不需要 key）

第一次 /search 会加载两个模型（约 500MB、20 秒），之后常驻内存。
"""
import argparse
import json
import os
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
RATE_LIMIT = 30          # 每分钟每 IP 的请求数
RATE_WINDOW = 60

_lock = threading.Lock()
_state = {"ready": False, "load_s": 0.0, "n": 0}
_hits = defaultdict(deque)


def load_stack():
    """延迟加载：只有真的来请求了才付这 500MB / 20 秒的代价"""
    global _state
    with _lock:
        if _state["ready"]:
            return
        t = time.perf_counter()
        import eval as E
        from eval import chunks, rank_of  # noqa: F401  (rank_of 供调用方使用)
        import rewrite
        import route
        from answer import ROUTE_BONUS, SUB_BONUS, ROUTE_PICK, CONF_GATE, VERSION

        globals().update(dict(E=E, chunks=chunks, rewrite=rewrite, route=route,
                              ROUTE_BONUS=ROUTE_BONUS, SUB_BONUS=SUB_BONUS,
                              ROUTE_PICK=ROUTE_PICK, CONF_GATE=CONF_GATE,
                              VERSION=VERSION))
        _state.update(ready=True, load_s=time.perf_counter() - t,
                      n=len(chunks), version=VERSION)


def do_search(q, k=3, use_route=True):
    load_stack()
    t0 = time.perf_counter()

    subs, picks, mode = [], [], "plain"
    # 从 .env 读，而不是 os.getenv：这个仓库的 key 一直放在 .env 里，
    # 由 llm.load_key() 负责读，环境变量里通常是空的。
    from llm import load_key
    key = load_key() or None
    if use_route and key:
        try:
            subs, _raw = route.route_allsubs(q, key)
            picks = route.secs_of_subs(subs)
            mode = "subs" if subs else "plain"
        except Exception as e:
            print("   ! 小节直选失败，降级为纯检索：%s" % e)

    # 不要 return_raw：那个口径返回 (cand, raw, picks) 三元组，是给参数扫描用的。
    # 默认口径返回 [(块id, 原始精排分)]，分数是精排的原始判断（不含加分）——
    # 这样 demo 上显示的分和「该不该拒答」的阈值是同一套含义。
    hits = rewrite.search_multi(
        q, None, k=k, picks=picks or None, bonus=0.0,
        route_pick=ROUTE_PICK,
        subs=subs or None, sub_bonus=SUB_BONUS if subs else 0.0)

    out = []
    for n, (idx, score) in enumerate(hits, 1):
        title = chunks[idx][1].strip()
        body = chunks[idx][2]
        out.append({
            "rank": n,
            "title": title,
            "sub": sorted(E.SUB_MAP.get(idx, set())),
            "score": round(float(score), 4),
            "chars": len(body),
            "snippet": " ".join(body.split())[:260],
        })

    return {
        "q": q, "mode": mode, "subs": subs, "picks": [p.strip() for p in picks],
        "hits": out, "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "note": ("未配置 key：跳过小节直选，只用原问题检索"
                 if not key else ("小节都没选中，退回纯检索" if mode == "plain" else "")),
    }


def rate_ok(ip):
    now = time.time()
    dq = _hits[ip]
    while dq and now - dq[0] > RATE_WINDOW:
        dq.popleft()
    if len(dq) >= RATE_LIMIT:
        return False
    dq.append(now)
    return True


def _has_key():
    try:
        from llm import load_key
        return bool(load_key())
    except Exception:
        return False


def load_cases():
    """离线案例：不加载模型，直接读候选池，用于页面首屏和 CI"""
    path = os.path.join(HERE, "demo_cases.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return {"cases": [], "note": "还没有 demo_cases.json，先跑 build_demo.py"}


class Handler(BaseHTTPRequestHandler):
    server_version = "raglab/1.0"

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(
            body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/healthz":
            load_stack()
            return self._send(200, {
                "ok": True, "version": _state.get("version", "?"),
                "chunks": _state["n"], "model_load_s": round(_state["load_s"], 1),
                "route": bool(_has_key()),
            })
        if p == "/api/cases":
            return self._send(200, load_cases())
        if p in ("/", "/index.html"):
            f = os.path.join(HERE, "demo.html")
            if not os.path.exists(f):
                return self._send(404, {"error": "还没有 demo.html，先跑 build_demo.py"})
            return self._send(200, open(f, "rb").read(), "text/html; charset=utf-8")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/search":
            return self._send(404, {"error": "not found"})
        ip = self.client_address[0]
        if not rate_ok(ip):
            return self._send(429, {"error": "太频繁了，一分钟 %d 次上限" % RATE_LIMIT})

        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            q = str(body.get("q", "")).strip()
            k = max(1, min(10, int(body.get("k", 3))))
        except Exception as e:
            return self._send(400, {"error": "请求体解析失败：%s" % e})
        if not q:
            return self._send(400, {"error": "q 不能为空"})
        if len(q) > 200:
            return self._send(400, {"error": "问题太长了（上限 200 字）"})

        use_route = not getattr(self.server, "no_route", False)
        try:
            return self._send(200, do_search(q, k, use_route))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})

    def log_message(self, fmt, *args):
        print("  %s %s" % (self.address_string(), fmt % args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-route", action="store_true", help="不调 LLM，只用检索")
    ap.add_argument("--warm", action="store_true", help="启动时就加载模型")
    args = ap.parse_args()

    if args.warm:
        print("预加载模型…")
        load_stack()
        print("加载完成 %.1fs，%d 块" % (_state["load_s"], _state["n"]))

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.no_route = args.no_route
    print("http://%s:%d  (route=%s)" % (args.host, args.port,
                                        "off" if args.no_route else "on"))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
