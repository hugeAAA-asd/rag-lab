"""生成 demo.html：把「章节路由 → 小节直选」的差异摊开给人数

为什么做这个页面，而不是一个聊天框
----------------------------------
简历上的 claim 是「小节直选把 @1 从 .700 提到 .950」。一个聊天框证明不了这件事——
访客问一句，看到一段回答，然后呢？他没法判断这是变好还是变坏。

逐题对照能：同一道题，左边是旧链路选的章节，右边是新链路选的小节，
正确小节加粗，错的标红，一眼看出「哪几题被救回来了」。

数据全部来自仓库里的候选池（pools_*.json / allsub_pools_*.json），
所以页面是**静态的、零依赖的**，直接 open demo.html 就能看；
想要「现场试」那一步，需要跑 `python api.py` 起服务（那才有模型）。

用法
----
    python build_demo.py          # 重新生成 demo.html
"""
import json

SETS = [
    # (标签, 名称, 新链路池子, 旧链路池子, 旧链路结果的补充字段来源)
    ("h3", "h3 干净留出集（20 题，唯一未污染）",
     "allsub_pools_h3.json", "eval_holdout3_pools.json", "eval_holdout3_result.json"),
    ("43", "43 题开发集（40 题有标注）",
     "allsub_pools_43.json", "pools_43_sub2.json", None),
]

import os
import reproduce as R   # 复用它的排序逻辑：两边口径必须一致，否则页面数字和 README 打架


def load(name):
    if not name or not os.path.exists(name):
        return []
    return json.load(open(name, encoding="utf-8"))


def build():
    """两边都用同一套排序复算 top-1，再看它落在不在正确小节里

    为什么不能直接比「模型选了哪个」：
      旧链路选的是**章节**（8.3），新链路选的是**小节**（8.3.4），
      粒度不同，命中率没有可比性。只有都落到「最终 top-1 的块属于哪一节」
      这个共同口径上，0.700 → 0.950 才是同一件事。
    """
    cfg, meta = R.load_config(), R.load_meta()
    cases = {}

    for tag, label, newf, oldf, resf in SETS:
        new_rows = {r["q"]: r for r in load(newf)}
        old_rows = {r["q"]: r for r in load(oldf)}
        extra = {r["q"]: r for r in load(resf)} if resf else {}

        rows = []
        for q, nr in new_rows.items():
            want = sorted(nr.get("subs_want") or [])
            sec = (nr.get("sec") or "").strip()
            if not want or not sec:
                continue

            def top1(row, subs):
                r = {"cand": row.get("cand") or [], "raw": row.get("raw") or [],
                     "picks": row.get("picks") or [], "subs": subs,
                     "confs": row.get("confs") or []}
                if not r["cand"]:
                    return None
                hits = R.rank(r, meta, cfg)
                return R.rank_of(hits, meta, sec, want)

            rk_new = top1(nr, nr.get("subs") or [])
            orow = old_rows.get(q) or extra.get(q) or {}
            rk_old = top1(orow, [])

            rows.append({
                "q": q,
                "want": " / ".join(want),
                "sec": sec.replace("## ", ""),
                "subs": (nr.get("subs") or [])[:3],
                "picks": [p.strip().replace("## ", "") for p in (orow.get("picks") or [])][:3],
                "rk_old": rk_old, "rk_new": rk_new,
                "ok_old": rk_old == 1, "ok_new": rk_new == 1,
            })

        # 排序要让「最想让人看到的那几行」在最上面：先救回，再弄坏，
        # 再都没中，最后都对。按布尔值排会把「都对」沉到底，但也会把
        # 「救回」埋到中间——而救回才是这个页面存在的理由。
        def order(x):
            if x["ok_new"] and not x["ok_old"]:
                return 0
            if x["ok_old"] and not x["ok_new"]:
                return 1
            if not x["ok_new"]:
                return 2
            return 3
        rows.sort(key=order)
        n = len(rows) or 1
        cases[tag] = {
            "label": label, "rows": rows, "n": len(rows),
            "old": sum(r["ok_old"] for r in rows) / n,
            "new": sum(r["ok_new"] for r in rows) / n,
            "fixed": sum(1 for r in rows if r["ok_new"] and not r["ok_old"]),
            "broken": sum(1 for r in rows if r["ok_old"] and not r["ok_new"]),
        }
    return cases


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rag-lab · 章节路由 vs 小节直选</title>
<style>
  :root {
    --bg: #fbfbfa; --card: #ffffff; --line: #e6e5e1;
    --ink: #1f1e1c; --mute: #6b6a65; --hint: #9a9892;
    --ok: #2f7d4f; --bad: #b23b3b; --accent: #2f5d8a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.65 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 34px 20px 60px; }
  h1 { font-size: 21px; font-weight: 600; margin: 0 0 6px; letter-spacing: .2px; }
  .sub { color: var(--mute); font-size: 14px; margin-bottom: 22px; }
  .kpis { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 22px; }
  .kpi {
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 12px 16px; min-width: 132px;
  }
  .kpi b { display: block; font-size: 20px; font-weight: 600; }
  .kpi span { color: var(--hint); font-size: 12px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  .tab {
    border: 1px solid var(--line); background: var(--card); color: var(--mute);
    padding: 7px 14px; border-radius: 999px; cursor: pointer; font-size: 13px;
  }
  .tab.on { background: var(--ink); color: #fff; border-color: var(--ink); }
  table { width: 100%; border-collapse: collapse; background: var(--card);
          border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
  th, td { text-align: left; padding: 11px 13px; border-bottom: 1px solid var(--line);
           vertical-align: top; font-size: 13.5px; }
  th { background: #f6f5f2; color: var(--mute); font-weight: 500; font-size: 12.5px; }
  tr:last-child td { border-bottom: none; }
  .q { max-width: 330px; }
  .mono { font-family: ui-monospace, Consolas, monospace; font-size: 12.5px; }
  .ok { color: var(--ok); }
  .bad { color: var(--bad); }
  .dim { color: var(--hint); }
  .chip { display: inline-block; padding: 1px 7px; border-radius: 5px;
          background: #f1f0ed; font-size: 12px; margin-right: 4px; }
  .chip.hit { background: #e4f0e7; color: var(--ok); }
  .chip.miss { background: #f7e6e6; color: var(--bad); }
  .panel { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
           padding: 18px; margin-top: 18px; }
  input[type=text] {
    width: 100%; padding: 11px 13px; border: 1px solid var(--line); border-radius: 9px;
    font-size: 14px; background: #fff; color: var(--ink);
  }
  button.go {
    margin-top: 10px; padding: 9px 18px; border-radius: 9px; border: 1px solid var(--ink);
    background: var(--ink); color: #fff; font-size: 14px; cursor: pointer;
  }
  button.go:disabled { opacity: .45; cursor: default; }
  .hit { border-top: 1px solid var(--line); padding: 14px 0 4px; }
  .hit h4 { margin: 0 0 6px; font-size: 14px; font-weight: 600; }
  .snip { color: var(--mute); font-size: 13px; }
  .foot { color: var(--hint); font-size: 12.5px; margin-top: 26px; line-height: 1.7; }
  a { color: var(--accent); }
  .empty { color: var(--hint); padding: 16px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>同一道题，旧链路选「章节」，新链路选「小节」</h1>
  <div class="sub" id="sub"></div>

  <div class="kpis" id="kpis"></div>

  <div class="tabs" id="tabs"></div>
  <div class="tabs" id="filters"></div>

  <table>
    <thead><tr>
      <th>问题</th><th>正确小节</th>
      <th>旧：章节路由 top3</th><th>新：小节直选 top3</th><th>结果</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table>

  <div class="panel">
    <div style="font-weight:600;margin-bottom:4px">现场试一句</div>
    <div class="dim" style="margin-bottom:12px;font-size:13px">
      需要先在本地跑 <span class="mono">python api.py</span>（会加载约 500MB 模型）。
      没配 key 时只做检索、不做小节直选，返回的 <span class="mono">mode</span> 会如实标成 plain。
    </div>
    <input type="text" id="q" placeholder="例：敲命令查出来的东西，怎么顺手存进记忆和笔记里">
    <button class="go" id="go">检索</button>
    <div id="out"></div>
  </div>

  <div class="foot">
    数据来自仓库里的候选池（<span class="mono">allsub_pools_*.json</span>），页面为静态生成，
    与 EXPERIMENTS.md ⑧ 的数字同源。<br>
    「小节直选」= 287 个小节一次交给模型选 3 个，章节号反推；
    「章节路由」= 79 个章节选 3 个，再在章内挑小节。
  </div>
</div>

<script>
const DATA = __DATA__;
let cur = "h3", filter = "all";

const pct = x => (x * 100).toFixed(1) + "%";

function renderTabs() {
  const t = document.getElementById("tabs");
  t.innerHTML = Object.entries(DATA).map(([k, v]) =>
    `<div class="tab ${k === cur ? "on" : ""}" data-k="${k}">${v.label}</div>`).join("");
  t.querySelectorAll(".tab").forEach(el => el.onclick = () => { cur = el.dataset.k; render(); });
}

function renderFilters() {
  const f = document.getElementById("filters");
  const opts = [["all", "全部"], ["fixed", "只看被救回的"], ["bad", "只看错的"]];
  f.innerHTML = opts.map(([k, l]) =>
    `<div class="tab ${k === filter ? "on" : ""}" data-f="${k}">${l}</div>`).join("");
  f.querySelectorAll(".tab").forEach(el => el.onclick = () => { filter = el.dataset.f; render(); });
}

function renderKpis() {
  const d = DATA[cur];
  document.getElementById("kpis").innerHTML = `
    <div class="kpi"><b>${pct(d.old)}</b><span>旧：章节路由命中</span></div>
    <div class="kpi"><b>${pct(d.new)}</b><span>新：小节直选命中</span></div>
    <div class="kpi"><b>${d.fixed}</b><span>被救回的题</span></div>
    <div class="kpi"><b>${d.broken}</b><span>被弄坏的题</span></div>
    <div class="kpi"><b>${d.n}</b><span>题量</span></div>`;
}

function render() {
  renderTabs(); renderFilters(); renderKpis();
  const d = DATA[cur];
  document.getElementById("sub").textContent =
    `${d.label}：旧 ${pct(d.old)} → 新 ${pct(d.new)}（小节级 @1）`;

  let rows = d.rows;
  if (filter === "fixed") rows = rows.filter(r => r.ok_new && !r.ok_old);
  if (filter === "bad") rows = rows.filter(r => !r.ok_new);

  document.getElementById("rows").innerHTML = rows.map(r => {
    const oldChips = r.picks.length
      ? r.picks.map(p => `<span class="chip ${p.startsWith(r.sec) ? "hit" : ""}">${p}</span>`).join("")
      : `<span class="dim">—</span>`;
    const newChips = r.subs.length
      ? r.subs.map(s => `<span class="chip ${r.want.split(" / ").includes(s) ? "hit" : ""}">${s}</span>`).join("")
      : `<span class="dim">—</span>`;
    let verdict;
    if (r.ok_new && !r.ok_old) verdict = `<span class="ok">救回</span>`;
    else if (r.ok_old && !r.ok_new) verdict = `<span class="bad">弄坏</span>`;
    else if (r.ok_new) verdict = `<span class="dim">都对</span>`;
    else verdict = `<span class="bad">都没中</span>`;
    return `<tr>
      <td class="q">${r.q}</td>
      <td class="mono">${r.want}</td>
      <td class="mono">${oldChips}</td>
      <td class="mono">${newChips}</td>
      <td>${verdict}</td></tr>`;
  }).join("") || `<tr><td colspan="5" class="empty">这个筛选下没有题</td></tr>`;
}

document.getElementById("go").onclick = async () => {
  const q = document.getElementById("q").value.trim();
  const out = document.getElementById("out");
  if (!q) return;
  const btn = document.getElementById("go");
  btn.disabled = true; out.innerHTML = `<div class="empty">检索中…（首次会加载模型，约 20 秒）</div>`;
  try {
    const r = await fetch("/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q, k: 3 })
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    const mode = d.mode === "subs"
      ? `<span class="ok">小节直选生效</span>：${d.subs.join(" / ")}`
      : `<span class="dim">纯检索（${d.note || "未启用小节直选"}）</span>`;
    out.innerHTML = `<div class="hit"><h4>${mode}</h4>
      <div class="dim" style="font-size:12.5px">耗时 ${d.elapsed_ms} ms</div></div>` +
      d.hits.map(h => `<div class="hit">
        <h4>${h.rank}. ${h.title} <span class="dim mono">精排 ${h.score}</span></h4>
        <div style="margin-bottom:6px">${h.sub.map(s => `<span class="chip">${s}</span>`).join("")}</div>
        <div class="snip">${h.snippet}…</div></div>`).join("");
  } catch (e) {
    // 这个页面会同时挂在 GitHub Pages 上，那里没有后端，/search 必然 404。
    // 所以别把「调不通」写成报错——在线访客看到红字会以为站点坏了。
    // 说清分工：在线版看对照数据，现场检索要本机跑服务。
    out.innerHTML = `<div class="empty">现场检索需要一个后端，这个静态页上没有。<br>
      上面的对照数据是离线算好的，照常可看；想试检索请在本机
      <span class="mono">python api.py</span> 后访问 http://127.0.0.1:8000。<br>
      <span style="opacity:.6">（${e.message}）</span></div>`;
  } finally { btn.disabled = false; }
};

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    cases = build()
    for k, v in cases.items():
        print("%s: %d 题  旧 %.3f → 新 %.3f  救回 %d / 弄坏 %d" % (
            k, v["n"], v["old"], v["new"], v["fixed"], v["broken"]))
    html = HTML.replace("__DATA__", json.dumps(cases, ensure_ascii=False))
    open("demo.html", "w", encoding="utf-8").write(html)
    json.dump(cases, open("demo_cases.json", "w", encoding="utf-8"),
              ensure_ascii=False)

    # 同一份内容再写一份到 docs/：GitHub Pages 从 /docs 发布。
    # 不把 index.html 放根目录——那会和仓库首页抢路径，且根目录该留给源码。
    os.makedirs("docs", exist_ok=True)
    open(os.path.join("docs", "index.html"), "w", encoding="utf-8").write(html)
    print("写入 demo.html 和 docs/index.html（各 %.0f KB），以及 demo_cases.json"
          % (len(html) / 1024))
