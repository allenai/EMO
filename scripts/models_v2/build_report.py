#!/usr/bin/env python3
"""Build the models_v2 experiment report (styled to match models_fullextend).

Pulls six training/eval curves from W&B for the models_v2 stdMoE runs and renders
a self-contained HTML report (dark header, tab nav, uPlot charts with log-y toggle
and drag-zoom) to claude_outputs/models_v2/report.html.

Run:  python scripts/models_v2/build_report.py            # pull from W&B + render
      python scripts/models_v2/build_report.py --no-wandb # render from cached curves.json

Registered in scripts/publish_reports.sh (deploys to https://emo-reports.pages.dev/).
"""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path

VENDOR = Path(__file__).parent / "vendor"
BASE = Path(__file__).resolve().parents[2] / "claude_outputs" / "models_v2"
CACHE = BASE / "curves.json"

ENTITY_PROJECT = "ryanyxw/emo-extension"
TOK_PER_STEP = 1024 * 4096  # global_batch_size(1024) * seq_len(4096) = 4,194,304 tokens/step

# (label, W&B run id) — pinned to the healthy/current runs (crashed restarts excluded).
RUNS = [
    ("stdmoe_64exp_25b",  "lsq79eb5"),
    ("stdmoe_64exp_50b",  "r5kyiexy"),
    ("stdmoe_128exp_50b", "yuafg0dw"),
]

# (chart title, W&B metric key)
METRICS = [
    ("CE loss",                      "train/CE loss"),
    ("Grad norm",                    "optim/total grad norm"),
    ("Load balancing loss",          "train/load balancing loss"),
    ("Unique experts used / batch",  "train/unique experts used per batch"),
    ("HellaSwag (soft loss v2)",     "eval/downstream/hellaswag (soft loss v2)"),
    ("ARC-Challenge (soft loss v2)", "eval/downstream/arc_challenge (soft loss v2)"),
]


def slug(t: str) -> str:
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", t.lower()))


# ---- styling lifted verbatim from scripts/models_fullextend/build_report.py ----
CSS = """
:root { --fg:#1e293b; --muted:#64748b; --bg:#f8fafc; --card:#ffffff; --line:#e2e8f0; }
* { box-sizing:border-box; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
       color:var(--fg); background:var(--bg); line-height:1.55; }
header { background:#0f172a; color:#f1f5f9; padding:18px 28px; }
header h1 { margin:0 0 2px; font-size:20px; }
header p { margin:0; color:#94a3b8; font-size:13px; }
.home-link { display:inline-block; margin-bottom:8px; color:#94a3b8; font-size:13px; text-decoration:none; }
.home-link:hover { color:#f1f5f9; }
.topbar { position:sticky; top:0; z-index:10; }
nav { display:flex; gap:6px; flex-wrap:wrap; padding:10px 28px; background:#1e293b; }
nav button { border:0; border-radius:6px; padding:7px 14px; font-size:14px; cursor:pointer;
             background:transparent; color:#cbd5e1; }
nav button:hover { background:#334155; }
nav button.active { background:#3b82f6; color:#fff; }
#subnav { display:flex; gap:6px 16px; flex-wrap:wrap; padding:8px 28px; background:#eef2f7;
          border-bottom:1px solid var(--line); font-size:13px; }
#subnav a { color:#475569; text-decoration:none; white-space:nowrap; }
#subnav a:hover { color:#2563eb; text-decoration:underline; }
#subnav:empty { display:none; }
main { max-width:1180px; margin:0 auto; padding:24px 28px 80px; }
section.tab { display:none; }
section.tab.active { display:block; }
.card { background:var(--card); border:1px solid var(--line); border-left:4px solid var(--line);
        border-radius:8px; padding:16px 20px; margin:16px 0; }
.card h3 { margin:0 0 8px; font-size:15px; text-transform:uppercase; letter-spacing:0.05em;
           scroll-margin-top:96px; }
.card.goal { border-left-color:#2563eb; } .card.goal h3 { color:#2563eb; }
.card.method { border-left-color:#7c3aed; } .card.method h3 { color:#7c3aed; }
.card.results { border-left-color:#059669; } .card.results h3 { color:#059669; }
table { border-collapse:collapse; margin:12px 0; font-size:14px; width:auto; }
th, td { border:1px solid var(--line); padding:6px 12px; text-align:left; vertical-align:top; }
th { background:#f1f5f9; }
tbody tr:nth-child(even) { background:#f8fafc; }
details { margin:12px 0; }
summary { cursor:pointer; color:#2563eb; font-size:14px; }
.note { font-size:13px; color:var(--muted); }
pre { background:#0f172a; color:#e2e8f0; padding:12px 14px; border-radius:6px; overflow:auto; font-size:13px; }
pre code { background:transparent; padding:0; color:inherit; }
code { background:#eef2f7; padding:1px 5px; border-radius:4px; font-size:0.9em; }
h4 { margin:18px 0 4px; }
.ce-chart { width:100%; margin-bottom:6px; }
.chart-controls { display:flex; align-items:center; gap:12px; margin:8px 0 4px; }
.chart-controls button { border:1px solid var(--line); background:#fff; border-radius:6px;
                         padding:5px 10px; font-size:13px; cursor:pointer; }
.chart-controls button:hover { background:#f1f5f9; }
.u-legend { font-size:12.5px; }
"""

# uPlot multi-chart init (from models_fullextend; spanGaps:true so per-run sampling overlays cleanly).
_CHARTS_JS = r"""
<script>
(function(){
  const C = __CURVES__;
  const palette = ["#64748b","#2563eb","#7c3aed","#059669","#dc2626","#d97706"];
  const colorOf = {};
  C.runs.forEach((r,i) => { colorOf[r.label] = palette[i % palette.length]; });
  const reg = {};
  function build(key){
    const st = reg[key], chart = st.chart, el = st.el;
    const data = [chart.x].concat(chart.series.map(s => s.y));
    const series = [{ value: (u,v) => v==null ? "--" : v }].concat(
      chart.series.map(s => ({
        label: s.label, stroke: colorOf[s.label] || "#888", width: 1.6,
        spanGaps: true, value: (u,v) => v==null ? "--" : (+v).toFixed(4),
      })));
    const opts = {
      width: (el.clientWidth || 900), height: 360,
      focus: { alpha: 0.25 },
      scales: { x: { time:false }, y: { distr: st.logY ? 3 : 1 } },
      cursor: { focus: { prox: 30 }, drag: { x:true, y:true, uni:10 } },
      axes: [
        { label: "step", values: (u,vals) => vals.map(v => v>=1000 ? (v/1000)+"k" : v) },
        { label: chart.title },
      ],
      series: series,
    };
    if (st.plot) st.plot.destroy();
    st.plot = new uPlot(opts, data, el);
  }
  C.charts.forEach(chart => {
    reg[chart.key] = { plot:null, logY:false, chart:chart, el:document.getElementById("chart-"+chart.key) };
    if (reg[chart.key].el) build(chart.key);
  });
  document.querySelectorAll("button.logtoggle").forEach(b => {
    b.addEventListener("click", () => { const k=b.dataset.chart; reg[k].logY=!reg[k].logY; build(k); });
  });
  window.__ceReg = Object.assign(window.__ceReg || {}, reg);
  if (!window.__ceResizeBound) {
    window.__ceResizeBound = true;
    window.ceResize = function(){
      Object.keys(window.__ceReg).forEach(k => { const st=window.__ceReg[k]; if (st.plot && st.el) st.plot.setSize({ width: st.el.clientWidth||900, height: 360 }); });
    };
    window.addEventListener("resize", window.ceResize);
  }
})();
</script>
"""

# tab nav (verbatim from models_fullextend)
JS = """
function slug(t){ return t.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/(^-|-$)/g,''); }
function buildSubnav(id) {
  const sec = document.getElementById(id);
  const sub = document.getElementById('subnav');
  sub.innerHTML = '';
  if (!sec) return;
  sec.querySelectorAll('.card > h3').forEach(h => {
    if (!h.id) h.id = id + '--' + slug(h.textContent);
    const a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    a.addEventListener('click', e => { e.preventDefault(); h.scrollIntoView({behavior:'smooth', block:'start'}); });
    sub.appendChild(a);
  });
}
function show(id) {
  document.querySelectorAll('section.tab').forEach(s => s.classList.toggle('active', s.id === id));
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('active', b.dataset.target === id));
  history.replaceState(null, '', '#' + id);
  buildSubnav(id);
  if (window.ceResize) window.ceResize();
}
document.querySelectorAll('nav button').forEach(b => b.addEventListener('click', () => show(b.dataset.target)));
show(location.hash && document.getElementById(location.hash.slice(1)) ? location.hash.slice(1) : 'overview');
"""


def fetch_from_wandb() -> dict:
    import wandb
    api = wandb.Api()
    runs_meta, per = [], {}
    for label, rid in RUNS:
        run = api.run(f"{ENTITY_PROJECT}/{rid}")
        last_ce, last_step = None, None
        for title, key in METRICS:
            d = per.setdefault(key, {}).setdefault(label, {})
            for row in run.history(keys=[key], samples=600, pandas=False):
                v, s = row.get(key), row.get("_step")
                if v is None or s is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fv) or math.isinf(fv):
                    continue
                d[int(s)] = fv
        ce = per["train/CE loss"][label]
        if ce:
            last_step = max(ce)
            last_ce = round(ce[last_step], 3)
        runs_meta.append({"label": label, "url": run.url, "state": run.state, "id": rid,
                          "last_ce": last_ce,
                          "last_tok": round(last_step * TOK_PER_STEP / 1e9, 1) if last_step else None})
    labels = [l for l, _ in RUNS]
    charts = []
    for title, key in METRICS:
        steps = sorted(set().union(*[set(per[key][l]) for l in labels]) or {0})
        series = [{"label": l, "y": [per[key][l].get(s) for s in steps]} for l in labels]
        charts.append({"key": slug(title), "title": title, "x": steps, "series": series})
    return {"runs": runs_meta, "charts": charts}


def chart_blocks(charts: list) -> str:
    out = []
    for c in charts:
        out.append(
            f'<h4>{c["title"]}</h4>'
            '<div class="chart-controls">'
            f'<button class="logtoggle" data-chart="{c["key"]}">toggle log-y</button>'
            '<span class="note">drag to zoom (x &amp; y) &middot; double-click reset '
            '&middot; hover to highlight a run &middot; click legend to toggle</span>'
            '</div>'
            f'<div class="ce-chart" id="chart-{c["key"]}"></div>'
        )
    return "".join(out)


def render(payload: dict, uplot_css: str, uplot_js: str) -> str:
    runs = payload["runs"]
    links = " &middot; ".join(
        f'<a href="{r["url"]}" target="_blank" rel="noopener">{r["label"]}</a>' for r in runs)
    body_rows = "".join(
        f"<tr><td>{r['label']}</td><td>{r['state']}</td>"
        f"<td>{r['last_tok']}B</td><td>{r['last_ce']}</td></tr>" for r in runs)
    run_table = ("<table><thead><tr><th>run</th><th>state</th><th>tokens</th>"
                 f"<th>latest CE</th></tr></thead><tbody>{body_rows}</tbody></table>")

    overview = (
        f'<div class="card goal"><h3>Goal</h3>'
        '<p>A standard top-k MoE (<code>moe_lbreducedp_sharedexp</code>) token-budget / '
        'expert-count sweep. Unlike the released baselines (130B LR schedule hard-stopped at '
        '50B), every run here decays the LR cosine <strong>directly over its true token '
        'budget</strong> &mdash; "what if we only had this many tokens." Three runs vary the '
        'expert pool (64 vs 128) and budget (25B vs 50B).</p></div>'
        f'<div class="card results"><h3>Runs</h3>{run_table}'
        f'<p class="note"><strong>W&amp;B runs:</strong> {links}</p></div>'
        '<div class="card method"><h3>Setup</h3>'
        '<p>Common to all: OLMoE-mix-0824, <code>d_model</code> 2048 / 16 layers, top-8 routed '
        '+ 1 shared expert, lr 4e-3, lb 1e-1, 8 nodes / 64 GPUs, global batch 4.19M tokens/step. '
        'Permanent checkpoints saved at 25/50/75% of each run.</p></div>'
    )
    curves = (
        '<div class="card results"><h3>Training &amp; eval curves</h3>'
        '<p class="note">x-axis = optimizer step (k). Pulled live from W&amp;B; grad norm reads '
        'best on a log y-axis (toggle per chart).</p>'
        f'{chart_blocks(payload["charts"])}</div>'
    )

    tabs = [("overview", "Overview", overview), ("curves", "Curves", curves)]
    nav = "".join(f'<button data-target="{t}">{n}</button>' for t, n, _ in tabs)
    sections = "".join(f'<section class="tab" id="{t}">{b}</section>' for t, _, b in tabs)
    charts_js = _CHARTS_JS.replace("__CURVES__", json.dumps(payload))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EMO models_v2: stdMoE token-budget / expert-count sweep</title>
<style>{CSS}</style>
<style>{uplot_css}</style>
<script>{uplot_js}</script>
</head>
<body>
<header>
<a class="home-link" href="/">&larr; all reports</a>
<h1>EMO models_v2: stdMoE token-budget / expert-count sweep</h1>
<p>models_v2 &middot; LR decayed over the true token budget &middot; live curves from W&amp;B
&middot; generated by scripts/models_v2/build_report.py</p>
</header>
<div class="topbar"><nav>{nav}</nav><div id="subnav"></div></div>
<main>{sections}</main>
{charts_js}
<script>{JS}</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-wandb", action="store_true", help="render from cached curves.json")
    ap.add_argument("--output", type=Path, default=BASE / "report.html")
    args = ap.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if args.no_wandb:
        payload = json.loads(CACHE.read_text())
    else:
        payload = fetch_from_wandb()
        CACHE.write_text(json.dumps(payload))
    uplot_css = (VENDOR / "uPlot.min.css").read_text() if (VENDOR / "uPlot.min.css").is_file() else ""
    uplot_js = (VENDOR / "uPlot.iife.min.js").read_text() if (VENDOR / "uPlot.iife.min.js").is_file() else ""
    args.output.write_text(render(payload, uplot_css, uplot_js))
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
