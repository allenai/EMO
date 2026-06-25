#!/usr/bin/env python3
"""Build the models_v2 experiment report (styled to match models_fullextend).

Pulls six training/eval curves from W&B for the models_v2 stdMoE runs and renders
a self-contained HTML report (dark header, tab nav, compact 2-up uPlot charts with
per-chart log-y toggle, drag-zoom, and a click-to-expand modal) to
claude_outputs/models_v2/report.html.

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

# (label, W&B run id-or-ids) — pinned to the healthy/current runs. A label may map to a LIST of
# run ids when the run crashed and resumed: histories are merged in order, later (resumed) runs
# winning on overlapping steps. stdmoe_64exp_50b_wsd is the WSD-scheduler twin of
# stdmoe_64exp_50b (same arch/data/budget, warmup-stable-decay instead of cosine) — kept adjacent
# so the LR/loss curves compare directly. Its run crashed at step ~2759 (n6zg596k) and resumed
# from step 2501 (96odpdqg), so both are merged to recover the full 1-11913 history.
# stdmoe_64exp_50b_wsd_anneal_s8941_12p5b is a WSD DECAY BRANCH: forked from the wsd trunk's 37.5B
# stable checkpoint (step8941) and decayed LR 4e-3->0 over 12.5B tokens to 50B (steps 8942-11921),
# so its curve diverges from the trunk's stable line at step 8941 — kept adjacent to the trunk.
RUNS = [
    ("stdmoe_64exp_25b",                        "lsq79eb5"),
    ("stdmoe_64exp_50b",                        "r5kyiexy"),
    ("stdmoe_64exp_50b_wsd",                    ["n6zg596k", "96odpdqg"]),
    ("stdmoe_64exp_50b_wsd_decay@37.5B_12.5B",  "hbq6004e"),
    ("stdmoe_128exp_50b",                       "yuafg0dw"),
    # lr2e-3 stable trunk (flat peak LR 2e-3, no baked-in decay) + its two decay branches, which
    # diverge from the trunk's flat LR line at their branch steps (45B/step10729, 40B/step9537).
    ("stdmoe_64exp_50b_wsd_lr2e-3",             "0ucu7x8n"),
    ("stdmoe_64exp_50b_wsd_lr2e-3_decay@45B_5B",  "69drqnz8"),
    ("stdmoe_64exp_50b_wsd_lr2e-3_decay@40B_10B", "opp7l86a"),
]

# (chart title, W&B metric key)
METRICS = [
    ("CE loss",                      "train/CE loss"),
    ("Grad norm",                    "optim/total grad norm"),
    ("Learning rate",                "optim/LR (group 0)"),
    ("Load balancing loss",          "train/load balancing loss"),
    ("Unique experts used / batch",  "train/unique experts used per batch"),
    ("HellaSwag (soft loss v2)",     "eval/downstream/hellaswag (soft loss v2)"),
    ("ARC-Challenge (soft loss v2)", "eval/downstream/arc_challenge (soft loss v2)"),
]


def slug(t: str) -> str:
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", t.lower()))


# ---- merged-test-split evals (scripts/models_v2/launch_merged_eval.sh outputs) ----
# Outputs live at models_v2/merged_evals/<run>/<group>/task-<task>-metrics.json. Each model is
# scored on the selective project's `*_merged` test splits (no selection/finetuning). We read
# every task's primary_metric and aggregate into MC9 / MMLU / Gen5 / GSM8K, matching the
# selective project's headline groupings.
EVAL_ROOT = Path(__file__).resolve().parents[2] / "models_v2" / "merged_evals"

# (run dir, display label) in report order. EMO baseline first, then the stdMoE sweep.
EVAL_MODELS = [
    ("emo_1b14b_50bof130b",    "EMO 128e · 50Bof130B"),
    ("stdmoe_1b14b_50bof130b", "stdMoE 128e · 50Bof130B"),
    ("stdmoe_128exp_50b",      "stdMoE 128e · 50B"),
    ("stdmoe_64exp_50b",       "stdMoE 64e · 50B"),
    # WSD family at 50B: the trunk's own final (5B end-of-run decay) and the 12.5B decay branch
    # forked at 37.5B. Branch run dir is hierarchical (under the trunk's anneals/).
    ("stdmoe_64exp_50b_wsd",                       "stdMoE 64e · 50B wsd (5B decay)"),
    ("stdmoe_64exp_50b_wsd/anneals/s8941_12p5b",   "stdMoE 64e · wsd decay@37.5B/12.5B"),
    # lr2e-3 stable trunk decay branches (peak LR 2e-3): 5B decay from 45B, 10B decay from 40B.
    ("stdmoe_64exp_50b_wsd_lr2e-3/anneals/s10729_5b", "stdMoE 64e · lr2e-3 decay@45B/5B"),
    ("stdmoe_64exp_50b_wsd_lr2e-3/anneals/s9537_10b", "stdMoE 64e · lr2e-3 decay@40B/10B"),
    ("stdmoe_64exp_25b",       "stdMoE 64e · 25B"),
]

# MC9 = 9 rank-classification tasks (incl. hellaswag); Gen5 = 5 generative QA tasks. Base task
# names (the `*_merged` prefix before the `:rc_test`/`:test` split suffix).
MC9_TASKS = ["arc_easy_merged", "arc_challenge_merged", "boolq_merged", "csqa_merged",
             "hellaswag_merged", "openbookqa_merged", "piqa_merged", "socialiqa_merged",
             "winogrande_merged"]
GEN5_TASKS = ["squad_merged", "coqa_merged", "naturalqs_merged", "triviaqa_merged", "drop_merged"]
GSM8K_TASK = "gsm8k_generation_8shot_merged"
# Short column headers for the per-task detail table.
TASK_SHORT = {"arc_easy_merged": "ARC-e", "arc_challenge_merged": "ARC-c", "boolq_merged": "BoolQ",
              "csqa_merged": "CSQA", "hellaswag_merged": "HSwag", "openbookqa_merged": "OBQA",
              "piqa_merged": "PIQA", "socialiqa_merged": "SIQA", "winogrande_merged": "WinoG",
              "squad_merged": "SQuAD", "coqa_merged": "CoQA", "naturalqs_merged": "NQ",
              "triviaqa_merged": "TriviaQA", "drop_merged": "DROP"}


def _load_eval_model(run: str) -> dict:
    """Return {base_task_name: {"metric","val","n"}} for one run, or {} if none scored yet."""
    res: dict = {}
    # Metrics live exactly one level below the run dir (<run>/<group>/task-*.json). Use a single
    # `*` (not `**`) so a trunk run (e.g. stdmoe_64exp_50b_wsd) does NOT recurse into a decay
    # branch nested beneath it (e.g. stdmoe_64exp_50b_wsd/anneals/s8941_12p5b/...).
    for f in sorted(EVAL_ROOT.glob(f"{run}/*/task-*-metrics.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        base = str(d.get("task_name", "")).split(":")[0]
        if not base:
            continue
        pm = d.get("primary_metric") or d.get("task_config", {}).get("primary_metric")
        val = (d.get("metrics") or {}).get(pm)
        res[base] = {"metric": pm, "val": val, "n": d.get("num_instances")}
    return res


def _avg(vals: list):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def load_evals() -> dict:
    """Aggregate per-model merged-eval scores into MC9 / MMLU / Gen5 / GSM8K + per-task detail."""
    models = []
    for run, label in EVAL_MODELS:
        r = _load_eval_model(run)
        mc9 = _avg([r[t]["val"] for t in MC9_TASKS if t in r])
        gen5 = _avg([r[t]["val"] for t in GEN5_TASKS if t in r])
        mmlu_vals = [v["val"] for k, v in r.items() if k.startswith("mmlu_merged_")]
        mmlu = _avg(mmlu_vals)
        gsm = r.get(GSM8K_TASK, {}).get("val")
        # Only treat a model as fully scored when every expected task landed (32 = 9 MC9 + 5
        # Gen5 + GSM8K + 17 MMLU). Avoids showing misleading partial averages mid-sweep.
        complete = (all(t in r for t in MC9_TASKS) and all(t in r for t in GEN5_TASKS)
                    and GSM8K_TASK in r and len(mmlu_vals) == 17)
        models.append({
            "label": label, "run": run, "present": complete,
            "summary": {"mc9": mc9, "mmlu": mmlu, "gen5": gen5, "gsm8k": gsm,
                        "n_mmlu": len(mmlu_vals)},
            "tasks": {t: r.get(t, {}).get("val") for t in MC9_TASKS + GEN5_TASKS},
        })
    return {"models": models}


# ---- styling: models_fullextend chrome + compact 2-up grid + expand modal ----
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
code { background:#eef2f7; padding:1px 5px; border-radius:4px; font-size:0.9em; }
.ce-chart { width:100%; }
/* compact 2-up grid */
.chart-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px 20px; margin-top:6px; }
@media (max-width:760px){ .chart-grid { grid-template-columns:1fr; } }
.chart-cell { min-width:0; border:1px solid var(--line); border-radius:8px; padding:8px 10px 4px; }
.chart-cell h4 { margin:0 0 2px; font-size:13px; }
.chart-controls { display:flex; align-items:center; gap:8px; margin:2px 0 4px; }
.chart-controls button { border:1px solid var(--line); background:#fff; border-radius:6px;
                         padding:3px 9px; font-size:12px; cursor:pointer; }
.chart-controls button:hover { background:#f1f5f9; }
.u-legend { font-size:12px; }
/* expand modal */
.ce-modal { position:fixed; inset:0; background:rgba(15,23,42,.55); display:none; z-index:50;
            align-items:center; justify-content:center; padding:24px; }
.ce-modal.open { display:flex; }
.ce-modal-inner { background:#fff; border-radius:10px; padding:14px 16px; box-shadow:0 10px 40px rgba(0,0,0,.3); }
.ce-modal-bar { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; gap:16px; }
.ce-modal-bar strong { font-size:15px; }
.ce-modal-bar button { border:1px solid var(--line); background:#fff; border-radius:6px; padding:5px 10px;
                       font-size:13px; cursor:pointer; }
.ce-modal-bar button:hover { background:#f1f5f9; }
"""

# uPlot init: compact grid charts + click-to-expand modal. Plain string (literal braces);
# data injected via __CURVES__ replace.
_CHARTS_JS = r"""
<script>
(function(){
  const C = __CURVES__;
  const palette = ["#64748b","#2563eb","#7c3aed","#059669","#dc2626","#d97706"];
  const colorOf = {};
  C.runs.forEach((r,i) => { colorOf[r.label] = palette[i % palette.length]; });
  const SMALL_H = 230, reg = {};
  function opts(chart, logY, w, h){
    return {
      width:w, height:h, focus:{ alpha:0.25 },
      scales:{ x:{ time:false }, y:{ distr: logY ? 3 : 1 } },
      cursor:{ focus:{ prox:30 }, drag:{ x:true, y:true, uni:10 } },
      axes:[ { label:"step", values:(u,vals)=>vals.map(v=>v>=1000?(v/1000)+"k":v) }, { label:chart.title } ],
      series:[ { value:(u,v)=>v==null?"--":v } ].concat(chart.series.map(s=>({
        label:s.label, stroke:colorOf[s.label]||"#888", width:1.6, spanGaps:true,
        value:(u,v)=>v==null?"--":(+v).toFixed(4) }))),
    };
  }
  const dataOf = chart => [chart.x].concat(chart.series.map(s=>s.y));
  function build(key){
    const st=reg[key], el=st.el; if(!el) return;
    if(st.plot) st.plot.destroy();
    st.plot = new uPlot(opts(st.chart, st.logY, el.clientWidth||430, SMALL_H), dataOf(st.chart), el);
  }
  C.charts.forEach(chart => {
    reg[chart.key] = { plot:null, logY:false, chart:chart, el:document.getElementById("chart-"+chart.key) };
    build(chart.key);
  });
  document.querySelectorAll("button.logtoggle").forEach(b =>
    b.addEventListener("click", () => { const k=b.dataset.chart; reg[k].logY=!reg[k].logY; build(k); }));

  // ---- expand modal ----
  const modal=document.createElement("div"); modal.className="ce-modal";
  modal.innerHTML='<div class="ce-modal-inner"><div class="ce-modal-bar"><strong></strong>'
    +'<span><button class="ce-mlog">toggle log-y</button> <button class="ce-mclose">close ✕</button></span>'
    +'</div><div class="ce-modal-chart"></div></div>';
  document.body.appendChild(modal);
  const mEl=modal.querySelector(".ce-modal-chart"), mTitle=modal.querySelector("strong");
  let m=null;  // { key, logY, plot }
  const mSize=()=>({ w:Math.min(window.innerWidth*0.94,1180)|0, h:Math.min(window.innerHeight*0.72,700)|0 });
  function drawM(){ const c=reg[m.key].chart, s=mSize(); if(m.plot) m.plot.destroy();
    m.plot=new uPlot(opts(c, m.logY, s.w, s.h), dataOf(c), mEl); }
  function openM(key){ m={ key:key, logY:reg[key].logY, plot:null }; mTitle.textContent=reg[key].chart.title;
    modal.classList.add("open"); drawM(); }
  function closeM(){ if(m&&m.plot) m.plot.destroy(); m=null; modal.classList.remove("open"); }
  document.querySelectorAll("button.expand").forEach(b => b.addEventListener("click", ()=>openM(b.dataset.chart)));
  modal.querySelector(".ce-mclose").addEventListener("click", closeM);
  modal.querySelector(".ce-mlog").addEventListener("click", ()=>{ if(m){ m.logY=!m.logY; drawM(); } });
  modal.addEventListener("click", e=>{ if(e.target===modal) closeM(); });
  document.addEventListener("keydown", e=>{ if(e.key==="Escape") closeM(); });

  window.__ceReg = Object.assign(window.__ceReg || {}, reg);
  if(!window.__ceResizeBound){
    window.__ceResizeBound=true;
    window.ceResize=function(){
      Object.keys(window.__ceReg).forEach(k=>{ const st=window.__ceReg[k];
        if(st.plot && st.el && st.el.clientWidth) st.plot.setSize({ width:st.el.clientWidth, height:SMALL_H }); });
      if(m && m.plot){ const s=mSize(); m.plot.setSize({ width:s.w, height:s.h }); }
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
        # A label may be backed by several W&B runs (a crashed run + its resume). Fetch each in
        # order and merge into the same per-step dict so later runs overwrite the overlap.
        rids = [rid] if isinstance(rid, str) else list(rid)
        last_run = None
        for one in rids:
            run = api.run(f"{ENTITY_PROJECT}/{one}")
            last_run = run
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
        last_step = max(ce) if ce else None
        runs_meta.append({"label": label, "url": last_run.url, "state": last_run.state,
                          "id": "+".join(rids),
                          "last_ce": round(ce[last_step], 3) if ce else None,
                          "last_tok": round(last_step * TOK_PER_STEP / 1e9, 1) if last_step else None})
    labels = [l for l, _ in RUNS]
    charts = []
    for title, key in METRICS:
        steps = sorted(set().union(*[set(per[key][l]) for l in labels]) or {0})
        series = [{"label": l, "y": [per[key][l].get(s) for s in steps]} for l in labels]
        charts.append({"key": slug(title), "title": title, "x": steps, "series": series})
    return {"runs": runs_meta, "charts": charts}


def chart_blocks(charts: list) -> str:
    cells = []
    for c in charts:
        cells.append(
            '<div class="chart-cell">'
            f'<h4>{c["title"]}</h4>'
            '<div class="chart-controls">'
            f'<button class="logtoggle" data-chart="{c["key"]}">log-y</button>'
            f'<button class="expand" data-chart="{c["key"]}">expand &#10530;</button>'
            '</div>'
            f'<div class="ce-chart" id="chart-{c["key"]}"></div>'
            '</div>'
        )
    return f'<div class="chart-grid">{"".join(cells)}</div>'


def _pct(x):
    return f"{x * 100:.1f}" if isinstance(x, (int, float)) else "&mdash;"


def eval_tab(evals: dict) -> str:
    models = evals["models"]
    any_present = any(m["present"] for m in models)

    # Summary table: models x {MC9, MMLU, Gen5, GSM8K}.
    head = ("<tr><th>model</th><th>MC9<br><span class='note'>acc</span></th>"
            "<th>MMLU<br><span class='note'>acc</span></th>"
            "<th>Gen5<br><span class='note'>F1</span></th>"
            "<th>GSM8K<br><span class='note'>EM</span></th></tr>")
    rows = []
    for m in models:
        s = m["summary"]
        if m["present"]:
            cells = (f"<td>{_pct(s['mc9'])}</td><td>{_pct(s['mmlu'])}</td>"
                     f"<td>{_pct(s['gen5'])}</td><td>{_pct(s['gsm8k'])}</td>")
        else:
            cells = "<td colspan='4' class='note'>evals in flight / not scored yet</td>"
        rows.append(f"<tr><td>{m['label']}</td>{cells}</tr>")
    summary = (f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>")

    # Per-task detail (MC9 9 + Gen5 5), collapsible.
    detail_tasks = MC9_TASKS + GEN5_TASKS
    dhead = "<tr><th>model</th>" + "".join(
        f"<th>{TASK_SHORT.get(t, t)}</th>" for t in detail_tasks) + "</tr>"
    drows = []
    for m in models:
        if not m["present"]:
            continue
        tds = "".join(f"<td>{_pct(m['tasks'].get(t))}</td>" for t in detail_tasks)
        drows.append(f"<tr><td>{m['label']}</td>{tds}</tr>")
    detail = (
        "<details><summary>Per-task breakdown (MC9 + Gen5, primary metric &times;100)</summary>"
        f"<div style='overflow-x:auto'><table><thead>{dhead}</thead><tbody>"
        f"{''.join(drows)}</tbody></table></div>"
        "<p class='note'>MC9 uses each task's OLMES primary metric (acc_per_char / acc_uncond / "
        "acc_raw); Gen5 uses F1. MMLU is the macro-average over 17 categories (acc_per_char).</p>"
        "</details>") if drows else ""

    note = "" if any_present else (
        "<p class='note'>No metrics found under <code>models_v2/merged_evals/</code> yet.</p>")

    return (
        '<div class="card goal"><h3>What</h3>'
        '<p>Plain base-model eval (no expert selection, no finetuning) of each models_v2 '
        'checkpoint on the selective project\'s <strong>merged test-split</strong> tasks &mdash; '
        'the same task definitions/splits the selective project reports, so these numbers are '
        'comparable to its base scores. Launched by '
        '<code>scripts/models_v2/launch_merged_eval.sh</code> (8 size-balanced Beaker jobs/model).'
        '</p></div>'
        f'<div class="card results"><h3>Headline scores</h3>{note}{summary}'
        '<p class="note">MC9 = 9-task rank-classification average (incl. HellaSwag); '
        'MMLU = 17-category macro-average; Gen5 = SQuAD/CoQA/NQ/TriviaQA/DROP F1 average; '
        'GSM8K = 8-shot exact-match. All values &times;100.</p>'
        f'{detail}</div>'
    )


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
        '<div class="card goal"><h3>Goal</h3>'
        '<p>A standard top-k MoE (<code>moe_lbreducedp_sharedexp</code>) token-budget / '
        'expert-count sweep. Unlike the released baselines (130B LR schedule hard-stopped at '
        '50B), every run here decays the LR <strong>directly over its true token '
        'budget</strong> &mdash; "what if we only had this many tokens." Runs vary the '
        'expert pool (64 vs 128) and budget (25B vs 50B); '
        '<code>stdmoe_64exp_50b_wsd</code> additionally swaps the cosine schedule for '
        'warmup-stable-decay (WSD: warmup 2000 / decay 1192 steps &asymp; 5B tokens) to compare '
        'schedulers at fixed 64e&middot;50B against its cosine twin <code>stdmoe_64exp_50b</code>.'
        '</p></div>'
        f'<div class="card results"><h3>Runs</h3>{run_table}'
        f'<p class="note"><strong>W&amp;B runs:</strong> {links}</p></div>'
        '<div class="card method"><h3>Setup</h3>'
        '<p>Common to all: OLMoE-mix-0824, <code>d_model</code> 2048 / 16 layers, top-8 routed '
        '+ 1 shared expert, lr 4e-3, lb 1e-1, 8 nodes / 64 GPUs, global batch 4.19M tokens/step. '
        'Permanent checkpoints saved at 25/50/75% of each run.</p></div>'
    )
    curves = (
        '<div class="card results"><h3>Training &amp; eval curves</h3>'
        '<p class="note">x-axis = optimizer step (k). Drag to zoom (double-click resets), toggle '
        'log-y per chart, or hit <strong>expand &#10530;</strong> to open a chart full-size. '
        'Pulled live from W&amp;B.</p>'
        f'{chart_blocks(payload["charts"])}</div>'
    )

    evals_html = eval_tab(payload["evals"])
    tabs = [("overview", "Overview", overview), ("evals", "Evals", evals_html),
            ("curves", "Curves", curves)]
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
    payload["evals"] = load_evals()  # always re-read merged_evals/ from disk
    uplot_css = (VENDOR / "uPlot.min.css").read_text() if (VENDOR / "uPlot.min.css").is_file() else ""
    uplot_js = (VENDOR / "uPlot.iife.min.js").read_text() if (VENDOR / "uPlot.iife.min.js").is_file() else ""
    args.output.write_text(render(payload, uplot_css, uplot_js))
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
