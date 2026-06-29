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

# ---------------------------------------------------------------------------
# Master run inventory + preset groupings (drives the interactive Curves tab).
#
# The Curves tab renders ONE shared set of metric charts; every run is a series on every chart,
# in THIS list's order — so a run's color is stable across all charts and matches its sidebar
# swatch. A side panel toggles individual runs on/off; PRESETS are one-click groupings that select
# a meaningful set of runs (a trunk + all of its decay/anneal branches, an upcycle family, ...).
# `ids` is a single W&B run id, or a LIST when a run crashed + resumed (histories merged in order,
# later wins on overlapping steps — e.g. the 64e wsd 4e-3 trunk crashed at ~2759/n6zg596k and
# resumed from 2501/96odpdqg).
RUNS = [
    # --- 64-expert baselines ---
    {"key": "64cos25",  "label": "64e·25B cos",  "cat": "64e baselines", "ids": "lsq79eb5"},
    {"key": "64cos50",  "label": "64e·50B cos",  "cat": "64e baselines", "ids": "r5kyiexy"},
    {"key": "64wsd4e3", "label": "64e wsd 4e-3", "cat": "64e baselines", "ids": ["n6zg596k", "96odpdqg"]},
    {"key": "64wsd2e3", "label": "64e wsd 2e-3", "cat": "64e baselines", "ids": "0ucu7x8n"},
    {"key": "64wsd4e4", "label": "64e wsd 4e-4", "cat": "64e baselines", "ids": "fcvnftxd"},
    # --- 64-expert WSD decay branches (forked off a flat-LR stable trunk) ---
    {"key": "64dec_4e3_125", "label": "64e 4e-3 decay@37.5B/12.5B", "cat": "64e decay branches", "ids": "hbq6004e"},
    {"key": "64dec_2e3_5",   "label": "64e 2e-3 decay@45B/5B",      "cat": "64e decay branches", "ids": "69drqnz8"},
    {"key": "64dec_2e3_10",  "label": "64e 2e-3 decay@40B/10B",     "cat": "64e decay branches", "ids": "opp7l86a"},
    {"key": "64dec_4e4_5",   "label": "64e 4e-4 decay@45B/5B",      "cat": "64e decay branches", "ids": "e1munm14"},
    {"key": "64dec_4e4_10",  "label": "64e 4e-4 decay@40B/10B",     "cat": "64e decay branches", "ids": "qx8e61ny"},
    # --- 128-expert baselines (+ a decay branch) ---
    {"key": "128cos",        "label": "128e·50B cos",              "cat": "128e baselines", "ids": "yuafg0dw"},
    {"key": "128wsd4e3",     "label": "128e wsd 4e-3",             "cat": "128e baselines", "ids": "f2u26et2"},
    {"key": "128wsd2e3",     "label": "128e wsd 2e-3",             "cat": "128e baselines", "ids": "sswartor"},
    {"key": "128dec_4e3_10", "label": "128e 4e-3 decay@40B/10B",   "cat": "128e baselines", "ids": "uk48bfrl"},
    # --- Extension methods: expert upcycling 64→128 (5B convergence check) ---
    {"key": "up_copy_cc",    "label": "upcycle copy·carry·copy",   "cat": "upcycle 64→128", "ids": "85nhg564"},
    {"key": "up_copy_cz",    "label": "upcycle copy·carry·zero",   "cat": "upcycle 64→128", "ids": "2hnes1fe"},
    {"key": "up_copy_reset", "label": "upcycle copy·reset",        "cat": "upcycle 64→128", "ids": "kkummxiv"},
    {"key": "up_rand_carry", "label": "upcycle random·carry",      "cat": "upcycle 64→128", "ids": "eus6qsqh"},
    {"key": "up_rand_reset", "label": "upcycle random·reset",      "cat": "upcycle 64→128", "ids": "s4bq4kvw"},
    {"key": "up_jit_cc",     "label": "upcycle jitter·carry·copy", "cat": "upcycle 64→128", "ids": "uar2nr44"},
    {"key": "up_jit_cz",     "label": "upcycle jitter·carry·zero", "cat": "upcycle 64→128", "ids": "r6yj95lk"},
    {"key": "up_jit_reset",  "label": "upcycle jitter·reset",      "cat": "upcycle 64→128", "ids": "t36ixfxn"},
]

_UP_ALL = ["up_copy_cc", "up_copy_cz", "up_copy_reset", "up_rand_carry", "up_rand_reset",
           "up_jit_cc", "up_jit_cz", "up_jit_reset"]
_ALL_64 = ["64cos25", "64cos50", "64wsd4e3", "64wsd2e3", "64wsd4e4",
           "64dec_4e3_125", "64dec_2e3_5", "64dec_2e3_10", "64dec_4e4_5", "64dec_4e4_10"]
_ALL_128 = ["128cos", "128wsd4e3", "128wsd2e3", "128dec_4e3_10"]

# One-click groupings shown as buttons in the Curves sidebar. Each selects a meaningful set of runs
# (a trunk + all its decay branches, an upcycle family, a head-to-head). `default: True` marks the
# set shown on first load.
PRESETS = [
    {"cat": "Baselines", "items": [
        {"name": "64e cosine",             "runs": ["64cos25", "64cos50"]},
        {"name": "64e WSD 4e-3 (+decay)",  "runs": ["64wsd4e3", "64dec_4e3_125"]},
        {"name": "64e WSD 2e-3 (+decay)",  "runs": ["64wsd2e3", "64dec_2e3_5", "64dec_2e3_10"]},
        {"name": "64e WSD 4e-4 (+decay)",  "runs": ["64wsd4e4", "64dec_4e4_5", "64dec_4e4_10"]},
        {"name": "128e cosine",            "runs": ["128cos"]},
        {"name": "128e WSD 4e-3 (+decay)", "runs": ["128wsd4e3", "128dec_4e3_10"]},
        {"name": "128e WSD 2e-3",          "runs": ["128wsd2e3"]},
    ]},
    {"cat": "By expert count", "items": [
        {"name": "All 64e",  "runs": _ALL_64},
        {"name": "All 128e", "runs": _ALL_128},
    ]},
    {"cat": "Extension methods", "items": [
        {"name": "Upcycle: all 8",  "runs": _UP_ALL},
        {"name": "Upcycle: copy",   "runs": ["up_copy_cc", "up_copy_cz", "up_copy_reset"]},
        {"name": "Upcycle: jitter", "runs": ["up_jit_cc", "up_jit_cz", "up_jit_reset"]},
        {"name": "Upcycle: random", "runs": ["up_rand_carry", "up_rand_reset"]},
        {"name": "Upcycle vs from-scratch 128e", "runs": _UP_ALL + ["128wsd2e3"], "default": True},
    ]},
]

# Stable per-run color palette (indexed by position in RUNS; reused for sidebar swatches AND chart
# strokes so they always agree). 22 runs -> 24 reasonably-distinct colors.
PALETTE = [
    "#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#db2777", "#65a30d",
    "#475569", "#9333ea", "#0d9488", "#e11d48", "#4f46e5", "#ca8a04", "#15803d", "#b91c1c",
    "#7c2d12", "#1d4ed8", "#be123c", "#047857", "#a21caf", "#92400e", "#0369a1", "#4d7c0f",
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
    ("stdmoe_128exp_50b_wsd",  "stdMoE 128e · 50B wsd"),
    ("stdmoe_128exp_50b_wsd/anneals/s9537_10b", "stdMoE 128e · wsd decay@40B/10B"),
    ("stdmoe_64exp_50b",       "stdMoE 64e · 50B"),
    # WSD family at 50B: the trunk's own final (5B end-of-run decay) and the 12.5B decay branch
    # forked at 37.5B. Branch run dir is hierarchical (under the trunk's anneals/).
    ("stdmoe_64exp_50b_wsd",                       "stdMoE 64e · 50B wsd (5B decay)"),
    ("stdmoe_64exp_50b_wsd/anneals/s8941_12p5b",   "stdMoE 64e · wsd decay@37.5B/12.5B"),
    # lr2e-3 stable trunk decay branches (peak LR 2e-3): 5B decay from 45B, 10B decay from 40B.
    ("stdmoe_64exp_50b_wsd_lr2e-3/anneals/s10729_5b", "stdMoE 64e · lr2e-3 decay@45B/5B"),
    ("stdmoe_64exp_50b_wsd_lr2e-3/anneals/s9537_10b", "stdMoE 64e · lr2e-3 decay@40B/10B"),
    # lr4e-4 stable trunk decay branches (peak LR 4e-4): 5B decay from 45B, 10B decay from 40B.
    ("stdmoe_64exp_50b_wsd_lr4e-4/anneals/s10729_5b", "stdMoE 64e · lr4e-4 decay@45B/5B"),
    ("stdmoe_64exp_50b_wsd_lr4e-4/anneals/s9537_10b", "stdMoE 64e · lr4e-4 decay@40B/10B"),
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
/* curves layout: sticky run sidebar + chart area */
.curves-layout { display:flex; gap:18px; align-items:flex-start; }
.run-panel { flex:0 0 252px; position:sticky; top:104px; max-height:calc(100vh - 124px);
             overflow:auto; background:var(--card); border:1px solid var(--line);
             border-radius:8px; padding:12px 14px; font-size:13px; }
.curves-main { flex:1 1 auto; min-width:0; }
.rp-h { font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.05em;
        color:var(--muted); margin:12px 0 6px; display:flex; justify-content:space-between;
        align-items:center; }
.rp-h:first-child { margin-top:0; }
.rp-mini button { border:1px solid var(--line); background:#fff; border-radius:5px; cursor:pointer;
                  font-size:11px; padding:2px 7px; margin-left:4px; }
.rp-mini button:hover { background:#f1f5f9; }
.rp-cat { font-size:11px; color:var(--muted); margin:9px 0 3px; font-weight:600; }
.rp-btns { display:flex; flex-wrap:wrap; gap:4px; margin-bottom:2px; }
.rp-btns button { border:1px solid #c7d2fe; background:#eef2ff; color:#3730a3; border-radius:12px;
                  font-size:11px; padding:3px 9px; cursor:pointer; }
.rp-btns button:hover { background:#e0e7ff; }
.rp-fitx-l { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted);
             margin:6px 0 2px; cursor:pointer; }
.rp-runs { margin-top:2px; }
.rp-run { display:flex; align-items:center; gap:7px; padding:2px 0; cursor:pointer; line-height:1.3; }
.rp-run input { margin:0; flex:0 0 auto; }
.rp-sw { display:inline-block; width:11px; height:11px; border-radius:3px; flex:0 0 auto; }
@media (max-width:900px){ .curves-layout { flex-direction:column; }
  .run-panel { position:static; flex-basis:auto; width:100%; max-height:none; } }
"""

# uPlot init: ONE shared set of metric charts; every run is a series (same order on every chart, so
# colors/indices line up). A side panel toggles runs on/off, presets select run groups, and the
# x-axis auto-fits to the visible selection. Plain string (literal braces); data injected via
# __CURVES__ / __PALETTE__ replace.
_CHARTS_JS = r"""
<script>
(function(){
  const C = __CURVES__;
  const palette = __PALETTE__;
  const nRuns = C.runs.length;
  const SMALL_H = 230, reg = {};
  // Per-run visibility, seeded from each sidebar checkbox's initial checked state (set server-side
  // from the default preset). Series index i  <->  run i  <->  palette[i]  <->  checkbox data-idx=i.
  const vis = new Array(nRuns).fill(false);
  const cbs = Array.from(document.querySelectorAll(".rp-run input[type=checkbox]"));
  cbs.forEach(cb => { vis[+cb.dataset.idx] = cb.checked; });
  const fitxEl = document.getElementById("rp-fitx");

  function opts(chart, logY, w, h, legend){
    return {
      width:w, height:h, focus:{ alpha:0.25 }, legend:{ show:!!legend },
      scales:{ x:{ time:false }, y:{ distr: logY ? 3 : 1 } },
      cursor:{ focus:{ prox:30 }, drag:{ x:true, y:true, uni:10 } },
      axes:[ { label:"step", values:(u,vals)=>vals.map(v=>v>=1000?(v/1000)+"k":v) }, { label:chart.title } ],
      series:[ { value:(u,v)=>v==null?"--":v } ].concat(chart.series.map((s,i)=>({
        label:s.label, stroke:palette[i % palette.length], width:1.6, spanGaps:true, show:vis[i],
        value:(u,v)=>v==null?"--":(+v).toFixed(4) }))),
    };
  }
  const dataOf = chart => [chart.x].concat(chart.series.map(s=>s.y));
  // x-range covering every visible series' non-null samples (null if nothing visible has data).
  function visRange(c){
    let lo=Infinity, hi=-Infinity;
    for(let i=0;i<c.series.length;i++){ if(!vis[i]) continue; const y=c.series[i].y;
      for(let j=0;j<y.length;j++){ if(y[j]!=null){ const x=c.x[j]; if(x<lo)lo=x; if(x>hi)hi=x; } } }
    return lo<=hi ? [lo,hi] : null;
  }
  function build(key){
    const st=reg[key], el=st.el; if(!el) return;
    if(st.plot) st.plot.destroy();
    st.plot = new uPlot(opts(st.chart, st.logY, el.clientWidth||430, SMALL_H, false), dataOf(st.chart), el);
  }
  C.charts.forEach(chart => {
    reg[chart.key] = { plot:null, logY:false, chart:chart, el:document.getElementById("chart-"+chart.key) };
    build(chart.key);
  });
  function applyVis(){
    Object.keys(reg).forEach(k => { const st=reg[k]; if(!st.plot) return;
      for(let i=0;i<nRuns;i++) st.plot.setSeries(i+1, { show:vis[i] });
      if(fitxEl && fitxEl.checked){ const r=visRange(st.chart); if(r) st.plot.setScale("x", { min:r[0], max:r[1] }); }
    });
    if(m && m.plot) drawM();
  }
  applyVis();  // sync initial fit to the default selection

  // ---- sidebar wiring: per-run checkboxes, presets, all/none, auto-fit ----
  cbs.forEach(cb => cb.addEventListener("change", () => { vis[+cb.dataset.idx]=cb.checked; applyVis(); }));
  function setVis(keys){
    const set = new Set(keys);
    for(let i=0;i<nRuns;i++) vis[i] = set.has(C.runs[i].key);
    cbs.forEach(cb => { cb.checked = vis[+cb.dataset.idx]; });
    applyVis();
  }
  document.querySelectorAll("button.preset").forEach(b =>
    b.addEventListener("click", () => setVis(b.dataset.runs.split(",").filter(Boolean))));
  const allBtn=document.getElementById("rp-all"), noneBtn=document.getElementById("rp-none");
  if(allBtn) allBtn.addEventListener("click", () => setVis(C.runs.map(r=>r.key)));
  if(noneBtn) noneBtn.addEventListener("click", () => setVis([]));
  if(fitxEl) fitxEl.addEventListener("change", applyVis);

  document.querySelectorAll("button.logtoggle").forEach(b =>
    b.addEventListener("click", () => { const k=b.dataset.chart; reg[k].logY=!reg[k].logY; build(k); applyVis(); }));

  // ---- expand modal (full-size single chart; legend on so values are readable) ----
  const modal=document.createElement("div"); modal.className="ce-modal";
  modal.innerHTML='<div class="ce-modal-inner"><div class="ce-modal-bar"><strong></strong>'
    +'<span><button class="ce-mlog">toggle log-y</button> <button class="ce-mclose">close ✕</button></span>'
    +'</div><div class="ce-modal-chart"></div></div>';
  document.body.appendChild(modal);
  const mEl=modal.querySelector(".ce-modal-chart"), mTitle=modal.querySelector("strong");
  let m=null;  // { key, logY, plot }
  const mSize=()=>({ w:Math.min(window.innerWidth*0.94,1180)|0, h:Math.min(window.innerHeight*0.72,700)|0 });
  function drawM(){ const c=reg[m.key].chart, s=mSize(); if(m.plot) m.plot.destroy();
    m.plot=new uPlot(opts(c, m.logY, s.w, s.h, true), dataOf(c), mEl);
    if(fitxEl && fitxEl.checked){ const r=visRange(c); if(r) m.plot.setScale("x", { min:r[0], max:r[1] }); } }
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


def _fetch_run(api, ids, cache) -> tuple:
    """Fetch + merge metric histories for one run (a list of ids merged in order, later wins on
    overlap). Returns (last_run, {key: {step: val}}). Caches per-id so runs shared across groups
    are only pulled once."""
    merged: dict = {}
    last_run = None
    for one in ids:
        if one not in cache:
            run = api.run(f"{ENTITY_PROJECT}/{one}")
            hist: dict = {}
            for _, key in METRICS:
                d: dict = {}
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
                hist[key] = d
            cache[one] = (run, hist)
        run, hist = cache[one]
        last_run = run
        for key, d in hist.items():
            merged.setdefault(key, {}).update(d)
    return last_run, merged


def fetch_from_wandb() -> dict:
    import wandb
    api = wandb.Api()
    cache: dict = {}
    # Pull each run's merged history once, in RUNS order.
    per_run: dict = {}   # run key -> {metric_key: {step: val}}
    runs_meta = []
    for r in RUNS:
        ids = [r["ids"]] if isinstance(r["ids"], str) else list(r["ids"])
        last_run, merged = _fetch_run(api, ids, cache)
        per_run[r["key"]] = merged
        ce = merged.get("train/CE loss", {})
        last_step = max(ce) if ce else None
        runs_meta.append({
            "key": r["key"], "label": r["label"], "cat": r["cat"], "id": "+".join(ids),
            "url": last_run.url, "state": last_run.state,
            "last_ce": round(ce[last_step], 3) if ce else None,
            "last_tok": round(last_step * TOK_PER_STEP / 1e9, 1) if last_step else None,
        })
    # One chart per metric; every run is a series, in RUNS order (so series index == run index ==
    # palette index == sidebar checkbox data-idx). Runs without data for a metric are all-null.
    charts = []
    for title, key in METRICS:
        steps = sorted(set().union(*[set(per_run[r["key"]].get(key, {})) for r in RUNS]) or {0})
        series = [{"label": r["label"], "y": [per_run[r["key"]].get(key, {}).get(s) for s in steps]}
                  for r in RUNS]
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


def run_panel_html(runs: list) -> str:
    """The Curves sidebar: preset group buttons + a per-run checkbox list (grouped by category,
    color swatches matching the chart strokes). `runs` is payload["runs"] in RUNS order, so the
    enumerate index == series index == PALETTE index == checkbox data-idx."""
    default_keys: set = set()
    for cat in PRESETS:
        for it in cat["items"]:
            if it.get("default"):
                default_keys = set(it["runs"])
    presets = []
    for cat in PRESETS:
        btns = "".join(
            f'<button class="preset" data-runs="{",".join(it["runs"])}">{it["name"]}</button>'
            for it in cat["items"])
        presets.append(f'<div class="rp-cat">{cat["cat"]}</div><div class="rp-btns">{btns}</div>')
    rows, last_cat = [], None
    for i, r in enumerate(runs):
        if r["cat"] != last_cat:
            rows.append(f'<div class="rp-cat">{r["cat"]}</div>')
            last_cat = r["cat"]
        checked = " checked" if r["key"] in default_keys else ""
        sw = PALETTE[i % len(PALETTE)]
        rows.append(
            f'<label class="rp-run"><input type="checkbox" data-idx="{i}"{checked}>'
            f'<span class="rp-sw" style="background:{sw}"></span>{r["label"]}</label>')
    return (
        '<div class="run-panel">'
        '<div class="rp-h">Presets</div>'
        f'<div class="rp-presets">{"".join(presets)}</div>'
        '<div class="rp-h">Runs <span class="rp-mini">'
        '<button id="rp-all">all</button><button id="rp-none">none</button></span></div>'
        '<label class="rp-fitx-l"><input type="checkbox" id="rp-fitx" checked>'
        ' auto-fit x to selection</label>'
        f'<div class="rp-runs">{"".join(rows)}</div>'
        '</div>')


def render(payload: dict, uplot_css: str, uplot_js: str) -> str:
    runs = payload["runs"]

    # Overview: intro + common setup + one run-table card per category.
    overview_cards = [
        '<div class="card goal"><h3>Goal</h3>'
        '<p>A standard top-k MoE (<code>moe_lbreducedp_sharedexp</code>) study. '
        '<strong>Baselines</strong> (now settled): the WSD scheduler, the peak-LR sweep, and '
        'decay-amount ablations, plus 64&rarr;128 expert scaling. <strong>Extension methods</strong> '
        '(in progress): ways to grow an already-trained model &mdash; starting with expert '
        'upcycling. The <strong>Curves</strong> tab plots every run on one shared set of charts; '
        'use the sidebar to toggle individual runs or click a <em>preset</em> to focus on one '
        'question (e.g. &ldquo;128e WSD 4e-3 (+decay)&rdquo; or &ldquo;Upcycle vs from-scratch&rdquo;).</p></div>'
        '<div class="card method"><h3>Setup</h3>'
        '<p>Common to all: OLMoE-mix-0824, <code>d_model</code> 2048 / 16 layers, top-8 routed '
        '+ 1 shared expert, lb 1e-1, 8 nodes / 64 GPUs, global batch 4.19M tokens/step. WSD runs '
        'decay the LR over their true token budget; trunks keep permanent checkpoints every 5B.</p></div>'
    ]
    cats: list = []
    by_cat: dict = {}
    for r in runs:
        if r["cat"] not in by_cat:
            cats.append(r["cat"])
            by_cat[r["cat"]] = []
        by_cat[r["cat"]].append(r)
    for cat in cats:
        crs = by_cat[cat]
        body = "".join(
            f"<tr><td>{r['label']}</td><td>{r['state']}</td>"
            f"<td>{r['last_tok']}B</td><td>{r['last_ce']}</td>"
            f'<td><a href="{r["url"]}" target="_blank" rel="noopener">W&amp;B</a></td></tr>' for r in crs)
        overview_cards.append(
            f'<div class="card results"><h3>{cat}</h3>'
            '<table><thead><tr><th>run</th><th>state</th><th>tokens</th><th>latest CE</th>'
            '<th>link</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>')
    overview = "".join(overview_cards)

    # Curves: how-to card, then a sticky run-selector sidebar + ONE shared chart grid.
    curves = (
        '<div class="card goal"><h3>How to read</h3>'
        '<p class="note">Every run is plotted on the same charts. In the sidebar, toggle individual '
        'runs or click a <strong>preset</strong> to show a meaningful group (a trunk + its decay '
        'branches, an upcycle family, a head-to-head). With <strong>auto-fit x</strong> on, the '
        'x-axis snaps to the selected runs&rsquo; range (so a 5B upcycle window isn&rsquo;t squashed '
        'next to a full 50B run); the y-axis always rescales to what&rsquo;s shown. Drag to zoom '
        '(double-click resets), toggle log-y per chart, or <strong>expand &#10530;</strong> for '
        'full size. Colors are stable per run and match the sidebar swatches. Live from W&amp;B.</p></div>'
        '<div class="curves-layout">'
        f'{run_panel_html(runs)}'
        f'<div class="curves-main">{chart_blocks(payload["charts"])}</div>'
        '</div>')

    evals_html = eval_tab(payload["evals"])
    tabs = [("overview", "Overview", overview), ("evals", "Evals", evals_html),
            ("curves", "Curves", curves)]
    nav = "".join(f'<button data-target="{t}">{n}</button>' for t, n, _ in tabs)
    sections = "".join(f'<section class="tab" id="{t}">{b}</section>' for t, _, b in tabs)
    charts_js = (_CHARTS_JS.replace("__CURVES__", json.dumps(payload))
                 .replace("__PALETTE__", json.dumps(PALETTE)))

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
