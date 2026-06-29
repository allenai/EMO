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
# Master run inventory (drives the interactive explorer tabs).
#
# Each explorer tab renders a SHARED set of metric charts on the right; every run is a series on
# every chart, in THIS list's order — so a run's color (PALETTE[index]) is stable everywhere. The
# left column lists named recipe GROUPS (see TABS); clicking a group toggles its runs (a trunk +
# all its decay branches) onto the plots. `ids` is a single W&B run id, or a LIST when a run
# crashed + resumed (histories merged in order, later wins on overlap — e.g. the 64e wsd 4e-3 trunk
# crashed at ~2759/n6zg596k and resumed from 2501/96odpdqg).
RUNS = [
    # --- 64-expert baselines ---
    {"key": "64cos25",  "label": "64e·25B cos",  "cat": "64e baselines", "ids": "lsq79eb5"},
    {"key": "64cos50",  "label": "64e·50B cos",  "cat": "64e baselines", "ids": "r5kyiexy"},
    {"key": "64wsd4e3", "label": "64e wsd 4e-3", "cat": "64e baselines", "ids": ["n6zg596k", "96odpdqg"]},
    {"key": "64wsd2e3", "label": "64e wsd 2e-3", "cat": "64e baselines", "ids": "0ucu7x8n"},
    {"key": "64wsd4e4", "label": "64e wsd 4e-4", "cat": "64e baselines", "ids": "fcvnftxd"},
    # --- 64-expert WSD decay branches (forked off a flat-LR stable trunk) ---
    # `branch` anchors the child's curves to the parent's value at the fork step so they visibly
    # branch off it (evals are logged sparsely, so the child's first eval lands well after the fork
    # -> a gap without this). fork_step = round(forkB * 5e9-ish / 4,194,304): 37.5B=8941, 40B=9537,
    # 45B=10729.
    {"key": "64dec_4e3_125", "label": "64e 4e-3 decay@37.5B/12.5B", "cat": "64e decay branches", "ids": "hbq6004e",
     "branch": {"parent": "64wsd4e3", "fork_step": 8941}},
    {"key": "64dec_2e3_5",   "label": "64e 2e-3 decay@45B/5B",      "cat": "64e decay branches", "ids": "69drqnz8",
     "branch": {"parent": "64wsd2e3", "fork_step": 10729}},
    {"key": "64dec_2e3_10",  "label": "64e 2e-3 decay@40B/10B",     "cat": "64e decay branches", "ids": "opp7l86a",
     "branch": {"parent": "64wsd2e3", "fork_step": 9537}},
    {"key": "64dec_4e4_5",   "label": "64e 4e-4 decay@45B/5B",      "cat": "64e decay branches", "ids": "e1munm14",
     "branch": {"parent": "64wsd4e4", "fork_step": 10729}},
    {"key": "64dec_4e4_10",  "label": "64e 4e-4 decay@40B/10B",     "cat": "64e decay branches", "ids": "qx8e61ny",
     "branch": {"parent": "64wsd4e4", "fork_step": 9537}},
    # --- 128-expert baselines (+ a decay branch) ---
    {"key": "128cos",        "label": "128e·50B cos",              "cat": "128e baselines", "ids": "yuafg0dw"},
    {"key": "128wsd4e3",     "label": "128e wsd 4e-3",             "cat": "128e baselines", "ids": "f2u26et2"},
    {"key": "128wsd2e3",     "label": "128e wsd 2e-3",             "cat": "128e baselines", "ids": "sswartor"},
    {"key": "128dec_4e3_10", "label": "128e 4e-3 decay@40B/10B",   "cat": "128e baselines", "ids": "uk48bfrl",
     "branch": {"parent": "128wsd4e3", "fork_step": 9537}},
    {"key": "128dec_2e3_10", "label": "128e 2e-3 decay@40B/10B",   "cat": "128e baselines", "ids": "rwgtpu7e",
     "branch": {"parent": "128wsd2e3", "fork_step": 9537}},
    # --- Extension methods: expert upcycling 64→128 (5B convergence check) ---
    {"key": "up_copy_cc",    "label": "upcycle copy·carry·copy",   "cat": "upcycle 64→128", "ids": "85nhg564"},
    {"key": "up_copy_cz",    "label": "upcycle copy·carry·zero",   "cat": "upcycle 64→128", "ids": "2hnes1fe"},
    {"key": "up_copy_reset", "label": "upcycle copy·reset",        "cat": "upcycle 64→128", "ids": "kkummxiv"},
    {"key": "up_rand_carry", "label": "upcycle random·carry",      "cat": "upcycle 64→128", "ids": "eus6qsqh"},
    {"key": "up_rand_reset", "label": "upcycle random·reset",      "cat": "upcycle 64→128", "ids": "s4bq4kvw"},
    # jitter·carry·copy: 5B-check run (uar2nr44, 25B->30B) merged with the 50B extend pass
    # (byow8llp, 30B-> ; in flight) so the curve shows the full continuation.
    {"key": "up_jit_cc",     "label": "upcycle jitter·carry·copy", "cat": "upcycle 64→128", "ids": ["uar2nr44", "byow8llp"]},
    # jitter·carry·zero: 5B-check run (r6yj95lk, 25B->30B) merged with the 50B extend pass
    # (nrjavgbl, 30B-> ; in flight).
    {"key": "up_jit_cz",     "label": "upcycle jitter·carry·zero", "cat": "upcycle 64→128", "ids": ["r6yj95lk", "nrjavgbl"]},
    {"key": "up_jit_reset",  "label": "upcycle jitter·reset",      "cat": "upcycle 64→128", "ids": "t36ixfxn"},
]

# ---------------------------------------------------------------------------
# Explorer tabs. Each tab = a left column of named recipe GROUPS + a shared right-column chart grid.
# Clicking a group toggles its runs (trunk + all its decay branches) onto every chart. `default` is
# the list of group indices shown on first load. Tab 1 is the LR-scheduling story (cosine vs WSD ×
# peak-LR × decay); tab 2 is the extension-method ablation (still being shaped — provisional).
TABS = [
    {
        "id": "lr",
        "label": "LR scheduling",
        "title": "Learning-rate scheduling: cosine vs WSD",
        "intro": "Cosine (peak LR 4e-3) vs warmup-stable-decay (WSD) at 64 and 128 experts. WSD keeps "
                 "a flat trunk then decays the LR to 0; I swept the peak LR (4e-3 / 2e-3 / 4e-4), and "
                 "for some trunks forked explicit decay branches over the last N·B tokens. Pick a "
                 "recipe on the left to plot its trunk plus all of its decay lines; pick several to "
                 "compare. The x-axis auto-fits to the selection and the y-axis rescales to what's "
                 "shown. Each decay branch is anchored to its parent trunk's value at the fork step, "
                 "so it visibly branches off the trunk (evals are logged too sparsely to otherwise "
                 "record a point at the fork).",
        "groups": [
            {"name": "64e cosine",              "runs": ["64cos25", "64cos50"]},
            {"name": "128e cosine",             "runs": ["128cos"]},
            {"name": "64e WSD 4e-3 (+decays)",  "runs": ["64wsd4e3", "64dec_4e3_125"]},
            {"name": "64e WSD 2e-3 (+decays)",  "runs": ["64wsd2e3", "64dec_2e3_5", "64dec_2e3_10"]},
            {"name": "64e WSD 4e-4 (+decays)",  "runs": ["64wsd4e4", "64dec_4e4_5", "64dec_4e4_10"]},
            {"name": "128e WSD 4e-3 (+decays)", "runs": ["128wsd4e3", "128dec_4e3_10"]},
            {"name": "128e WSD 2e-3 (+decays)", "runs": ["128wsd2e3", "128dec_2e3_10"]},
        ],
        "default": [2],  # 64e WSD 4e-3 (+decays)
    },
    {
        "id": "ext",
        "label": "Extension methods",
        "title": "Extension methods: expert upcycling 64→128",
        "intro": "Can we grow a trained 64e model into a 128e one cheaply? Take the 64e WSD-2e-3 trunk "
                 "at 25B (step5960), expand it to 128 experts (63 standard kept + 64 new + shared "
                 "moved to the last slot), and continue WSD training — a 5B convergence check (25B→30B, "
                 "flat LR 2e-3). Three init families × optimizer treatments: <strong>copy</strong> "
                 "(new experts duplicate sources), <strong>jitter</strong> (copy + noise), "
                 "<strong>random</strong> (fresh). Two reference bounds bracket the result (drawn "
                 "dashed): the <strong>upperbound</strong> = from-scratch 128e WSD-2e-3 (the ceiling "
                 "if you trained 128 experts from scratch), and the <strong>lowerbound</strong> = the "
                 "64e WSD-2e-3 trunk we extended from. The reference lines are excluded from the "
                 "x-auto-fit, so the 5B upcycle window stays readable while the bounds show through it.",
        "groups": [
            {"name": "Upcycle: copy",   "runs": ["up_copy_cc", "up_copy_cz", "up_copy_reset"]},
            {"name": "Upcycle: jitter", "runs": ["up_jit_cc", "up_jit_cz", "up_jit_reset"]},
            {"name": "Upcycle: random", "runs": ["up_rand_carry", "up_rand_reset"]},
            {"name": "Upperbound: 128e WSD 2e-3 (from-scratch)", "runs": ["128wsd2e3"], "ref": True},
            {"name": "Lowerbound: 64e WSD 2e-3 (source)",        "runs": ["64wsd2e3"],  "ref": True},
        ],
        "default": [0, 3, 4],  # copy family bracketed by both bounds
    },
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
/* explorer: left recipe selector + right shared chart grid */
.explorer { display:flex; gap:18px; align-items:flex-start; }
.exp-panel { flex:0 0 232px; position:sticky; top:104px; max-height:calc(100vh - 124px);
             overflow:auto; background:var(--card); border:1px solid var(--line);
             border-radius:8px; padding:12px 12px 14px; }
.exp-charts { flex:1 1 auto; min-width:0; }
.exp-h { font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.05em;
         color:var(--muted); margin:0 0 8px; }
.exp-group { display:flex; align-items:center; gap:8px; width:100%; text-align:left; cursor:pointer;
             border:1px solid var(--line); background:#fff; border-radius:7px; padding:8px 10px;
             margin:5px 0; font-size:13px; color:var(--fg); line-height:1.25; }
.exp-group:hover { background:#f8fafc; }
.exp-group[aria-pressed="true"] { border-color:#2563eb; background:#eff6ff; font-weight:600; }
.exp-dot { width:11px; height:11px; border-radius:3px; flex:0 0 auto; border:1px solid rgba(0,0,0,.15);
           background:#fff; }
.exp-group[aria-pressed="true"] .exp-dot { background:#2563eb; border-color:#2563eb; }
.exp-group.ref { border-style:dashed; }  /* reference bound -> dashed line on the charts */
.exp-fitx-l { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted);
              margin:10px 2px 0; cursor:pointer; }
@media (max-width:900px){ .explorer { flex-direction:column; }
  .exp-panel { position:static; flex-basis:auto; width:100%; max-height:none; } }
"""

# uPlot init for the explorer tabs. Each `.explorer` container (one per tab) has a left column of
# `.exp-group` toggle buttons (data-runs="key,key,...") and a right `.exp-charts` grid of
# `.chart-cell` (data-metric=slug). All runs are series on every chart in RUNKEYS order, so series
# index i <-> RUNKEYS[i] <-> PALETTE[i]; a group toggle just unions its run indices into a per-tab
# visibility set and rebuilds. x auto-fits to the selection; y rescales to shown series. Plain
# string (literal braces); data injected via __CHARTS__ / __RUNKEYS__ / __PALETTE__ replace.
_CHARTS_JS = r"""
<script>
(function(){
  const CHARTS = __CHARTS__, RUNKEYS = __RUNKEYS__, palette = __PALETTE__;
  const idxOf = {}; RUNKEYS.forEach((k,i)=>idxOf[k]=i);
  const chartByKey = {}; CHARTS.forEach(c=>chartByKey[c.key]=c);
  const SMALL_H = 240;
  const allReg = [];   // every {plot,el} across tabs, for resize

  function makeOpts(chart, logY, w, h, vis, legend, refSet){
    return {
      width:w, height:h, focus:{ alpha:0.3 }, legend:{ show:!!legend },
      scales:{ x:{ time:false }, y:{ distr: logY ? 3 : 1 } },
      cursor:{ focus:{ prox:30 }, drag:{ x:true, y:true, uni:10 } },
      axes:[ { label:"step", values:(u,vals)=>vals.map(v=>v>=1000?(v/1000)+"k":v) }, { label:chart.title } ],
      series:[ { value:(u,v)=>v==null?"--":v } ].concat(chart.series.map((s,i)=>({
        label:s.label, stroke:palette[i % palette.length], width:1.8, spanGaps:true, show:!!vis[i],
        dash:(refSet && refSet.has(i)) ? [7,4] : undefined,
        value:(u,v)=>v==null?"--":(+v).toFixed(4) }))),
    };
  }
  const dataOf = chart => [chart.x].concat(chart.series.map(s=>s.y));
  // x-range over visible series' non-null samples (null if nothing visible has data).
  function visRange(chart, vis){
    let lo=Infinity, hi=-Infinity;
    for(let i=0;i<chart.series.length;i++){ if(!vis[i]) continue; const y=chart.series[i].y;
      for(let j=0;j<y.length;j++){ if(y[j]!=null){ const x=chart.x[j]; if(x<lo)lo=x; if(x>hi)hi=x; } } }
    return lo<=hi ? [lo,hi] : null;
  }

  // ---- shared expand modal ----
  const modal=document.createElement("div"); modal.className="ce-modal";
  modal.innerHTML='<div class="ce-modal-inner"><div class="ce-modal-bar"><strong></strong>'
    +'<span><button class="ce-mlog">toggle log-y</button> <button class="ce-mclose">close &#10005;</button></span>'
    +'</div><div class="ce-modal-chart"></div></div>';
  document.body.appendChild(modal);
  const mEl=modal.querySelector(".ce-modal-chart"), mTitle=modal.querySelector("strong");
  let M=null;  // { chart, vis, fitVis, refSet, logY, plot }
  const mSize=()=>({ w:Math.min(window.innerWidth*0.94,1180)|0, h:Math.min(window.innerHeight*0.72,700)|0 });
  function drawModal(){ const s=mSize(); if(M.plot) M.plot.destroy();
    M.plot=new uPlot(makeOpts(M.chart, M.logY, s.w, s.h, M.vis, true, M.refSet), dataOf(M.chart), mEl);
    const r=visRange(M.chart, M.fitVis); if(r) M.plot.setScale("x", { min:r[0], max:r[1] }); }
  function openModal(chart, vis, fitVis, refSet, logY){ M={ chart, vis, fitVis, refSet, logY, plot:null };
    mTitle.textContent=chart.title; modal.classList.add("open"); drawModal(); }
  function closeModal(){ if(M&&M.plot) M.plot.destroy(); M=null; modal.classList.remove("open"); }
  modal.querySelector(".ce-mclose").addEventListener("click", closeModal);
  modal.querySelector(".ce-mlog").addEventListener("click", ()=>{ if(M){ M.logY=!M.logY; drawModal(); } });
  modal.addEventListener("click", e=>{ if(e.target===modal) closeModal(); });
  document.addEventListener("keydown", e=>{ if(e.key==="Escape") closeModal(); });

  // ---- per-explorer wiring ----
  function setupExplorer(root){
    const groupBtns = Array.from(root.querySelectorAll(".exp-group"));
    const fitxEl = root.querySelector(".exp-fitx");
    const fitx = () => !fitxEl || fitxEl.checked;
    const reg = {};   // metric slug -> { plot, logY, chart, el }
    // Reference groups (e.g. upper/lower bounds): drawn dashed and excluded from the x-auto-fit so
    // the focus runs define the window. Built once from the static group markup.
    const refSet = new Set();
    groupBtns.forEach(b => { if(b.dataset.ref==="1")
      b.dataset.runs.split(",").filter(Boolean).forEach(k => { if(k in idxOf) refSet.add(idxOf[k]); }); });
    function selected(includeRef){
      const s = new Set();
      groupBtns.forEach(b => { if(b.getAttribute("aria-pressed")==="true" && (includeRef || b.dataset.ref!=="1"))
        b.dataset.runs.split(",").filter(Boolean).forEach(k => { if(k in idxOf) s.add(idxOf[k]); }); });
      return s;
    }
    const vis = () => { const s=selected(true);  return RUNKEYS.map((k,i)=>s.has(i)); };
    // x-fit driven by non-ref (focus) runs; if only refs are selected, fall back to all selected.
    function fitVis(){ const f=selected(false); const s=f.size?f:selected(true); return RUNKEYS.map((k,i)=>s.has(i)); }
    function build(slug){
      const st=reg[slug], el=st.el; if(!el) return;
      const v=vis();
      if(st.plot) st.plot.destroy();
      st.plot=new uPlot(makeOpts(st.chart, st.logY, el.clientWidth||440, SMALL_H, v, false, refSet), dataOf(st.chart), el);
      if(fitx()){ const r=visRange(st.chart, fitVis()); if(r) st.plot.setScale("x", { min:r[0], max:r[1] }); }
    }
    function rebuildAll(){ Object.keys(reg).forEach(build); }
    root.querySelectorAll(".chart-cell").forEach(cell => {
      const slug=cell.dataset.metric, el=cell.querySelector(".ce-chart");
      reg[slug]={ plot:null, logY:false, chart:chartByKey[slug], el:el };
      allReg.push(reg[slug]);
    });
    groupBtns.forEach(b => b.addEventListener("click", () => {
      b.setAttribute("aria-pressed", b.getAttribute("aria-pressed")==="true" ? "false" : "true");
      rebuildAll();
    }));
    if(fitxEl) fitxEl.addEventListener("change", rebuildAll);
    root.querySelectorAll("button.logtoggle").forEach(b => b.addEventListener("click", () => {
      reg[b.dataset.metric].logY=!reg[b.dataset.metric].logY; build(b.dataset.metric); }));
    root.querySelectorAll("button.expand").forEach(b => b.addEventListener("click", () => {
      const st=reg[b.dataset.metric]; openModal(st.chart, vis(), fitVis(), refSet, st.logY); }));
    rebuildAll();
  }
  document.querySelectorAll(".explorer").forEach(setupExplorer);

  // resize (also re-fit after a tab becomes visible: width was 0 while display:none)
  window.ceResize = function(){
    allReg.forEach(st => { if(st.plot && st.el && st.el.clientWidth)
      st.plot.setSize({ width:st.el.clientWidth, height:SMALL_H }); });
    if(M && M.plot){ const s=mSize(); M.plot.setSize({ width:s.w, height:s.h }); }
  };
  window.addEventListener("resize", window.ceResize);
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
    # Branch stitching: a decay child resumes the trunk's global step but its first eval is logged
    # well after the fork (evals are sparse), leaving a visual gap. For each branched run, anchor an
    # extra point at the fork step equal to the parent's value there (nearest sample at/below the
    # fork), so the child's curve starts ON the parent's curve and visibly branches off it. Only
    # added where the child has no real sample at/before the fork, so real data is never overwritten.
    for r in RUNS:
        br = r.get("branch")
        if not br:
            continue
        child, parent = per_run[r["key"]], per_run.get(br["parent"], {})
        fs = br["fork_step"]
        for _, key in METRICS:
            cd = child.setdefault(key, {})
            if any(s <= fs for s in cd):
                continue  # child already has a sample at/before the fork
            pk = parent.get(key, {})
            below = [s for s in pk if s <= fs]
            above = [s for s in pk if s > fs]
            if below and above:
                # Linearly interpolate the parent between the two samples straddling the fork, so
                # the anchor lands exactly ON the parent's drawn (straight-segment) line at the fork
                # -- not on the previous sample's stale value, which sits just off the line.
                a, b = max(below), min(above)
                cd[fs] = pk[a] + (fs - a) / (b - a) * (pk[b] - pk[a])
            elif below:
                cd[fs] = pk[max(below)]
    # One chart per metric; every run is a series, in RUNS order (so series index == run index ==
    # palette index == sidebar checkbox data-idx). Runs without data for a metric are all-null.
    charts = []
    for title, key in METRICS:
        steps = sorted(set().union(*[set(per_run[r["key"]].get(key, {})) for r in RUNS]) or {0})
        series = [{"label": r["label"], "y": [per_run[r["key"]].get(key, {}).get(s) for s in steps]}
                  for r in RUNS]
        charts.append({"key": slug(title), "title": title, "x": steps, "series": series})
    return {"runs": runs_meta, "charts": charts}


def chart_blocks(charts: list, tab_id: str) -> str:
    cells = []
    for c in charts:
        cells.append(
            f'<div class="chart-cell" data-metric="{c["key"]}">'
            f'<h4>{c["title"]}</h4>'
            '<div class="chart-controls">'
            f'<button class="logtoggle" data-metric="{c["key"]}">log-y</button>'
            f'<button class="expand" data-metric="{c["key"]}">expand &#10530;</button>'
            '</div>'
            f'<div class="ce-chart" id="chart-{tab_id}-{c["key"]}"></div>'
            '</div>'
        )
    return f'<div class="chart-grid">{"".join(cells)}</div>'


def explorer_html(tab: dict, charts: list) -> str:
    """One explorer: left column of recipe-group toggles + right shared chart grid. Group buttons
    carry data-runs (comma-joined run keys); the JS unions selected groups into chart visibility."""
    default = set(tab.get("default", []))
    btns = []
    for i, g in enumerate(tab["groups"]):
        pressed = "true" if i in default else "false"
        ref = ' data-ref="1"' if g.get("ref") else ""
        cls = "exp-group ref" if g.get("ref") else "exp-group"
        btns.append(
            f'<button class="{cls}" aria-pressed="{pressed}" data-runs="{",".join(g["runs"])}"{ref}>'
            f'<span class="exp-dot"></span><span>{g["name"]}</span></button>')
    panel = (
        '<div class="exp-panel">'
        '<div class="exp-h">Recipe</div>'
        f'{"".join(btns)}'
        '<label class="exp-fitx-l"><input type="checkbox" class="exp-fitx" checked>'
        ' auto-fit x to selection</label>'
        '</div>')
    return (f'<div class="explorer" data-tab="{tab["id"]}">'
            f'{panel}<div class="exp-charts">{chart_blocks(charts, tab["id"])}</div></div>')


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

    # Overview: intro + common setup + one run-table card per category.
    overview_cards = [
        '<div class="card goal"><h3>Goal</h3>'
        '<p>A standard top-k MoE (<code>moe_lbreducedp_sharedexp</code>) study. '
        '<strong>Baselines</strong> (now settled): the WSD scheduler, the peak-LR sweep, and '
        'decay-amount ablations, plus 64&rarr;128 expert scaling. <strong>Extension methods</strong> '
        '(in progress): ways to grow an already-trained model &mdash; starting with expert '
        'upcycling. Use the <strong>LR scheduling</strong> and <strong>Extension methods</strong> '
        'tabs: pick a recipe on the left and its runs (a trunk plus all of its decay branches) plot '
        'on the right.</p></div>'
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

    # One explorer tab per TABS entry (intro card + left recipe selector + right shared chart grid).
    tab_sections = []
    for t in TABS:
        body = (f'<div class="card goal"><h3>{t["title"]}</h3>'
                f'<p class="note">{t["intro"]}</p></div>'
                f'{explorer_html(t, payload["charts"])}')
        tab_sections.append((t["id"], t["label"], body))

    evals_html = eval_tab(payload["evals"])
    tabs = [("overview", "Overview", overview)] + tab_sections + [("evals", "Evals", evals_html)]
    nav = "".join(f'<button data-target="{t}">{n}</button>' for t, n, _ in tabs)
    sections = "".join(f'<section class="tab" id="{t}">{b}</section>' for t, _, b in tabs)
    charts_js = (_CHARTS_JS
                 .replace("__CHARTS__", json.dumps(payload["charts"]))
                 .replace("__RUNKEYS__", json.dumps([r["key"] for r in runs]))
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
