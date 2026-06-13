"""Export training/eval curves for the models_fullextend ghost runs + the no-ghost
baseline into a single JSON that build_report.py embeds as interactive (uPlot) charts.

Pulls each run's history for a set of metrics from WandB, resamples every series onto
a per-metric shared step grid, records each run's WandB URL, and writes
``claude_outputs/models_fullextend/ce_curves.json``.

Re-run whenever you want the report's charts refreshed (e.g. as sweep configs
progress / finish), then rebuild + publish the report.

    python scripts/models_fullextend/export_ce_curves.py

Requires WANDB_API_KEY in the environment.
"""

import argparse
import json
from bisect import bisect_left
from math import ceil
from pathlib import Path
from statistics import median

import wandb

TOKENS_PER_STEP = 1024 * 4096  # global batch (tokens)

# (wandb project, run display name, chart label). Missing runs are skipped, so new
# sweep configs can be added here before they exist.
RUNS = [
    (
        "ryanyxw/olmoe-modular",
        "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301",
        "no-ghost baseline (128e)",
    ),
    ("ryanyxw/emo-extension", "emo_1b14b_130b_ghost_usage_always_detachF", "usage / always / detachF"),
    ("ryanyxw/emo-extension", "emo_1b14b_130b_ghost_uniform_always_detachF", "uniform / always / detachF"),
    ("ryanyxw/emo-extension", "emo_1b14b_130b_ghost_random_always_detachF", "random / always / detachF"),
]

# (chart key, wandb metric key, chart title, make_chart). make_chart=False metrics are
# still fetched (for the speed summary) but not drawn -- e.g. MFU, whose convention
# differs in the older baseline project so it can't share an axis.
METRICS = [
    ("ce", "train/CE loss", "CE loss", True),
    ("grad_norm", "optim/total grad norm", "Grad norm", True),
    ("lb", "train/load balancing loss", "Load-balancing loss", True),
    ("unique_experts", "train/unique experts used per batch", "Unique experts used / batch", True),
    ("hellaswag", "eval/downstream/hellaswag (soft loss v2)", "HellaSwag (soft loss v2)", True),
    ("arc", "eval/downstream/arc_challenge (soft loss v2)", "ARC-Challenge (soft loss v2)", True),
    ("tps", "throughput/device/TPS", "Throughput (tokens/sec/device)", True),
    ("mfu", "throughput/device/MFU", "MFU (%)", False),
]

# Steady-state speed summary: median over steps >= this (skips compile/warmup).
SPEED_WARMUP_STEPS = 1000

GRID_POINTS = 2000  # max shared x-grid resolution per metric


def fetch(project: str, name: str):
    """Return (run, {wandb_key: [(step, val), ...]}) or None.

    One scan_history per metric (single-key): a multi-key scan over metrics of
    different cadences collapses a *running* run to the sparsest metric's steps.
    """
    runs = list(wandb.Api().runs(project, filters={"display_name": name}))
    if not runs:
        return None
    r = runs[0]
    series = {}
    for _, wk, *_ in METRICS:
        pairs = []
        for row in r.scan_history(keys=["_step", wk]):
            s, v = row.get("_step"), row.get(wk)
            if s is not None and v is not None:
                pairs.append((int(s), float(v)))
        series[wk] = pairs
    return r, series


def nearest(pairs, x):
    """val at the step nearest to x; pairs is sorted [(step,val)]."""
    steps = [p[0] for p in pairs]
    i = bisect_left(steps, x)
    if i == 0:
        return pairs[0][1]
    if i >= len(pairs):
        return pairs[-1][1]
    return pairs[i][1] if (steps[i] - x) < (x - steps[i - 1]) else pairs[i - 1][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=Path("claude_outputs/models_fullextend/ce_curves.json"))
    args = ap.parse_args()

    fetched = []  # (label, url, {wandb_key: sorted pairs})
    for proj, name, label in RUNS:
        got = fetch(proj, name)
        if got is None:
            print(f"  skip (not found): {label}")
            continue
        r, series = got
        series = {k: sorted(v) for k, v in series.items()}
        fetched.append((label, r.url, series))
        ce = series.get("train/CE loss", [])
        print(f"  {label}: url={r.url} ce_steps={len(ce)}"
              + (f" last_step={ce[-1][0]}" if ce else ""))
    if not fetched:
        raise SystemExit("no runs fetched")

    charts = []
    for key, wk, title, make_chart in METRICS:
        if not make_chart:
            continue
        present = [(label, s[wk]) for label, _, s in fetched if s.get(wk)]
        if not present:
            print(f"  metric '{key}': no data, skipping")
            continue
        all_steps = sorted({st for _, pairs in present for st, _ in pairs})
        if len(all_steps) > GRID_POINTS:
            stride = ceil(len(all_steps) / GRID_POINTS)
            grid = all_steps[::stride]
        else:
            grid = all_steps
        cseries = []
        for label, pairs in present:
            run_max = pairs[-1][0]
            y = [round(nearest(pairs, x), 6) if x <= run_max else None for x in grid]
            cseries.append({"label": label, "y": y})
        charts.append({"key": key, "title": title, "x": grid, "series": cseries})
        print(f"  metric '{key}': {len(cseries)} series, {len(grid)} x-points")

    # Steady-state speed summary (median of per-step TPS/MFU after warmup).
    def steady(pairs):
        vals = [v for s, v in pairs if s >= SPEED_WARMUP_STEPS] or [v for _, v in pairs]
        return round(median(vals), 2) if vals else None

    speed = []
    for label, _, s in fetched:
        tps = s.get("throughput/device/TPS", [])
        mfu = s.get("throughput/device/MFU", [])
        speed.append({"label": label, "tps": steady(tps) if tps else None,
                      "mfu": steady(mfu) if mfu else None})
        print(f"  speed[{label}]: TPS={speed[-1]['tps']} MFU={speed[-1]['mfu']}")

    out = {
        "tokens_per_step": TOKENS_PER_STEP,
        "runs": [{"label": label, "url": url} for label, url, _ in fetched],
        "charts": charts,
        "speed": speed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out))
    print(f"Wrote {args.output} ({args.output.stat().st_size/1e3:.1f} KB, {len(charts)} charts)")


if __name__ == "__main__":
    main()
