"""Export new-expert activation-fraction curves for the four extension continual-pretrain
runs (no-ghost baseline + three ghost variants) into a JSON build_report.py charts.

During the FineMath extension run the randpool router logs, per batch, what fraction of
tokens / documents route to the freshly added expert. The un-prefixed metric WandB shows
(`train/new expert {document,token} activation fraction`) is SUMMED across the 16 MoE
layers, so it ranges ~[0, 16]; divide by 16 for the per-layer fraction. We plot the raw
(summed) metric to match WandB, and annotate the layer-sum in the report.

Same JSON schema as export_ce_curves.py (runs + charts) so it reuses the report's uPlot
charting. Writes claude_outputs/models_fullextend/extension_curves.json.

    python scripts/models_fullextend/export_extension_curves.py

Requires WANDB_API_KEY in the environment.
"""

import argparse
import json
from bisect import bisect_left
from math import ceil
from pathlib import Path

import wandb

TOKENS_PER_STEP = 1024 * 4096  # global batch (tokens) -- same as pretraining
GRID_POINTS = 2000

# (wandb project, run display name, chart-series label). Order = legend/color order.
RUNS = [
    ("ryanyxw/emo-extension", "emo_1b14b_130b_noghost_extend1_finemath_frz", "no-ghost baseline"),
    ("ryanyxw/emo-extension", "emo_1b14b_130b_ghost_uniform_extend1_finemath_frz", "uniform ghost"),
    ("ryanyxw/emo-extension", "emo_1b14b_130b_ghost_usage_extend1_finemath_frz", "usage ghost"),
    ("ryanyxw/emo-extension", "emo_1b14b_130b_ghost_random_extend1_finemath_frz", "random ghost"),
]

# (chart key, wandb metric key, chart title).
METRICS = [
    ("ext_doc_activation", "train/new expert document activation fraction",
     "New-expert document activation fraction (summed over 16 layers; /16 = per-layer)"),
    ("ext_token_activation", "train/new expert token activation fraction",
     "New-expert token activation fraction (summed over 16 layers; /16 = per-layer)"),
]


def fetch(project: str, name: str):
    runs = list(wandb.Api().runs(project, filters={"display_name": name}))
    if not runs:
        return None
    r = runs[0]
    series = {}
    for _, wk, _ in METRICS:
        pairs = []
        for row in r.scan_history(keys=["_step", wk]):
            s, v = row.get("_step"), row.get(wk)
            if s is not None and v is not None:
                pairs.append((int(s), float(v)))
        series[wk] = sorted(pairs)
    return r, series


def nearest(pairs, x):
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
                    default=Path("claude_outputs/models_fullextend/extension_curves.json"))
    args = ap.parse_args()

    fetched = []  # (label, url, {wandb_key: pairs})
    for proj, name, label in RUNS:
        got = fetch(proj, name)
        if got is None:
            print(f"  skip (not found): {label}")
            continue
        r, series = got
        fetched.append((label, r.url, series))
        n = len(series.get(METRICS[0][1], []))
        print(f"  {label}: url={r.url} points={n}")
    if not fetched:
        raise SystemExit("no runs fetched")

    charts = []
    for key, wk, title in METRICS:
        present = [(label, s[wk]) for label, _, s in fetched if s.get(wk)]
        if not present:
            print(f"  metric '{key}': no data, skipping")
            continue
        all_steps = sorted({st for _, pairs in present for st, _ in pairs})
        grid = all_steps[::ceil(len(all_steps) / GRID_POINTS)] if len(all_steps) > GRID_POINTS else all_steps
        cseries = []
        for label, pairs in present:
            run_max = pairs[-1][0]
            y = [round(nearest(pairs, x), 6) if x <= run_max else None for x in grid]
            cseries.append({"label": label, "y": y})
        charts.append({"key": key, "title": title, "x": grid, "series": cseries})
        print(f"  metric '{key}': {len(cseries)} series, {len(grid)} x-points")

    out = {
        "tokens_per_step": TOKENS_PER_STEP,
        "runs": [{"label": label, "url": url} for label, url, _ in fetched],
        "charts": charts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out))
    print(f"Wrote {args.output} ({args.output.stat().st_size/1e3:.1f} KB, {len(charts)} charts)")


if __name__ == "__main__":
    main()
