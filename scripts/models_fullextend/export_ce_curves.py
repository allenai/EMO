"""Export CE-loss curves for the models_fullextend ghost runs + the no-ghost baseline
into a single JSON that build_report.py embeds as an interactive (uPlot) chart.

Pulls each run's full train/CE-loss history from WandB, resamples every series onto
one shared step grid (so uPlot can plot them together), and writes
``claude_outputs/models_fullextend/ce_curves.json``.

Re-run this whenever you want the report's chart refreshed with the latest steps
(e.g. as sweep configs progress / finish), then rebuild + publish the report.

    python scripts/models_fullextend/export_ce_curves.py

Requires WANDB_API_KEY in the environment.
"""

import argparse
import json
from bisect import bisect_left
from pathlib import Path

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
    (
        "ryanyxw/emo-extension",
        "emo_1b14b_130b_ghost_usage_always_detachF",
        "usage / always / detachF",
    ),
    (
        "ryanyxw/emo-extension",
        "emo_1b14b_130b_ghost_uniform_always_detachF",
        "uniform / always / detachF",
    ),
]

GRID_POINTS = 2000  # shared x-grid resolution


def fetch(project: str, name: str):
    runs = list(wandb.Api().runs(project, filters={"display_name": name}))
    if not runs:
        return None
    r = runs[0]
    steps, ces = [], []
    for row in r.scan_history(keys=["_step", "train/CE loss"]):
        s, v = row.get("_step"), row.get("train/CE loss")
        if s is None or v is None:
            continue
        steps.append(int(s))
        ces.append(float(v))
    if not steps:
        return None
    pairs = sorted(zip(steps, ces))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def nearest(steps, ces, x):
    """CE at the step nearest to x (steps sorted ascending)."""
    i = bisect_left(steps, x)
    if i == 0:
        return ces[0]
    if i >= len(steps):
        return ces[-1]
    return ces[i] if (steps[i] - x) < (x - steps[i - 1]) else ces[i - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=Path("claude_outputs/models_fullextend/ce_curves.json"))
    args = ap.parse_args()

    fetched = []
    for proj, name, label in RUNS:
        cur = fetch(proj, name)
        if cur is None:
            print(f"  skip (not found / no data): {label}")
            continue
        fetched.append((label, name, cur))
        print(f"  {label}: {len(cur[0])} points, steps {cur[0][0]}..{cur[0][-1]}")

    if not fetched:
        raise SystemExit("no runs fetched")

    max_step = max(cur[0][-1] for _, _, cur in fetched)
    n = min(GRID_POINTS, max_step)
    grid = [round(1 + i * (max_step - 1) / (n - 1)) for i in range(n)]

    series = []
    for label, name, (steps, ces) in fetched:
        run_max = steps[-1]
        y = [round(nearest(steps, ces, x), 5) if x <= run_max else None for x in grid]
        series.append({"label": label, "run": name, "max_step": run_max, "y": y})

    out = {
        "metric": "train/CE loss",
        "tokens_per_step": TOKENS_PER_STEP,
        "x": grid,
        "series": series,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out))
    print(f"Wrote {args.output} ({args.output.stat().st_size/1e3:.1f} KB, "
          f"{len(series)} series, {len(grid)} x-points)")


if __name__ == "__main__":
    main()
