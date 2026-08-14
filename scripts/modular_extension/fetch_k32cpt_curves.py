#!/usr/bin/env python3
"""Fetch W&B train-CE histories for every k=32 CPT stage run + the 130B baseline,
into a JSON cache the report's uPlot explorer renders (display pattern per
scripts/models_v2/build_report.py).

Each stage run (k32cpt_<arm>_c<XX>) trains the 33-expert subset on ONE cluster's data,
starting at trainer step 0; x = tokens trained within the stage. The baseline run
(emo64_100b130b_baseline) resumed the 64-expert model at step 23842; its x = tokens
since the 100B fork, so all curves share "tokens trained in this run" semantics.
Values are lightly EMA-smoothed (alpha 0.85) over W&B's ~600-sample history.

Cache: modular_extension/k32_cpt_runs/evals/train_curves.json
Reruns skip runs already cached in a terminal state (finished/failed/crashed).

Run:  python scripts/modular_extension/fetch_k32cpt_curves.py
"""
import json
from pathlib import Path

import wandb

OUT = Path("/root/EMO/modular_extension/k32_cpt_runs/evals/train_curves.json")
ENTITY_PROJECT = "ryanyxw/emo-extension"
TOK_PER_STEP = 1024 * 4096
BASELINE = "emo64_100b130b_baseline"
BASELINE_START_STEP = 23842
ARMS = ["carry", "carry_shuf", "fresh"]
KEY = "train/CE loss"
EMA = 0.85


def smooth(pairs):
    out, m = [], None
    for s, v in pairs:
        m = v if m is None else EMA * m + (1 - EMA) * v
        out.append((s, round(m, 4)))
    return out


def fetch_history(run):
    pairs = []
    for row in run.history(keys=[KEY], samples=600, pandas=False):
        v, s = row.get(KEY), row.get("_step")
        if v is None or s is None:
            continue
        pairs.append((int(s), float(v)))
    return sorted(pairs)


def main():
    cache = json.loads(OUT.read_text()) if OUT.exists() else {"runs": {}}
    api = wandb.Api()
    wanted = [(BASELINE, BASELINE)] + [
        (f"k32cpt_{arm}_c{c:02d}", arm) for arm in ARMS for c in range(32)
    ]
    by_name = {}
    for r in api.runs(ENTITY_PROJECT):
        if r.name in {w[0] for w in wanted} and r.name not in by_name:
            by_name[r.name] = r  # newest first; keep the latest run per name

    n_new = 0
    for name, arm in wanted:
        prev = cache["runs"].get(name)
        if prev and prev["state"] in ("finished", "failed", "crashed"):
            continue
        run = by_name.get(name)
        if run is None:
            continue
        pairs = smooth(fetch_history(run))
        start = BASELINE_START_STEP if name == BASELINE else 0
        cache["runs"][name] = {
            "arm": arm,
            "state": run.state,
            "tokens_b": [round((s - start) * TOK_PER_STEP / 1e9, 4) for s, _ in pairs],
            "ce": [v for _, v in pairs],
        }
        n_new += 1
        print(f"fetched {name} ({run.state}, {len(pairs)} pts)", flush=True)

    OUT.write_text(json.dumps(cache))
    done = {a: sum(1 for v in cache["runs"].values()
                   if v["arm"] == a and v["state"] == "finished") for a in ARMS}
    print(json.dumps({"newly_fetched": n_new, "finished_stage_runs": done,
                      "baseline": cache["runs"].get(BASELINE, {}).get("state")}))


if __name__ == "__main__":
    main()
