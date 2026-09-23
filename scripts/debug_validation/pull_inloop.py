#!/usr/bin/env python3
"""Pull the in-loop v3-small ppl validation curves (eval/lm/<set>-validation/CE loss) and the train/CE loss curve of
debug_validation runs from W&B into claude_outputs/debug_validation/ppl_validation/inloop_beta_arms.json
({run_name: {step: {set: ce}}}) and train_ce_curves.json ({arm: {run, id, train_ce: [[step, ce], ...]}}), the files
plot_validation.py / build_report.py read. A run name can have many W&B runs (every relaunch / twin logs its own); their rows
are merged by step, later runs win.
  python scripts/debug_validation/pull_inloop.py randsel:olmoe3_275m_emo_randsel_10b [more arm:run_name]
"""
import json, sys
from pathlib import Path
import wandb

OUT = Path("claude_outputs/debug_validation/ppl_validation/inloop_beta_arms.json")
CURVES = Path("claude_outputs/debug_validation/ppl_validation/train_ce_curves.json")
SETS = ["c4_en", "dolma_books", "dolma_common-crawl", "dolma_pes2o", "dolma_reddit", "dolma_stack", "dolma_wiki", "ice", "m2d2_s2orc", "pile", "wikitext_103"]
KEYS = {s: f"eval/lm/{s}-validation/CE loss" for s in SETS}


def pull(name):
    api = wandb.Api(timeout=180)
    runs = sorted(api.runs("ryanyxw/emo-extension", filters={"display_name": name}), key=lambda r: r.created_at)
    out, train = {}, {}
    for r in runs:
        for row in r.scan_history(page_size=10000):   # keys= would require every key in a row; eval rows carry only their own set
            st = row.get("_step")
            if st is None: continue
            vals = {s: row[k] for s, k in KEYS.items() if row.get(k) is not None}
            if vals: out.setdefault(str(int(st)), {}).update(vals)
            if row.get("train/CE loss") is not None: train[int(st)] = row["train/CE loss"]
    print(f"{name}: {len(runs)} W&B runs, {len(out)} eval steps, sets/step {sorted({len(v) for v in out.values()})}, last step {max(map(int, out)) if out else None}, {len(train)} train-CE points")
    return out, dict(run=name, id=runs[-1].id if runs else None, train_ce=sorted(train.items()))


if __name__ == "__main__":
    data = json.load(open(OUT)) if OUT.exists() else {}
    curves = json.load(open(CURVES)) if CURVES.exists() else {}
    for spec in sys.argv[1:]:
        arm, name = spec.split(":", 1) if ":" in spec else (spec, spec)
        data[name], curves[arm] = pull(name)
    json.dump(data, open(OUT, "w"), indent=1); json.dump(curves, open(CURVES, "w"))
    print("wrote", OUT, "runs:", list(data), "| curves arms:", list(curves))
