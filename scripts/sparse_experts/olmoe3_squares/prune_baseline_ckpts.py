#!/usr/bin/env python3
"""Prune the joint-training baselines of the random-pool controls to ~5 checkpoints once a checkpoint's evaluations exist (user
approval 2026-09-23: "after evaluating the 10 fixed checkpoints, keep ~5"). A fixed checkpoint is deleted when it is not in KEEP and
both its held-out pass (meta.json) and its v3-small ppl json exist. Idempotent; the monitor runs it every loop.
  python scripts/sparse_experts/olmoe3_squares/prune_baseline_ckpts.py [--dry-run]
"""
import shutil, sys
from pathlib import Path

S = Path("sparse_experts"); DRY = "--dry-run" in sys.argv
KEEP = {20000, 30000, 38148, 49073, 57221}   # both window ends + an even spread
FIXED = [20000, 25000, 30000, 35000, 38148, 39073, 44073, 49073, 54073, 57221]
# run dir | held-out baseline dir | ppl json dir
RUNS = [("olmoe3_275m_randsel_30b", "runs_heldout300b_randsel", "olmoe3_squares_randsel4/ppl_validation/olmoe3_275m_randsel_30b"),
        ("olmoe3_275m_randsel64_30b", "runs_heldout300b_randsel64", "olmoe3_squares_randsel64k4/ppl_validation/olmoe3_275m_randsel64_30b")]

freed = 0
for run, hrb, ppl in RUNS:
    for st in FIXED:
        d = S / run / f"step{st}"
        if st in KEEP or not d.exists(): continue
        if not (S / "olmoe3_routing" / hrb / f"baseline_step{st}/none/meta.json").exists() or not (S / ppl / f"step{st}.json").exists(): continue
        size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()); freed += size
        print(f"  {'would delete' if DRY else 'deleting'} {d} ({size / 1e9:.0f} GB, evaluated)")
        if not DRY: shutil.rmtree(d)
print(f"prune_baseline_ckpts: {'would free' if DRY else 'freed'} {freed / 1e9:.0f} GB")
