#!/usr/bin/env python3
"""Delete the per-document arrays of MERGED held-out routing passes (user approval 2026-09-23): doc_scores.npy, doc_usage.npy,
raw_topk.npy and cross.npy under sparse_experts/olmoe3_routing/runs_heldout*/<tag>/none/ (and none/rank0/). Kept: meta.json, doc_stats.npz,
counts.npz, docs.npz, DONE (everything the reports read). Passes without meta.json (not merged yet / in flight) are left alone, and
the two folders the routing-similarity analysis reads (KEEP_DIRS) are kept whole. Idempotent; the monitor runs it every loop.
  python scripts/sparse_experts/olmoe3_squares/cleanup_heldout_arrays.py [--dry-run]
"""
import sys
from pathlib import Path

R = Path("sparse_experts/olmoe3_routing"); DRY = "--dry-run" in sys.argv
KEEP_DIRS = {"runs_heldout300b_std", "runs_heldout300b_stdrand"}   # routing_similarity.py inputs (std k=4 control + its baseline)
ARRAYS = ("doc_scores.npy", "doc_usage.npy", "raw_topk.npy", "cross.npy")

freed = n = 0
for hr in sorted(R.glob("runs_heldout*")):
    if hr.name in KEEP_DIRS or not hr.is_dir(): continue
    for none in hr.glob("*/none"):
        if not (none / "meta.json").exists(): continue
        for f in ARRAYS:
            for p in (none / f, none / "rank0" / f):   # merge_routing consolidates rank0/ into none/; older passes keep both layouts
                if p.exists():
                    freed += p.stat().st_size; n += 1
                    if not DRY: p.unlink()
print(f"cleanup_heldout_arrays: {'would delete' if DRY else 'deleted'} {n} files, {freed / 1e12:.2f} TB")
