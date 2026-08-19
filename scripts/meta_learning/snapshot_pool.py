#!/usr/bin/env python3
"""Weights-only bf16 snapshot of a k32-CPT pool checkpoint (for the per-stage heatmap
evals). Equivalent to the opt-in --snapshot in expert_subset_surgery.py writeback, but
runnable out-of-band by the snapshot daemon: whenever a new wroteback marker appears and
no surgery is active, the daemon calls this to persist pool_after_cNN.pt.

Usage:
  python scripts/meta_learning/snapshot_pool.py --pool <pool_dir> --cluster 11
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--cluster", type=int, required=True)
    args = p.parse_args()

    from olmo_core.distributed.checkpoint import load_keys

    pool = Path(args.pool)
    marker = pool / f"wroteback_cluster{args.cluster:02d}.json"
    assert marker.exists(), f"no marker {marker} -- pool is not at/after this stage"
    snap_dir = pool / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    snap = snap_dir / f"pool_after_c{args.cluster:02d}.pt"
    if snap.exists():
        print(f"SKIP: {snap} exists")
        return
    (model_sd,) = list(load_keys(str(pool / "model_and_optim"), ["model"]))
    tmp = snap.with_suffix(".pt.tmp")
    torch.save({k: v.to(torch.bfloat16) for k, v in model_sd.items()}, str(tmp))
    tmp.rename(snap)
    print(f"snapshot: {snap} ({snap.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
