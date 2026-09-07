#!/usr/bin/env python3
"""Evaluate one already-retained Pool-333M pre-decay checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_dense_dclm333m_checkpoint_producer as integrated


POLICY = "dense_dclm333m_retained_checkpoint_evaluator_v1"
TARGETS = {
    "dense-1b-dclm333m-bs64-lr1e-3-wd1.0": (4, 12, 20, 28),
    "dense-474m-dclm333m-bs64-lr2e-3-wd0.3": (8, 24, 40),
    "dense-153m-dclm333m-bs64-lr2e-3-wd0.3": (16, 48, 80, 112, 144, 176),
}


def load(manifest_path: Path, coordinate_id: str, epoch: int):
    manifest = integrated.load_manifest(manifest_path)
    item = integrated.coordinate_for_target(manifest, coordinate_id, None)
    if coordinate_id not in TARGETS or epoch not in TARGETS[coordinate_id]:
        raise ValueError(f"unauthorized retained-checkpoint evaluator {coordinate_id} E{epoch}")
    if epoch not in {int(value) for value in item["retainedCheckpointEpochs"]}:
        # The selected 474M and 153M coordinates were continued beyond their
        # original manifest ceilings; expand only the in-memory retained ladder.
        continuation_targets = integrated.authorized_continuation_targets(item)
        valid_ceiling = max(value for value in continuation_targets if value >= epoch)
        item = integrated.coordinate_for_target(manifest, coordinate_id, valid_ceiling)
    if epoch in {int(value) for value in item["evaluationEpochs"]}:
        raise ValueError(f"E{epoch} is already part of the integrated POST ladder")
    return manifest, item


def source_checkpoint(item, epoch: int) -> Path:
    return Path(str(item["output"])) / f"step{integrated.stable_step(epoch, int(item['batchSequences']))}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coordinate", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    _, item = load(args.manifest, args.coordinate, args.epoch)
    if args.validate_only:
        print(f"validated retained Pool-333M evaluator {args.coordinate} E{args.epoch}")
        return
    print(
        f"DENSE_DCLM333M_RETAINED_EVAL_START id={args.coordinate} epoch={args.epoch}",
        flush=True,
    )
    result = integrated.evaluate(item, args.epoch)
    print(
        "DENSE_DCLM333M_RETAINED_EVAL_RESULT "
        f"id={args.coordinate} epoch={args.epoch} "
        f"json={json.dumps(result, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )
    print(
        f"DENSE_DCLM333M_RETAINED_EVAL_COMPLETE id={args.coordinate} epoch={args.epoch}",
        flush=True,
    )


if __name__ == "__main__":
    main()
