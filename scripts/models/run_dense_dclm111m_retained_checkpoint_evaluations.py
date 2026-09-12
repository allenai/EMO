#!/usr/bin/env python3
"""Decay and evaluate the retained 153M Pool-111M E40/E48/E56 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_dense_dclm111m_checkpoint_producer as pool111m


POLICY = "dense_dclm111m_retained_checkpoint_e40_e48_e56_v1"
COORDINATE = "dense-153m-dclm111m-bs64-lr2e-3-wd0.3"
EPOCHS = (40, 48, 56)


def load(manifest_path: Path):
    pool111m.configure_base()
    manifest = pool111m.load_manifest(manifest_path)
    item = pool111m.coordinate_for_target(manifest, COORDINATE, None)
    retained = {int(epoch) for epoch in item["retainedCheckpointEpochs"]}
    if not set(EPOCHS).issubset(retained):
        raise ValueError("the retained-checkpoint ladder does not include E40/E48/E56")
    pool111m.validate_coordinate(item)
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    item = load(args.manifest)
    if args.validate_only:
        print(f"validated {COORDINATE} retained evaluations {EPOCHS}")
        return
    for epoch in EPOCHS:
        print(
            f"DENSE_DCLM111M_RETAINED_EVAL_START id={COORDINATE} epoch={epoch}",
            flush=True,
        )
        result = pool111m.base.evaluate(item, epoch)
        print(
            "DENSE_DCLM111M_RETAINED_EVAL_RESULT "
            f"id={COORDINATE} epoch={epoch} "
            f"json={json.dumps(result, separators=(',', ':'), sort_keys=True)}",
            flush=True,
        )
    print(
        f"DENSE_DCLM111M_RETAINED_EVAL_COMPLETE id={COORDINATE} epochs={EPOCHS}",
        flush=True,
    )


if __name__ == "__main__":
    main()
