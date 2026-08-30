#!/usr/bin/env python3
"""Run the authorized standalone Dense-1B Pool-3B BS64 E24 evaluator."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_1b_checkpoint_evaluator as base
import run_dense_constant_checkpoint_producer as producer

PRODUCER_ID = "dense-1b-dclm3b-bs64-lr1e-3-wd0.3"
EPOCH = 24
PREVIOUS_EPOCH = 16


def load(manifest_path: Path) -> dict[str, Any]:
    manifest = producer.load_manifest(manifest_path)
    item = producer.coordinate(manifest, PRODUCER_ID)
    if (
        item["model"] != "1b"
        or item["pool"] != "dclm3b"
        or int(item["poolTokens"]) != 3_000_000_000
        or int(item["batchSequences"]) != 64
        or Decimal(str(item["learningRate"])) != Decimal("1e-3")
        or Decimal(str(item["weightDecay"])) != Decimal("0.3")
    ):
        raise ValueError("standalone E24 producer recipe mismatch")
    producer.validate_coordinate(item, int(manifest["maxEpoch"]), check_source=False)
    return item


def run(item: dict[str, Any]) -> None:
    current = base.run_epoch(item, EPOCH)
    previous_path = base.state_dir(item) / "post_decay" / f"e{PREVIOUS_EPOCH}.result.json"
    if not previous_path.is_file():
        raise FileNotFoundError(f"missing prior POST result {previous_path}")
    previous = json.loads(previous_path.read_text())
    saturated = Decimal(str(current["validationExact"])) >= Decimal(
        str(previous["validationExact"])
    )
    decision = {
        "policy": base.POLICY,
        "status": "saturated" if saturated else "improving",
        "comparisonGroup": "post_decay_only",
        "criterion": "strict_non_improvement",
        "epochs": [PREVIOUS_EPOCH, EPOCH],
        "validationExact": {
            str(PREVIOUS_EPOCH): float(previous["validationExact"]),
            str(EPOCH): float(current["validationExact"]),
        },
        "producerCancellationRequested": False,
        "producerStoppedAfterEpoch": EPOCH,
    }
    base.atomic_json(base.state_dir(item) / "decision.e24.json", decision)
    base.atomic_json(base.state_dir(item) / "decision.json", decision)
    print(
        f"DENSE1B_CHECKPOINT_EVALUATOR_COMPLETE id={item['id']} status={decision['status']} "
        f"json={json.dumps(decision, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    item = load(args.manifest)
    if args.validate_only:
        print(f"validated standalone evaluator {PRODUCER_ID} E{EPOCH}")
        return
    run(item)


if __name__ == "__main__":
    main()
