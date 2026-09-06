#!/usr/bin/env python3
"""Continue Dense-474M Original BS32 LR5e-4/WD0.1 from E4 through E32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_dense_474m_original_tuning as base

POLICY = "dense_474m_original_lr5e4_e32_v1"
MODE = "bs32-lr5e4-e32"
STATE_DIRECTORY = ".dense_474m_original_lr5e4_e32_v1"
EVALUATION_EPOCHS = (16, 24, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    return parser.parse_args()


def extension_coordinate(value: dict[str, Any]) -> dict[str, Any]:
    extension = value.get("bs32Lr5e4Extension")
    if not isinstance(extension, dict):
        raise ValueError("manifest is missing bs32Lr5e4Extension")
    expected = {
        "globalSequences": 32,
        "nprocPerNode": 2,
        "rankMicrobatchSequences": 16,
        "gradientAccumulationSteps": 1,
        "warmupSteps": 768,
        "learningRate": "5e-4",
        "weightDecay": "0.1",
        "sourceEpoch": 4,
        "targetEpoch": 32,
        "evaluationEpochs": list(EVALUATION_EPOCHS),
        "output": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm1b/bs32_lr5e-4_wd0.1",
    }
    mismatches = [key for key, expected_value in expected.items() if extension.get(key) != expected_value]
    if mismatches:
        raise ValueError(f"BS32 LR5e-4 extension mismatch: {mismatches}")
    expected_source = Path(str(extension["output"])) / f"step{base.stable_step(4, 32)}"
    if Path(str(extension.get("sourceCheckpoint"))) != expected_source:
        raise ValueError(f"extension must start from exact E4 PD {expected_source}")
    return {
        "id": "lr5e-4-wd0.1",
        **extension,
        "policy": POLICY,
        "stateDirectory": STATE_DIRECTORY,
    }


def run(value: dict[str, Any]) -> None:
    base.POLICY = POLICY
    coordinate = extension_coordinate(value)
    base.claim_output(value, MODE, coordinate)
    results: list[dict[str, Any]] = []
    source_epoch = int(coordinate["sourceEpoch"])
    source_checkpoint = Path(str(coordinate["sourceCheckpoint"]))
    for epoch in EVALUATION_EPOCHS:
        stage = {
            **coordinate,
            "sourceEpoch": source_epoch,
            "sourceCheckpoint": str(source_checkpoint),
            "targetEpoch": epoch,
            "fixedEpochs": [epoch],
        }
        state_path = base.state_root(stage) / "workflow.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.is_file()
            else {
                "policy": POLICY,
                "workflowId": value["id"],
                "mode": MODE,
                "coordinate": stage["id"],
                "status": "starting",
                "minRuntimeOmitted": True,
            }
        )
        base.atomic_json(state_path, state)
        base.ensure_predecay(value, MODE, stage, state)
        result = base.evaluate(value, MODE, stage, state)
        results.append(result)
        state.update({"status": "stage_complete", "activeEpoch": epoch})
        base.atomic_json(state_path, state)
        source_epoch = epoch
        source_checkpoint = Path(str(stage["output"])) / f"step{base.stable_step(epoch, 32)}"
    state.update({"status": "complete", "activeEpoch": EVALUATION_EPOCHS[-1]})
    base.atomic_json(state_path, state)
    print(
        "DENSE474M_ORIGINAL_LR5E4_E32_COMPLETE json="
        + json.dumps(
            {
                "policy": POLICY,
                "mode": MODE,
                "evaluationEpochs": list(EVALUATION_EPOCHS),
                "validationExact": {
                    str(result["epoch"]): float(result["validationExact"]) for result in results
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    print(f"DENSE474M_ORIGINAL_WORKFLOW_COMPLETE mode={MODE}", flush=True)


def main() -> None:
    args = parse_args()
    value = base.load_manifest(args.manifest)
    coordinate = extension_coordinate(value)
    if args.validate_only:
        print(f"validated {args.manifest} for {MODE}")
        return
    if args.print_plan:
        print(
            json.dumps(
                {
                    "mode": MODE,
                    "coordinate": coordinate["id"],
                    "sourceEpoch": 4,
                    "targetEpoch": 32,
                    "evaluationEpochs": list(EVALUATION_EPOCHS),
                },
                indent=2,
            )
        )
        return
    run(value)


if __name__ == "__main__":
    main()
