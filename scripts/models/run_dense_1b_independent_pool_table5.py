#!/usr/bin/env python3
"""Run one Table-5 1.5B Original trajectory on independent pool B or C."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_dense_1b_dr_wt_embedwd_grid as base


POLICY = "dense_1b_independent_pool_table5_replication_v1"
SEQUENCE_LENGTH = 4096
PLAN = Path("scripts/models/manifests/dense-1b-independent-pools-table5-v1.json")


def coordinate(plan: dict[str, Any], pool: str, batch: int) -> dict[str, Any]:
    matches = [
        item
        for item in plan["trajectories"]
        if item["pool"] == pool and int(item["batchSequences"]) == batch
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one pool-{pool}/BS{batch} trajectory")
    return matches[0]


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("policy") != POLICY:
        raise ValueError(f"plan policy must be {POLICY}")
    expected = {
        (pool, batch)
        for pool in ("b", "c")
        for batch in (32, 64, 128, 256)
    }
    actual = {(item["pool"], int(item["batchSequences"])) for item in plan["trajectories"]}
    if actual != expected or len(plan["trajectories"]) != 8:
        raise ValueError("plan must contain exactly the 2 pools x 4 Table-5 batches")
    table5 = {
        32: ("5e-4", "0.3", 20),
        64: ("1e-3", "0.3", 20),
        128: ("1e-3", "0.3", 16),
        256: ("2e-3", "0.333", 12),
    }
    outputs = set()
    for item in plan["trajectories"]:
        batch = int(item["batchSequences"])
        if (str(item["lr"]), str(item["wd"]), int(item["targetEpoch"])) != table5[batch]:
            raise ValueError(f"pool-{item['pool']}/BS{batch} does not match Table 5")
        if item.get("variant") != "Original":
            raise ValueError("all replications must use the Original recipe")
        if item.get("dataOrder") != "ordinary_shuffled":
            raise ValueError("Original replication must use ordinary shuffled loading")
        if item.get("weightTying") is not False or item.get("decayEmbeddings") is not False:
            raise ValueError("Original replication must be untied with zero embedding WD")
        if int(item["gpuCount"]) != 8 or int(item["nprocPerNode"]) != 8:
            raise ValueError("every trajectory must use one full 8-GPU node")
        if int(item["rankMicrobatchSequences"]) != 4:
            raise ValueError("Table-5 replication requires rank microbatch four")
        if int(item["gradientAccumulation"]) != batch // 32:
            raise ValueError("gradient accumulation does not produce the requested batch")
        if int(item["warmupSteps"]) != 24_576 // batch:
            raise ValueError("warmup is not token-matched to the original sweep")
        if not item.get("checkpointEveryEpoch"):
            raise ValueError("unallocated trajectories must checkpoint every epoch")
        output = str(item["output"])
        if output in outputs:
            raise ValueError(f"duplicate output writer {output}")
        outputs.add(output)


def base_config(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "globalSequences": int(item["batchSequences"]),
        "nprocPerNode": 8,
        "rankMicrobatchSequences": 4,
        "gradientAccumulation": int(item["gradientAccumulation"]),
        "warmupSteps": int(item["warmupSteps"]),
        "coordinates": [
            {"lr": str(item["lr"]), "wd": str(item["wd"]), "output": str(item["output"])}
        ],
        "initialTargets": [int(item["targetEpoch"])],
        "epochIncrement": 4,
        "maxEpoch": int(item["targetEpoch"]),
        "outputRoot": str(Path(item["output"]).parent),
        "runSuffix": f"independent-pool-{item['pool']}-table5-bs{item['batchSequences']}-v1",
        "variant": "Original",
        "policy": POLICY,
        "comparisonPolicy": "terminal_post_decay_only",
        "checkpointOnlyEpochs": [],
        "postDecayStartEpoch": int(item["targetEpoch"]),
        "postDecayEvaluation": "reported_table5_terminal_epoch_only",
        "postDecaySourceCount": 1,
        "postDecaySaturationCriterion": "not_applicable_fixed_horizon",
        "checkpointEveryEpoch": True,
        "hardStopAtMaxEpoch": True,
    }


def configure_base(item: dict[str, Any]) -> None:
    base.POLICY = POLICY
    base.POST_DECAY_START_EPOCH = int(item["targetEpoch"])
    base.CHECKPOINT_ONLY_EPOCHS = []
    base.POST_DECAY_SOURCE_COUNT = 1
    base.OUTPUT_ROOT = str(Path(item["output"]).parent)
    base.COMMON_ARGUMENTS = tuple(
        (
            f"--dataset.subset_manifest={item['dataManifest']}"
            if argument.startswith("--dataset.subset_manifest=")
            else argument
        )
        for argument in base.COMMON_ARGUMENTS
    )


def write_terminal_selection(item: dict[str, Any], result: dict[str, Any]) -> None:
    config = base_config(item)
    path = base.selection_path(config, str(item["lr"]), str(item["wd"]))
    value = {
        "status": "complete",
        "policy": POLICY,
        "trigger": "reported_table5_horizon",
        "triggerEpoch": int(item["targetEpoch"]),
        "selectedPostDecayEpoch": int(item["targetEpoch"]),
        "selectedPostDecayValidationExact": result["validationExact"],
        "selectedCheckpoint": result["checkpoint"],
        "pool": item["pool"],
        "batchSequences": int(item["batchSequences"]),
        "completedAt": datetime.now(timezone.utc).isoformat(),
    }
    base.atomic_json(path, value)
    print(
        "DENSE1B_INDEPENDENT_TABLE5_COMPLETE "
        f"pool={item['pool']} bs={item['batchSequences']} epoch={item['targetEpoch']} "
        f"validation={result['validationExact']} output={item['output']}",
        flush=True,
    )


def run(item: dict[str, Any]) -> None:
    configure_base(item)
    config = base_config(item)
    lr, wd = str(item["lr"]), str(item["wd"])
    selection = base.selection_path(config, lr, wd)
    if selection.is_file():
        state = json.loads(selection.read_text())
        if state.get("status") == "complete":
            print(
                "DENSE1B_INDEPENDENT_TABLE5_ALREADY_COMPLETE "
                f"pool={item['pool']} bs={item['batchSequences']}"
            )
            return
    epoch = int(item["targetEpoch"])
    base.train_predecay(config, lr, wd, None, epoch)
    result = base.run_postdecay(config, lr, wd, epoch)
    write_terminal_selection(item, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--pool", choices=("b", "c"), required=True)
    parser.add_argument("--batch", type=int, choices=(32, 64, 128, 256), required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    item = coordinate(plan, args.pool, args.batch)
    if args.validate_only:
        print(json.dumps(item, indent=2))
        return
    run(item)


if __name__ == "__main__":
    main()
