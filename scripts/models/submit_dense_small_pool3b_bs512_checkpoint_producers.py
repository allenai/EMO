#!/usr/bin/env python3
"""Guardedly submit and register the four one-node small-model Pool-3B BS512 producers."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_dense_small_pool3b_checkpoint_producer as runner
import submit_dense_small_pool3b_checkpoint_producers as base

MANIFEST = Path("scripts/models/manifests/dense-small-pool3b-bs512-checkpoint-producers-v1.json")
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")


def load_manifest() -> dict[str, Any]:
    config = runner.load_manifest(MANIFEST)
    for item in config["producerCoordinates"]:
        runner.validate_coordinate(config, item, check_filesystem=False)
    return config


def guarded_name(item: dict[str, Any]) -> str:
    return f"{item['id']}-constant-producer-v1"


def spec_for(item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    previous_manifest = base.MANIFEST
    try:
        base.MANIFEST = MANIFEST
        spec = base.spec_for(item, revision, priority)
    finally:
        base.MANIFEST = previous_manifest
    task = spec["tasks"][0]
    task["resources"]["gpuCount"] = int(item["gpuCount"])
    task["context"]["minRuntime"] = "0s"
    spec["description"] = (
        f"Dense-{item['model']} DCLM-3B BS512 LR{item['learningRate']} "
        f"WD{item['weightDecay']} DR+WT+EmbedWD constant-LR checkpoint producer; "
        f"one node/{item['gpuCount']} GPUs, rank microbatch "
        f"{item['rankMicrobatchSequences']} sequences, gradient accumulation "
        f"{item['gradientAccumulationSteps']}; retain every "
        f"{item['epochIncrement']} epochs for independent decay/evaluation."
    )
    return spec


def create(item: dict[str, Any], revision: str, priority: str) -> str:
    name = guarded_name(item)
    existing = base.existing_named_experiment(name)
    if existing:
        return existing
    output = base.command(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", base.WORKSPACE],
        input_text=json.dumps(spec_for(item, revision, priority)),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError(f"submission returned no experiment ID for {name}")
    return identifiers[0]


def register_report(config: dict[str, Any], experiments: dict[str, str], revision: str) -> None:
    report = json.loads(REPORT.read_text())
    producers = report.setdefault("producers", [])
    by_id = {str(record["id"]): record for record in producers}
    for item in config["producerCoordinates"]:
        coordinate_id = str(item["id"])
        if coordinate_id in by_id:
            record = by_id[coordinate_id]
            if str(record.get("experiment")) != experiments[coordinate_id]:
                raise RuntimeError(f"registered experiment mismatch for {coordinate_id}")
            continue
        record = {
            "id": coordinate_id,
            "role": "constant_lr_checkpoint_producer",
            "policy": str(config["policy"]),
            "model": item["model"],
            "pool": "dclm3b",
            "sourcePool": "dclm1b",
            "batchSequences": item["batchSequences"],
            "gpuCount": item["gpuCount"],
            "rankMicrobatchSequences": item["rankMicrobatchSequences"],
            "gradientAccumulationSteps": item["gradientAccumulationSteps"],
            "learningRate": item["learningRate"],
            "weightDecay": item["weightDecay"],
            "status": "submitted",
            "sourceEpoch": 1,
            "sourceExperiment": item["sourceExperiment"],
            "sourceCheckpoint": item["sourceCheckpoint"],
            "output": item["output"],
            "resolvedCheckpointEpochs": [],
            "targetEpochs": runner.target_epochs(item, int(config["maxEpoch"])),
            "currentEpoch": 1,
            "currentPhase": "fresh_2b_bridge_to_predecay_e1",
            "evaluationEnabled": False,
            "experiment": experiments[coordinate_id],
            "revision": revision,
            "submittedAt": datetime.now(tz=UTC).isoformat(),
            "beakerStatus": "submitted",
            "wandbHealth": {
                "status": "pending",
                "beakerState": "submitted",
                "criticalSignals": [],
            },
            "earlyStopComparison": {
                "baselineBatchSequences": 256,
                "metric": "validationExact",
                "matchedEpoch": True,
                "requireHealthyPost": True,
            },
        }
        producers.append(record)
        by_id[coordinate_id] = record
    report["producerCount"] = len(producers)
    report["smallPool3bBs512Policy"] = str(config["policy"])
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_CHECKPOINT_PRODUCER_GRID=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--print-specs", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    config = load_manifest()
    if args.print_plan:
        for item in config["producerCoordinates"]:
            print(json.dumps(base.plan_row(config, item), sort_keys=True))
        return
    if not args.revision:
        raise SystemExit("--revision is required for spec generation or submission")
    base.validate_revision(args.revision)
    registrations: dict[str, str] = {}
    for item in config["producerCoordinates"]:
        if args.print_specs:
            print(json.dumps(spec_for(item, args.revision, args.priority), indent=2))
        elif args.submit_if_ready:
            experiment = create(item, args.revision, args.priority)
            registrations[str(item["id"])] = experiment
            print(f"{item['id']}: {experiment}")
        else:
            print(f"{item['id']}: ready; pass --submit-if-ready")
    if args.submit_if_ready:
        register_report(config, registrations, args.revision)


if __name__ == "__main__":
    main()
