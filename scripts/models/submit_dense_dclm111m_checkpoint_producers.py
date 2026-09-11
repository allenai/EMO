#!/usr/bin/env python3
"""Print or guardedly submit the twelve integrated DCLM-111M trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dense_dclm111m_checkpoint_producer as runner
import submit_dense_dclm333m_checkpoint_producers as base

MANIFEST = Path("scripts/models/manifests/dense-dclm111m-checkpoint-producers-v1.json")
RUNNER = "scripts/models/run_dense_dclm111m_checkpoint_producer.py"
REGISTRY = Path("reports/0802/data/wsd_pool111m_grid.json")
REGISTRY_JS = REGISTRY.with_suffix(".js")


def configure_submitter() -> None:
    runner.configure_base()
    base.runner = runner
    base.MANIFEST = MANIFEST
    base.RUNNER = RUNNER
    base.POOL_DISPLAY_NAME = runner.POOL_DISPLAY_NAME


def load_manifest() -> dict[str, Any]:
    configure_submitter()
    return runner.load_manifest(MANIFEST)


def plan_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in config["producerCoordinates"]:
        estimate = runner.runtime_estimate(item, config["runtimeEstimate"])
        rows.append(
            {
                "id": item["id"],
                "model": item["model"],
                "batch": item["batchSequences"],
                "gpus": runner.gpu_count(item),
                "lr": item["learningRate"],
                "wd": item["weightDecay"],
                "rawSeconds": estimate["rawSeconds"],
                "bufferedSeconds": estimate["bufferedSeconds"],
            }
        )
    return rows


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def write_registry(created: list[tuple[dict[str, Any], str]], revision: str) -> None:
    records = []
    for item, experiment in created:
        records.append(
            {
                "id": item["id"],
                "model": item["model"],
                "pool": "dclm111m",
                "batchSequences": item["batchSequences"],
                "learningRate": item["learningRate"],
                "weightDecay": item["weightDecay"],
                "gpuCount": runner.gpu_count(item),
                "retainedCheckpointEpochs": item["retainedCheckpointEpochs"],
                "evaluationEpochs": item["evaluationEpochs"],
                "stopOnAdjacentPostNonImprovement": True,
                "status": "submitted",
                "experiment": experiment,
                "revision": revision,
                "output": item["output"],
            }
        )
    value = {
        "policy": runner.POLICY,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "datasetManifest": str(runner.EXPECTED_DATASET_MANIFEST),
        "scheduling": "unallocated",
        "minRuntimeOmitted": True,
        "trajectoryCount": len(records),
        "trajectories": records,
    }
    rendered = json.dumps(value, indent=2) + "\n"
    atomic_text(REGISTRY, rendered)
    atomic_text(REGISTRY_JS, "window.ICSL_POOL111M_GRID=" + rendered.rstrip() + ";\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--coordinate", action="append", default=[])
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--print-specs", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    config = load_manifest()
    selected = [
        item
        for item in config["producerCoordinates"]
        if not args.coordinate or item["id"] in args.coordinate
    ]
    if len(selected) != len(set(args.coordinate)) and args.coordinate:
        raise SystemExit("every requested coordinate must be present exactly once")
    if args.print_plan or not (args.print_specs or args.submit_if_ready):
        print(json.dumps(plan_rows(config), indent=2))
        return
    if not args.revision:
        raise SystemExit("--revision is required for spec generation or submission")
    base.validate_revision(args.revision)
    created: list[tuple[dict[str, Any], str]] = []
    for item in selected:
        configure_submitter()
        if args.print_specs:
            print(
                json.dumps(
                    base.spec_for(
                        item,
                        args.revision,
                        args.priority,
                        omit_min_runtime=True,
                    ),
                    indent=2,
                )
            )
        else:
            experiment = base.create(
                item,
                args.revision,
                args.priority,
                omit_min_runtime=True,
            )
            created.append((item, experiment))
            print(f"{item['id']}: {experiment}")
    if created:
        write_registry(created, args.revision)


if __name__ == "__main__":
    main()
