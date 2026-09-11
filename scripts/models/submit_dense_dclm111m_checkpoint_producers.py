#!/usr/bin/env python3
"""Print or guardedly submit the twelve integrated DCLM-111M trajectories."""

from __future__ import annotations

import argparse
import json
import os
import re
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
                "minRuntimeSeconds": min(
                    int(estimate["minRuntimeSeconds"]), base.MAX_MIN_RUNTIME_SECONDS
                ),
            }
        )
    return rows


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def allocated_spec(
    item: dict[str, Any], revision: str, priority: str
) -> tuple[dict[str, Any], int]:
    spec = base.spec_for(item, revision, priority, omit_min_runtime=True)
    estimate = runner.runtime_estimate(item, load_manifest()["runtimeEstimate"])
    min_runtime = min(
        int(estimate["minRuntimeSeconds"]), base.MAX_MIN_RUNTIME_SECONDS
    )
    spec["tasks"][0]["context"]["minRuntime"] = f"{min_runtime}s"
    spec["description"] = re.sub(
        r"minRuntime omitted\.$", f"minRuntime={min_runtime}s.", spec["description"]
    )
    return spec, min_runtime


def create_allocated(item: dict[str, Any], revision: str, priority: str) -> str:
    spec, _ = allocated_spec(item, revision, priority)
    name = f"{item['id']}-integrated-producer-eval-allocated-v2"
    output = base.command(
        [
            "beaker",
            "experiment",
            "create",
            "-",
            "--name",
            name,
            "--workspace",
            base.WORKSPACE,
        ],
        input_text=json.dumps(spec),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError(f"submission returned no experiment ID for {name}")
    return identifiers[0]


def write_registry(
    created: list[tuple[dict[str, Any], str]], revision: str, *, allocated: bool
) -> None:
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
                "minRuntimeSeconds": (
                    min(
                        int(
                            runner.runtime_estimate(
                                item, load_manifest()["runtimeEstimate"]
                            )["minRuntimeSeconds"]
                        ),
                        base.MAX_MIN_RUNTIME_SECONDS,
                    )
                    if allocated
                    else None
                ),
            }
        )
    value = {
        "policy": runner.POLICY,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "datasetManifest": str(runner.EXPECTED_DATASET_MANIFEST),
        "scheduling": "allocated" if allocated else "unallocated",
        "minRuntimeOmitted": not allocated,
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
    parser.add_argument("--allocated", action="store_true")
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
            spec = (
                allocated_spec(item, args.revision, args.priority)[0]
                if args.allocated
                else base.spec_for(
                    item,
                    args.revision,
                    args.priority,
                    omit_min_runtime=True,
                )
            )
            print(json.dumps(spec, indent=2))
        else:
            experiment = (
                create_allocated(item, args.revision, args.priority)
                if args.allocated
                else base.create(
                    item,
                    args.revision,
                    args.priority,
                    omit_min_runtime=True,
                )
            )
            created.append((item, experiment))
            print(f"{item['id']}: {experiment}")
    if created:
        write_registry(created, args.revision, allocated=args.allocated)


if __name__ == "__main__":
    main()
