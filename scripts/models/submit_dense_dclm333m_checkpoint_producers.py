#!/usr/bin/env python3
"""Print or guardedly submit integrated DCLM-333M producer/evaluators."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dense_dclm333m_checkpoint_producer as runner

WORKSPACE = "ai2/flex2"
MANIFEST = Path("scripts/models/manifests/dense-dclm333m-checkpoint-producers-v1.json")
RUNNER = "scripts/models/run_dense_dclm333m_checkpoint_producer.py"
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
REPORT_JS_PREFIX = "window.ICSL_CHECKPOINT_PRODUCER_GRID="
MAX_MIN_RUNTIME_SECONDS = 8 * 60 * 60


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(
        arguments,
        check=True,
        input=input_text,
        text=True,
        capture_output=True,
    ).stdout


def load_manifest() -> dict[str, Any]:
    return runner.load_manifest(MANIFEST)


def format_duration(seconds: float) -> str:
    minutes = math.ceil(seconds / 60)
    days, remainder = divmod(minutes, 24 * 60)
    hours, minutes = divmod(remainder, 60)
    pieces = []
    if days:
        pieces.append(f"{days}d")
    if hours:
        pieces.append(f"{hours}h")
    if minutes or not pieces:
        pieces.append(f"{minutes}m")
    return " ".join(pieces)


def reserved_min_runtime(
    item: dict[str, Any], runtime_policy: dict[str, Any], *, omitted: bool = False
) -> int:
    if omitted or item["model"] not in {"1b", "474m"}:
        return 0
    estimate = runner.runtime_estimate(item, runtime_policy)
    return min(int(estimate["minRuntimeSeconds"]), MAX_MIN_RUNTIME_SECONDS)


def plan_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in config["producerCoordinates"]:
        estimate = runner.runtime_estimate(item, config["runtimeEstimate"])
        use_allocated_slot = item["model"] in {"1b", "474m"}
        reservation = reserved_min_runtime(item, config["runtimeEstimate"])
        rows.append(
            {
                "model": item["model"],
                "batch": int(item["batchSequences"]),
                "gpus": runner.gpu_count(item),
                "lr": item["learningRate"],
                "wd": item["weightDecay"],
                "checkpoints": ",".join(f"E{epoch}" for epoch in item["retainedCheckpointEpochs"]),
                "evaluations": ",".join(f"E{epoch}" for epoch in item["evaluationEpochs"]),
                "output": item["output"],
                "training": format_duration(estimate["trainingSeconds"]),
                "evaluation": format_duration(estimate["evaluationSeconds"]),
                "raw": format_duration(estimate["rawSeconds"]),
                "buffered": format_duration(estimate["minRuntimeSeconds"]),
                "reserved": format_duration(reservation),
                "slot": "allocated" if use_allocated_slot else "unallocated",
            }
        )
    return rows


def print_plan(config: dict[str, Any]) -> None:
    print("Pool: DCLM-333M, exact whole-document subset of sealed Pool-1B")
    print(f"Dataset manifest: {config['datasetManifest']}")
    print(
        "All coordinates: one node, per-coordinate GPU count, gradient accumulation 1, "
        "DR+WT+EmbedWD; isolated WSD+heldout "
        "branches never advance the constant frontier"
    )
    print()
    print(
        "| Model | BS | GPUs | LR | WD | Retained PD | Decay+eval | Output directory | "
        "Constant train | POST work | Full raw | Scheduling | minRuntime |"
    )
    print("|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|---:|")
    for row in plan_rows(config):
        print(
            f"| {row['model']} | {row['batch']} | {row['gpus']} | {row['lr']} | {row['wd']} | "
            f"{row['checkpoints']} | {row['evaluations']} | `{row['output']}` | "
            f"{row['training']} | {row['evaluation']} | {row['raw']} | {row['slot']} | "
            f"{row['reserved'] if row['slot'] == 'allocated' else 'omitted'} |"
        )


def validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("--revision must be a full 40-character commit hash")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "origin/sewonm/icsl"],
        check=True,
    )


def set_env(task: dict[str, Any], name: str, value: str) -> None:
    for variable in task.get("envVars", []):
        if variable.get("name") == name:
            variable["value"] = value
            return
    task.setdefault("envVars", []).append({"name": name, "value": value})


def guarded_name(item: dict[str, Any], target_epoch: int | None = None) -> str:
    if target_epoch is not None:
        if runner.gpu_count(item) != runner.NPROC_PER_NODE:
            return (
                f"{item['id']}-continuation-e{target_epoch}-"
                f"gpus{runner.gpu_count(item)}-v2"
            )
        return f"{item['id']}-continuation-e{target_epoch}-v1"
    return f"{item['id']}-integrated-producer-eval-v1"


def existing_named_experiment(name: str) -> str | None:
    payload = json.loads(
        command(
            [
                "beaker",
                "workspace",
                "experiments",
                WORKSPACE,
                "--text",
                name,
                "--format",
                "json",
            ]
        )
    )
    values = payload if isinstance(payload, list) else payload.get("experiments", [])
    matches = [item for item in values if item.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError(f"multiple experiments use guarded name {name}")
    return str(matches[0]["id"]) if matches else None


def spec_for(
    item: dict[str, Any],
    revision: str,
    priority: str,
    *,
    target_epoch: int | None = None,
    omit_min_runtime: bool = False,
) -> dict[str, Any]:
    config = load_manifest()
    estimate = runner.runtime_estimate(item, config["runtimeEstimate"])
    min_runtime = reserved_min_runtime(
        item,
        config["runtimeEstimate"],
        omitted=omit_min_runtime,
    )
    spec = copy.deepcopy(
        json.loads(
            command(
                [
                    "beaker",
                    "experiment",
                    "spec",
                    str(item["baseExperiment"]),
                    "--format",
                    "json",
                ]
            )
        )
    )
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("trusted base experiment must contain exactly one task")
    task = spec["tasks"][0]
    if "/weka/oe-training-default" not in {
        dataset.get("mountPath") for dataset in task.get("datasets", [])
    }:
        raise RuntimeError("trusted base experiment is missing the Weka mount")
    task["name"] = "main"
    task["arguments"] = [
        "python",
        RUNNER,
        "--manifest",
        str(MANIFEST),
        "--coordinate",
        str(item["id"]),
    ]
    if target_epoch is not None:
        task["arguments"].extend(["--target-epoch", str(target_epoch)])
    blocked = {
        "GANTRY_USE_TORCHRUN",
        "GANTRY_RDZV_ID",
        "GANTRY_RDZV_PORT",
        "NUM_NODES",
        "PYTORCH_CUDA_ALLOC_CONF",
    }
    task["envVars"] = [
        variable
        for variable in task.get("envVars", [])
        if variable.get("name") not in blocked
        and not (
            str(variable.get("name", "")).startswith("BEAKER_")
            and variable.get("name") != "BEAKER_TOKEN"
        )
    ]
    set_env(task, "GIT_REF", revision)
    set_env(task, "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    task["resources"] = {"gpuCount": runner.gpu_count(item), "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
    if min_runtime:
        task["context"]["minRuntime"] = f"{min_runtime}s"
    task["hostNetworking"] = False
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"{item['model']} DCLM-333M BS{item['batchSequences']} DR+WT+EmbedWD "
        f"LR{item['learningRate']} WD{item['weightDecay']}; {runner.gpu_count(item)} GPUs; "
        "gradient accumulation 1; "
        + (
            f"continue the exact clean E{runner.continuation_source_epoch(item, target_epoch)} producer frontier; "
            if target_epoch is not None
            else "start from scratch; "
        )
        + f"constant-LR frontier through E{item['maxEpoch']}; retain exact pre-decay "
        f"checkpoints at {item['retainedCheckpointEpochs']}; independently WSD-decay and "
        f"heldout-evaluate {item['evaluationEpochs']} before continuing from each exact PD "
        f"source; output {item['output']}; "
        + (f"minRuntime={min_runtime}s." if min_runtime else "minRuntime omitted.")
    )
    return spec


def create(
    item: dict[str, Any],
    revision: str,
    priority: str,
    *,
    target_epoch: int | None = None,
    omit_min_runtime: bool = False,
) -> str:
    name = guarded_name(item, target_epoch)
    existing = existing_named_experiment(name)
    if existing:
        return existing
    output = command(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        input_text=json.dumps(
            spec_for(
                item,
                revision,
                priority,
                target_epoch=target_epoch,
                omit_min_runtime=omit_min_runtime,
            )
        ),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError(f"submission returned no experiment ID for {name}")
    return identifiers[0]


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def register_new_runs(
    config: dict[str, Any], created: list[tuple[dict[str, Any], str]], revision: str
) -> None:
    if not created:
        return
    report = json.loads(REPORT.read_text())
    records = {
        str(record["id"]): record
        for record in report.get("dclm333mIntegratedRuns", [])
    }
    outputs = {str(record.get("output")) for record in records.values()}
    for item, experiment in created:
        coordinate_id = str(item["id"])
        output = str(item["output"])
        existing = records.get(coordinate_id)
        if existing is not None:
            if existing.get("experiment") != experiment or existing.get("output") != output:
                raise RuntimeError(f"report already contains conflicting coordinate {coordinate_id}")
            continue
        if output in outputs:
            raise RuntimeError(f"report already contains output writer {output}")
        record = {
            "id": coordinate_id,
            "role": "integrated_checkpoint_producer_and_evaluator",
            "policy": runner.POLICY,
            "model": str(item["model"]),
            "pool": "dclm333m",
            "batchSequences": int(item["batchSequences"]),
            "learningRate": str(item["learningRate"]),
            "weightDecay": str(item["weightDecay"]),
            "gpuCount": runner.gpu_count(item),
            "gradientAccumulationSteps": 1,
            "retainedCheckpointEpochs": list(item["retainedCheckpointEpochs"]),
            "evaluationEpochs": list(item["evaluationEpochs"]),
            "resolvedCheckpointEpochs": [],
            "resolvedPostEpochs": [],
            "postDecayResults": {},
            "currentEpoch": int(item["retainedCheckpointEpochs"][0]),
            "currentPhase": "producer",
            "status": "submitted",
            "experiment": experiment,
            "revision": revision,
            "minRuntime": (
                f"{reserved_min_runtime(item, config['runtimeEstimate'])}s"
                if reserved_min_runtime(item, config["runtimeEstimate"])
                else "omitted"
            ),
            "output": output,
            "beakerStatus": "submitted",
        }
        if item["model"] == "1b":
            record["comparisonGate"] = {
                "matchedPostEpochs": [8, 16],
                "criterion": "same_coordinate_strictly_lower_healthy_matched_post_at_e8_and_e16",
                "action": "notify_only_user_decides_pruning",
                "status": "pending",
            }
            record["hardStopEpoch"] = 32
            record["futureConditionalCoordinate"] = (
                "dense-1b-dclm333m-bs32-lr5e-4-wd1.0"
            )
        else:
            record["wdPruningGate"] = {
                "decisionEpoch": 16 if item["model"] == "474m" else 32,
                "candidateToStop": "0.1",
                "comparator": "0.3",
                "criterion": "wd0.3_strictly_lower_healthy_matched_post_validationExact",
                "status": "pending",
            }
        records[coordinate_id] = record
        outputs.add(output)
    ordered_ids = [str(item["id"]) for item in config["producerCoordinates"]]
    if set(records) != set(ordered_ids):
        raise RuntimeError("report and manifest Pool-333M coordinates differ after registration")
    report["dclm333mIntegratedRuns"] = [records[item_id] for item_id in ordered_ids]
    report["dclm333mIntegratedCount"] = len(ordered_ids)
    report["updatedAt"] = datetime.now(timezone.utc).isoformat()
    rendered = json.dumps(report, indent=2) + "\n"
    atomic_text(REPORT, rendered)
    atomic_text(REPORT_JS, REPORT_JS_PREFIX + rendered.rstrip() + ";\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--coordinate", action="append", default=[])
    parser.add_argument("--omit-min-runtime", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--print-specs", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    config = load_manifest()
    if args.print_plan or not (args.print_specs or args.submit_if_ready):
        print_plan(config)
        return
    if not args.revision:
        raise SystemExit("--revision is required for spec generation or submission")
    validate_revision(args.revision)
    selected = [
        item
        for item in config["producerCoordinates"]
        if not args.coordinate or item["id"] in args.coordinate
    ]
    if len(selected) != len(set(args.coordinate)) and args.coordinate:
        raise SystemExit("every requested coordinate must be present exactly once")
    created: list[tuple[dict[str, Any], str]] = []
    for item in selected:
        if args.print_specs:
            print(
                json.dumps(
                    spec_for(
                        item,
                        args.revision,
                        args.priority,
                        omit_min_runtime=args.omit_min_runtime,
                    ),
                    indent=2,
                )
            )
        else:
            experiment = create(
                item,
                args.revision,
                args.priority,
                omit_min_runtime=args.omit_min_runtime,
            )
            created.append((item, experiment))
            print(f"{item['id']}: {experiment}")
    register_new_runs(config, created, args.revision)


if __name__ == "__main__":
    main()
