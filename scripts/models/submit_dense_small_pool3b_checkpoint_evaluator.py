#!/usr/bin/env python3
"""Guardedly submit one allocated-slot small-model Pool-3B evaluator."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import evaluator_min_runtime
import run_dense_small_pool3b_checkpoint_evaluator as evaluator
import run_dense_small_pool3b_checkpoint_producer as producer

WORKSPACE = "ai2/flex2"
DEFAULT_MANIFEST = Path("scripts/models/manifests/dense-small-pool3b-checkpoint-producers-v2.json")
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
RUNNER = "scripts/models/run_dense_small_pool3b_checkpoint_evaluator.py"


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(
        arguments, check=True, input=input_text, text=True, capture_output=True
    ).stdout


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


def guarded_name(item: dict[str, Any], epoch: int) -> str:
    return f"{item['id']}-post-e{epoch}-v1"


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


def runtime(item: dict[str, Any], epoch: int) -> dict[str, Any]:
    return evaluator_min_runtime.estimate_min_runtime(
        model=str(item["model"]),
        pool_tokens=producer.TARGET_POOL_TOKENS,
        batch_sequences=int(item["batchSequences"]),
        epochs=[epoch],
    )


def report_producer(item: dict[str, Any], epoch: int) -> dict[str, Any]:
    report = json.loads(REPORT.read_text())
    matches = [record for record in report.get("producers", []) if record.get("id") == item["id"]]
    if len(matches) != 1:
        raise RuntimeError(f"report is missing producer {item['id']}")
    resolved = {int(value) for value in matches[0].get("resolvedCheckpointEpochs", [])}
    if epoch not in resolved:
        raise RuntimeError(f"E{epoch} pre-decay checkpoint is not resolved for {item['id']}")
    return matches[0]


def spec_for(
    item: dict[str, Any],
    epoch: int,
    revision: str,
    priority: str,
    output: str,
    manifest: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
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
        str(manifest),
        "--coordinate",
        str(item["id"]),
        "--epoch",
        str(epoch),
        "--producer-output",
        output,
    ]
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
    if item["model"] == "474m":
        set_env(task, "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    reserved = runtime(item, epoch)
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {
        "priority": priority,
        "minRuntime": str(reserved["minRuntime"]),
        "autoResume": True,
    }
    task["hostNetworking"] = False
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"Dense-{item['model']} DCLM-3B BS{item['batchSequences']} LR{item['learningRate']} "
        f"WD{item['weightDecay']} independent E{epoch} WSD decay and heldout evaluation; "
        f"allocated-slot scheduling with buffered minRuntime={reserved['minRuntime']}."
    )
    return spec


def create(
    item: dict[str, Any],
    epoch: int,
    revision: str,
    priority: str,
    output: str,
    manifest: Path = DEFAULT_MANIFEST,
) -> str:
    name = guarded_name(item, epoch)
    existing = existing_named_experiment(name)
    if existing:
        return existing
    output = command(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        input_text=json.dumps(spec_for(item, epoch, revision, priority, output, manifest)),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError(f"submission returned no experiment ID for {name}")
    return identifiers[0]


def register(
    item: dict[str, Any],
    epoch: int,
    experiment: str,
    revision: str,
    output: str,
    manifest: Path = DEFAULT_MANIFEST,
) -> None:
    report = json.loads(REPORT.read_text())
    records = report.setdefault("smallEvaluators", [])
    evaluator_id = guarded_name(item, epoch)
    matches = [record for record in records if record.get("id") == evaluator_id]
    if matches:
        if matches[0].get("experiment") != experiment:
            raise RuntimeError(f"registered evaluator mismatch for {evaluator_id}")
        return
    source = evaluator.source_checkpoint(item, epoch, output)
    endpoint_step = producer.total_step(
        epoch, producer.TARGET_POOL_TOKENS, int(item["batchSequences"])
    )
    records.append(
        {
            "id": evaluator_id,
            "role": "independent_wsd_decay_and_eval",
            "policy": evaluator.POLICY,
            "producerId": item["id"],
            "model": item["model"],
            "pool": "dclm3b",
            "batchSequences": item["batchSequences"],
            "learningRate": item["learningRate"],
            "weightDecay": item["weightDecay"],
            "epoch": epoch,
            "sourceCheckpoint": str(source),
            "producerOutput": output,
            "endpointStep": endpoint_step,
            "status": "submitted",
            "experiment": experiment,
            "revision": revision,
            "manifest": str(manifest),
            "minRuntime": runtime(item, epoch)["minRuntime"],
            "submittedAt": datetime.now(tz=UTC).isoformat(),
            "beakerStatus": "submitted",
            "wandbHealth": {
                "status": "pending",
                "beakerState": "submitted",
                "criticalSignals": [],
            },
        }
    )
    report["smallEvaluatorPolicy"] = "standalone_allocated_per_epoch"
    report["smallEvaluatorCount"] = len(records)
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_CHECKPOINT_PRODUCER_GRID=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--coordinate", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--submit-if-ready", action="store_true")
    parser.add_argument("--print-spec", action="store_true")
    args = parser.parse_args()
    validate_revision(args.revision)
    _, item = evaluator.load(args.manifest, args.coordinate, args.epoch)
    producer_record = report_producer(item, args.epoch)
    output = str(producer_record["output"])
    evaluator.producer_output(item, output)
    if args.print_spec:
        print(
            json.dumps(
                spec_for(
                    item,
                    args.epoch,
                    args.revision,
                    args.priority,
                    output,
                    args.manifest,
                ),
                indent=2,
            )
        )
        return
    reserved = runtime(item, args.epoch)
    if not args.submit_if_ready:
        print(
            f"{guarded_name(item, args.epoch)}: ready with minRuntime={reserved['minRuntime']}; "
            "pass --submit-if-ready"
        )
        return
    experiment = create(item, args.epoch, args.revision, args.priority, output, args.manifest)
    register(item, args.epoch, experiment, args.revision, output, args.manifest)
    print(f"{guarded_name(item, args.epoch)}: {experiment} minRuntime={reserved['minRuntime']}")


if __name__ == "__main__":
    main()
