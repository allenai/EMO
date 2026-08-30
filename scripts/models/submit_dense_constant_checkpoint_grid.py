#!/usr/bin/env python3
"""Guardedly submit ten checkpoint producers and the two 1B E8/E16 evaluators."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKSPACE = "ai2/flex2"
POLICY = "dense_constant_checkpoint_producers_v1"
MANIFEST = Path("scripts/models/manifests/dense-constant-checkpoint-producers-v1.json")
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
PRODUCER_RUNNER = "scripts/models/run_dense_constant_checkpoint_producer.py"
EVALUATOR_RUNNER = "scripts/models/run_dense_1b_checkpoint_evaluator.py"


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(
        arguments,
        check=True,
        input=input_text,
        text=True,
        capture_output=True,
    ).stdout


def manifest() -> dict[str, Any]:
    value = json.loads(MANIFEST.read_text())
    if value.get("policy") != POLICY:
        raise ValueError(f"manifest policy must be {POLICY}")
    if len(value.get("producerCoordinates", [])) != 10:
        raise ValueError("manifest must contain exactly ten producers")
    if len(value.get("evaluatorCoordinates", [])) != 2:
        raise ValueError("manifest must contain exactly two 1B evaluators")
    return value


def initial_report(config: dict[str, Any]) -> dict[str, Any]:
    producers = []
    for item in config["producerCoordinates"]:
        existing = [int(epoch) for epoch in item.get("existingScheduledEpochs", [])]
        targets = list(
            range(
                int(item["firstCheckpointEpoch"]),
                int(config["maxEpoch"]) + 1,
                int(item["epochIncrement"]),
            )
        )
        next_epoch = next(
            (epoch for epoch in targets if epoch > int(item["sourceEpoch"])), None
        )
        producers.append(
            {
                "id": item["id"],
                "role": "constant_lr_checkpoint_producer",
                "model": item["model"],
                "pool": item["pool"],
                "batchSequences": item["batchSequences"],
                "learningRate": item["learningRate"],
                "weightDecay": item["weightDecay"],
                "status": "planned",
                "sourceEpoch": item["sourceEpoch"],
                "sourceExperiment": item["sourceExperiment"],
                "sourceCheckpoint": item["sourceCheckpoint"],
                "output": item["output"],
                "resolvedCheckpointEpochs": existing,
                "targetEpochs": targets,
                "currentEpoch": next_epoch,
                "evaluationEnabled": False,
            }
        )
    by_id = {item["id"]: item for item in config["producerCoordinates"]}
    evaluators = []
    for evaluator in config["evaluatorCoordinates"]:
        item = by_id[evaluator["producerId"]]
        evaluators.append(
            {
                "id": evaluator["id"],
                "role": "independent_wsd_decay_and_eval",
                "producerId": evaluator["producerId"],
                "model": "1b",
                "pool": "dclm3b",
                "batchSequences": item["batchSequences"],
                "learningRate": item["learningRate"],
                "weightDecay": item["weightDecay"],
                "epochs": evaluator["epochs"],
                "resolvedPostEpochs": [],
                "status": "planned",
            }
        )
    return {
        "title": "Dense DR+WT+EmbedWD constant-LR checkpoint producers",
        "policy": POLICY,
        "maxEpoch": config["maxEpoch"],
        "producerCount": 10,
        "evaluatorCount": 2,
        "smallEvaluatorPolicy": "not_launched",
        "producers": producers,
        "evaluators": evaluators,
    }


def load_report(config: dict[str, Any]) -> dict[str, Any]:
    if not REPORT.is_file():
        return initial_report(config)
    report = json.loads(REPORT.read_text())
    expected = initial_report(config)
    old_producers = {item["id"]: item for item in report.get("producers", [])}
    old_evaluators = {item["id"]: item for item in report.get("evaluators", [])}
    for item in expected["producers"]:
        old = old_producers.get(item["id"], {})
        for key in ("experiment", "revision", "beakerStatus", "submittedAt", "wandbHealth"):
            if key in old:
                item[key] = old[key]
        if old.get("status") not in {None, "planned"}:
            item["status"] = old["status"]
    for item in expected["evaluators"]:
        old = old_evaluators.get(item["id"], {})
        for key in (
            "experiment",
            "revision",
            "beakerStatus",
            "submittedAt",
            "wandbHealth",
            "resolvedPostEpochs",
            "decision",
        ):
            if key in old:
                item[key] = old[key]
        if old.get("status") not in {None, "planned"}:
            item["status"] = old["status"]
    return expected


def write_report(report: dict[str, Any]) -> None:
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_CHECKPOINT_PRODUCER_GRID="
        + json.dumps(report, separators=(",", ":"))
        + ";\n"
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


def guarded_name(record: dict[str, Any]) -> str:
    suffix = "constant-producer-v1" if record["role"] == "constant_lr_checkpoint_producer" else "v1"
    return f"{record['id']}-{suffix}"


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


def base_experiment(record: dict[str, Any], config: dict[str, Any]) -> str:
    if record["role"] == "constant_lr_checkpoint_producer":
        item = next(item for item in config["producerCoordinates"] if item["id"] == record["id"])
    else:
        evaluator = next(item for item in config["evaluatorCoordinates"] if item["id"] == record["id"])
        item = next(
            item for item in config["producerCoordinates"] if item["id"] == evaluator["producerId"]
        )
    return str(item["baseExperiment"])


def spec_for(
    record: dict[str, Any], config: dict[str, Any], revision: str, priority: str
) -> dict[str, Any]:
    spec = copy.deepcopy(
        json.loads(
            command(
                ["beaker", "experiment", "spec", base_experiment(record, config), "--format", "json"]
            )
        )
    )
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("trusted base experiment must contain one task")
    task = spec["tasks"][0]
    if "/weka/oe-training-default" not in {
        dataset.get("mountPath") for dataset in task.get("datasets", [])
    }:
        raise RuntimeError("trusted base experiment is missing the Weka mount")
    task["name"] = "main"
    if record["role"] == "constant_lr_checkpoint_producer":
        task["arguments"] = [
            "python",
            PRODUCER_RUNNER,
            "--manifest",
            str(MANIFEST),
            "--coordinate",
            record["id"],
        ]
    else:
        task["arguments"] = [
            "python",
            EVALUATOR_RUNNER,
            "--manifest",
            str(MANIFEST),
            "--evaluator",
            record["id"],
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
    if record["model"] == "474m":
        set_env(task, "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "minRuntime": "0s", "autoResume": True}
    task["hostNetworking"] = False
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    if record["role"] == "constant_lr_checkpoint_producer":
        spec["description"] = (
            f"{record['model']} {record['pool']} BS{record['batchSequences']} "
            f"DR+WT+EmbedWD LR{record['learningRate']} WD{record['weightDecay']}; "
            "exact clean-checkpoint resume; constant LR; checkpoint production only; no decay or eval."
        )
    else:
        spec["description"] = (
            f"Dense-1B DCLM-3B BS{record['batchSequences']} independent WSD decay and "
            "heldout evaluation for E8 and E16 only."
        )
    return spec


def create(record: dict[str, Any], config: dict[str, Any], revision: str, priority: str) -> str:
    name = guarded_name(record)
    existing = existing_named_experiment(name)
    if existing:
        return existing
    output = command(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        input_text=json.dumps(spec_for(record, config, revision, priority)),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError(f"submission returned no experiment ID for {name}")
    return identifiers[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--submit-if-ready", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    validate_revision(args.revision)
    config = manifest()
    report = load_report(config)
    records = [*report["producers"], *report["evaluators"]]
    if len(records) != 12:
        raise RuntimeError("submission set must contain exactly 12 records")
    for record in records:
        if record.get("experiment"):
            print(f"{record['id']}: already registered {record['experiment']}")
            continue
        if args.print_only:
            print(json.dumps(spec_for(record, config, args.revision, args.priority), indent=2))
            continue
        if not args.submit_if_ready:
            print(f"{record['id']}: ready; pass --submit-if-ready")
            continue
        experiment = create(record, config, args.revision, args.priority)
        record.update(
            {
                "experiment": experiment,
                "revision": args.revision,
                "status": "submitted",
                "beakerStatus": "submitted",
                "submittedAt": datetime.now(tz=UTC).isoformat(),
                "wandbHealth": {
                    "status": "pending",
                    "beakerState": "submitted",
                    "criticalSignals": [],
                },
            }
        )
        write_report(report)
        print(f"{record['id']}: {experiment}")
    write_report(report)


if __name__ == "__main__":
    main()
