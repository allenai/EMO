#!/usr/bin/env python3
"""Submit isolated recovery attempts for authorized slow small-model producers."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import run_dense_small_pool3b_checkpoint_producer as producer
import run_dense_small_pool3b_throughput_recovery as recovery


WORKSPACE = "ai2/flex2"
MANIFEST = Path("scripts/models/manifests/dense-small-pool3b-checkpoint-producers-v2.json")
RUNNER = "scripts/models/run_dense_small_pool3b_throughput_recovery.py"


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(
        arguments,
        check=True,
        input=input_text,
        text=True,
        capture_output=True,
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


def guarded_name(item: dict[str, Any], attempt: int) -> str:
    return f"{item['id']}-throughput-recovery-r{attempt}"


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
    matches = [value for value in values if value.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError(f"multiple experiments use guarded name {name}")
    return str(matches[0]["id"]) if matches else None


def spec_for(
    item: dict[str, Any], attempt: int, revision: str, priority: str
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
        str(MANIFEST),
        "--coordinate",
        str(item["id"]),
        "--attempt",
        str(attempt),
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
    if str(item["model"]) == "474m":
        set_env(task, "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {
        "priority": priority,
        "minRuntime": "0s",
        "autoResume": True,
    }
    task["hostNetworking"] = False
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"Isolated throughput recovery r{attempt} for {item['id']}; load exact clean "
        "Pool-3B E1 model/trainer/optimizer checkpoint, reset only data-loader state, "
        "continue the sealed repacked/shuffled Pool-3B constant-LR producer, and write "
        f"only to {recovery.recovery_output(item, attempt)}."
    )
    return spec


def create(
    item: dict[str, Any], attempt: int, revision: str, priority: str
) -> str:
    name = guarded_name(item, attempt)
    existing = existing_named_experiment(name)
    if existing:
        return existing
    output = command(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        input_text=json.dumps(spec_for(item, attempt, revision, priority)),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError(f"submission returned no experiment ID for {name}")
    return identifiers[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinate", choices=sorted(recovery.ALLOWED_COORDINATES), required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-spec", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    validate_revision(args.revision)
    config = producer.load_manifest(MANIFEST)
    item = producer.coordinate(config, args.coordinate)
    producer.validate_coordinate(config, item, check_filesystem=False)
    recovery.validate_recovery_source(item, check_filesystem=False)
    if args.print_spec:
        print(json.dumps(spec_for(item, args.attempt, args.revision, args.priority), indent=2))
        return
    if not args.submit_if_ready:
        print(
            json.dumps(
                {
                    "coordinate": item["id"],
                    "attempt": args.attempt,
                    "source": str(recovery.recovery_source(item)),
                    "output": str(recovery.recovery_output(item, args.attempt)),
                    "status": "ready",
                },
                sort_keys=True,
            )
        )
        return
    experiment = create(item, args.attempt, args.revision, args.priority)
    print(
        json.dumps(
            {
                "coordinate": item["id"],
                "attempt": args.attempt,
                "experiment": experiment,
                "source": str(recovery.recovery_source(item)),
                "output": str(recovery.recovery_output(item, args.attempt)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
