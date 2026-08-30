#!/usr/bin/env python3
"""Guardedly submit the eight corrected small-model Pool-3B producers."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import run_dense_small_pool3b_checkpoint_producer as runner

WORKSPACE = "ai2/flex2"
MANIFEST = Path(
    "scripts/models/manifests/dense-small-pool3b-checkpoint-producers-v2.json"
)
RUNNER = "scripts/models/run_dense_small_pool3b_checkpoint_producer.py"


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(
        arguments,
        check=True,
        input=input_text,
        text=True,
        capture_output=True,
    ).stdout


def load_manifest() -> dict[str, Any]:
    config = runner.load_manifest(MANIFEST)
    for item in config["producerCoordinates"]:
        runner.validate_coordinate(config, item, check_filesystem=False)
    return config


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


def guarded_name(item: dict[str, Any]) -> str:
    return f"{item['id']}-constant-producer-v2"


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


def spec_for(item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
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
        f"{item['model']} DCLM-3B BS{item['batchSequences']} DR+WT+EmbedWD "
        f"LR{item['learningRate']} WD{item['weightDecay']}; exact Pool-1B pre-decay E1 "
        "model/trainer/optimizer resume; reset onto fresh 2B extension to Pool-3B "
        "pre-decay E1; then reset onto repacked shuffled combined 3B at constant LR; "
        "checkpoint production only; no decay or evaluation."
    )
    return spec


def create(item: dict[str, Any], revision: str, priority: str) -> str:
    name = guarded_name(item)
    existing = existing_named_experiment(name)
    if existing:
        return existing
    output = command(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        input_text=json.dumps(spec_for(item, revision, priority)),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError(f"submission returned no experiment ID for {name}")
    return identifiers[0]


def plan_row(config: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    batch = int(item["batchSequences"])
    bridge_step = runner.stable_step(1, runner.TARGET_POOL_TOKENS, batch)
    epochs = runner.target_epochs(item, int(config["maxEpoch"]))
    return {
        "id": item["id"],
        "source": item["sourceCheckpoint"],
        "bridge": f"fresh 2B extension -> Pool-3B pre-decay E1 step{bridge_step}",
        "repeatedPool": config["repeatedManifest"],
        "retainedEpochs": epochs,
        "output": item["output"],
        "evaluation": False,
    }


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
            print(json.dumps(plan_row(config, item), sort_keys=True))
        return
    if not args.revision:
        raise SystemExit("--revision is required for spec generation or submission")
    validate_revision(args.revision)
    for item in config["producerCoordinates"]:
        if args.print_specs:
            print(json.dumps(spec_for(item, args.revision, args.priority), indent=2))
        elif args.submit_if_ready:
            print(f"{item['id']}: {create(item, args.revision, args.priority)}")
        else:
            print(f"{item['id']}: ready; pass --submit-if-ready only after user confirmation")


if __name__ == "__main__":
    main()
