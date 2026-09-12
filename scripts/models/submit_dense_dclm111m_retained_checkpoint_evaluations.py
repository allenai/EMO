#!/usr/bin/env python3
"""Guardedly submit the 153M Pool-111M retained E40/E48/E56 POST workflow."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import run_dense_dclm111m_checkpoint_producer as pool111m
import run_dense_dclm111m_retained_checkpoint_evaluations as evaluator
import submit_dense_dclm333m_checkpoint_producers as submit


MANIFEST = Path("scripts/models/manifests/dense-dclm111m-checkpoint-producers-v1.json")
RUNNER = "scripts/models/run_dense_dclm111m_retained_checkpoint_evaluations.py"
NAME = "dense-153m-dclm111m-bs64-lr2e-3-wd0.3-retained-post-e40-e48-e56-v1"
MIN_RUNTIME_SECONDS = 2 * 60 * 60


def set_env(task: dict[str, Any], name: str, value: str) -> None:
    submit.set_env(task, name, value)


def spec_for(item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    spec = copy.deepcopy(
        json.loads(
            submit.command(
                ["beaker", "experiment", "spec", str(item["baseExperiment"]), "--format", "json"]
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
    task["arguments"] = ["python", RUNNER, "--manifest", str(MANIFEST)]
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
    task["resources"] = {"gpuCount": pool111m.gpu_count(item), "sharedMemory": "10 GiB"}
    task["context"] = {
        "priority": priority,
        "autoResume": True,
        "minRuntime": f"{MIN_RUNTIME_SECONDS}s",
    }
    task["hostNetworking"] = False
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        "Evaluation-only workflow for retained 153M Pool-111M BS64 LR2e-3 WD0.3 "
        "pre-decay E40/E48/E56 checkpoints; sequential isolated WSD decay and heldout "
        "evaluation; one 4-GPU node; two-hour protected allocation; no producer stages."
    )
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-spec", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    submit.validate_revision(args.revision)
    item = evaluator.load(MANIFEST)
    spec = spec_for(item, args.revision, args.priority)
    if args.print_spec:
        print(json.dumps(spec, indent=2))
        return
    if not args.submit_if_ready:
        print(f"{NAME}: ready")
        return
    existing = submit.existing_named_experiment(NAME)
    if existing:
        print(existing)
        return
    output = submit.command(
        ["beaker", "experiment", "create", "-", "--name", NAME, "--workspace", submit.WORKSPACE],
        input_text=json.dumps(spec),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError("submission returned no experiment ID")
    print(identifiers[0])


if __name__ == "__main__":
    main()
