#!/usr/bin/env python3
"""Replay only the BS256 WSD decay from its retained pre-decay checkpoint.

The parent warmdown intentionally disables downstream evaluation. This launcher
loads the exact BS256 step-2145 checkpoint, restores trainer and optimizer state,
runs the remaining 239 decay steps through step 2384, and evaluates held-out
DCLM plus the inherited downstream suite. It never reapplies batch-size moment
recalibration or optimizer-hyperparameter transitions.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PARENT_EXPERIMENT = "01M00EJZDCJSY5QTC5YVZPZBS6"
OUTPUT_ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"
SOURCE_NAME = "bs256_dr_lr1e-3_wd0.333_init=bs1024e2_lr5e-04_wd0.333"
RUN_NAME = SOURCE_NAME + "_decay_eval"
SOURCE_CHECKPOINT = f"{OUTPUT_ROOT}/{SOURCE_NAME}/step2145"
OUTPUT = f"{OUTPUT_ROOT}/{RUN_NAME}"
PRE_DECAY_STEP = 2145
END_STEP = 2384
DECAY_STEPS = END_STEP - PRE_DECAY_STEP
TRAIN_SCRIPT = "src/scripts/train/olmo2-1B.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", default="dense-1b-best-coordinate-bs256-decay-eval")
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full Git commit SHA")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.name):
        parser.error("--name must be a lowercase Beaker name component")
    return args


def parent_spec() -> dict[str, Any]:
    completed = subprocess.run(
        ["beaker", "experiment", "spec", PARENT_EXPERIMENT, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def first_bs256_command(task: dict[str, Any]) -> tuple[str, list[str]]:
    shell = task.get("arguments", [])
    if shell[:2] != ["bash", "-lc"] or len(shell) != 3:
        raise RuntimeError("parent task is not a bash launcher")
    for line in shell[2].splitlines():
        parts = shlex.split(line)
        if not parts or parts[0] != "torchrun" or TRAIN_SCRIPT not in parts:
            continue
        script_index = parts.index(TRAIN_SCRIPT)
        if parts[script_index + 1] != SOURCE_NAME:
            continue
        return parts[script_index], parts[script_index + 2 :]
    raise RuntimeError("could not find the exact parent BS256 torchrun command")


def upsert(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    output: list[str] = []
    found = False
    for argument in arguments:
        if argument.startswith(prefix):
            if not found:
                output.append(replacement)
                found = True
        else:
            output.append(argument)
    if not found:
        output.append(replacement)
    return output


def build_spec(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    built = copy.deepcopy(spec)
    if len(built.get("tasks", [])) != 1:
        raise RuntimeError("parent experiment must have exactly one task")
    task = built["tasks"][0]
    script, parent_arguments = first_bs256_command(task)
    excluded = (
        "--trainer.callbacks.batch_warmdown_recalibration_",
        "--trainer.callbacks.batch_warmdown_hyperparameters_",
        "--trainer.load_path=",
        "--trainer.load_trainer_state=",
        "--trainer.load_optim_state=",
        "--trainer.prefer_explicit_load_path=",
    )
    arguments = [value for value in parent_arguments if not value.startswith(excluded)]
    replacements = (
        ("--save-folder=", f"--save-folder={OUTPUT}"),
        ("--trainer.max_duration=", f"--trainer.max_duration={{value: {END_STEP}, unit: steps}}"),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={RUN_NAME}"),
        (
            "--trainer.callbacks.wandb.tags=",
            "--trainer.callbacks.wandb.tags="
            "[pretraining,step1,batch-warmdown,best-coordinate,dr,bs256,decay-replay,heldout,downstream-nine]",
        ),
        (
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=",
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=true",
        ),
        (
            "--trainer.callbacks.checkpointer.fixed_steps=",
            f"--trainer.callbacks.checkpointer.fixed_steps=[{END_STEP}]",
        ),
        ("--trainer.load_path=", f"--trainer.load_path={SOURCE_CHECKPOINT}"),
        ("--trainer.load_trainer_state=", "--trainer.load_trainer_state=true"),
        ("--trainer.load_optim_state=", "--trainer.load_optim_state=true"),
        ("--trainer.prefer_explicit_load_path=", "--trainer.prefer_explicit_load_path=true"),
    )
    for prefix, replacement in replacements:
        arguments = upsert(arguments, prefix, replacement)

    commands = [
        "set -euo pipefail",
        shlex.join(["test", "-d", SOURCE_CHECKPOINT]),
        f"if compgen -G {shlex.quote(OUTPUT + '/step*')} > /dev/null; then "
        f"echo {shlex.quote('BS256_DECAY_EVAL_PREFLIGHT_ERROR checkpoint already exists under: ' + OUTPUT)} >&2; exit 20; fi",
        f'test "$(git rev-parse HEAD)" = "{args.revision}" || '
        f'{{ echo "BS256_DECAY_EVAL_PREFLIGHT_ERROR revision mismatch: expected {args.revision}, got $(git rev-parse HEAD)" >&2; exit 21; }}',
        shlex.join(
            [
                "echo",
                "BS256_DECAY_EVAL_PREFLIGHT "
                f"source_step={PRE_DECAY_STEP} endpoint_step={END_STEP} "
                f"decay_steps={DECAY_STEPS} lr=1e-3 wd=0.333 "
                "load_trainer_state=true load_optim_state=true recalibration=false",
            ]
        ),
        shlex.join(["python", script, RUN_NAME, "--dry-run", *arguments]),
        shlex.join(["torchrun", "--nproc-per-node=8", script, RUN_NAME, *arguments]),
        shlex.join(["test", "-d", f"{OUTPUT}/step{END_STEP}"]),
    ]
    task["name"] = "main"
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": "0s", "autoResume": False}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for env in task.setdefault("envVars", []):
        if env.get("name") == "GIT_REF":
            env["value"] = args.revision
        elif env.get("name") == "GIT_BRANCH":
            env["value"] = "sewonm/icsl"
    built.pop("description", None)
    return built


def validate_spec(spec: dict[str, Any]) -> None:
    shell = spec["tasks"][0]["arguments"][2]
    required = (
        SOURCE_CHECKPOINT,
        OUTPUT,
        "source_step=2145 endpoint_step=2384 decay_steps=239",
        "--trainer.max_duration={value: 2384, unit: steps}",
        "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=true",
        "--trainer.prefer_explicit_load_path=true",
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=true",
        "--trainer.callbacks.checkpointer.fixed_steps=[2384]",
    )
    missing = [value for value in required if value not in shell]
    if missing:
        raise RuntimeError(f"decay-eval spec is missing required configuration: {missing}")
    forbidden = ("AdamSecondMomentBatchRecalibrationCallback", "OptimizerHyperparameterTransitionCallback")
    present = [value for value in forbidden if value in shell]
    if present:
        raise RuntimeError(f"decay replay must not reapply transition callbacks: {present}")


def main() -> None:
    args = parse_args()
    spec = build_spec(parent_spec(), args)
    validate_spec(spec)
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    created = subprocess.run(
        [
            "beaker",
            "experiment",
            "create",
            "-",
            "--name",
            args.name,
            "--workspace",
            args.workspace,
            "--priority",
            args.priority,
        ],
        check=True,
        input=json.dumps(spec),
        stdout=subprocess.PIPE,
        text=True,
    )
    print(created.stdout, end="")


if __name__ == "__main__":
    main()
