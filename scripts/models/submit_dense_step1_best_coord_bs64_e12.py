#!/usr/bin/env python3
"""Submit the authorized Dense-1B best-coordinate BS64 E8 -> E12 continuation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


SOURCE_EXPERIMENT = "01M00EJZDCJSY5QTC5YVZPZBS6"
ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"
SOURCE_NAME = (
    "bs64_dr_lr1e-3_wd0.3_init=bs256e4_lr1e-03_wd0.333_"
    "init=bs1024e2_lr5e-04_wd0.333"
)
OUTPUT_NAME = (
    "bs64_dr_lr1e-3_wd0.3_init=bs64e8_lr1e-03_wd0.3_"
    "init=bs256e4_lr1e-03_wd0.333_init=bs1024e2_lr5e-04_wd0.333"
)
SOURCE_STEP = 15_879
SOURCE = f"{ROOT}/{SOURCE_NAME}/step{SOURCE_STEP}"
OUTPUT = f"{ROOT}/{OUTPUT_NAME}"
TRAIN_SCRIPT = "src/scripts/train/olmo2-1B.py"
TRAIN_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json"
VALIDATION_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_validation.json"
TOKENS_PER_EPOCH = 1_000_000_000
STAGE_EPOCHS = 4
SEQUENCE_LENGTH = 4096
BATCH_SEQUENCES = 64
ADDED_STEPS = math.ceil(
    STAGE_EPOCHS * TOKENS_PER_EPOCH / (BATCH_SEQUENCES * SEQUENCE_LENGTH)
)
END_STEP = SOURCE_STEP + ADDED_STEPS
PRE_DECAY_STEP = END_STEP - round(END_STEP * 0.1) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--name", default="dense-1b-best-coordinate-batch-warmdown-e12"
    )
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full Git commit SHA")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.name):
        parser.error("--name must be a lowercase Beaker name component")
    return args


def source_spec() -> dict[str, Any]:
    completed = subprocess.run(
        ["beaker", "experiment", "spec", SOURCE_EXPERIMENT, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def extract_source_args(spec: dict[str, Any]) -> list[str]:
    for task in spec.get("tasks", []):
        shell = task.get("arguments", [])
        if shell[:2] != ["bash", "-lc"] or len(shell) != 3:
            continue
        for line in shell[2].splitlines():
            if not line.startswith("torchrun "):
                continue
            parts = shlex.split(line)
            if TRAIN_SCRIPT not in parts:
                continue
            index = parts.index(TRAIN_SCRIPT)
            if parts[index + 1] == SOURCE_NAME:
                return parts[index + 2 :]
    raise RuntimeError("could not extract the completed best-coordinate BS64 arguments")


def one(arguments: list[str], prefix: str) -> str:
    values = [value.split("=", 1)[1] for value in arguments if value.startswith(prefix)]
    if len(values) != 1:
        raise RuntimeError(f"expected one {prefix}, found {values}")
    return values[0]


def validate_source(arguments: list[str]) -> None:
    expected = {
        "--data_loader.global_batch_size=": str(BATCH_SEQUENCES * SEQUENCE_LENGTH),
        "--lr=": "1e-3",
        "--train_module.optim.weight_decay=": "0.3",
    }
    for prefix, value in expected.items():
        actual = one(arguments, prefix)
        if actual != value:
            raise RuntimeError(f"source {prefix} is {actual}, expected {value}")
    if "--dynamic-repacking" not in arguments:
        raise RuntimeError("source experiment is not dynamic repacking")


def upsert(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    output: list[str] = []
    found = False
    for value in arguments:
        if value.startswith(prefix):
            if not found:
                output.append(replacement)
                found = True
        else:
            output.append(value)
    if not found:
        output.append(replacement)
    return output


def continuation_args(source_args: list[str]) -> list[str]:
    excluded = (
        "--load_path=",
        "--load_trainer_state=",
        "--load_optim_state=",
        "--trainer.load_path=",
        "--trainer.load_trainer_state=",
        "--trainer.load_optim_state=",
        "--trainer.prefer_explicit_load_path=",
        "--trainer.reset_data_loader_state_on_load_path=",
        "--train_module.validate_optimizer_hyperparameters_on_load=",
        "--trainer.callbacks.batch_warmdown_",
    )
    arguments = [value for value in source_args if not value.startswith(excluded)]
    replacements = (
        ("--save-folder=", f"--save-folder={OUTPUT}"),
        (
            "--trainer.max_duration=",
            f"--trainer.max_duration={{value: {END_STEP}, unit: steps}}",
        ),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={OUTPUT_NAME}"),
        (
            "--trainer.callbacks.wandb.tags=",
            "--trainer.callbacks.wandb.tags="
            "[pretraining,step1,batch-warmdown,best-coordinate,e12,dr,bs64]",
        ),
        (
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=",
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
        ),
        (
            "--trainer.callbacks.heldout_evaluator=",
            "--trainer.callbacks.heldout_evaluator="
            + one(source_args, "--trainer.callbacks.heldout_evaluator="),
        ),
        (
            "--trainer.callbacks.checkpointer.fixed_steps=",
            f"--trainer.callbacks.checkpointer.fixed_steps=[{PRE_DECAY_STEP}, {END_STEP}]",
        ),
        (
            "--trainer.callbacks.checkpointer.save_interval=",
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
        ),
        (
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        ),
        (
            "--data_loader.global_batch_size=",
            f"--data_loader.global_batch_size={BATCH_SEQUENCES * SEQUENCE_LENGTH}",
        ),
        (
            "--data_loader.restore_data_order_from_state=",
            "--data_loader.restore_data_order_from_state=true",
        ),
        (
            "--data_loader.ignore_fingerprint_mismatch=",
            "--data_loader.ignore_fingerprint_mismatch=false",
        ),
        (
            "--train_module.scheduler=",
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, "
            "units: steps, warmup: 24, decay_fraction: 0.1}",
        ),
        ("--train_module.optim.weight_decay=", "--train_module.optim.weight_decay=0.3"),
        ("--lr=", "--lr=1e-3"),
    )
    for prefix, replacement in replacements:
        arguments = upsert(arguments, prefix, replacement)
    arguments.extend(
        [
            f"--trainer.load_path={SOURCE}",
            "--trainer.load_trainer_state=true",
            "--trainer.load_optim_state=true",
            "--trainer.prefer_explicit_load_path=true",
            "--trainer.reset_data_loader_state_on_load_path=false",
            "--train_module.validate_optimizer_hyperparameters_on_load=true",
        ]
    )
    return arguments


def set_env(task: dict[str, Any], name: str, value: str) -> None:
    for env in task.setdefault("envVars", []):
        if env.get("name") == name:
            env["value"] = value
            return
    task["envVars"].append({"name": name, "value": value})


def build_spec(
    spec: dict[str, Any], args: argparse.Namespace, train_args: list[str]
) -> dict[str, Any]:
    built = copy.deepcopy(spec)
    task = copy.deepcopy(built["tasks"][0])
    built["tasks"] = [task]
    commands = [
        "set -euo pipefail",
        f"test -d {shlex.quote(SOURCE)} || {{ echo {shlex.quote('E12_PREFLIGHT_ERROR missing source: ' + SOURCE)} >&2; exit 40; }}",
        f"test -f {shlex.quote(TRAIN_MANIFEST)} || exit 41",
        f"test -f {shlex.quote(VALIDATION_MANIFEST)} || exit 42",
        f"if compgen -G {shlex.quote(OUTPUT + '/step*')} > /dev/null; then echo {shlex.quote('E12_PREFLIGHT_ERROR output checkpoint exists: ' + OUTPUT)} >&2; exit 43; fi",
        f'test "$(git rev-parse HEAD)" = "{args.revision}" || {{ echo "E12_PREFLIGHT_ERROR revision mismatch" >&2; exit 44; }}',
        shlex.join(
            [
                "echo",
                "BEST_COORD_BS64_E12_PREFLIGHT source_step=15879 source_epoch=8 "
                f"added={ADDED_STEPS} pre_decay={PRE_DECAY_STEP} end={END_STEP} "
                "batch=64 lr=1e-3 wd=0.3 dr=true recalibration=false",
            ]
        ),
        shlex.join(["python", TRAIN_SCRIPT, OUTPUT_NAME, "--dry-run", *train_args]),
        shlex.join(
            ["torchrun", "--nproc-per-node=8", TRAIN_SCRIPT, OUTPUT_NAME, *train_args]
        ),
        shlex.join(["test", "-d", f"{OUTPUT}/step{PRE_DECAY_STEP}"]),
        shlex.join(["test", "-d", f"{OUTPUT}/step{END_STEP}"]),
    ]
    task["name"] = "main"
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    blocked = {"GANTRY_RDZV_ID", "GANTRY_RDZV_PORT", "GANTRY_USE_TORCHRUN", "NUM_NODES"}
    task["envVars"] = [
        env
        for env in task.get("envVars", [])
        if env.get("name") not in blocked
        and not str(env.get("name", "")).startswith("BEAKER_REPLICA_")
    ]
    set_env(task, "GIT_REF", args.revision)
    set_env(task, "GIT_BRANCH", "sewonm/icsl")
    set_env(task, "NUM_NODES", "1")
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": "0s", "autoResume": False}
    task["hostNetworking"] = False
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for field in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
        task.pop(field, None)
    built.pop("description", None)
    return built


def validate(spec: dict[str, Any]) -> None:
    shell = spec["tasks"][0]["arguments"][2]
    required = (
        SOURCE,
        OUTPUT,
        "source_step=15879 source_epoch=8 added=15259 pre_decay=28023 end=31138",
        "--trainer.max_duration={value: 31138, unit: steps}",
        "--trainer.callbacks.checkpointer.fixed_steps=[28023, 31138]",
        "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=true",
        "--train_module.validate_optimizer_hyperparameters_on_load=true",
        "--data_loader.restore_data_order_from_state=true",
        "--dynamic-repacking",
    )
    missing = [value for value in required if value not in shell]
    if missing:
        raise RuntimeError(f"E12 spec missing: {missing}")
    forbidden = ("AdamSecondMomentBatchRecalibrationCallback", "OptimizerHyperparameterTransitionCallback")
    present = [value for value in forbidden if value in shell]
    if present:
        raise RuntimeError(f"same-batch E12 continuation must not recalibrate: {present}")


def main() -> None:
    args = parse_args()
    spec = source_spec()
    source_args = extract_source_args(spec)
    validate_source(source_args)
    train_args = continuation_args(source_args)
    built = build_spec(spec, args, train_args)
    validate(built)
    if args.print_only:
        json.dump(built, sys.stdout, indent=2)
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
        input=json.dumps(built),
        stdout=subprocess.PIPE,
        text=True,
    )
    print(created.stdout, end="")


if __name__ == "__main__":
    main()
