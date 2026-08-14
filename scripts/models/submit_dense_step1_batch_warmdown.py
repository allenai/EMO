#!/usr/bin/env python3
"""Submit the guarded Dense-1B BS1024 -> BS256 -> BS64 DR warmdown chain."""

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


SOURCE_EXPERIMENT = "01KZRTKNSSX9H1DPZ7NHJSM28F"
SOURCE_CHECKPOINT = (
    "/weka/oe-training-default/sewonm/icsl/models/"
    "dense_1b_step1_0802_repeated_dclm1b_wsd_bs1024_dr_e2_"
    "lr1e-3_wd0.333_warmup24_dataloader-cd-r01/step477"
)
OUTPUT_ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"
BS256_NAME = "bs256_dr_init=bs1024e2_lr1e-03_wd0.333"
BS64_NAME = "bs64_dr_init=bs256e4_init=bs1024e2_lr1e-03_wd0.333"
BS256_OUTPUT = f"{OUTPUT_ROOT}/{BS256_NAME}"
BS64_OUTPUT = f"{OUTPUT_ROOT}/{BS64_NAME}"
TRAIN_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json"
VALIDATION_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_validation.json"
TRAIN_SCRIPT = "src/scripts/train/olmo2-1B.py"
LR = "1e-3"
WD = "0.333"
SEQUENCE_LENGTH = 4096
RANK_MICROBATCH_SEQUENCES = 8
BS256_END_STEP = 2384
BS64_END_STEP = 17643


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", default="dense-1b-batch-warmdown-bs256-bs64")
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.name):
        parser.error("--name must be a lowercase Beaker name component")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full Git commit SHA")
    return args


def source_spec() -> dict[str, Any]:
    completed = subprocess.run(
        ["beaker", "experiment", "spec", SOURCE_EXPERIMENT, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def extract_training_arguments(spec: dict[str, Any]) -> list[str]:
    for task in spec.get("tasks", []):
        shell = task.get("arguments", [])
        if shell[:2] != ["bash", "-lc"] or len(shell) != 3:
            continue
        for line in shell[2].splitlines():
            if not line.startswith("torchrun "):
                continue
            parts = shlex.split(line)
            indices = [index for index, value in enumerate(parts) if value == TRAIN_SCRIPT]
            if parts[:1] == ["torchrun"] and len(indices) == 1:
                index = indices[0]
                return parts[index + 2 :]
    raise RuntimeError("could not extract the source torchrun arguments")


def values_for(arguments: list[str], prefix: str) -> list[str]:
    return [value.split("=", 1)[1] for value in arguments if value.startswith(prefix)]


def unique_value(arguments: list[str], prefix: str) -> str:
    values = values_for(arguments, prefix)
    if len(values) != 1:
        raise RuntimeError(f"expected one {prefix}, found {values}")
    return values[0]


def validate_source(arguments: list[str]) -> None:
    expected = {
        "--data_loader.global_batch_size=": str(1024 * SEQUENCE_LENGTH),
        "--lr=": LR,
        "--train_module.optim.weight_decay=": WD,
    }
    for prefix, value in expected.items():
        actual = unique_value(arguments, prefix)
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


def stage_arguments(
    source_arguments: list[str],
    *,
    batch_sequences: int,
    end_step: int,
    source_checkpoint: str,
    output: str,
    callback_name: str,
    expected_step: int,
) -> list[str]:
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
        "--trainer.callbacks.batch_warmdown_recalibration_",
    )
    arguments = [value for value in source_arguments if not value.startswith(excluded)]
    name = Path(output).name
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        ("--trainer.max_duration=", f"--trainer.max_duration={{value: {end_step}, unit: steps}}"),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
        (
            "--trainer.callbacks.wandb.tags=",
            "--trainer.callbacks.wandb.tags="
            f"[pretraining,step1,batch-warmdown,dr,bs{batch_sequences},lr1e-3,wd0p333]",
        ),
        ("--trainer.callbacks.downstream_evaluator.eval_on_finish=", "--trainer.callbacks.downstream_evaluator.eval_on_finish=false"),
        ("--trainer.callbacks.checkpointer.fixed_steps=", f"--trainer.callbacks.checkpointer.fixed_steps=[{end_step}]"),
        ("--trainer.callbacks.checkpointer.save_interval=", "--trainer.callbacks.checkpointer.save_interval=1000000000"),
        ("--trainer.callbacks.checkpointer.ephemeral_save_interval=", "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999"),
        ("--data_loader.global_batch_size=", f"--data_loader.global_batch_size={batch_sequences * SEQUENCE_LENGTH}"),
        ("--data_loader.restore_data_order_from_state=", "--data_loader.restore_data_order_from_state=true"),
        ("--data_loader.ignore_fingerprint_mismatch=", "--data_loader.ignore_fingerprint_mismatch=false"),
        ("--train_module.rank_microbatch_size=", f"--train_module.rank_microbatch_size={RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH}"),
        (
            "--train_module.scheduler=",
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: 24, decay_fraction: 0.1}",
        ),
        ("--train_module.optim.weight_decay=", f"--train_module.optim.weight_decay={WD}"),
        ("--lr=", f"--lr={LR}"),
    )
    for prefix, replacement in replacements:
        arguments = upsert(arguments, prefix, replacement)
    arguments.extend(
        [
            f"--trainer.load_path={source_checkpoint}",
            "--trainer.load_trainer_state=true",
            "--trainer.load_optim_state=true",
            "--trainer.prefer_explicit_load_path=true",
            "--trainer.reset_data_loader_state_on_load_path=false",
            "--train_module.validate_optimizer_hyperparameters_on_load=true",
            f"--trainer.callbacks.{callback_name}="
            "{_CLASS_: olmo_core.train.callbacks.optimizer_recalibration."
            "AdamSecondMomentBatchRecalibrationCallback, batch_size_ratio: 4.0, "
            f"expected_step: {expected_step}}}",
        ]
    )
    return arguments


def set_env(task: dict[str, Any], name: str, value: str) -> None:
    for env in task.setdefault("envVars", []):
        if env.get("name") == name:
            env["value"] = value
            return
    task["envVars"].append({"name": name, "value": value})


def build_spec(spec: dict[str, Any], args: argparse.Namespace, source_args: list[str]) -> dict[str, Any]:
    built = copy.deepcopy(spec)
    if not built.get("tasks"):
        raise RuntimeError("source experiment has no tasks")
    task = copy.deepcopy(built["tasks"][0])
    built["tasks"] = [task]
    first = stage_arguments(
        source_args,
        batch_sequences=256,
        end_step=BS256_END_STEP,
        source_checkpoint=SOURCE_CHECKPOINT,
        output=BS256_OUTPUT,
        callback_name="batch_warmdown_recalibration_bs256",
        expected_step=477,
    )
    second = stage_arguments(
        source_args,
        batch_sequences=64,
        end_step=BS64_END_STEP,
        source_checkpoint=f"{BS256_OUTPUT}/step{BS256_END_STEP}",
        output=BS64_OUTPUT,
        callback_name="batch_warmdown_recalibration_bs64",
        expected_step=BS256_END_STEP,
    )
    commands = [
        "set -euo pipefail",
        shlex.join(["test", "-d", SOURCE_CHECKPOINT]),
        shlex.join(["test", "-f", TRAIN_MANIFEST]),
        shlex.join(["test", "-f", VALIDATION_MANIFEST]),
        shlex.join(["test", "!", "-e", BS256_OUTPUT]),
        shlex.join(["test", "!", "-e", BS64_OUTPUT]),
        f'test "$(git rev-parse HEAD)" = "{args.revision}"',
        shlex.join(["echo", "BATCH_WARMDOWN_PREFLIGHT source_step=477 bs256_end=2384 bs64_end=17643 lr=1e-3 wd=0.333 dr=true v_recal_ratios=4,4"]),
        shlex.join(["python", TRAIN_SCRIPT, BS256_NAME, "--dry-run", *first]),
        shlex.join(["torchrun", "--nproc-per-node=8", TRAIN_SCRIPT, BS256_NAME, *first]),
        shlex.join(["test", "-d", f"{BS256_OUTPUT}/step{BS256_END_STEP}"]),
        shlex.join(["python", TRAIN_SCRIPT, BS64_NAME, "--dry-run", *second]),
        shlex.join(["torchrun", "--nproc-per-node=8", TRAIN_SCRIPT, BS64_NAME, *second]),
        shlex.join(["test", "-d", f"{BS64_OUTPUT}/step{BS64_END_STEP}"]),
    ]
    task["name"] = "main"
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    blocked = {"GANTRY_RDZV_ID", "GANTRY_RDZV_PORT", "GANTRY_USE_TORCHRUN", "NUM_NODES"}
    task["envVars"] = [
        env for env in task.get("envVars", [])
        if env.get("name") not in blocked and not str(env.get("name", "")).startswith("BEAKER_REPLICA_")
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


def validate_spec(spec: dict[str, Any]) -> None:
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("warmdown spec must contain exactly one task")
    task = spec["tasks"][0]
    shell = task.get("arguments", [None, None, ""])[2]
    required = [
        SOURCE_CHECKPOINT,
        BS256_OUTPUT,
        BS64_OUTPUT,
        "batch_warmdown_recalibration_bs256",
        "batch_warmdown_recalibration_bs64",
        "batch_size_ratio: 4.0",
        "expected_step: 477",
        "expected_step: 2384",
        "--trainer.max_duration={value: 2384, unit: steps}",
        "--trainer.max_duration={value: 17643, unit: steps}",
        "--data_loader.global_batch_size=1048576",
        "--data_loader.global_batch_size=262144",
        "--data_loader.restore_data_order_from_state=true",
        "--data_loader.ignore_fingerprint_mismatch=false",
        "--trainer.load_optim_state=true",
        "--trainer.load_trainer_state=true",
        "--trainer.prefer_explicit_load_path=true",
        "--train_module.validate_optimizer_hyperparameters_on_load=true",
        "--dynamic-repacking",
        "step2384",
        "step17643",
    ]
    missing = [value for value in required if value not in shell]
    if missing:
        raise RuntimeError(f"warmdown spec is missing guards/config: {missing}")
    if task.get("resources", {}).get("gpuCount") != 8:
        raise RuntimeError("warmdown must request one 8-GPU node")


def experiment_id(output: str) -> str:
    matches = re.findall(r"\b01[A-Z0-9]{24}\b", output)
    if not matches:
        raise RuntimeError(f"could not parse experiment ID from {output!r}")
    return matches[0]


def main() -> None:
    args = parse_args()
    spec = source_spec()
    source_args = extract_training_arguments(spec)
    validate_source(source_args)
    built = build_spec(spec, args, source_args)
    validate_spec(built)
    if args.print_only:
        json.dump(built, sys.stdout, indent=2)
        print()
        return
    created = subprocess.run(
        [
            "beaker", "experiment", "create", "-", "--name", args.name,
            "--workspace", args.workspace, "--priority", args.priority,
        ],
        check=True,
        input=json.dumps(built),
        stdout=subprocess.PIPE,
        text=True,
    )
    print(experiment_id(created.stdout))
    print(created.stdout, end="")


if __name__ == "__main__":
    main()
