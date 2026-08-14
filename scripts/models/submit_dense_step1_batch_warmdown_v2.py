#!/usr/bin/env python3
"""Submit Dense-1B batch-warmdown Version 2 (BS1024 E4 -> BS256 E8 -> BS64 E12)."""

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


SOURCE_EXPERIMENT = "01KZSBTR8N37J6F9J3A5GYTNBC"
ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"
SOURCE = f"{ROOT}/bs1024_dr_lr1e-3_wd0.333/step858"
BS256_NAME = "bs256_dr_lr1e-3_wd0.333_init=bs1024e4_lr1e-03_wd0.333"
BS64_NAME = (
    "bs64_dr_lr1e-3_wd0.3_init=bs256e8_lr1e-03_wd0.333_"
    "init=bs1024e4_lr1e-03_wd0.333"
)
BS256_OUTPUT = f"{ROOT}/{BS256_NAME}"
BS64_OUTPUT = f"{ROOT}/{BS64_NAME}"
TRAIN_SCRIPT = "src/scripts/train/olmo2-1B.py"
TRAIN_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json"
VALIDATION_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_validation.json"
TOKENS_PER_EPOCH = 1_000_000_000
SEQUENCE_LENGTH = 4096
SOURCE_STEP = 858
SOURCE_EPOCH = 4
STAGE_EPOCHS = 4


def added_steps(batch_sequences: int) -> int:
    return math.ceil(STAGE_EPOCHS * TOKENS_PER_EPOCH / (batch_sequences * SEQUENCE_LENGTH))


def pre_decay_step(endpoint: int) -> int:
    return endpoint - round(endpoint * 0.1) - 1


BS256_ADDED = added_steps(256)
BS256_END = SOURCE_STEP + BS256_ADDED
BS256_PRE_DECAY = pre_decay_step(BS256_END)
BS64_ADDED = added_steps(64)
BS64_END = BS256_PRE_DECAY + BS64_ADDED
BS64_PRE_DECAY = pre_decay_step(BS64_END)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", default="dense-1b-batch-warmdown-v2")
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
            return parts[index + 2 :]
    raise RuntimeError("could not extract source E4 training arguments")


def one(arguments: list[str], prefix: str) -> str:
    values = [value.split("=", 1)[1] for value in arguments if value.startswith(prefix)]
    if len(values) != 1:
        raise RuntimeError(f"expected one {prefix}, found {values}")
    return values[0]


def validate_source(arguments: list[str]) -> None:
    expected = {
        "--data_loader.global_batch_size=": str(1024 * SEQUENCE_LENGTH),
        "--lr=": "1e-3",
        "--train_module.optim.weight_decay=": "0.333",
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


def stage_args(
    source_args: list[str],
    *,
    batch: int,
    end: int,
    pre_decay: int,
    load_path: str,
    output: str,
    expected_step: int,
    source_wd: str,
    target_wd: str,
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
        "--trainer.callbacks.batch_warmdown_",
    )
    arguments = [value for value in source_args if not value.startswith(excluded)]
    name = Path(output).name
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        ("--trainer.max_duration=", f"--trainer.max_duration={{value: {end}, unit: steps}}"),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
        (
            "--trainer.callbacks.wandb.tags=",
            f"--trainer.callbacks.wandb.tags=[pretraining,step1,batch-warmdown,v2,dr,bs{batch}]",
        ),
        (
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=",
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
        ),
        (
            "--trainer.callbacks.checkpointer.fixed_steps=",
            f"--trainer.callbacks.checkpointer.fixed_steps=[{pre_decay}, {end}]",
        ),
        ("--trainer.callbacks.checkpointer.save_interval=", "--trainer.callbacks.checkpointer.save_interval=1000000000"),
        ("--trainer.callbacks.checkpointer.ephemeral_save_interval=", "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999"),
        ("--data_loader.global_batch_size=", f"--data_loader.global_batch_size={batch * SEQUENCE_LENGTH}"),
        ("--data_loader.restore_data_order_from_state=", "--data_loader.restore_data_order_from_state=true"),
        ("--data_loader.ignore_fingerprint_mismatch=", "--data_loader.ignore_fingerprint_mismatch=false"),
        ("--train_module.rank_microbatch_size=", f"--train_module.rank_microbatch_size={8 * SEQUENCE_LENGTH}"),
        (
            "--train_module.scheduler=",
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: 24, decay_fraction: 0.1}",
        ),
        ("--train_module.optim.weight_decay=", f"--train_module.optim.weight_decay={target_wd}"),
        ("--lr=", "--lr=1e-3"),
    )
    for prefix, replacement in replacements:
        arguments = upsert(arguments, prefix, replacement)
    suffix = f"bs{batch}"
    arguments.extend(
        [
            f"--trainer.load_path={load_path}",
            "--trainer.load_trainer_state=true",
            "--trainer.load_optim_state=true",
            "--trainer.prefer_explicit_load_path=true",
            "--trainer.reset_data_loader_state_on_load_path=false",
            "--train_module.validate_optimizer_hyperparameters_on_load=false",
            f"--trainer.callbacks.batch_warmdown_hyperparameters_{suffix}="
            "{_CLASS_: olmo_core.train.callbacks.optimizer_recalibration."
            "OptimizerHyperparameterTransitionCallback, "
            f"expected_step: {expected_step}, source_lr: 1e-3, target_lr: 1e-3, "
            f"source_weight_decay: {source_wd}, target_weight_decay: {target_wd}}}",
            f"--trainer.callbacks.batch_warmdown_recalibration_{suffix}="
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
    bs256 = stage_args(
        source_args,
        batch=256,
        end=BS256_END,
        pre_decay=BS256_PRE_DECAY,
        load_path=SOURCE,
        output=BS256_OUTPUT,
        expected_step=SOURCE_STEP,
        source_wd="0.333",
        target_wd="0.333",
    )
    bs64 = stage_args(
        source_args,
        batch=64,
        end=BS64_END,
        pre_decay=BS64_PRE_DECAY,
        load_path=f"{BS256_OUTPUT}/step{BS256_PRE_DECAY}",
        output=BS64_OUTPUT,
        expected_step=BS256_PRE_DECAY,
        source_wd="0.333",
        target_wd="0.3",
    )
    commands = [
        "set -euo pipefail",
        f"test -d {shlex.quote(SOURCE)} || {{ echo {shlex.quote('V2_PREFLIGHT_ERROR missing source: ' + SOURCE)} >&2; exit 30; }}",
        f"test -f {shlex.quote(TRAIN_MANIFEST)} || exit 31",
        f"test -f {shlex.quote(VALIDATION_MANIFEST)} || exit 32",
        f"if compgen -G {shlex.quote(BS256_OUTPUT + '/step*')} > /dev/null; then echo {shlex.quote('V2_PREFLIGHT_ERROR BS256 checkpoint exists')} >&2; exit 33; fi",
        f"if compgen -G {shlex.quote(BS64_OUTPUT + '/step*')} > /dev/null; then echo {shlex.quote('V2_PREFLIGHT_ERROR BS64 checkpoint exists')} >&2; exit 34; fi",
        f'test "$(git rev-parse HEAD)" = "{args.revision}" || {{ echo "V2_PREFLIGHT_ERROR revision mismatch" >&2; exit 35; }}',
        shlex.join(
            [
                "echo",
                "BATCH_WARMDOWN_V2_PREFLIGHT source_step=858 source_epoch=4 "
                f"bs256_added={BS256_ADDED} bs256_pre_decay={BS256_PRE_DECAY} bs256_end={BS256_END} "
                f"bs64_added={BS64_ADDED} bs64_pre_decay={BS64_PRE_DECAY} bs64_end={BS64_END} "
                "ratios=4,4 lr=1e-3 wds=0.333,0.3 dr=true",
            ]
        ),
        shlex.join(["python", TRAIN_SCRIPT, BS256_NAME, "--dry-run", *bs256]),
        shlex.join(["torchrun", "--nproc-per-node=8", TRAIN_SCRIPT, BS256_NAME, *bs256]),
        shlex.join(["test", "-d", f"{BS256_OUTPUT}/step{BS256_PRE_DECAY}"]),
        shlex.join(["test", "-d", f"{BS256_OUTPUT}/step{BS256_END}"]),
        shlex.join(["python", TRAIN_SCRIPT, BS64_NAME, "--dry-run", *bs64]),
        shlex.join(["torchrun", "--nproc-per-node=8", TRAIN_SCRIPT, BS64_NAME, *bs64]),
        shlex.join(["test", "-d", f"{BS64_OUTPUT}/step{BS64_PRE_DECAY}"]),
        shlex.join(["test", "-d", f"{BS64_OUTPUT}/step{BS64_END}"]),
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


def validate(spec: dict[str, Any]) -> None:
    shell = spec["tasks"][0]["arguments"][2]
    required = (
        SOURCE,
        BS256_OUTPUT,
        BS64_OUTPUT,
        "source_step=858",
        "bs256_added=3815 bs256_pre_decay=4205 bs256_end=4673",
        "bs64_added=15259 bs64_pre_decay=17517 bs64_end=19464",
        "expected_step: 858",
        "expected_step: 4205",
        "batch_size_ratio: 4.0",
        "source_weight_decay: 0.333, target_weight_decay: 0.3",
        "--trainer.load_optim_state=true",
        "--trainer.prefer_explicit_load_path=true",
        "--dynamic-repacking",
    )
    missing = [value for value in required if value not in shell]
    if missing:
        raise RuntimeError(f"Version 2 spec missing: {missing}")


def main() -> None:
    args = parse_args()
    spec = source_spec()
    source_args = extract_source_args(spec)
    validate_source(source_args)
    built = build_spec(spec, args, source_args)
    validate(built)
    if args.print_only:
        json.dump(built, sys.stdout, indent=2)
        print()
        return
    created = subprocess.run(
        ["beaker", "experiment", "create", "-", "--name", args.name, "--workspace", args.workspace, "--priority", args.priority],
        check=True,
        input=json.dumps(built),
        stdout=subprocess.PIPE,
        text=True,
    )
    print(created.stdout, end="")


if __name__ == "__main__":
    main()
