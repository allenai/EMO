#!/usr/bin/env python3
"""Submit one exact-checkpoint decay endpoint for a dense-1B batch simulation.

This launcher is deliberately narrow: it resumes a registered structured-noise
or Local-SGD trajectory from an exact retained pre-decay checkpoint, restores
trainer state, runs only the target endpoint's 10% WSD decay, and evaluates the
held-out DCLM set plus all downstream tasks inherited from the parent job.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any


TOKENS_PER_EPOCH = 1_000_000_000
SEQUENCE_LENGTH = 4096
GLOBAL_SEQUENCES = 512
DECAY_FRACTION = 0.1
REPORT = Path("reports/0802/data/wsd_batch_simulation_1b.json")
METHODS = (
    "structured_noise",
    "local_sgd_h1",
    "local_sgd_h2",
    "local_sgd_h4",
    "local_sgd_h16",
)


def total_steps(epoch: int) -> int:
    return math.ceil(epoch * TOKENS_PER_EPOCH / (GLOBAL_SEQUENCES * SEQUENCE_LENGTH))


def pre_decay_step(epoch: int) -> int:
    endpoint = total_steps(epoch)
    return endpoint - round(DECAY_FRACTION * endpoint) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--target-epoch", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--learning-rate", default="1e-3")
    parser.add_argument("--weight-decay", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    expected = pre_decay_step(args.target_epoch)
    if not args.source_checkpoint.endswith(f"/step{expected}"):
        parser.error(
            f"target E{args.target_epoch} requires exact source ending in /step{expected}"
        )
    return args


def get_spec(experiment: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def extract_training_command(task: dict[str, Any]) -> tuple[str, str, list[str], int]:
    arguments = task.get("arguments", [])
    if arguments[:2] != ["bash", "-lc"] or len(arguments) != 3:
        raise RuntimeError("expected a bash task containing torchrun")
    for line in arguments[2].splitlines():
        parts = shlex.split(line)
        if not parts or parts[0] != "torchrun":
            continue
        nproc_values = [part for part in parts[1:] if part.startswith("--nproc-per-node=")]
        if len(nproc_values) != 1:
            raise RuntimeError("parent torchrun has no unique nproc-per-node")
        nproc = int(nproc_values[0].split("=", 1)[1])
        script_index = parts.index(nproc_values[0]) + 1
        return parts[script_index], parts[script_index + 1], parts[script_index + 2 :], nproc
    raise RuntimeError("could not find parent torchrun command")


def upsert(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    found = False
    output: list[str] = []
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


def registered_plan(args: argparse.Namespace) -> dict[str, Any]:
    report = json.loads(REPORT.read_text())
    matches = [
        run
        for run in report["runs"]
        if run.get("method") == args.method
        and run.get("targetEpoch") == args.target_epoch
        and run.get("batchSequences") == GLOBAL_SEQUENCES
        and run.get("simulatedBatchSequences") == 64
        and run.get("lr") == args.learning_rate
        and run.get("wd") == args.weight_decay
        and run.get("sourceCheckpoint") == args.source_checkpoint
    ]
    if len(matches) != 1 or matches[0].get("status") not in {
        "planned",
        "print-only-verified",
    }:
        raise RuntimeError(
            "expected exactly one planned registered simulation-decay tuple; "
            f"found {len(matches)}"
        )
    parents = [
        run
        for run in report["runs"]
        if run.get("beaker") == args.base_experiment
        and run.get("method") == args.method
        and run.get("batchSequences") == GLOBAL_SEQUENCES
        and run.get("simulatedBatchSequences") == 64
        and run.get("lr") == args.learning_rate
        and run.get("wd") == args.weight_decay
        and pre_decay_step(args.target_epoch) in run.get("retainedPreDecaySteps", [])
    ]
    if len(parents) != 1:
        raise RuntimeError(
            "source experiment is not the unique registered matching parent retaining "
            f"step {pre_decay_step(args.target_epoch)}"
        )
    return matches[0]


def build_spec(base_spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("expected one parent task")
    task = spec["tasks"][0]
    script, parent_name, parent_arguments, nproc = extract_training_command(task)
    if args.method not in parent_name:
        raise RuntimeError("parent run name does not match requested simulation method")

    endpoint = total_steps(args.target_epoch)
    source = pre_decay_step(args.target_epoch)
    name = f"{parent_name}_e{args.target_epoch}_decay_{args.suffix}"
    output = f"/weka/oe-training-default/sewonm/icsl/models/{name}"
    arguments = [
        value
        for value in parent_arguments
        if not value.startswith(("--load_path=", "--load_trainer_state="))
    ]
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        (
            "--trainer.max_duration=",
            f"--trainer.max_duration={{value: {args.target_epoch * TOKENS_PER_EPOCH}, unit: tokens}}",
        ),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
        (
            "--trainer.callbacks.wandb.tags=",
            "--trainer.callbacks.wandb.tags="
            + f"[pretraining,step1,0802,repeated-data,wsd,bs512,simbs64,{args.method},intermediate-decay,e{args.target_epoch},wd{args.weight_decay}]",
        ),
        (
            "--trainer.callbacks.checkpointer.fixed_steps=",
            f"--trainer.callbacks.checkpointer.fixed_steps=[{endpoint}]",
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
            "--train_module.scheduler=",
            "--train_module.scheduler="
            + "{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: 48, decay_fraction: 0.1}",
        ),
        ("--train_module.optim.weight_decay=", f"--train_module.optim.weight_decay={args.weight_decay}"),
        ("--lr=", f"--lr={args.learning_rate}"),
    )
    for prefix, replacement in replacements:
        arguments = upsert(arguments, prefix, replacement)
    arguments.extend(
        (f"--load_path={args.source_checkpoint}", "--load_trainer_state=true")
    )

    dry_run = [script, name, "--dry-run", *arguments]
    train = ["torchrun", f"--nproc-per-node={nproc}", script, name, *arguments]
    commands = [
        "set -euo pipefail",
        shlex.join(["test", "-d", args.source_checkpoint]),
        shlex.join(["test", "!", "-e", output]),
        shlex.join(
            [
                "echo",
                f"SIM_DECAY_PREFLIGHT method={args.method} epoch={args.target_epoch} "
                f"source_step={source} endpoint_step={endpoint} load_trainer_state=true",
            ]
        ),
        shlex.join(["python", *dry_run]),
        shlex.join(train),
    ]
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["resources"] = {"gpuCount": nproc, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": 0, "autoResume": False}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for env_var in task.setdefault("envVars", []):
        if env_var.get("name") == "GIT_REF":
            env_var["value"] = args.revision
            break
    else:
        task["envVars"].append({"name": "GIT_REF", "value": args.revision})
    spec.pop("description", None)
    return spec


def main() -> None:
    args = parse_args()
    registered_plan(args)
    spec = build_spec(get_spec(args.base_experiment), args)
    if args.print_only:
        print(json.dumps(spec, indent=2))
        return
    completed = subprocess.run(
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
    print(completed.stdout, end="")


if __name__ == "__main__":
    main()
