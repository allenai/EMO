#!/usr/bin/env python3
"""Submit a persistent dense-1B WSD batch-size/LR chain to Beaker.

The chain evaluates target-specific WSD endpoints while sharing the stable
trajectory. Warmup is held fixed at 24 * 1024 * 4096 tokens, and every saved
or loaded pre-decay checkpoint is derived from the requested global sequence
batch size.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shlex
import subprocess
import sys
from typing import Any


TARGETS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24)
PRECEDING_TARGET = dict(zip(TARGETS[1:], TARGETS[:-1]))
TOKENS_PER_EPOCH = 1_000_000_000
SEQUENCE_LENGTH = 4096
REFERENCE_GLOBAL_SEQUENCES = 1024
REFERENCE_WARMUP_STEPS = 24
DECAY_FRACTION = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--global-sequences", type=int, choices=(64, 128, 256, 512, 1024), required=True)
    parser.add_argument("--learning-rate", choices=("5e-4", "1e-3"), required=True)
    parser.add_argument("--weight-decay", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def get_base_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def upsert_argument(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    replaced = False
    output: list[str] = []
    for value in arguments:
        if value.startswith(prefix):
            if not replaced:
                output.append(replacement)
                replaced = True
        else:
            output.append(value)
    if not replaced:
        output.append(replacement)
    return output


def total_steps(epoch: int, global_sequences: int) -> int:
    global_batch_tokens = global_sequences * SEQUENCE_LENGTH
    return math.ceil(epoch * TOKENS_PER_EPOCH / global_batch_tokens)


def stable_step(epoch: int, global_sequences: int) -> int:
    end_step = total_steps(epoch, global_sequences)
    decay_steps = round(DECAY_FRACTION * end_step)
    return end_step - decay_steps - 1


def fixed_steps(target: int, global_sequences: int) -> tuple[int, ...]:
    previous = PRECEDING_TARGET.get(target, 0)
    return tuple(stable_step(epoch, global_sequences) for epoch in range(previous + 1, target + 1))


def warmup_steps(global_sequences: int) -> int:
    numerator = REFERENCE_WARMUP_STEPS * REFERENCE_GLOBAL_SEQUENCES
    if numerator % global_sequences:
        raise ValueError("global sequence batch does not preserve an integer token-matched warmup")
    return numerator // global_sequences


def run_prefix_from_base(base_name: str) -> str:
    prefix, replacements = re.subn(r"_e\d+_lr[^_]+_wd[^_]+_warmup\d+.*$", "", base_name)
    if replacements != 1:
        raise ValueError(f"Unexpected Step 1 base run name: {base_name}")
    return prefix.replace("_step2_", "_step1_", 1)


def run_name(
    prefix: str,
    *,
    target: int,
    global_sequences: int,
    learning_rate: str,
    weight_decay: str,
    warmup: int,
) -> str:
    return (
        f"{prefix}_bs{global_sequences}_e{target}_lr{learning_rate}_"
        f"wd{weight_decay}_warmup{warmup}"
    )


def stage_arguments(
    base_arguments: list[str],
    *,
    name: str,
    output: str,
    target: int,
    global_sequences: int,
    learning_rate: str,
    weight_decay: str,
    warmup: int,
    load_path: str | None,
) -> list[str]:
    arguments = [
        value
        for value in base_arguments
        if not value.startswith(("--load_path=", "--load_trainer_state="))
    ]
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        ("--trainer.max_duration=", f"--trainer.max_duration={{value: {target * TOKENS_PER_EPOCH}, unit: tokens}}"),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
        (
            "--trainer.callbacks.wandb.tags=",
            "--trainer.callbacks.wandb.tags="
            + f"[pretraining, step1, 0802, repeated-data, dclm-train-only, batch-size-tune, wsd, bs{global_sequences}, warmup{warmup}]",
        ),
        (
            "--trainer.callbacks.checkpointer.fixed_steps=",
            "--trainer.callbacks.checkpointer.fixed_steps="
            + json.dumps(fixed_steps(target, global_sequences), separators=(",", ":")),
        ),
        ("--data_loader.global_batch_size=", f"--data_loader.global_batch_size={global_sequences * SEQUENCE_LENGTH}"),
        (
            "--train_module.scheduler=",
            "--train_module.scheduler="
            + f"{{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: {warmup}, decay_fraction: {DECAY_FRACTION}}}",
        ),
        ("--train_module.optim.weight_decay=", f"--train_module.optim.weight_decay={weight_decay}"),
        ("--lr=", f"--lr={learning_rate}"),
    )
    for argument_prefix, replacement in replacements:
        arguments = upsert_argument(arguments, argument_prefix, replacement)
    if load_path is not None:
        arguments.extend((f"--load_path={load_path}", "--load_trainer_state=true"))
    return arguments


def build_chain(
    base_spec: dict[str, Any],
    *,
    revision: str,
    global_sequences: int,
    learning_rate: str,
    weight_decay: str,
    nproc: int,
    priority: str,
) -> dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    original = task["arguments"]
    if original[:1] != ["python"]:
        raise ValueError("Expected a single-endpoint Gantry Python training task")
    training_script = original[1]
    prefix = run_prefix_from_base(original[2])
    base_arguments = original[3:]
    warmup = warmup_steps(global_sequences)
    commands = ["set -euo pipefail"]
    previous_output: str | None = None
    for target in TARGETS:
        name = run_name(
            prefix,
            target=target,
            global_sequences=global_sequences,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup=warmup,
        )
        output = f"/weka/oe-training-default/sewonm/icsl/models/{name}"
        load_path = None
        if previous_output is not None:
            load_path = f"{previous_output}/step{stable_step(PRECEDING_TARGET[target], global_sequences)}"
        arguments = stage_arguments(
            base_arguments,
            name=name,
            output=output,
            target=target,
            global_sequences=global_sequences,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup=warmup,
            load_path=load_path,
        )
        if load_path is not None:
            commands.append(shlex.join(["test", "-d", load_path]))
        commands.extend(
            (
                shlex.join(["test", "!", "-e", output]),
                shlex.join(
                    [
                        "echo",
                        f"CHAIN_PREFLIGHT_OK bs={global_sequences} epoch={target} source={load_path or 'fresh'}",
                    ]
                ),
                shlex.join(["python", training_script, name, "--dry-run", *arguments]),
                shlex.join(
                    ["torchrun", f"--nproc-per-node={nproc}", training_script, name, *arguments]
                ),
            )
        )
        previous_output = output

    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["envVars"] = [
        env_var
        for env_var in task.get("envVars", [])
        if env_var.get("name") != "GANTRY_USE_TORCHRUN"
    ]
    for env_var in task["envVars"]:
        if env_var.get("name") == "GIT_REF":
            env_var["value"] = revision
            break
    else:
        task["envVars"].append({"name": "GIT_REF", "value": revision})
    task["resources"] = {"gpuCount": nproc, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "minRuntime": "0s", "autoResume": False}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec.pop("description", None)
    return spec


def main() -> None:
    args = parse_args()
    spec = build_chain(
        get_base_spec(args.base_experiment),
        revision=args.revision,
        global_sequences=args.global_sequences,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        nproc=args.nproc,
        priority=args.priority,
    )
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    result = subprocess.run(
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
        text=True,
        stdout=subprocess.PIPE,
    )
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
