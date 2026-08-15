#!/usr/bin/env python3
"""Submit a persistent small-model WSD chain by cloning a verified Beaker spec.

The generated task keeps one allocation while it performs a single-process
configuration dry-run and an 8-GPU training/evaluation run for each requested
target. Later targets resume the preceding target's pre-decay checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from typing import Any


TARGETS = (1, 2, 4, 8, 12, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--lr", required=True, choices=("1e-3", "2e-3", "4e-3"))
    parser.add_argument("--start-target", type=int, choices=TARGETS, required=True)
    parser.add_argument("--name", required=True)
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


def argument_value(arguments: list[str], prefix: str) -> str:
    return next(value.removeprefix(prefix) for value in arguments if value.startswith(prefix))


def pre_decay_step(target: int, global_batch_tokens: int) -> int:
    max_steps = math.ceil(target * 1_000_000_000 / global_batch_tokens)
    # Python round is ties-to-even, matching the WSD scheduler.
    decay_steps = round(max_steps / 10)
    return max_steps - decay_steps - 1


def canonical_run_name(base_name: str, target: int, lr: str) -> str:
    name = re.sub(r"_e1_lr[^_]+", f"_e{target}_lr{lr}", base_name)
    return name.replace("_retry1", "")


def replace_argument(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    return [replacement if value.startswith(prefix) else value for value in arguments]


def target_arguments(
    base_arguments: list[str],
    base_name: str,
    target: int,
    lr: str,
    global_batch_tokens: int,
    checkpoint_start_epoch: int,
    load_path: str | None,
) -> tuple[str, list[str], str]:
    run_name = canonical_run_name(base_name, target, lr)
    output = f"/weka/oe-training-default/sewonm/icsl/models/{run_name}"
    fixed_steps = [
        pre_decay_step(epoch, global_batch_tokens)
        for epoch in range(checkpoint_start_epoch, target + 1)
    ]
    arguments = [
        value
        for value in base_arguments
        if not value.startswith(("--load_path=", "--load_trainer_state="))
    ]
    arguments = replace_argument(arguments, "--save-folder=", f"--save-folder={output}")
    arguments = replace_argument(
        arguments,
        "--trainer.max_duration=",
        f"--trainer.max_duration={{value: {target * 1_000_000_000}, unit: tokens}}",
    )
    arguments = replace_argument(
        arguments,
        "--trainer.callbacks.wandb.name=",
        f"--trainer.callbacks.wandb.name={run_name}",
    )
    arguments = replace_argument(
        arguments,
        "--trainer.callbacks.checkpointer.fixed_steps=",
        f"--trainer.callbacks.checkpointer.fixed_steps={json.dumps(fixed_steps, separators=(',', ':'))}",
    )
    arguments = replace_argument(arguments, "--lr=", f"--lr={lr}")
    if load_path is not None:
        arguments.extend((f"--load_path={load_path}", "--load_trainer_state=true"))
    return run_name, arguments, output


def build_chain(spec: dict[str, Any], lr: str, start_target: int, nproc: int) -> dict[str, Any]:
    task = spec["tasks"][0]
    original = task["arguments"]
    if original[:1] != ["python"]:
        raise ValueError("Expected a Gantry Python training task")
    training_script = original[1]
    base_name = original[2]
    base_arguments = original[3:]
    global_batch_tokens = int(
        argument_value(base_arguments, "--data_loader.global_batch_size=")
        if any(value.startswith("--data_loader.global_batch_size=") for value in base_arguments)
        else int(argument_value(base_arguments, "--global_batch_size=")) * 4096
    )
    base_output = argument_value(base_arguments, "--save-folder=")
    selected_targets = TARGETS[TARGETS.index(start_target) :]
    commands: list[str] = ["set -euo pipefail"]
    previous_output: str | None = None
    previous_target: int | None = None
    for target in selected_targets:
        if target == 1:
            checkpoint_start_epoch = 1
            load_path = None
        elif previous_output is None:
            checkpoint_start_epoch = target
            load_path = f"{base_output}/step{pre_decay_step(1, global_batch_tokens)}"
        else:
            assert previous_target is not None
            checkpoint_start_epoch = previous_target + 1
            load_path = f"{previous_output}/step{pre_decay_step(previous_target, global_batch_tokens)}"
        run_name, arguments, output = target_arguments(
            base_arguments,
            base_name,
            target,
            lr,
            global_batch_tokens,
            checkpoint_start_epoch,
            load_path,
        )
        commands.append(
            shlex.join(["python", training_script, run_name, "--dry-run", *arguments])
        )
        commands.append(
            shlex.join(
                ["torchrun", f"--nproc-per-node={nproc}", training_script, run_name, *arguments]
            )
        )
        previous_output = output
        previous_target = target
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    # Gantry treats the presence of this variable as truthy, even when its value
    # is "0". Remove it so the outer bash chain runs exactly once; each training
    # stage launches its own 8-process torchrun below.
    task["envVars"] = [
        env_var
        for env_var in task.get("envVars", [])
        if env_var.get("name") != "GANTRY_USE_TORCHRUN"
    ]
    spec.pop("description", None)
    return spec


def main() -> None:
    args = parse_args()
    spec = build_chain(get_base_spec(args.base_experiment), args.lr, args.start_target, args.nproc)
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
