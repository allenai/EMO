#!/usr/bin/env python3
"""Submit a one-GPU, short-step microbatch capacity probe for a small dense model."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
from typing import Any


DEFAULTS = {
    "153M": ("01KZ893CWXEYFNGWHNB92BNT4T", (16, 32, 64, 128, 256)),
    "474M": ("01KZ893KHCB6JVKGTRWF1YM7V9", (8, 16, 32, 64, 128)),
    "1B": ("01KZFPJT4AKFQ9GX1BVFE84ADS", (1, 2, 4, 8, 16)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", required=True, choices=tuple(DEFAULTS))
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--microbatches", help="Comma-separated sequence microbatches")
    parser.add_argument("--expandable-segments", action="store_true")
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def get_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def replace(arguments: list[str], prefix: str, value: str) -> list[str]:
    found = False
    output = []
    for argument in arguments:
        if argument.startswith(prefix):
            if not found:
                output.append(value)
                found = True
        else:
            output.append(argument)
    if not found:
        output.append(value)
    return output


def probe_arguments(base: list[str], model: str, microbatch: int, steps: int) -> list[str]:
    output = [
        argument
        for argument in base
        if not argument.startswith(
            (
                "--load_path=",
                "--load_trainer_state=",
                "--trainer.callbacks.heldout_evaluator=",
                "--trainer.callbacks.downstream_evaluator.",
                "--trainer.callbacks.checkpointer.",
            )
        )
    ]
    run_name = f"dense_{model.lower()}_microbatch_probe_mb{microbatch}"
    batch_tokens = microbatch * 4096
    output = replace(output, "--save-folder=", f"--save-folder=/results/{run_name}")
    output = replace(
        output,
        "--trainer.max_duration=",
        f"--trainer.max_duration={{value: {steps}, unit: steps}}",
    )
    output = replace(
        output,
        "--trainer.callbacks.wandb.enabled=",
        "--trainer.callbacks.wandb.enabled=false",
    )
    output = replace(
        output,
        "--trainer.callbacks.wandb.name=",
        f"--trainer.callbacks.wandb.name={run_name}",
    )
    output = replace(
        output,
        "--data_loader.global_batch_size=",
        f"--data_loader.global_batch_size={batch_tokens}",
    )
    output.extend(
        (
            f"--train_module.rank_microbatch_size={batch_tokens}",
            "--trainer.callbacks.downstream_evaluator.enabled=false",
            "--trainer.callbacks.checkpointer.enabled=false",
        )
    )
    return output


def build_spec(
    model: str,
    steps: int,
    requested_microbatches: tuple[int, ...] | None = None,
    expandable_segments: bool = False,
) -> dict[str, Any]:
    base_experiment, default_microbatches = DEFAULTS[model]
    microbatches = requested_microbatches or default_microbatches
    spec = copy.deepcopy(get_spec(base_experiment))
    task = spec["tasks"][0]
    shell = task["arguments"][2]
    training_line = next(line for line in shell.splitlines() if line.startswith("torchrun "))
    tokens = shlex.split(training_line)
    script = tokens[2]
    base_arguments = tokens[4:]
    commands = ["set -uo pipefail"]
    for microbatch in microbatches:
        run_name = f"dense_{model.lower()}_microbatch_probe_mb{microbatch}"
        arguments = probe_arguments(base_arguments, model, microbatch, steps)
        command = shlex.join(
            ["torchrun", "--nproc-per-node=1", script, run_name, *arguments]
        )
        commands.extend(
            (
                f"echo PROBE_START model={model} microbatch={microbatch} sequences tokens={microbatch * 4096}",
                command,
                "probe_status=$?",
                f"echo PROBE_END model={model} microbatch={microbatch} exit_code=$probe_status",
                "if [ \"$probe_status\" -ne 0 ]; then exit 0; fi",
            )
        )
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["resources"]["gpuCount"] = 1
    task["context"]["autoResume"] = False
    task["envVars"] = [
        env for env in task.get("envVars", []) if env.get("name") != "GANTRY_USE_TORCHRUN"
    ]
    if expandable_segments:
        task["envVars"] = [
            env
            for env in task["envVars"]
            if env.get("name") != "PYTORCH_CUDA_ALLOC_CONF"
        ]
        task["envVars"].append(
            {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"}
        )
    spec.pop("description", None)
    return spec


def main() -> None:
    args = parse_args()
    requested_microbatches = None
    if args.microbatches:
        requested_microbatches = tuple(int(value) for value in args.microbatches.split(","))
        if any(value <= 0 for value in requested_microbatches):
            raise ValueError("microbatches must be positive")
        if any(value & (value - 1) for value in requested_microbatches):
            raise ValueError("microbatches must be powers of two")
    spec = build_spec(
        args.model_size,
        args.steps,
        requested_microbatches=requested_microbatches,
        expandable_segments=args.expandable_segments,
    )
    if args.print_only:
        json.dump(spec, fp=__import__("sys").stdout, indent=2)
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
