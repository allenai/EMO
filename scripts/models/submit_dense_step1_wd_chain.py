#!/usr/bin/env python3
"""Submit dense Step-1 WD endpoints as one persistent Beaker job.

The generated job keeps one 8-GPU allocation while it runs every requested
endpoint sequentially.  Each endpoint performs an in-job source/output
preflight, a configuration dry-run, training, held-out DCLM validation, and the
nine downstream evaluations.  Later endpoints resume only the preceding
endpoint's fixed pre-decay checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import subprocess
import sys
from typing import Any


TARGETS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
MAX_TOKENS = {target: target * 1_000_000_000 for target in TARGETS}
FIXED_STEPS = {
    1: (214,),
    2: (428,),
    3: (643,),
    4: (858,),
    5: (1073,),
    6: (1287,),
    8: (1502, 1716),
    10: (1931, 2145),
    12: (2360, 2574),
    16: (2789, 3003, 3218, 3432),
}
LOAD_STEPS = {
    2: 214,
    3: 428,
    4: 643,
    5: 858,
    6: 1073,
    8: 1287,
    10: 1716,
    12: 2145,
    16: 2574,
}
PRECEDING_TARGET = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 10: 8, 12: 10, 16: 12}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--start-epoch", type=int, choices=TARGETS, default=1)
    parser.add_argument("--weight-decay", required=True)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument(
        "--gate-epoch",
        type=int,
        choices=TARGETS,
        help="After this endpoint, continue only if its DCLM validation CE improves.",
    )
    parser.add_argument(
        "--continue-if-below",
        type=float,
        help="DCLM validation CE threshold paired with --gate-epoch.",
    )
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if (args.gate_epoch is None) != (args.continue_if_below is None):
        parser.error("--gate-epoch and --continue-if-below must be supplied together")
    return args


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


def replace_argument(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    return [replacement if value.startswith(prefix) else value for value in arguments]


def run_name_for(base_name: str, target: int) -> str:
    name, replacements = re.subn(
        r"_e\d+_(lr[^_]+_wd[^_]+_warmup\d+)$", rf"_e{target}_\1", base_name
    )
    if replacements != 1:
        raise ValueError(f"Unexpected Step1 base run name: {base_name}")
    return name


def base_name_for_weight_decay(base_name: str, weight_decay: str) -> str:
    name, replacements = re.subn(
        r"_wd[^_]+_warmup", f"_wd{weight_decay}_warmup", base_name
    )
    if replacements != 1:
        raise ValueError(f"Unexpected Step1 base run name: {base_name}")
    return name


def epoch_from_run_name(run_name: str) -> int:
    match = re.search(r"_e(\d+)_lr", run_name)
    if match is None:
        raise ValueError(f"Could not infer endpoint from run name: {run_name}")
    return int(match.group(1))


def stage_arguments(
    base_arguments: list[str],
    base_name: str,
    target: int,
    load_path: str | None,
) -> tuple[str, list[str], str]:
    run_name = run_name_for(base_name, target)
    output = f"/weka/oe-training-default/sewonm/icsl/models/{run_name}"
    arguments = [
        value
        for value in base_arguments
        if not value.startswith(("--load_path=", "--load_trainer_state="))
    ]
    arguments = replace_argument(arguments, "--save-folder=", f"--save-folder={output}")
    arguments = replace_argument(
        arguments,
        "--trainer.max_duration=",
        f"--trainer.max_duration={{value: {MAX_TOKENS[target]}, unit: tokens}}",
    )
    arguments = replace_argument(
        arguments,
        "--trainer.callbacks.wandb.name=",
        f"--trainer.callbacks.wandb.name={run_name}",
    )
    arguments = replace_argument(
        arguments,
        "--trainer.callbacks.checkpointer.fixed_steps=",
        "--trainer.callbacks.checkpointer.fixed_steps="
        + json.dumps(FIXED_STEPS[target], separators=(",", ":")),
    )
    if load_path is not None:
        arguments.extend((f"--load_path={load_path}", "--load_trainer_state=true"))
    return run_name, arguments, output


def build_chain(
    base_spec: dict[str, Any],
    *,
    revision: str,
    start_epoch: int,
    weight_decay: str,
    nproc: int,
    priority: str,
    gate_epoch: int | None,
    continue_if_below: float | None,
) -> dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    original = task["arguments"]
    if original[:1] != ["python"]:
        raise ValueError("Expected a Gantry Python training task")
    training_script = original[1]
    base_name = base_name_for_weight_decay(original[2], weight_decay)
    base_epoch = epoch_from_run_name(base_name)
    base_arguments = original[3:]
    if argument_value(base_arguments, "--lr=") != "1e-3":
        raise ValueError("The authorized WD chain must use LR1e-3")
    base_arguments = replace_argument(
        base_arguments,
        "--train_module.optim.weight_decay=",
        f"--train_module.optim.weight_decay={weight_decay}",
    )
    base_output = f"/weka/oe-training-default/sewonm/icsl/models/{base_name}"
    base_arguments = replace_argument(
        base_arguments, "--save-folder=", f"--save-folder={base_output}"
    )
    base_arguments = replace_argument(
        base_arguments,
        "--trainer.callbacks.wandb.name=",
        f"--trainer.callbacks.wandb.name={base_name}",
    )

    selected_targets = TARGETS[TARGETS.index(start_epoch) :]
    if start_epoch != 1 and PRECEDING_TARGET[start_epoch] != base_epoch:
        raise ValueError(
            f"Endpoint {start_epoch} must start from endpoint {PRECEDING_TARGET[start_epoch]}, "
            f"but the base experiment is endpoint {base_epoch}"
        )
    if gate_epoch is not None:
        if gate_epoch not in selected_targets[:-1]:
            raise ValueError("--gate-epoch must be a non-terminal selected endpoint")
    commands = ["set -euo pipefail"]
    previous_output: str | None = None if start_epoch == 1 else base_output
    for target in selected_targets:
        load_path = (
            None if target == 1 else f"{previous_output}/step{LOAD_STEPS[target]}"
        )
        run_name, arguments, output = stage_arguments(base_arguments, base_name, target, load_path)
        if load_path is not None:
            commands.append(shlex.join(["test", "-d", load_path]))
        commands.extend(
            (
                shlex.join(["test", "!", "-e", output]),
                shlex.join(
                    [
                        "echo",
                        f"CHAIN_PREFLIGHT_OK epoch={target} source={load_path or 'fresh'}",
                    ]
                ),
                shlex.join(["python", training_script, run_name, "--dry-run", *arguments]),
            )
        )
        train_command = shlex.join(
            ["torchrun", f"--nproc-per-node={nproc}", training_script, run_name, *arguments]
        )
        if target == gate_epoch:
            gate_log = f"/tmp/{run_name}.log"
            commands.extend(
                (
                    f"{train_command} 2>&1 | tee {shlex.quote(gate_log)}",
                    "gate_ce=$(sed -n "
                    + shlex.quote(
                        r"s/.*dclm-validation-0802\/CE loss=\([0-9][0-9.]*\).*/\1/p"
                    )
                    + f" {shlex.quote(gate_log)} | tail -n 1)",
                    'test -n "${gate_ce}"',
                    f'echo "CHAIN_GATE epoch={target} observed=${{gate_ce}} '
                    f'threshold={continue_if_below}"',
                    f"if ! awk -v observed=\"${{gate_ce}}\" -v threshold={continue_if_below} "
                    + shlex.quote("BEGIN { exit !(observed < threshold) }")
                    + f'; then echo "CHAIN_STOP epoch={target} observed=${{gate_ce}}"; exit 0; fi',
                    f'echo "CHAIN_CONTINUE next_epoch={selected_targets[selected_targets.index(target) + 1]}"',
                )
            )
        else:
            commands.append(train_command)
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
        start_epoch=args.start_epoch,
        weight_decay=args.weight_decay,
        nproc=args.nproc,
        priority=args.priority,
        gate_epoch=args.gate_epoch,
        continue_if_below=args.continue_if_below,
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
