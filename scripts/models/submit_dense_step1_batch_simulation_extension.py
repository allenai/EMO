#!/usr/bin/env python3
"""Submit a selected dense-1B batch-simulation E8 -> E12 extension.

This launcher is intentionally narrow. It continues the single selected
BS512/simulated-BS64 SN or Local-SGD trajectory from its exact E4 pre-decay
checkpoint, evaluates E8, and runs E12 only when E8 strictly improves on the
registered E4 validation CE. Each stage resumes the preceding stage's exact
pre-decay checkpoint with trainer state and retains only the pre-decay
checkpoint needed by the next authorized endpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


TOKENS_PER_EPOCH = 1_000_000_000
SEQUENCE_LENGTH = 4096
GLOBAL_SEQUENCES = 512
SIMULATED_SEQUENCES = 64
DECAY_FRACTION = 0.1
TARGETS = (8, 12)
DEFAULT_METHOD = "local_sgd_h4"
ALLOWED_METHODS = ("local_sgd_h4", "structured_noise")
LR = "1e-3"
WD = "0.1"
SYNC_INTERVAL = 4
REPORT = Path("reports/0802/data/wsd_batch_simulation_1b.json")


def total_steps(epoch: int) -> int:
    return math.ceil(
        epoch * TOKENS_PER_EPOCH / (GLOBAL_SEQUENCES * SEQUENCE_LENGTH)
    )


def pre_decay_step(epoch: int) -> int:
    endpoint = total_steps(epoch)
    return endpoint - round(DECAY_FRACTION * endpoint) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--method", choices=ALLOWED_METHODS, default=DEFAULT_METHOD)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    expected = pre_decay_step(4)
    if not args.source_checkpoint.endswith(f"/step{expected}"):
        parser.error(f"E8 extension requires exact E4 pre-decay /step{expected}")
    if args.method == "local_sgd_h4" and expected % SYNC_INTERVAL:
        parser.error("source checkpoint is not aligned to the Local-SGD sync interval")
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


def registered_plan(args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    report = json.loads(REPORT.read_text())
    plans = [
        run
        for run in report["runs"]
        if run.get("method") == args.method
        and run.get("startEpoch") == 8
        and run.get("chainThrough") == 12
        and run.get("batchSequences") == GLOBAL_SEQUENCES
        and run.get("simulatedBatchSequences") == SIMULATED_SEQUENCES
        and run.get("lr") == LR
        and run.get("wd") == WD
        and run.get("sourceCheckpoint") == args.source_checkpoint
    ]
    if len(plans) != 1 or plans[0].get("status") not in {
        "planned",
        "print-only-verified",
    }:
        raise RuntimeError(
            "expected exactly one planned registered E8->E12 extension; "
            f"found {len(plans)}"
        )
    parents = [
        run
        for run in report["runs"]
        if run.get("beaker") == args.base_experiment
        and run.get("method") == args.method
        and run.get("status") == "complete"
        and run.get("lr") == LR
        and run.get("wd") == WD
        and pre_decay_step(4) in run.get("retainedPreDecaySteps", [])
        and run.get("results", {}).get("4", {}).get("status") == "complete"
    ]
    if len(parents) != 1:
        raise RuntimeError("base experiment is not the unique completed selected E4 parent")
    validation = parents[0]["results"]["4"].get("validation")
    if not isinstance(validation, (int, float)):
        raise RuntimeError("selected LS E4 parent has no validation CE")
    attempted_later = [
        run
        for run in report["runs"]
        if run is not plans[0]
        and run.get("method") == args.method
        and run.get("lr") == LR
        and run.get("wd") == WD
        and (
            run.get("targetEpoch") in TARGETS
            or run.get("startEpoch") in TARGETS
            or any(str(epoch) in run.get("results", {}) for epoch in TARGETS)
        )
    ]
    if attempted_later:
        raise RuntimeError(f"a {args.method} E8/E12 continuation is already represented")
    return plans[0], float(validation)


def stage(
    parent_arguments: list[str],
    parent_name: str,
    *,
    epoch: int,
    suffix: str,
    load_path: str,
    method: str,
) -> tuple[str, str, list[str]]:
    base = re.sub(r"_e4_", f"_e{epoch}_", parent_name, count=1)
    if base == parent_name:
        raise RuntimeError("parent name does not contain a unique E4 marker")
    name = f"{base}_{suffix}"
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
            f"--trainer.max_duration={{value: {epoch * TOKENS_PER_EPOCH}, unit: tokens}}",
        ),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
        (
            "--trainer.callbacks.wandb.tags=",
            "--trainer.callbacks.wandb.tags="
            + f"[pretraining,step1,0802,repeated-data,wsd,bs512,simbs64,{method},selected-horizon-extension,e{epoch},wd{WD}]",
        ),
        (
            "--trainer.callbacks.checkpointer.fixed_steps=",
            f"--trainer.callbacks.checkpointer.fixed_steps=[{pre_decay_step(epoch)}]",
        ),
        (
            "--trainer.callbacks.checkpointer.save_interval=",
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
        ),
        (
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        ),
        ("--train_module.optim.weight_decay=", f"--train_module.optim.weight_decay={WD}"),
        ("--lr=", f"--lr={LR}"),
    )
    for prefix, replacement in replacements:
        arguments = upsert(arguments, prefix, replacement)
    arguments.extend((f"--load_path={load_path}", "--load_trainer_state=true"))
    return name, output, arguments


def build_spec(
    base_spec: dict[str, Any],
    args: argparse.Namespace,
    initial_validation_ce: float,
) -> dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("expected one parent task")
    task = spec["tasks"][0]
    script, parent_name, parent_arguments, nproc = extract_training_command(task)
    if args.method not in parent_name:
        raise RuntimeError(f"parent run name does not match {args.method}")
    commands = ["set -euo pipefail", f"previous_gate_ce={initial_validation_ce}"]
    load_path = args.source_checkpoint
    for index, epoch in enumerate(TARGETS):
        name, output, arguments = stage(
            parent_arguments,
            parent_name,
            epoch=epoch,
            suffix=args.suffix,
            load_path=load_path,
            method=args.method,
        )
        commands.extend(
            (
                shlex.join(["test", "-d", load_path]),
                shlex.join(["test", "!", "-e", output]),
                shlex.join(
                    [
                        "echo",
                        f"SIM_EXTENSION_PREFLIGHT method={args.method} epoch={epoch} "
                        f"source={load_path} pre_decay_step={pre_decay_step(epoch)} "
                        f"endpoint_step={total_steps(epoch)} load_trainer_state=true",
                    ]
                ),
                shlex.join(["python", script, name, "--dry-run", *arguments]),
            )
        )
        train_command = shlex.join(
            ["torchrun", f"--nproc-per-node={nproc}", script, name, *arguments]
        )
        if index < len(TARGETS) - 1:
            gate_log = f"/tmp/{name}.log"
            commands.extend(
                (
                    f"{train_command} 2>&1 | tee {shlex.quote(gate_log)}",
                    "gate_ce=$(sed -n "
                    + shlex.quote(
                        r"s/.*dclm-validation-0802\/CE loss=\([0-9][0-9.]*\).*/\1/p"
                    )
                    + f" {shlex.quote(gate_log)} | tail -n 1)",
                    'test -n "${gate_ce}"',
                    f'echo "SIM_EXTENSION_GATE epoch={epoch} observed=${{gate_ce}} '
                    'previous=${previous_gate_ce}"',
                    'if ! awk -v observed="${gate_ce}" -v previous="${previous_gate_ce}" '
                    + shlex.quote("BEGIN { exit !(observed < previous) }")
                    + f'; then echo "SIM_EXTENSION_STOP epoch={epoch} observed=${{gate_ce}}"; exit 0; fi',
                    'previous_gate_ce="${gate_ce}"',
                    f'echo "SIM_EXTENSION_CONTINUE next_epoch={TARGETS[index + 1]}"',
                )
            )
        else:
            commands.append(train_command)
        load_path = f"{output}/step{pre_decay_step(epoch)}"
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["resources"] = {"gpuCount": nproc, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": 0, "autoResume": False}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    task["envVars"] = [
        env_var
        for env_var in task.get("envVars", [])
        if env_var.get("name") != "GANTRY_USE_TORCHRUN"
    ]
    for env_var in task["envVars"]:
        if env_var.get("name") == "GIT_REF":
            env_var["value"] = args.revision
            break
    else:
        task["envVars"].append({"name": "GIT_REF", "value": args.revision})
    spec.pop("description", None)
    return spec


def main() -> None:
    args = parse_args()
    _, initial_validation_ce = registered_plan(args)
    spec = build_spec(get_spec(args.base_experiment), args, initial_validation_ce)
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
