#!/usr/bin/env python3
"""Submit a 1B-through-5B unique-pool WSD chain for report Step 2-1."""

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


TARGETS = (1, 2, 3, 4, 5)
TOKENS_PER_TARGET = 1_000_000_000
SEQUENCE_LENGTH = 4096
REFERENCE_BATCH_SEQUENCES = 1024
REFERENCE_WARMUP_STEPS = 24
DECAY_FRACTION = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--global-sequences", type=int, choices=(64, 256), required=True)
    parser.add_argument("--learning-rate", choices=("5e-4", "1e-3"), required=True)
    parser.add_argument("--start-target", type=int, choices=TARGETS, default=1)
    parser.add_argument(
        "--resume-checkpoint",
        help="Exact same-coordinate checkpoint used to resume --start-target after infrastructure failure.",
    )
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.start_target == 1 and args.resume_checkpoint is not None:
        parser.error("--resume-checkpoint is only valid when --start-target is after 1B")
    if args.start_target > 1 and args.resume_checkpoint is None:
        parser.error("--resume-checkpoint is required when --start-target is after 1B")
    return args


def get_base_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def upsert(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    output = [value for value in arguments if not value.startswith(prefix)]
    output.append(replacement)
    return output


def total_steps(target: int, global_sequences: int) -> int:
    return math.ceil(target * TOKENS_PER_TARGET / (global_sequences * SEQUENCE_LENGTH))


def stable_step(target: int, global_sequences: int) -> int:
    end_step = total_steps(target, global_sequences)
    return end_step - round(DECAY_FRACTION * end_step) - 1


def warmup_steps(global_sequences: int) -> int:
    return REFERENCE_WARMUP_STEPS * REFERENCE_BATCH_SEQUENCES // global_sequences


def base_prefix(base_name: str) -> str:
    prefix, count = re.subn(r"_e\d+_lr[^_]+_wd[^_]+_warmup\d+.*$", "", base_name)
    if count != 1:
        raise ValueError(f"Unexpected report-2 base run name: {base_name}")
    return prefix.replace("_step2_", "_step2_1_", 1)


def build_chain(
    base_spec: dict[str, Any],
    *,
    revision: str,
    global_sequences: int,
    learning_rate: str,
    start_target: int,
    resume_checkpoint: str | None,
    nproc: int,
    priority: str,
) -> dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    original = task["arguments"]
    if original[:1] != ["python"]:
        raise ValueError("Expected a single-endpoint Python training task")
    training_script, original_name = original[1:3]
    arguments = original[3:]
    prefix = base_prefix(original_name)
    warmup = warmup_steps(global_sequences)
    output_root = "/weka/oe-training-default/sewonm/icsl/models"
    commands = ["set -euo pipefail"]
    previous_output: str | None = None

    for target in TARGETS[TARGETS.index(start_target) :]:
        name = (
            f"{prefix}_bs{global_sequences}_e{target}_lr{learning_rate}_"
            f"wd0.033_warmup{warmup}"
        )
        output = f"{output_root}/{name}"
        phase_args = [
            value
            for value in arguments
            if not value.startswith(("--load_path=", "--load_trainer_state="))
        ]
        replacements = (
            ("--save-folder=", f"--save-folder={output}"),
            (
                "--trainer.max_duration=",
                f"--trainer.max_duration={{value: {target * TOKENS_PER_TARGET}, unit: tokens}}",
            ),
            ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
            (
                "--trainer.callbacks.wandb.tags=",
                "--trainer.callbacks.wandb.tags="
                + f"[pretraining, step2-1, 0802, unique-data, dclm-train-only, document-disjoint, uniform-document-sample, batch-size-tune, wsd, bs{global_sequences}, warmup{warmup}, single-job-chain]",
            ),
            (
                "--data_loader.global_batch_size=",
                f"--data_loader.global_batch_size={global_sequences * SEQUENCE_LENGTH}",
            ),
            (
                "--train_module.scheduler=",
                "--train_module.scheduler="
                + f"{{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: {warmup}, decay_fraction: {DECAY_FRACTION}}}",
            ),
            (
                "--trainer.callbacks.checkpointer.fixed_steps=",
                f"--trainer.callbacks.checkpointer.fixed_steps=[{stable_step(target, global_sequences)}]",
            ),
            ("--train_module.optim.weight_decay=", "--train_module.optim.weight_decay=0.033"),
            ("--lr=", f"--lr={learning_rate}"),
        )
        for argument_prefix, replacement in replacements:
            phase_args = upsert(phase_args, argument_prefix, replacement)

        load_path = resume_checkpoint if target == start_target else None
        if load_path is None and previous_output is not None:
            load_path = f"{previous_output}/step{stable_step(target - 1, global_sequences)}"
        if load_path is not None:
            phase_args.extend((f"--load_path={load_path}", "--load_trainer_state=true"))
            commands.append(shlex.join(["test", "-d", load_path]))
        output_preflight = ["test", "-d", output] if target == start_target and resume_checkpoint else ["test", "!", "-e", output]
        commands.extend(
            (
                shlex.join(output_preflight),
                shlex.join(
                    [
                        "echo",
                        f"CHAIN_START bs={global_sequences} lr={learning_rate} target={target} source={load_path or 'fresh'}",
                    ]
                ),
                shlex.join(["python", training_script, name, "--dry-run", *phase_args]),
                shlex.join(
                    ["torchrun", f"--nproc-per-node={nproc}", training_script, name, *phase_args]
                ),
                shlex.join(["test", "-d", f"{output}/step{stable_step(target, global_sequences)}"]),
                shlex.join(["echo", f"CHAIN_FINISH bs={global_sequences} lr={learning_rate} target={target}"]),
            )
        )
        previous_output = output

    resumed_targets = ",".join(str(target) for target in TARGETS[TARGETS.index(start_target) :])
    commands.append(
        shlex.join(
            [
                "echo",
                f"CHAIN_COMPLETE bs={global_sequences} lr={learning_rate} targets={resumed_targets}",
            ]
        )
    )
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["envVars"] = [item for item in task.get("envVars", []) if item.get("name") != "GANTRY_USE_TORCHRUN"]
    for item in task["envVars"]:
        if item.get("name") == "GIT_REF":
            item["value"] = revision
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
        start_target=args.start_target,
        resume_checkpoint=args.resume_checkpoint,
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
