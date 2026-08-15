#!/usr/bin/env python3
"""Submit one unique-data WSD chain at 0.125B, 0.25B, and 0.5B tokens."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any


REPORT_JSON = Path("reports/0802/data/wsd_unique_vs_repeated_batch_tuned_1b.json")
TARGETS = (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2))
TOKENS_PER_BILLION = 1_000_000_000
SEQUENCE_LENGTH = 4096
REFERENCE_BATCH_SEQUENCES = 1024
REFERENCE_WARMUP_STEPS = 24
DECAY_FRACTION = Fraction(1, 10)
ATTEMPTED_STATUSES = {"pending", "queued", "running", "complete", "failed", "canceled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--global-sequences", type=int, choices=(64, 256), required=True)
    parser.add_argument("--learning-rate", choices=("1e-3",), default="1e-3")
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def target_key(target: Fraction) -> str:
    return format(Decimal(target.numerator) / Decimal(target.denominator), "f").rstrip("0")


def target_slug(target: Fraction) -> str:
    return target_key(target).replace(".", "p")


def target_tokens(target: Fraction) -> int:
    numerator = target.numerator * TOKENS_PER_BILLION
    if numerator % target.denominator:
        raise ValueError(f"Target {target_key(target)}B does not have an integer token count")
    return numerator // target.denominator


def end_step(target: Fraction, global_sequences: int) -> int:
    numerator = target.numerator * TOKENS_PER_BILLION
    denominator = target.denominator * global_sequences * SEQUENCE_LENGTH
    return (numerator + denominator - 1) // denominator


def stable_step(target: Fraction, global_sequences: int) -> int:
    endpoint = end_step(target, global_sequences)
    decay_steps = round(float(DECAY_FRACTION) * endpoint)
    return endpoint - decay_steps - 1


def warmup_steps(global_sequences: int) -> int:
    return REFERENCE_WARMUP_STEPS * REFERENCE_BATCH_SEQUENCES // global_sequences


def get_base_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def upsert(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    return [value for value in arguments if not value.startswith(prefix)] + [replacement]


def base_prefix(base_name: str) -> str:
    prefix, count = re.subn(r"_e\d+_lr[^_]+_wd[^_]+_warmup\d+.*$", "", base_name)
    if count != 1:
        raise ValueError(f"Unexpected Step-2 base run name: {base_name}")
    return prefix.replace("_step2_", "_step2_1_", 1)


def reject_duplicate_submission(global_sequences: int, learning_rate: str) -> None:
    report = json.loads(REPORT_JSON.read_text())
    requested = {float(target) for target in TARGETS}
    duplicates: list[str] = []
    for row in report.get("uniqueRuns", []):
        if row.get("batchSequences") != global_sequences or row.get("lr") != learning_rate:
            continue
        if float(row.get("epoch", -1)) in requested:
            duplicates.append(
                f"{row.get('epoch')}B endpoint status={row.get('status')} "
                f"experiment={row.get('beaker', 'unknown')}"
            )
    for chain in report.get("chainExperiments", []):
        if chain.get("batchSequences") != global_sequences or chain.get("lr") != learning_rate:
            continue
        targets = {float(value) for value in chain.get("targets", [])}
        status = str(chain.get("status", "")).lower()
        if targets & requested and status in ATTEMPTED_STATUSES:
            duplicates.append(
                f"fractional chain status={status} experiment={chain.get('beaker', 'unknown')}"
            )
    if duplicates:
        raise SystemExit("Refusing duplicate fractional submission:\n  - " + "\n  - ".join(duplicates))


def build_chain(
    base_spec: dict[str, Any],
    *,
    revision: str,
    global_sequences: int,
    learning_rate: str,
    nproc: int,
    priority: str,
) -> dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    original = task["arguments"]
    if original[:1] != ["python"]:
        raise ValueError("Expected a single-endpoint Python training task")
    training_script, original_name = original[1:3]
    base_arguments = original[3:]
    prefix = base_prefix(original_name)
    warmup = warmup_steps(global_sequences)
    output_root = "/weka/oe-training-default/sewonm/icsl/models"
    commands = ["set -euo pipefail"]
    previous_output: str | None = None
    previous_target: Fraction | None = None

    for target in TARGETS:
        key = target_key(target)
        name = (
            f"{prefix}_bs{global_sequences}_e{target_slug(target)}_lr{learning_rate}_"
            f"wd0.033_warmup{warmup}"
        )
        output = f"{output_root}/{name}"
        load_path = None
        if previous_output is not None and previous_target is not None:
            load_path = f"{previous_output}/step{stable_step(previous_target, global_sequences)}"
        phase_args = [
            value
            for value in base_arguments
            if not value.startswith(("--load_path=", "--load_trainer_state="))
        ]
        replacements = (
            ("--save-folder=", f"--save-folder={output}"),
            (
                "--trainer.max_duration=",
                f"--trainer.max_duration={{value: {target_tokens(target)}, unit: tokens}}",
            ),
            ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
            (
                "--trainer.callbacks.wandb.tags=",
                "--trainer.callbacks.wandb.tags="
                + "[pretraining, step2-1, 0802, unique-data, dclm-train-only, "
                + "document-disjoint, uniform-document-sample, batch-size-tune, "
                + f"fractional-endpoint, wsd, bs{global_sequences}, warmup{warmup}, single-job-chain]",
            ),
            (
                "--data_loader.global_batch_size=",
                f"--data_loader.global_batch_size={global_sequences * SEQUENCE_LENGTH}",
            ),
            (
                "--train_module.scheduler=",
                "--train_module.scheduler="
                + f"{{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: {warmup}, "
                + f"decay_fraction: {float(DECAY_FRACTION)}}}",
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
        if load_path is not None:
            phase_args.extend((f"--load_path={load_path}", "--load_trainer_state=true"))
            commands.append(shlex.join(["test", "-d", load_path]))
        commands.extend(
            (
                shlex.join(["test", "!", "-e", output]),
                shlex.join(
                    [
                        "echo",
                        f"FRACTIONAL_CHAIN_START bs={global_sequences} lr={learning_rate} "
                        f"target={key} source={load_path or 'fresh'}",
                    ]
                ),
                shlex.join(["python", training_script, name, "--dry-run", *phase_args]),
                shlex.join(
                    ["torchrun", f"--nproc-per-node={nproc}", training_script, name, *phase_args]
                ),
                shlex.join(["test", "-d", f"{output}/step{stable_step(target, global_sequences)}"]),
                shlex.join(
                    [
                        "echo",
                        f"FRACTIONAL_CHAIN_FINISH bs={global_sequences} lr={learning_rate} target={key}",
                    ]
                ),
            )
        )
        previous_output = output
        previous_target = target

    commands.append(
        shlex.join(
            [
                "echo",
                f"FRACTIONAL_CHAIN_COMPLETE bs={global_sequences} lr={learning_rate} "
                "targets=0.125,0.25,0.5",
            ]
        )
    )
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["envVars"] = [
        item for item in task.get("envVars", []) if item.get("name") != "GANTRY_USE_TORCHRUN"
    ]
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
    reject_duplicate_submission(args.global_sequences, args.learning_rate)
    spec = build_chain(
        get_base_spec(args.base_experiment),
        revision=args.revision,
        global_sequences=args.global_sequences,
        learning_rate=args.learning_rate,
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
