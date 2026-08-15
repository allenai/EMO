#!/usr/bin/env python3
"""Submit target-specific WSD chains at E0.125, E0.25, and E0.5.

Each stage evaluates its own WSD endpoint and the following stage resumes from
the exact same-coordinate pre-decay checkpoint.  BS16 restarts may optionally
continue through E1 after a user-authorized cancellation; larger batches stop
at E0.5.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from submit_dense_step1_batch_coordinate import extract_training_command
from submit_dense_step1_batch_lr_chain import (
    DECAY_FRACTION,
    SEQUENCE_LENGTH,
    TOKENS_PER_EPOCH,
    run_prefix_from_base,
    upsert_argument,
    warmup_steps,
)


ATTEMPT_REGISTRY = Path("reports/0802/data/wsd_batch_size_1b.json")
ATTEMPTED_STATUSES = {"pending", "queued", "running", "complete", "failed", "canceled"}
FRACTIONAL_TARGETS = (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2))
RANK_MICROBATCH_SEQUENCES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--global-sequences",
        type=int,
        choices=(16, 32, 64, 128, 256, 512),
        required=True,
    )
    parser.add_argument(
        "--learning-rate",
        choices=("1.25e-4", "2.5e-4", "5e-4", "1e-3"),
        required=True,
    )
    parser.add_argument("--weight-decay", required=True)
    parser.add_argument(
        "--include-epoch-one",
        action="store_true",
        help="Continue the fractional chain through E1; allowed only for BS16 restarts.",
    )
    parser.add_argument(
        "--restart-canceled-experiment",
        help=(
            "Exact BS16 E1 experiment canceled so its fresh restart could retain and "
            "evaluate the fractional endpoints."
        ),
    )
    parser.add_argument(
        "--recover-canceled-experiment",
        help=(
            "Exact prior fractional-chain experiment canceled before startup by an "
            "infrastructure failure. The registry must show no completed fractional "
            "result and no model-health failure."
        ),
    )
    parser.add_argument(
        "--checkpoint-source-root",
        help=(
            "Completed E1 run directory containing the exact same-coordinate fractional "
            "pre-decay checkpoints. Each fractional endpoint resumes its own checkpoint."
        ),
    )
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    if args.include_epoch_one and args.global_sequences != 16:
        parser.error("--include-epoch-one is reserved for the BS16 restart")
    if args.include_epoch_one != (args.restart_canceled_experiment is not None):
        parser.error(
            "--include-epoch-one and --restart-canceled-experiment must be supplied together"
        )
    if args.include_epoch_one and args.checkpoint_source_root is not None:
        parser.error("the BS16 restart cannot also use --checkpoint-source-root")
    if args.restart_canceled_experiment and args.recover_canceled_experiment:
        parser.error(
            "--restart-canceled-experiment and --recover-canceled-experiment are mutually exclusive"
        )
    if args.global_sequences % (args.nproc * RANK_MICROBATCH_SEQUENCES):
        parser.error(
            "global sequence batch must be divisible by "
            f"nproc ({args.nproc}) x rank microbatch ({RANK_MICROBATCH_SEQUENCES})"
        )
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", args.weight_decay):
        parser.error("unsupported weight-decay format")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.suffix):
        parser.error("--suffix must be a lowercase run-name component")
    if numeric_value(args.learning_rate) >= Decimal("0.002"):
        parser.error("LR2e-3 or higher is prohibited")
    return args


def numeric_value(value: object) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"Invalid numeric coordinate value: {value!r}") from error


def target_key(target: Fraction) -> str:
    if target.denominator == 1:
        return str(target.numerator)
    return str(Decimal(target.numerator) / Decimal(target.denominator)).rstrip("0")


def target_slug(target: Fraction) -> str:
    return target_key(target).replace(".", "p")


def end_step(target: Fraction, global_sequences: int) -> int:
    numerator = target.numerator * TOKENS_PER_EPOCH
    denominator = target.denominator * global_sequences * SEQUENCE_LENGTH
    return (numerator + denominator - 1) // denominator


def stable_step(target: Fraction, global_sequences: int) -> int:
    endpoint = end_step(target, global_sequences)
    decay_steps = round(DECAY_FRACTION * endpoint)
    return endpoint - decay_steps - 1


def target_tokens(target: Fraction) -> int:
    numerator = target.numerator * TOKENS_PER_EPOCH
    if numerator % target.denominator:
        raise ValueError(f"Target E{target_key(target)} does not have an integer token count")
    return numerator // target.denominator


def get_base_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def selected_epoch_one(args: argparse.Namespace) -> dict[str, Any]:
    """Return and verify the best admissible completed E1 coordinate."""
    registry = ATTEMPT_REGISTRY
    if not registry.is_file():
        raise FileNotFoundError(f"Required selection registry not found: {registry}")
    with registry.open() as report_file:
        report = json.load(report_file)

    unhealthy = report.get("healthAudit", {}).get("unhealthy", {})
    candidates: list[dict[str, Any]] = []
    for sweep in report.get("batchSweeps", []):
        if sweep.get("batchSequences") != args.global_sequences:
            continue
        result = sweep.get("results", {}).get("1")
        if not result or str(result.get("status", "")).lower() != "complete":
            continue
        if result.get("wandb") in unhealthy:
            continue
        if numeric_value(sweep.get("lr")) >= Decimal("0.002"):
            continue
        candidates.append(
            {
                "lr": sweep.get("lr"),
                "wd": sweep.get("wd"),
                "validation": result.get("validation"),
                "beaker": sweep.get("beaker"),
                "output": result.get("output"),
            }
        )

    if not candidates:
        raise SystemExit(f"No admissible completed E1 candidates found for BS{args.global_sequences}")
    selected = min(candidates, key=lambda item: numeric_value(item["validation"]))
    requested = (numeric_value(args.learning_rate), numeric_value(args.weight_decay))
    registered = (numeric_value(selected["lr"]), numeric_value(selected["wd"]))
    if requested != registered or args.base_experiment != selected["beaker"]:
        raise SystemExit(
            f"Requested BS{args.global_sequences} E1 source is not the registered winner: "
            f"requested LR{args.learning_rate}/WD{args.weight_decay} experiment={args.base_experiment}; "
            f"selected LR{selected['lr']}/WD{selected['wd']} experiment={selected['beaker']} "
            f"validation={selected['validation']}"
        )
    expected_root = str(selected.get("output", "")).rstrip("/")
    actual_root = str(args.checkpoint_source_root or "").rstrip("/")
    if not expected_root or actual_root != expected_root:
        raise SystemExit(
            "Fractional checkpoint source must be the exact selected E1 output root: "
            f"expected={expected_root!r} actual={actual_root!r}"
        )
    return selected


def argument_value(arguments: list[str], prefix: str) -> str | None:
    for argument in reversed(arguments):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return None


def audit_base_spec(base_spec: dict[str, Any], args: argparse.Namespace) -> None:
    task = base_spec["tasks"][0]
    _, _, arguments = extract_training_command(task)
    expected_batch = str(args.global_sequences * SEQUENCE_LENGTH)
    actual_batch = argument_value(arguments, "--data_loader.global_batch_size=")
    actual_lr = argument_value(arguments, "--lr=")
    actual_wd = argument_value(arguments, "--train_module.optim.weight_decay=")
    if actual_batch != expected_batch:
        raise SystemExit(f"Base experiment batch mismatch: expected {expected_batch}, got {actual_batch}")
    if actual_lr is None or numeric_value(actual_lr) != numeric_value(args.learning_rate):
        raise SystemExit(f"Base experiment LR mismatch: expected {args.learning_rate}, got {actual_lr}")
    if actual_wd is None or numeric_value(actual_wd) != numeric_value(args.weight_decay):
        raise SystemExit(f"Base experiment WD mismatch: expected {args.weight_decay}, got {actual_wd}")


def reject_duplicate_submission(args: argparse.Namespace) -> None:
    if not ATTEMPT_REGISTRY.is_file():
        raise FileNotFoundError(f"Required attempt registry not found: {ATTEMPT_REGISTRY}")
    with ATTEMPT_REGISTRY.open() as report_file:
        report = json.load(report_file)

    requested_lr = numeric_value(args.learning_rate)
    requested_wd = numeric_value(args.weight_decay)
    requested_targets = list(FRACTIONAL_TARGETS)
    if args.include_epoch_one:
        requested_targets.append(Fraction(1, 1))

    restart_match = None
    recovery_match = None
    duplicates: list[str] = []
    unhealthy = report.get("healthAudit", {}).get("unhealthy", {})
    for sweep in report.get("batchSweeps", []):
        if sweep.get("batchSequences") != args.global_sequences:
            continue
        if numeric_value(sweep.get("lr")) != requested_lr:
            continue
        if numeric_value(sweep.get("wd")) != requested_wd:
            continue

        if sweep.get("beaker") == args.restart_canceled_experiment:
            restart_match = sweep
        if sweep.get("beaker") == args.recover_canceled_experiment:
            recovery_match = sweep

        for epoch, result in sweep.get("results", {}).items():
            wandb = result.get("wandb")
            if wandb in unhealthy:
                raise SystemExit(
                    "Refusing fractional continuation of an unhealthy coordinate: "
                    f"E{epoch} W&B={wandb}"
                )
            try:
                recorded_target = Fraction(str(epoch))
            except ValueError:
                continue
            if recorded_target in requested_targets:
                duplicates.append(
                    f"E{epoch} result status={result.get('status', 'recorded')} "
                    f"experiment={sweep.get('beaker', 'unknown')}"
                )

        active_epoch = sweep.get("activeEpoch")
        try:
            active_target = Fraction(str(active_epoch))
        except (TypeError, ValueError):
            active_target = None
        status = str(sweep.get("status", "")).lower()
        is_restart_exception = (
            sweep.get("beaker") == args.restart_canceled_experiment
            and status == "canceled"
            and active_target == Fraction(1, 1)
        )
        is_recovery_exception = (
            sweep.get("beaker") == args.recover_canceled_experiment
            and status == "canceled"
            and active_target == Fraction(1, 2)
            and not sweep.get("results")
            and str(sweep.get("failureClass", "")).startswith("infrastructure-")
        )
        if (
            active_target in requested_targets
            and status in ATTEMPTED_STATUSES
            and not is_restart_exception
            and not is_recovery_exception
        ):
            duplicates.append(
                f"E{target_key(active_target)} sweep status={status} "
                f"experiment={sweep.get('beaker', 'unknown')}"
            )

    if args.restart_canceled_experiment is not None:
        if restart_match is None:
            raise SystemExit("Restart experiment is missing from the report registry")
        if str(restart_match.get("status", "")).lower() != "canceled":
            raise SystemExit("Restart experiment must be recorded as canceled")
        if Fraction(str(restart_match.get("activeEpoch"))) != Fraction(1, 1):
            raise SystemExit("Restart experiment must be the exact canceled BS16 E1 attempt")
        if "1" in restart_match.get("results", {}):
            raise SystemExit("Restart experiment already has a completed E1 result")

    if args.recover_canceled_experiment is not None:
        if recovery_match is None:
            raise SystemExit("Recovery experiment is missing from the report registry")
        if str(recovery_match.get("status", "")).lower() != "canceled":
            raise SystemExit("Recovery experiment must be recorded as canceled")
        if Fraction(str(recovery_match.get("activeEpoch"))) != Fraction(1, 2):
            raise SystemExit("Recovery experiment must be the exact canceled fractional chain")
        if recovery_match.get("results"):
            raise SystemExit("Recovery experiment already has a completed fractional result")
        if not str(recovery_match.get("failureClass", "")).startswith("infrastructure-"):
            raise SystemExit("Recovery experiment must have an infrastructure failure class")

    if duplicates:
        details = "\n  - ".join(sorted(set(duplicates)))
        raise SystemExit(f"Refusing duplicate fractional submission:\n  - {details}")


def stage_arguments(
    base_arguments: list[str],
    *,
    name: str,
    output: str,
    target: Fraction,
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
    endpoint = end_step(target, global_sequences)
    pre_decay = stable_step(target, global_sequences)
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
            + "[pretraining, step1, 0802, repeated-data, dclm-train-only, "
            + f"batch-size-tune, fractional-endpoint, wsd, bs{global_sequences}, warmup{warmup}]",
        ),
        (
            "--trainer.callbacks.checkpointer.fixed_steps=",
            "--trainer.callbacks.checkpointer.fixed_steps="
            + json.dumps([pre_decay, endpoint], separators=(",", ":")),
        ),
        ("--trainer.callbacks.checkpointer.save_interval=", "--trainer.callbacks.checkpointer.save_interval=1000000000"),
        (
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        ),
        (
            "--data_loader.global_batch_size=",
            f"--data_loader.global_batch_size={global_sequences * SEQUENCE_LENGTH}",
        ),
        (
            "--train_module.scheduler=",
            "--train_module.scheduler="
            + f"{{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: {warmup}, "
            + f"decay_fraction: {DECAY_FRACTION}}}",
        ),
        ("--train_module.optim.weight_decay=", f"--train_module.optim.weight_decay={weight_decay}"),
        ("--lr=", f"--lr={learning_rate}"),
    )
    for prefix, replacement in replacements:
        arguments = upsert_argument(arguments, prefix, replacement)
    if load_path is not None:
        arguments.extend((f"--load_path={load_path}", "--load_trainer_state=true"))
    return arguments


def build_chain(base_spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    training_script, base_name, base_arguments = extract_training_command(task)
    prefix = re.sub(r"_bs\d+$", "", run_prefix_from_base(base_name))
    warmup = warmup_steps(args.global_sequences)
    targets = list(FRACTIONAL_TARGETS)
    if args.include_epoch_one:
        targets.append(Fraction(1, 1))

    commands = ["set -euo pipefail"]
    previous_output: str | None = None
    previous_target: Fraction | None = None
    for target in targets:
        endpoint_name = (
            f"{prefix}_bs{args.global_sequences}_e{target_slug(target)}_lr{args.learning_rate}_"
            f"wd{args.weight_decay}_warmup{warmup}_{args.suffix}"
        )
        output = f"/weka/oe-training-default/sewonm/icsl/models/{endpoint_name}"
        load_path = None
        if args.checkpoint_source_root is not None:
            load_path = (
                f"{args.checkpoint_source_root.rstrip('/')}/"
                f"step{stable_step(target, args.global_sequences)}"
            )
        elif previous_output is not None and previous_target is not None:
            load_path = f"{previous_output}/step{stable_step(previous_target, args.global_sequences)}"
        endpoint_arguments = stage_arguments(
            base_arguments,
            name=endpoint_name,
            output=output,
            target=target,
            global_sequences=args.global_sequences,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
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
                        f"FRACTIONAL_PREFLIGHT_OK bs={args.global_sequences} "
                        f"epoch={target_key(target)} lr={args.learning_rate} "
                        f"wd={args.weight_decay} source={load_path or 'fresh'}",
                    ]
                ),
                shlex.join(["python", training_script, endpoint_name, "--dry-run", *endpoint_arguments]),
                shlex.join(
                    [
                        "torchrun",
                        f"--nproc-per-node={args.nproc}",
                        training_script,
                        endpoint_name,
                        *endpoint_arguments,
                    ]
                ),
            )
        )
        previous_output = output
        previous_target = target

    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
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
    task["resources"] = {"gpuCount": args.nproc, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": "0s", "autoResume": False}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec.pop("description", None)
    return spec


def main() -> None:
    args = parse_args()
    selected_epoch_one(args)
    reject_duplicate_submission(args)
    base_spec = get_base_spec(args.base_experiment)
    audit_base_spec(base_spec, args)
    spec = build_chain(base_spec, args)
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
        ],
        check=True,
        input=json.dumps(spec),
        text=True,
        stdout=subprocess.PIPE,
    )
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
