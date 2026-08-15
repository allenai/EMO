#!/usr/bin/env python3
"""Submit an adaptive dense-1B batch-size coordinate trajectory to Beaker.

The base experiment may itself be a persistent batch-size chain.  This script
extracts its first torchrun command and rewrites the training coordinate.  By
default it submits one endpoint; ``--chain-through`` instead submits one job
that advances the same coordinate through every configured search target up
to the requested later endpoint.  Every skipped integer epoch's pre-decay
checkpoint is retained so that an intermediate endpoint can be evaluated later
without replaying training.
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

from submit_dense_step1_batch_lr_chain import (
    DECAY_FRACTION,
    SEQUENCE_LENGTH,
    TOKENS_PER_EPOCH,
    run_name,
    run_prefix_from_base,
    stage_arguments,
    stable_step,
    warmup_steps,
)


ATTEMPT_REGISTRY = Path("reports/0802/data/wsd_batch_size_1b.json")
ATTEMPTED_STATUSES = {"pending", "queued", "running", "complete", "failed", "canceled"}
MAX_LEARNING_RATE = Decimal("1e-3")
RANK_MICROBATCH_SEQUENCES = 4
STANDARD_TARGETS = (1, 2, 4, 8, 16)
E12_EXTENSION_TARGETS = (1, 2, 4, 8, 12, 16)
SELECTED_HORIZON_EXTENSION_TARGETS = (1, 2, 4, 8, 12, 16, 20, 24, 28)
BS128_CONDITIONAL_TARGETS = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32)
ALL_TARGETS = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32)
INTERMEDIATE_DECAY_TARGETS = {
    16: {12},
    32: {16},
    64: {20, 28},
}
FRACTIONAL_CHECKPOINT_BATCHES = {16, 32, 64, 128, 256, 512}
FRACTIONAL_TARGETS = (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2))
SELECTED_HORIZON_COORDINATES = {
    16: (Decimal("5e-4"), Decimal("0.1")),
    32: (Decimal("5e-4"), Decimal("0.3")),
}
SSH_HOST: str | None = None


def beaker_command(*args: str) -> list[str]:
    command = ["beaker", *args]
    if not SSH_HOST:
        return command
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        SSH_HOST,
        shlex.join(command),
    ]


def target_ladder(global_sequences: int) -> tuple[int, ...]:
    if global_sequences == 128:
        return BS128_CONDITIONAL_TARGETS
    if global_sequences in {16, 32}:
        return SELECTED_HORIZON_EXTENSION_TARGETS
    if global_sequences == 512:
        return E12_EXTENSION_TARGETS
    if global_sequences == 64:
        return STANDARD_TARGETS + (24, 32)
    return STANDARD_TARGETS


def fractional_stable_step(target: Fraction, global_sequences: int) -> int:
    numerator = target.numerator * TOKENS_PER_EPOCH
    denominator = target.denominator * global_sequences * SEQUENCE_LENGTH
    end_step = (numerator + denominator - 1) // denominator
    decay_steps = round(DECAY_FRACTION * end_step)
    return end_step - decay_steps - 1


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
    parser.add_argument("--target-epoch", type=int, choices=ALL_TARGETS, required=True)
    parser.add_argument(
        "--learning-rate",
        choices=("1.25e-4", "2.5e-4", "5e-4", "1e-3", "2e-3"),
        required=True,
    )
    parser.add_argument("--weight-decay", required=True)
    parser.add_argument(
        "--source-checkpoint",
        help="Exact matching-coordinate pre-decay checkpoint; required after epoch 1.",
    )
    parser.add_argument(
        "--recover-canceled-experiment",
        help=(
            "Narrow exception for resuming a canceled epoch-1 attempt from --source-checkpoint. "
            "The experiment must be recorded as canceled for this exact coordinate."
        ),
    )
    parser.add_argument(
        "--recover-failed-experiment",
        help=(
            "Narrow exception for retrying a clear infrastructure failure from "
            "--source-checkpoint. The experiment must be recorded as failed for "
            "this exact coordinate."
        ),
    )
    parser.add_argument(
        "--chain-through",
        type=int,
        choices=ALL_TARGETS,
        help="Run every target from --target-epoch through this epoch in the same job.",
    )
    parser.add_argument(
        "--decay-from-intermediate-checkpoint",
        action="store_true",
        help=(
            "Run only the target-specific WSD decay from the exact registered "
            "pre-decay checkpoint for an explicitly authorized intermediate endpoint."
        ),
    )
    parser.add_argument(
        "--selected-horizon-extension",
        action="store_true",
        help=(
            "Narrow one-job BS16/BS32 selected-coordinate extension from E16 through "
            "E20/E24/E28. The job stops after E20 or E24 unless validation CE "
            "strictly improves on the preceding endpoint."
        ),
    )
    parser.add_argument("--suffix", default="coord")
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--ssh-host")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if (
        args.target_epoch == 1
        and args.source_checkpoint is not None
        and args.recover_canceled_experiment is None
        and args.recover_failed_experiment is None
    ):
        parser.error("epoch 1 must start fresh unless --recover-canceled-experiment is provided")
    if args.target_epoch != 1 and args.source_checkpoint is None:
        parser.error("--source-checkpoint is required after epoch 1")
    if args.recover_canceled_experiment is not None:
        if args.source_checkpoint is None:
            parser.error("canceled recovery requires --source-checkpoint")
    if (
        args.recover_failed_experiment is not None
        and args.target_epoch != 1
        and args.source_checkpoint is None
    ):
        parser.error("failed recovery requires --source-checkpoint")
    if args.recover_canceled_experiment is not None and args.recover_failed_experiment is not None:
        parser.error("choose only one recovery mode")
    if args.selected_horizon_extension:
        selected = SELECTED_HORIZON_COORDINATES.get(args.global_sequences)
        if selected is None:
            parser.error("selected horizon extension is authorized only for BS16 and BS32")
        if args.recover_failed_experiment is None:
            if args.target_epoch != 20 or args.chain_through != 28:
                parser.error(
                    "selected horizon extension must start at E20 and use --chain-through 28"
                )
        else:
            if args.global_sequences != 16 or args.target_epoch != 24 or args.chain_through != 28:
                parser.error(
                    "selected horizon recovery is narrowly authorized only for "
                    "BS16 E24→E28"
                )
            if args.source_checkpoint is None or not args.source_checkpoint.endswith(
                "/step288390"
            ):
                parser.error(
                    "BS16 selected horizon recovery requires exact retained "
                    "checkpoint step288390"
                )
        if (
            numeric_value(args.learning_rate) != selected[0]
            or numeric_value(args.weight_decay) != selected[1]
        ):
            parser.error(
                "selected horizon extension must use the fixed selected coordinate "
                f"LR={selected[0]}, WD={selected[1]} for BS{args.global_sequences}"
            )
        if args.recover_failed_experiment is None:
            expected_source_step = stable_step(16, args.global_sequences)
            if args.source_checkpoint is None or not args.source_checkpoint.endswith(
                f"/step{expected_source_step}"
            ):
                parser.error(
                    "selected horizon extension requires the exact E16 pre-decay checkpoint "
                    f"ending in /step{expected_source_step}"
                )
        if args.decay_from_intermediate_checkpoint:
            parser.error(
                "selected horizon extension cannot be combined with intermediate decay"
            )
        if args.recover_canceled_experiment:
            parser.error("selected horizon extension does not support canceled recovery")
    targets = target_ladder(args.global_sequences)
    if args.decay_from_intermediate_checkpoint:
        allowed = INTERMEDIATE_DECAY_TARGETS.get(args.global_sequences, set())
        if args.target_epoch not in allowed:
            parser.error(
                "intermediate decay is authorized only for "
                "BS16/E12, BS32/E16, BS64/E20, and BS64/E28"
            )
        if args.chain_through is not None:
            parser.error("intermediate decay cannot be combined with --chain-through")
        expected_step = stable_step(args.target_epoch, args.global_sequences)
        if args.source_checkpoint is None or not args.source_checkpoint.endswith(
            f"/step{expected_step}"
        ):
            parser.error(
                "intermediate decay requires the exact target pre-decay checkpoint "
                f"ending in /step{expected_step}"
            )
    elif args.target_epoch not in targets:
        parser.error(
            f"epoch {args.target_epoch} is not a BS{args.global_sequences} search target; "
            f"choose one of {targets}"
        )
    if args.chain_through is not None and args.chain_through not in targets:
        parser.error(
            f"epoch {args.chain_through} is not a BS{args.global_sequences} chain target; "
            f"choose one of {targets}"
        )
    if args.chain_through is not None and targets.index(args.chain_through) < targets.index(args.target_epoch):
        parser.error("--chain-through must be the same as or later than --target-epoch")
    if (
        args.global_sequences in SELECTED_HORIZON_COORDINATES
        and (args.chain_through or args.target_epoch) > 16
        and not args.selected_horizon_extension
    ):
        parser.error(
            "BS16/BS32 work beyond E16 requires --selected-horizon-extension; "
            "no coordinate search is authorized"
        )
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:e-[0-9]+)?", args.learning_rate):
        parser.error("unsupported learning-rate format")
    if Decimal(args.learning_rate) > MAX_LEARNING_RATE:
        parser.error("New Step 1 batch coordinate work is capped at 1e-3 by user policy")
    accumulation_denominator = args.nproc * RANK_MICROBATCH_SEQUENCES
    if args.global_sequences % accumulation_denominator:
        parser.error(
            "global sequence batch must be divisible by "
            f"nproc ({args.nproc}) x rank microbatch ({RANK_MICROBATCH_SEQUENCES}); "
            f"got BS{args.global_sequences}"
        )
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", args.weight_decay):
        parser.error("unsupported weight-decay format")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.suffix):
        parser.error("--suffix must be a lowercase run-name component")
    return args


def get_base_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        beaker_command("experiment", "spec", experiment, "--format", "json"),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def numeric_value(value: object) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"Invalid numeric coordinate value: {value!r}") from error


def best_healthy_validation(report: dict[str, Any], batch: int, epoch: int) -> Decimal | None:
    unhealthy = report.get("healthAudit", {}).get("unhealthy", {})
    values: list[Decimal] = []
    for sweep in report.get("batchSweeps", []):
        if sweep.get("batchSequences") != batch:
            continue
        result = sweep.get("results", {}).get(str(epoch))
        if not isinstance(result, dict) or str(result.get("status", "")).lower() != "complete":
            continue
        if result.get("wandb") in unhealthy or result.get("validation") is None:
            continue
        values.append(numeric_value(result["validation"]))
    return min(values) if values else None


def best_healthy_coordinate(
    report: dict[str, Any], batch: int, epoch: int
) -> tuple[Decimal, Decimal, Decimal] | None:
    unhealthy = report.get("healthAudit", {}).get("unhealthy", {})
    values: list[tuple[Decimal, Decimal, Decimal]] = []
    for sweep in report.get("batchSweeps", []):
        if sweep.get("batchSequences") != batch:
            continue
        result = sweep.get("results", {}).get(str(epoch))
        if not isinstance(result, dict) or str(result.get("status", "")).lower() != "complete":
            continue
        if result.get("wandb") in unhealthy or result.get("validation") is None:
            continue
        values.append(
            (
                numeric_value(result["validation"]),
                numeric_value(sweep.get("lr")),
                numeric_value(sweep.get("wd")),
            )
        )
    return min(values) if values else None


def enforce_conditional_bs64_extension(report: dict[str, Any], target_epoch: int) -> None:
    if target_epoch not in {24, 32}:
        return
    predecessor = 16 if target_epoch == 24 else 24
    comparison = 8 if target_epoch == 24 else 16
    live = [
        sweep.get("beaker", "unknown")
        for sweep in report.get("batchSweeps", [])
        if sweep.get("batchSequences") == 64
        and sweep.get("activeEpoch") == predecessor
        and str(sweep.get("status", "")).lower() in {"pending", "queued", "running"}
    ]
    if live:
        raise SystemExit(
            f"Refusing BS64 E{target_epoch}: E{predecessor} is not fully resolved; "
            f"live experiments: {', '.join(sorted(set(live)))}"
        )
    predecessor_ce = best_healthy_validation(report, 64, predecessor)
    comparison_ce = best_healthy_validation(report, 64, comparison)
    if predecessor_ce is None or comparison_ce is None:
        raise SystemExit(
            f"Refusing BS64 E{target_epoch}: missing a healthy completed validation CE "
            f"for E{comparison} or E{predecessor}."
        )
    if predecessor_ce >= comparison_ce:
        raise SystemExit(
            f"Refusing BS64 E{target_epoch}: selected E{predecessor} CE {predecessor_ce} "
            f"did not improve on E{comparison} CE {comparison_ce}."
        )


def enforce_conditional_bs128_extension(
    report: dict[str, Any],
    target_epoch: int,
    learning_rate: str,
    weight_decay: str,
) -> None:
    if target_epoch not in {20, 24, 28, 32}:
        return
    predecessor = target_epoch - 4
    comparison = predecessor - 4
    live = [
        sweep.get("beaker", "unknown")
        for sweep in report.get("batchSweeps", [])
        if sweep.get("batchSequences") == 128
        and sweep.get("activeEpoch") == predecessor
        and str(sweep.get("status", "")).lower() in {"pending", "queued", "running"}
    ]
    if live:
        raise SystemExit(
            f"Refusing BS128 E{target_epoch}: E{predecessor} is not fully resolved; "
            f"live experiments: {', '.join(sorted(set(live)))}"
        )
    predecessor_coordinate = best_healthy_coordinate(report, 128, predecessor)
    comparison_ce = best_healthy_validation(report, 128, comparison)
    if predecessor_coordinate is None or comparison_ce is None:
        raise SystemExit(
            f"Refusing BS128 E{target_epoch}: missing a healthy completed validation CE "
            f"for E{comparison} or E{predecessor}."
        )
    predecessor_ce, predecessor_lr, predecessor_wd = predecessor_coordinate
    if predecessor_ce >= comparison_ce:
        raise SystemExit(
            f"Refusing BS128 E{target_epoch}: selected E{predecessor} CE {predecessor_ce} "
            f"did not improve on E{comparison} CE {comparison_ce}."
        )
    if (
        numeric_value(learning_rate) != predecessor_lr
        or numeric_value(weight_decay) != predecessor_wd
    ):
        raise SystemExit(
            f"Refusing BS128 E{target_epoch}: only the improving selected E{predecessor} "
            f"coordinate LR={predecessor_lr}, WD={predecessor_wd} may continue."
        )


def attempted_endpoints(
    report: dict[str, Any],
    *,
    global_sequences: int,
    learning_rate: str,
    weight_decay: str,
    targets: list[int],
    recover_canceled_experiment: str | None = None,
    recover_failed_experiment: str | None = None,
    ignored_result_searches: set[str] | None = None,
) -> list[str]:
    """Return exact coordinate endpoints already recorded as attempted.

    Weight decay is compared numerically but exactly: 0.3 equals 0.300, while
    0.3 and 0.333 remain distinct checkpoint trajectories.
    """

    requested_lr = numeric_value(learning_rate)
    requested_wd = numeric_value(weight_decay)
    requested_targets = set(targets)
    recovery_ancestors: set[str] = set()
    recovery_cursor = recover_failed_experiment or recover_canceled_experiment
    while recovery_cursor is not None and recovery_cursor not in recovery_ancestors:
        recovery_ancestors.add(recovery_cursor)
        recovery_cursor = next(
            (
                str(sweep["recoveryOf"])
                for sweep in report.get("batchSweeps", [])
                if sweep.get("beaker") == recovery_cursor and sweep.get("recoveryOf")
            ),
            None,
        )
    matches: list[str] = []
    ignored_result_searches = ignored_result_searches or set()
    for sweep in report.get("batchSweeps", []):
        if sweep.get("batchSequences") != global_sequences:
            continue
        if numeric_value(sweep.get("lr")) != requested_lr:
            continue
        if numeric_value(sweep.get("wd")) != requested_wd:
            continue

        results = sweep.get("results", {})
        active_targets = {sweep.get("activeEpoch")}
        if (
            sweep.get("chainThrough") == sweep.get("activeEpoch")
            and "chain" in str(sweep.get("search", ""))
            and not sweep.get("sourceCheckpoint")
        ):
            # Fresh backfill chains attempt every configured target from E1
            # through activeEpoch, even before the earlier endpoints have
            # completed and appeared in results.
            active_targets.update(
                target
                for target in target_ladder(global_sequences)
                if target <= sweep["activeEpoch"]
            )
        for target in requested_targets:
            result = results.get(str(target))
            if result is not None:
                if (
                    str(result.get("status", "")).lower() == "failed"
                    and recover_failed_experiment is not None
                    and sweep.get("beaker") in recovery_ancestors
                ):
                    continue
                if str(sweep.get("search", "")) in ignored_result_searches:
                    continue
                matches.append(
                    f"E{target} result status={result.get('status', 'recorded')} "
                    f"experiment={sweep.get('beaker', 'unknown')}"
                )
                continue
            if target not in active_targets:
                continue
            status = str(sweep.get("status", "")).lower()
            if (
                status == "canceled"
                and recover_canceled_experiment is not None
                and sweep.get("beaker") in recovery_ancestors
            ):
                continue
            if (
                status == "failed"
                and recover_failed_experiment is not None
                and sweep.get("beaker") in recovery_ancestors
            ):
                continue
            if status in ATTEMPTED_STATUSES:
                matches.append(
                    f"E{target} sweep status={status} "
                    f"experiment={sweep.get('beaker', 'unknown')}"
                )
    return sorted(set(matches))


def reject_duplicate_submission(args: argparse.Namespace) -> dict[str, Any]:
    if not ATTEMPT_REGISTRY.is_file():
        raise FileNotFoundError(
            f"Required attempt registry not found: {ATTEMPT_REGISTRY}. "
            "Run this launcher from the repository root."
        )
    with ATTEMPT_REGISTRY.open() as report_file:
        report = json.load(report_file)
    if args.global_sequences == 64:
        enforce_conditional_bs64_extension(report, args.target_epoch)
    if args.global_sequences == 128:
        enforce_conditional_bs128_extension(
            report,
            args.target_epoch,
            args.learning_rate,
            args.weight_decay,
        )
    unhealthy = report.get("healthAudit", {}).get("unhealthy", {})
    requested_lr = numeric_value(args.learning_rate)
    requested_wd = numeric_value(args.weight_decay)
    unhealthy_predecessors: list[str] = []
    for sweep in report.get("batchSweeps", []):
        if sweep.get("batchSequences") != args.global_sequences:
            continue
        if numeric_value(sweep.get("lr")) != requested_lr or numeric_value(sweep.get("wd")) != requested_wd:
            continue
        for epoch, result in sweep.get("results", {}).items():
            wandb = result.get("wandb")
            if wandb in unhealthy and float(epoch) < args.target_epoch:
                unhealthy_predecessors.append(
                    f"E{epoch} W&B={wandb}: {unhealthy[wandb].get('reason', 'unhealthy')}"
                )
        active_wandb = sweep.get("activeWandb")
        active_epoch = sweep.get("activeEpoch")
        if active_wandb in unhealthy and isinstance(active_epoch, int) and active_epoch < args.target_epoch:
            unhealthy_predecessors.append(
                f"E{active_epoch} W&B={active_wandb}: "
                f"{unhealthy[active_wandb].get('reason', 'unhealthy')}"
            )
    if unhealthy_predecessors:
        details = "\n  - ".join(sorted(set(unhealthy_predecessors)))
        raise SystemExit(
            "Refusing to continue an unhealthy coordinate trajectory for "
            f"BS{args.global_sequences}, LR={args.learning_rate}, WD={args.weight_decay}:\n"
            f"  - {details}\n"
            "Choose a different coordinate; unhealthy runs remain provenance-only."
        )
    if args.recover_canceled_experiment is not None:
        recoveries = [
            sweep
            for sweep in report.get("batchSweeps", [])
            if sweep.get("beaker") == args.recover_canceled_experiment
            and sweep.get("batchSequences") == args.global_sequences
            and numeric_value(sweep.get("lr")) == requested_lr
            and numeric_value(sweep.get("wd")) == requested_wd
            and str(sweep.get("status", "")).lower() == "canceled"
            and sweep.get("activeEpoch") == (args.chain_through or args.target_epoch)
            and str(args.target_epoch) not in sweep.get("results", {})
        ]
        if len(recoveries) != 1:
            raise SystemExit(
                "Refusing canceled recovery: expected exactly one matching canceled attempt "
                f"for experiment {args.recover_canceled_experiment}."
            )
    if args.recover_failed_experiment is not None:
        recoveries = [
            sweep
            for sweep in report.get("batchSweeps", [])
            if sweep.get("beaker") == args.recover_failed_experiment
            and sweep.get("batchSequences") == args.global_sequences
            and numeric_value(sweep.get("lr")) == requested_lr
            and numeric_value(sweep.get("wd")) == requested_wd
            and str(sweep.get("status", "")).lower() == "failed"
            and sweep.get("activeEpoch")
            in {args.target_epoch, args.chain_through or args.target_epoch}
            and (
                str(args.target_epoch) not in sweep.get("results", {})
                or str(
                    sweep.get("results", {})
                    .get(str(args.target_epoch), {})
                    .get("status", "")
                ).lower()
                == "failed"
            )
        ]
        if len(recoveries) != 1:
            raise SystemExit(
                "Refusing failed recovery: expected exactly one matching failed attempt "
                f"for experiment {args.recover_failed_experiment}."
            )
    end_target = args.chain_through or args.target_epoch
    if args.decay_from_intermediate_checkpoint:
        targets = [args.target_epoch]
    else:
        ladder = target_ladder(args.global_sequences)
        targets = ladder[ladder.index(args.target_epoch) : ladder.index(end_target) + 1]
    matches = attempted_endpoints(
        report,
        global_sequences=args.global_sequences,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        targets=targets,
        recover_canceled_experiment=args.recover_canceled_experiment,
        recover_failed_experiment=args.recover_failed_experiment,
        ignored_result_searches=(
            {
                "incidental-checkpoint-evaluation",
                "pre-decay-checkpoint-evaluation-provenance",
            }
            if args.decay_from_intermediate_checkpoint
            else None
        ),
    )
    if matches:
        details = "\n  - ".join(matches)
        raise SystemExit(
            "Refusing duplicate coordinate submission for "
            f"BS{args.global_sequences}, LR={args.learning_rate}, WD={args.weight_decay}:\n"
            f"  - {details}\n"
            "Start at the first unattempted epoch and resume the exact matching checkpoint."
        )
    return report


def audit_intermediate_decay_source(
    args: argparse.Namespace,
    report: dict[str, Any],
    base_spec: dict[str, Any],
) -> None:
    if not args.decay_from_intermediate_checkpoint:
        return
    requested_lr = numeric_value(args.learning_rate)
    requested_wd = numeric_value(args.weight_decay)
    expected_step = stable_step(args.target_epoch, args.global_sequences)
    matches = [
        sweep
        for sweep in report.get("batchSweeps", [])
        if sweep.get("beaker") == args.base_experiment
        and sweep.get("batchSequences") == args.global_sequences
        and numeric_value(sweep.get("lr")) == requested_lr
        and numeric_value(sweep.get("wd")) == requested_wd
        and expected_step in sweep.get("retainedPreDecaySteps", [])
    ]
    if len(matches) != 1:
        raise SystemExit(
            "Refusing intermediate decay: expected exactly one registered matching parent "
            f"experiment retaining step {expected_step}; found {len(matches)}."
        )
    parent = matches[0]
    parent_status = str(parent.get("status", "")).lower()
    if parent_status not in {"complete", "failed"}:
        raise SystemExit(
            f"Refusing intermediate decay from parent status {parent_status!r}."
        )
    if parent_status == "failed":
        failure_text = " ".join(
            str(parent.get(key, "")) for key in ("failureClass", "reason")
        ).lower()
        if not any(token in failure_text for token in ("infrastructure", "enospc", "weka")):
            raise SystemExit(
                "Refusing intermediate decay from a failed parent without explicit "
                "infrastructure provenance."
            )
    unhealthy = report.get("healthAudit", {}).get("unhealthy", {})
    if parent.get("activeWandb") in unhealthy:
        raise SystemExit("Refusing intermediate decay from an unhealthy exact trajectory")
    _, _, base_arguments = extract_training_command(base_spec["tasks"][0])
    save_folders = [
        value.split("=", 1)[1]
        for value in base_arguments
        if value.startswith("--save-folder=")
    ]
    if len(save_folders) != 1:
        raise SystemExit("Refusing intermediate decay: parent has no unique save folder")
    expected_checkpoint = f"{save_folders[0]}/step{expected_step}"
    if args.source_checkpoint != expected_checkpoint:
        raise SystemExit(
            "Refusing intermediate decay: source checkpoint does not exactly match the "
            f"registered parent save folder ({expected_checkpoint})."
        )


def audit_selected_horizon_extension_source(
    args: argparse.Namespace,
    report: dict[str, Any],
    base_spec: dict[str, Any],
) -> Decimal | None:
    if not args.selected_horizon_extension:
        return None
    selected = SELECTED_HORIZON_COORDINATES[args.global_sequences]
    resolved = best_healthy_coordinate(report, args.global_sequences, 16)
    if resolved is None:
        raise SystemExit(
            f"Refusing BS{args.global_sequences} selected extension: E16 is unresolved."
        )
    validation_ce, selected_lr, selected_wd = resolved
    if (selected_lr, selected_wd) != selected:
        raise SystemExit(
            f"Refusing BS{args.global_sequences} selected extension: report winner is "
            f"LR={selected_lr}, WD={selected_wd}, expected LR={selected[0]}, WD={selected[1]}."
        )
    unhealthy = report.get("healthAudit", {}).get("unhealthy", {})
    if args.recover_failed_experiment is not None:
        parents = [
            sweep
            for sweep in report.get("batchSweeps", [])
            if sweep.get("beaker") == args.recover_failed_experiment
            and sweep.get("batchSequences") == args.global_sequences
            and numeric_value(sweep.get("lr")) == selected_lr
            and numeric_value(sweep.get("wd")) == selected_wd
            and str(sweep.get("status", "")).lower() == "failed"
            and str(
                sweep.get("results", {})
                .get(str(args.target_epoch), {})
                .get("status", "")
            ).lower()
            == "failed"
        ]
        if len(parents) != 1:
            raise SystemExit(
                "Refusing selected extension recovery: expected exactly one registered "
                f"failed parent; found {len(parents)}."
            )
        parent = parents[0]
        result = parent["results"][str(args.target_epoch)]
        failure_text = " ".join(
            str(value)
            for value in (
                parent.get("failureClass"),
                parent.get("reason"),
                result.get("failure"),
                result.get("reason"),
            )
        ).lower()
        if not any(
            token in failure_text
            for token in ("infrastructure", "nccl", "watchdog", "timeout")
        ):
            raise SystemExit(
                "Refusing selected extension recovery without explicit "
                "infrastructure provenance"
            )
        if parent.get("activeWandb") in unhealthy or result.get("wandb") in unhealthy:
            raise SystemExit("Refusing selected extension recovery from a red trajectory")
        if (
            args.source_checkpoint != result.get("retainedCheckpoint")
            or 288390 not in parent.get("retainedPreDecaySteps", [])
        ):
            raise SystemExit(
                "Refusing selected extension recovery: source is not the exact "
                "registered retained checkpoint step288390"
            )
        ladder = target_ladder(args.global_sequences)
        previous_target = ladder[ladder.index(args.target_epoch) - 1]
        previous_validation_ce = best_healthy_validation(
            report, args.global_sequences, previous_target
        )
        if previous_validation_ce is None:
            raise SystemExit(
                "Refusing selected extension recovery: preceding endpoint is unresolved"
            )
        validation_ce = previous_validation_ce
    else:
        parents = [
            sweep
            for sweep in report.get("batchSweeps", [])
            if sweep.get("beaker") == args.base_experiment
            and sweep.get("batchSequences") == args.global_sequences
            and numeric_value(sweep.get("lr")) == selected_lr
            and numeric_value(sweep.get("wd")) == selected_wd
            and str(sweep.get("results", {}).get("16", {}).get("status", "")).lower()
            == "complete"
            and numeric_value(sweep["results"]["16"].get("validation")) == validation_ce
        ]
        if len(parents) != 1:
            raise SystemExit(
                "Refusing selected extension: expected exactly one registered selected E16 "
                f"parent experiment; found {len(parents)}."
            )
        parent = parents[0]
        result = parent["results"]["16"]
        if result.get("wandb") in unhealthy:
            raise SystemExit("Refusing selected extension from an unhealthy E16 winner")
        expected_step = stable_step(16, args.global_sequences)
        registered_sources = {
            value
            for value in (
                parent.get("sourceCheckpoint"),
                result.get("resumeCheckpoint"),
                (
                    f"{result.get('output')}/step{expected_step}"
                    if result.get("output")
                    else None
                ),
            )
            if value is not None and str(value).endswith(f"/step{expected_step}")
        }
        if args.source_checkpoint not in registered_sources:
            raise SystemExit(
                "Refusing selected extension: source checkpoint is not the exact registered "
                f"selected E16 pre-decay checkpoint. Registered choices: {sorted(registered_sources)}"
            )
    _, _, base_arguments = extract_training_command(base_spec["tasks"][0])

    def unique_argument(prefix: str) -> str:
        values = [
            value.split("=", 1)[1]
            for value in base_arguments
            if value.startswith(prefix)
        ]
        if len(values) != 1:
            raise SystemExit(
                f"Refusing selected extension: base Beaker spec has {len(values)} {prefix} values."
            )
        return values[0]

    if numeric_value(unique_argument("--lr=")) != selected_lr:
        raise SystemExit("Refusing selected extension: Beaker LR does not match the report")
    if numeric_value(unique_argument("--train_module.optim.weight_decay=")) != selected_wd:
        raise SystemExit("Refusing selected extension: Beaker WD does not match the report")
    expected_global_batch = args.global_sequences * SEQUENCE_LENGTH
    if int(unique_argument("--data_loader.global_batch_size=")) != expected_global_batch:
        raise SystemExit(
            "Refusing selected extension: Beaker global batch does not match the report"
        )
    return validation_ce


def extract_training_command(task: dict[str, Any]) -> tuple[str, str, list[str]]:
    arguments = task.get("arguments", [])
    if arguments[:2] != ["bash", "-lc"] or len(arguments) != 3:
        raise ValueError("Expected a persistent-chain bash task")
    for line in arguments[2].splitlines():
        parts = shlex.split(line)
        if not parts or parts[0] != "torchrun":
            continue
        if len(parts) < 5 or not parts[1].startswith("--nproc-per-node="):
            continue
        # A persistent-chain stage may pipe torchrun through ``tee`` so the
        # validation gate can read its endpoint metric.  Those shell tokens
        # are not training-script arguments and must not be inherited by a
        # later exact-coordinate recovery built from this experiment.
        training_arguments = parts[4:]
        for index, value in enumerate(training_arguments):
            if value in {"|", "||", "&&", ";"} or re.fullmatch(
                r"(?:\d*>>?|\d*>&\d+)", value
            ):
                training_arguments = training_arguments[:index]
                break
        return parts[2], parts[3], training_arguments
    raise ValueError("Could not find a torchrun training command in the base experiment")


def build_endpoint(
    base_spec: dict[str, Any],
    *,
    revision: str,
    global_sequences: int,
    target_epoch: int,
    learning_rate: str,
    weight_decay: str,
    source_checkpoint: str | None,
    chain_through: int | None,
    suffix: str,
    nproc: int,
    priority: str,
    decay_from_intermediate_checkpoint: bool = False,
    selected_horizon_extension: bool = False,
    initial_validation_ce: Decimal | None = None,
) -> dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    training_script, base_name, base_arguments = extract_training_command(task)
    prefix = re.sub(r"_bs\d+$", "", run_prefix_from_base(base_name))
    warmup = warmup_steps(global_sequences)
    commands = ["set -euo pipefail"]
    end_target = chain_through or target_epoch
    ladder = target_ladder(global_sequences)
    if decay_from_intermediate_checkpoint:
        targets = (target_epoch,)
    else:
        start_index = ladder.index(target_epoch)
        end_index = ladder.index(end_target)
        targets = ladder[start_index : end_index + 1]
    previous_output: str | None = None
    if selected_horizon_extension:
        if initial_validation_ce is None:
            raise ValueError("selected horizon extension requires the E16 validation CE")
        commands.append(f"previous_gate_ce={shlex.quote(str(initial_validation_ce))}")
    for index, target in enumerate(targets):
        endpoint_name = run_name(
            prefix,
            target=target,
            global_sequences=global_sequences,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup=warmup,
        ) + f"_{suffix}"
        output = f"/weka/oe-training-default/sewonm/icsl/models/{endpoint_name}"
        if index == 0:
            load_path = source_checkpoint
        else:
            previous_target = targets[index - 1]
            load_path = f"{previous_output}/step{stable_step(previous_target, global_sequences)}"
        endpoint_arguments = stage_arguments(
            base_arguments,
            name=endpoint_name,
            output=output,
            target=target,
            global_sequences=global_sequences,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup=warmup,
            load_path=load_path,
        )
        # The search retains every required skipped-integer pre-decay checkpoint
        # explicitly through fixed_steps. Routine/ephemeral saves are redundant
        # for this workflow and can exhaust the shared Weka quota when several
        # long-frontier neighbors run concurrently.
        endpoint_arguments = [
            value
            for value in endpoint_arguments
            if not value.startswith("--trainer.callbacks.checkpointer.save_interval=")
            and not value.startswith("--trainer.callbacks.checkpointer.ephemeral_save_interval=")
        ]
        endpoint_arguments.extend(
            [
                "--trainer.callbacks.checkpointer.save_interval=1000000000",
                # OLMo requires the ephemeral interval to be strictly smaller
                # than the durable save interval, even when both are placed far
                # beyond this experiment's duration.
                "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
            ]
        )
        if decay_from_intermediate_checkpoint:
            preserved_steps = [stable_step(target, global_sequences)]
        else:
            previous_search_target = ladder[ladder.index(target) - 1] if ladder.index(target) else 0
            preserved_steps = [
                stable_step(epoch, global_sequences)
                for epoch in range(previous_search_target + 1, target + 1)
            ]
        if (
            not decay_from_intermediate_checkpoint
            and target == 1
            and global_sequences in FRACTIONAL_CHECKPOINT_BATCHES
        ):
            preserved_steps = [
                fractional_stable_step(fractional_target, global_sequences)
                for fractional_target in FRACTIONAL_TARGETS
            ] + preserved_steps
        preserved_steps = sorted(set(preserved_steps))
        checkpoint_argument = (
            "--trainer.callbacks.checkpointer.fixed_steps="
            + json.dumps(preserved_steps, separators=(",", ":"))
        )
        endpoint_arguments = [
            checkpoint_argument
            if value.startswith("--trainer.callbacks.checkpointer.fixed_steps=")
            else value
            for value in endpoint_arguments
        ]
        endpoint_arguments = [
            value.replace(
                "batch-size-tune, wsd",
                (
                    "batch-size-tune, intermediate-decay, wsd"
                    if decay_from_intermediate_checkpoint
                    else (
                        "batch-size-tune, selected-horizon-extension, wsd"
                        if selected_horizon_extension
                        else "batch-size-tune, coordinate-descent, wsd"
                    )
                ),
            )
            if value.startswith("--trainer.callbacks.wandb.tags=")
            else value
            for value in endpoint_arguments
        ]
        if load_path is not None:
            commands.append(shlex.join(["test", "-d", load_path]))
        commands.extend(
            (
                shlex.join(["test", "!", "-e", output]),
                shlex.join(
                    [
                        "echo",
                        f"COORD_PREFLIGHT_OK bs={global_sequences} epoch={target} "
                        f"lr={learning_rate} wd={weight_decay} "
                        f"source={load_path or 'fresh'}",
                    ]
                ),
                shlex.join(["python", training_script, endpoint_name, "--dry-run", *endpoint_arguments]),
            )
        )
        train_command = shlex.join(
            [
                "torchrun",
                f"--nproc-per-node={nproc}",
                training_script,
                endpoint_name,
                *endpoint_arguments,
            ]
        )
        if selected_horizon_extension and target != end_target:
            gate_log = f"/tmp/{endpoint_name}.log"
            commands.extend(
                (
                    f"{train_command} 2>&1 | tee {shlex.quote(gate_log)}",
                    "gate_ce=$(sed -n "
                    + shlex.quote(
                        r"s/.*dclm-validation-0802\/CE loss=\([0-9][0-9.]*\).*/\1/p"
                    )
                    + f" {shlex.quote(gate_log)} | tail -n 1)",
                    'test -n "${gate_ce}"',
                    f'echo "HORIZON_GATE epoch={target} observed=${{gate_ce}} '
                    'previous=${previous_gate_ce}"',
                    'if ! awk -v observed="${gate_ce}" -v previous="${previous_gate_ce}" '
                    + shlex.quote("BEGIN { exit !(observed < previous) }")
                    + f'; then echo "HORIZON_STOP epoch={target} observed=${{gate_ce}}"; exit 0; fi',
                    'previous_gate_ce="${gate_ce}"',
                    f'echo "HORIZON_CONTINUE next_epoch={targets[index + 1]}"',
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
    global SSH_HOST
    args = parse_args()
    SSH_HOST = args.ssh_host
    report = reject_duplicate_submission(args)
    base_spec = get_base_spec(args.base_experiment)
    audit_intermediate_decay_source(args, report, base_spec)
    initial_validation_ce = audit_selected_horizon_extension_source(
        args, report, base_spec
    )
    spec = build_endpoint(
        base_spec,
        revision=args.revision,
        global_sequences=args.global_sequences,
        target_epoch=args.target_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        source_checkpoint=args.source_checkpoint,
        chain_through=args.chain_through,
        suffix=args.suffix,
        nproc=args.nproc,
        priority=args.priority,
        decay_from_intermediate_checkpoint=args.decay_from_intermediate_checkpoint,
        selected_horizon_extension=args.selected_horizon_extension,
        initial_validation_ce=initial_validation_ce,
    )
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    result = subprocess.run(
        beaker_command(
            "experiment",
            "create",
            "-",
            "--name",
            args.name,
            "--workspace",
            args.workspace,
        ),
        check=True,
        input=json.dumps(spec),
        text=True,
        stdout=subprocess.PIPE,
    )
    print(result.stdout, end="")
    if args.register:
        ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", result.stdout)
        if not ids:
            raise RuntimeError("submission succeeded without a parsed experiment ID")
        _, _, submitted_args = extract_training_command(spec["tasks"][0])
        outputs = [
            value.split("=", 1)[1]
            for value in submitted_args
            if value.startswith("--save-folder=")
        ]
        if len(outputs) != 1:
            raise RuntimeError("submitted endpoint has no unique output")
        report["batchSweeps"].append(
            {
                "batchSequences": args.global_sequences,
                "contextLength": SEQUENCE_LENGTH,
                "lr": args.learning_rate,
                "wd": args.weight_decay,
                "warmupSteps": warmup_steps(args.global_sequences),
                "gpuCount": args.nproc,
                "rankMicrobatchSequences": RANK_MICROBATCH_SEQUENCES,
                "gradAccumSteps": args.global_sequences
                // (args.nproc * RANK_MICROBATCH_SEQUENCES),
                "status": "pending",
                "activeEpoch": args.target_epoch,
                "search": "pool3b-wd-source-backfill",
                "beaker": ids[0],
                "revision": args.revision,
                "targetLadder": list(target_ladder(args.global_sequences)),
                "e1StableStep": stable_step(1, args.global_sequences),
                "retainedPreDecaySteps": [
                    stable_step(1, args.global_sequences)
                ],
                "output": outputs[0],
                "results": {},
                "reason": (
                    "Exact WD-matched 1B-pool E1 source backfill required by the "
                    "nested Pool-3B repair sweep."
                ),
            }
        )
        report["updated"] = "2026-08-10"
        ATTEMPT_REGISTRY.write_text(json.dumps(report, indent=2) + "\n")
        ATTEMPT_REGISTRY.with_suffix(".js").write_text(
            "window.ICSL_REPORT_DATA="
            + json.dumps(report, separators=(",", ":"))
            + ";\n"
        )


if __name__ == "__main__":
    main()
