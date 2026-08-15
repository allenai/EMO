#!/usr/bin/env python3
"""Submit one guarded small-dense Step 1-1 batch/LR/WD endpoint.

The two small-model reports are independent attempt registries.  Epoch 1
starts fresh; every later endpoint must resume the exact same-coordinate
pre-decay checkpoint from the preceding configured frontier.  All skipped
integer pre-decay checkpoints are retained so that incidental endpoints can
be evaluated without replaying training.
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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SEQUENCE_LENGTH = 4096
TOKENS_PER_EPOCH = 1_000_000_000
DECAY_FRACTION = 0.1
RANK_MICROBATCH_SEQUENCES = 16
REFERENCE_WARMUP_SEQUENCE_STEPS = 24 * 1024
# Coordinate tuning ends at E16.  The user-authorized long-horizon schedule then
# advances the selected exact coordinate in eight-epoch increments.  Completed
# E20/E28 endpoints remain report provenance, but they are no longer launch
# targets.
TARGETS = (
    1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64,
    72, 80, 88, 96, 104, 112, 120, 128,
)
PREDECESSOR = dict(zip(TARGETS[1:], TARGETS[:-1]))
ATTEMPTED_STATUSES = {"pending", "queued", "running", "complete", "failed", "canceled"}
WD_CAP = {
    32: Decimal("0.3"),
    64: Decimal("0.3"),
    128: Decimal("0.3"),
    256: Decimal("0.333"),
    512: Decimal("1.0"),
}
MODELS = {
    "153m": {
        "label": "153M",
        "base": "01KZ6Q4DJ8J994A6SQ39MEGTZ2",
        "revision": "9bca4f9dfc49bbb0eb506b4d761662c2e43f0981",
        "registry": Path("reports/0802/data/wsd_batch_size_153m.json"),
    },
    "474m": {
        "label": "474M",
        "base": "01KZ7307CK7ZZQ1XCJ2QQ08KD4",
        "revision": "e9c7d0ba0fa9af06543149d9fab03700cae05f08",
        "registry": Path("reports/0802/data/wsd_batch_size_474m.json"),
    },
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


def numeric(value: object) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"invalid numeric value: {value!r}") from error


def nproc_for_batch(batch: int) -> int:
    if batch == 32:
        return 2
    return 4 if batch == 64 else 8


def warmup_steps(batch: int) -> int:
    if REFERENCE_WARMUP_SEQUENCE_STEPS % batch:
        raise ValueError(f"BS{batch} does not preserve the token-matched warmup")
    return REFERENCE_WARMUP_SEQUENCE_STEPS // batch


def total_steps(epoch: int, batch: int) -> int:
    return math.ceil(epoch * TOKENS_PER_EPOCH / (batch * SEQUENCE_LENGTH))


def stable_step(epoch: int, batch: int) -> int:
    end_step = total_steps(epoch, batch)
    return end_step - round(DECAY_FRACTION * end_step) - 1


def retained_steps(target: int, batch: int) -> list[int]:
    previous = PREDECESSOR.get(target, 0)
    return [stable_step(epoch, batch) for epoch in range(previous + 1, target + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument(
        "--global-sequences", type=int, choices=(32, 64, 128, 256, 512), required=True
    )
    parser.add_argument("--target-epoch", type=int, choices=TARGETS, required=True)
    parser.add_argument(
        "--learning-rate",
        choices=("5e-4", "1e-3", "1.5e-3", "2e-3", "3e-3", "4e-3"),
        required=True,
    )
    parser.add_argument(
        "--weight-decay",
        choices=("0.01", "0.033", "0.1", "0.3", "0.333", "1.0"),
        required=True,
    )
    parser.add_argument("--source-checkpoint")
    parser.add_argument(
        "--recover-experiment",
        help=(
            "Replace one failed/canceled infrastructure, deterministic preflight, or "
            "explicitly reauthorized policy-pause attempt for the exact tuple while "
            "preserving its provenance in attemptHistory"
        ),
    )
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--ssh-host")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.target_epoch == 1 and args.source_checkpoint and not args.recover_experiment:
        parser.error("E1 must start fresh")
    if args.target_epoch != 1 and not args.source_checkpoint:
        parser.error("later epochs require --source-checkpoint")
    if numeric(args.weight_decay) > WD_CAP[args.global_sequences]:
        parser.error(
            f"WD {args.weight_decay} exceeds the tuned per-batch cap "
            f"{WD_CAP[args.global_sequences]} for BS{args.global_sequences}"
        )
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.suffix):
        parser.error("--suffix must be a lowercase run-name component")
    nproc = nproc_for_batch(args.global_sequences)
    denominator = nproc * RANK_MICROBATCH_SEQUENCES
    if args.global_sequences % denominator:
        parser.error(
            f"BS{args.global_sequences} is not divisible by {nproc} GPUs x "
            f"microbatch {RANK_MICROBATCH_SEQUENCES}"
        )
    return args


def get_base_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        beaker_command("experiment", "spec", experiment, "--format", "json"),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def upsert(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    found = False
    output: list[str] = []
    for argument in arguments:
        if argument.startswith(prefix):
            if not found:
                output.append(replacement)
                found = True
        else:
            output.append(argument)
    if not found:
        output.append(replacement)
    return output


def read_registry(model: str) -> dict[str, Any]:
    registry = MODELS[model]["registry"]
    if not registry.is_file():
        raise FileNotFoundError(f"missing attempt registry: {registry}")
    with registry.open() as report_file:
        return json.load(report_file)


def matching_sweeps(
    report: dict[str, Any], batch: int, lr: str, wd: str
) -> list[dict[str, Any]]:
    return [
        sweep
        for sweep in report.get("batchSweeps", [])
        if sweep.get("batchSequences") == batch
        and numeric(sweep.get("lr")) == numeric(lr)
        and numeric(sweep.get("wd")) == numeric(wd)
    ]


def best_coordinate(
    report: dict[str, Any], batch: int, epoch: int
) -> tuple[Decimal, Decimal, Decimal] | None:
    unhealthy = report.get("healthAudit", {}).get("unhealthy", {})
    max_lr = numeric(report.get("selectionPolicy", {}).get("maxLearningRate", "Infinity"))
    previous_epoch = PREDECESSOR.get(epoch)
    previous_best = None
    while previous_epoch is not None and previous_best is None:
        previous_best = best_coordinate(report, batch, previous_epoch)
        previous_epoch = PREDECESSOR.get(previous_epoch)
    wd_floor = previous_best[2] if previous_best is not None else Decimal("-Infinity")
    candidates: list[tuple[Decimal, Decimal, Decimal]] = []
    for sweep in report.get("batchSweeps", []):
        if sweep.get("batchSequences") != batch:
            continue
        result = sweep.get("results", {}).get(str(epoch))
        if not isinstance(result, dict) or result.get("status") != "complete":
            continue
        if (
            result.get("wandb") in unhealthy
            or result.get("validation") is None
            or numeric(sweep["lr"]) > max_lr
            or numeric(sweep["wd"]) < wd_floor
        ):
            continue
        candidates.append(
            (
                numeric(result["validation"]),
                numeric(sweep["lr"]),
                numeric(sweep["wd"]),
            )
        )
    overrides = report.get("selectionPolicy", {}).get(
        "selectedCoordinateOverrides", {}
    )
    override = overrides.get(str(batch), {}).get(str(epoch))
    if isinstance(override, dict):
        selected = [
            candidate
            for candidate in candidates
            if candidate[1] == numeric(override.get("lr"))
            and candidate[2] == numeric(override.get("wd"))
        ]
        if len(selected) != 1:
            raise SystemExit(
                f"Selected-coordinate override for BS{batch} E{epoch} does not match "
                "exactly one healthy completed result"
            )
        return selected[0]
    return min(candidates) if candidates else None


def batch_policy_value(
    report: dict[str, Any], policy_name: str, batch: int
) -> object | None:
    policy = report.get("selectionPolicy", {}).get(policy_name, {})
    if not isinstance(policy, dict):
        return None
    return policy.get(str(batch), policy.get(batch))


def fixed_continuation_policy(
    report: dict[str, Any], batch: int, lr: str, wd: str, target: int
) -> dict[str, Any] | None:
    policies = report.get("selectionPolicy", {}).get(
        "fixedContinuationChains", []
    )
    for policy in policies:
        if (
            isinstance(policy, dict)
            and policy.get("batchSequences") == batch
            and numeric(policy.get("lr")) == numeric(lr)
            and numeric(policy.get("wd")) == numeric(wd)
            and target > int(policy.get("afterEpoch", 0))
            and target <= int(policy.get("throughEpoch", 0))
        ):
            return policy
    return None


def audit_request(args: argparse.Namespace, report: dict[str, Any]) -> None:
    report_wd_cap = batch_policy_value(
        report, "maxWeightDecayByBatch", args.global_sequences
    )
    if report_wd_cap is not None and numeric(args.weight_decay) > numeric(report_wd_cap):
        raise SystemExit(
            f"WD {args.weight_decay} exceeds the model-specific BS{args.global_sequences} "
            f"cap {report_wd_cap}"
        )

    matches = matching_sweeps(
        report, args.global_sequences, args.learning_rate, args.weight_decay
    )
    duplicate: list[str] = []
    for sweep in matches:
        result = sweep.get("results", {}).get(str(args.target_epoch))
        if result is not None:
            duplicate.append(
                f"result status={result.get('status')} experiment={sweep.get('beaker')}"
            )
        if sweep.get("activeEpoch") == args.target_epoch and str(sweep.get("status", "")).lower() in ATTEMPTED_STATUSES:
            duplicate.append(
                f"sweep status={sweep.get('status')} experiment={sweep.get('beaker')}"
            )
    if duplicate:
        if not args.recover_experiment:
            raise SystemExit("Refusing duplicate exact tuple:\n  - " + "\n  - ".join(duplicate))
        recoverable = [
            sweep
            for sweep in matches
            if sweep.get("beaker") == args.recover_experiment
            and sweep.get("activeEpoch") == args.target_epoch
            and str(sweep.get("status", "")).lower() in {"failed", "canceled"}
            and str(sweep.get("failureClass", "")).startswith(
                ("infrastructure-", "preflight-", "user-policy-pause")
            )
            and sweep.get("results", {}).get(str(args.target_epoch)) is None
        ]
        if len(recoverable) != 1 or len(duplicate) != 1:
            raise SystemExit(
                "Recovery requires exactly one failed/canceled infrastructure, "
                "deterministic preflight, or explicitly reauthorized policy-pause "
                "exact-tuple attempt and no completed or active duplicate"
            )
    elif args.recover_experiment:
        raise SystemExit("--recover-experiment did not match a duplicate exact tuple")

    if args.target_epoch == 1:
        if not args.recover_experiment:
            return
        recovery_sweep = next(
            sweep
            for sweep in matches
            if sweep.get("beaker") == args.recover_experiment
        )
        recovery_failure_class = str(recovery_sweep.get("failureClass", ""))
        if recovery_failure_class.startswith(("preflight-", "user-policy-pause")):
            if args.source_checkpoint is not None:
                raise SystemExit(
                    "No-step E1 recovery must replay fresh rather than load a checkpoint"
                )
            return
        completed_steps = [
            int(step)
            for step in recovery_sweep.get("retainedPreDecaySteps", [])
            if int(step) <= int(recovery_sweep.get("progressStep") or -1)
        ]
        if not completed_steps:
            raise SystemExit(
                "Infrastructure E1 recovery has no surviving exact-coordinate pre-decay checkpoint"
            )
        expected = f"{recovery_sweep['output']}/step{max(completed_steps)}"
        if args.source_checkpoint != expected:
            raise SystemExit(
                "Refusing cross-resume or wrong E1 recovery step: expected " + expected
            )
        return
    predecessor = PREDECESSOR[args.target_epoch]
    healthy = report.get("healthAudit", {}).get("unhealthy", {})
    predecessor_results: list[dict[str, Any]] = []
    for sweep in matches:
        result = sweep.get("results", {}).get(str(predecessor))
        if (
            isinstance(result, dict)
            and result.get("status") == "complete"
            and result.get("wandb") not in healthy
            and result.get("output")
        ):
            predecessor_results.append(result)
    if len(predecessor_results) != 1:
        raise SystemExit(
            f"Expected exactly one healthy exact-coordinate E{predecessor} predecessor; "
            f"found {len(predecessor_results)}"
        )
    predecessor_checkpoint = str(
        predecessor_results[0].get("resumeCheckpoint")
        or f"{predecessor_results[0]['output']}/"
        f"step{stable_step(predecessor, args.global_sequences)}"
    )
    if args.recover_experiment:
        recovery_sweep = next(
            sweep
            for sweep in matches
            if sweep.get("beaker") == args.recover_experiment
        )
        recovery_failure_class = str(recovery_sweep.get("failureClass", ""))
        if recovery_failure_class.startswith(("preflight-", "user-policy-pause")):
            # No scientific endpoint was accepted. Replay the reauthorized tuple from
            # its ordinary exact predecessor instead of assuming partial work survived.
            expected = predecessor_checkpoint
        else:
            completed_steps = [
                int(step)
                for step in recovery_sweep.get("retainedPreDecaySteps", [])
                if int(step) <= int(recovery_sweep.get("progressStep") or -1)
            ]
            if not completed_steps:
                raise SystemExit(
                    "Infrastructure recovery has no surviving exact-coordinate pre-decay checkpoint"
                )
            expected = f"{recovery_sweep['output']}/step{max(completed_steps)}"
    else:
        expected = predecessor_checkpoint
    if args.source_checkpoint != expected:
        raise SystemExit(
            "Refusing cross-resume or wrong step: expected " + expected
        )

    freeze_after = batch_policy_value(
        report, "learningRateFreezeAfterEpoch", args.global_sequences
    )
    if freeze_after is not None and args.target_epoch > int(freeze_after):
        frozen = best_coordinate(report, args.global_sequences, int(freeze_after))
        if frozen is None:
            raise SystemExit(
                f"E{freeze_after} is not resolved, so its learning rate cannot be frozen"
            )
        frozen_lr = frozen[1]
        if numeric(args.learning_rate) != frozen_lr:
            raise SystemExit(
                f"LR is frozen after E{freeze_after} at {frozen_lr}; refusing "
                f"LR {args.learning_rate}"
            )

    fixed_policy = fixed_continuation_policy(
        report,
        args.global_sequences,
        args.learning_rate,
        args.weight_decay,
        args.target_epoch,
    )
    if fixed_policy is not None:
        comparison_epoch = PREDECESSOR.get(predecessor)
        if comparison_epoch is not None and fixed_policy.get(
            "stopOnNonImprovement", True
        ):
            comparison_results: list[dict[str, Any]] = []
            for sweep in matches:
                result = sweep.get("results", {}).get(str(comparison_epoch))
                if (
                    isinstance(result, dict)
                    and result.get("status") == "complete"
                    and result.get("wandb") not in healthy
                    and result.get("validation") is not None
                ):
                    comparison_results.append(result)
            if len(comparison_results) != 1:
                raise SystemExit(
                    f"Expected exactly one healthy exact-coordinate E{comparison_epoch} "
                    f"comparison for the fixed continuation; found {len(comparison_results)}"
                )
            if numeric(predecessor_results[0]["validation"]) >= numeric(
                comparison_results[0]["validation"]
            ):
                raise SystemExit(
                    f"Fixed-chain early-stop guard: E{predecessor} CE "
                    f"{predecessor_results[0]['validation']} did not improve on "
                    f"E{comparison_epoch} CE {comparison_results[0]['validation']}"
                )
        return

    hold_after = batch_policy_value(report, "holdAfterEpoch", args.global_sequences)
    if hold_after is not None and args.target_epoch > int(hold_after):
        raise SystemExit(
            f"BS{args.global_sequences} is held after E{hold_after} pending new user "
            "authorization"
        )

    single_after = batch_policy_value(
        report, "singleCoordinateAfterEpoch", args.global_sequences
    )
    if single_after is not None and args.target_epoch > int(single_after):
        selected = best_coordinate(report, args.global_sequences, int(single_after))
        if selected is None:
            raise SystemExit(
                f"E{single_after} is not resolved, so its single continuation cannot start"
            )
        _, selected_lr, selected_wd = selected
        if (
            numeric(args.learning_rate) != selected_lr
            or numeric(args.weight_decay) != selected_wd
        ):
            raise SystemExit(
                f"After E{single_after}, BS{args.global_sequences} must continue only "
                f"the selected LR {selected_lr}, WD {selected_wd} coordinate"
            )

    previous_best = best_coordinate(report, args.global_sequences, predecessor)
    if previous_best is None:
        raise SystemExit(f"E{predecessor} is not resolved")
    _, _, wd_floor = previous_best
    if numeric(args.weight_decay) < wd_floor:
        raise SystemExit(
            f"WD {args.weight_decay} is below the selected E{predecessor} floor {wd_floor}"
        )
    comparison_epoch = PREDECESSOR.get(predecessor)
    if comparison_epoch is not None:
        comparison_best = best_coordinate(report, args.global_sequences, comparison_epoch)
        if comparison_best is None:
            raise SystemExit(f"E{comparison_epoch} is not resolved")
        if previous_best[0] >= comparison_best[0]:
            raise SystemExit(
                f"Early-stop guard: selected E{predecessor} CE {previous_best[0]} did not "
                f"improve on E{comparison_epoch} CE {comparison_best[0]}"
            )


def run_name(args: argparse.Namespace) -> str:
    warmup = warmup_steps(args.global_sequences)
    return (
        f"dense_{args.model}_step1_0802_repeated_dclm1b_bs{args.global_sequences}_"
        f"e{args.target_epoch}_lr{args.learning_rate}_wd{args.weight_decay}_"
        f"warmup{warmup}_{args.suffix}"
    )


def audit_beaker_name(workspace: str, name: str) -> None:
    result = subprocess.run(
        beaker_command("workspace", "experiments", workspace, "--text", name, "--format", "json"),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)
    experiments = payload if isinstance(payload, list) else payload.get("experiments", [])
    exact = [item for item in experiments if item.get("name") == name]
    if exact:
        raise SystemExit(f"Refusing Beaker duplicate experiment name {name}")


def build_spec(args: argparse.Namespace, base_spec: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    original = task["arguments"]
    if original[:1] != ["python"]:
        raise ValueError("expected a single-endpoint Gantry Python task")
    script = original[1]
    arguments = [
        argument
        for argument in original[3:]
        if not argument.startswith(("--load_path=", "--load_trainer_state="))
    ]
    name = run_name(args)
    output = f"/weka/oe-training-default/sewonm/icsl/models/{name}"
    warmup = warmup_steps(args.global_sequences)
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        ("--trainer.max_duration=", f"--trainer.max_duration={{value: {args.target_epoch * TOKENS_PER_EPOCH}, unit: tokens}}"),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
        (
            "--trainer.callbacks.wandb.tags=",
            f"--trainer.callbacks.wandb.tags=[pretraining, step1, 0802, dense-{args.model}, repeated-data, dclm-train-only, batch-size-tune, coordinate-descent, wsd, bs{args.global_sequences}, warmup{warmup}]",
        ),
        (
            "--trainer.callbacks.checkpointer.fixed_steps=",
            "--trainer.callbacks.checkpointer.fixed_steps=" + json.dumps(retained_steps(args.target_epoch, args.global_sequences), separators=(",", ":")),
        ),
        ("--data_loader.global_batch_size=", f"--data_loader.global_batch_size={args.global_sequences * SEQUENCE_LENGTH}"),
        ("--train_module.rank_microbatch_size=", f"--train_module.rank_microbatch_size={RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH}"),
        (
            "--train_module.scheduler=",
            f"--train_module.scheduler={{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: {warmup}, decay_fraction: {DECAY_FRACTION}}}",
        ),
        ("--train_module.optim.weight_decay=", f"--train_module.optim.weight_decay={args.weight_decay}"),
        ("--lr=", f"--lr={args.learning_rate}"),
    )
    for prefix, replacement in replacements:
        arguments = upsert(arguments, prefix, replacement)
    arguments = [
        argument
        for argument in arguments
        if not argument.startswith(
            (
                "--trainer.callbacks.checkpointer.save_interval=",
                "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
            )
        )
    ]
    arguments.extend(
        (
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        )
    )
    if args.source_checkpoint:
        arguments.extend((f"--load_path={args.source_checkpoint}", "--load_trainer_state=true"))
    commands = ["set -euo pipefail"]
    if args.source_checkpoint:
        commands.append(shlex.join(["test", "-d", args.source_checkpoint]))
    commands.extend(
        (
            shlex.join(["test", "!", "-e", output]),
            shlex.join(["echo", f"SMALL_COORD_PREFLIGHT_OK model={args.model} bs={args.global_sequences} epoch={args.target_epoch} lr={args.learning_rate} wd={args.weight_decay} source={args.source_checkpoint or 'fresh'}"]),
            shlex.join(["python", script, name, "--dry-run", *arguments]),
            shlex.join(["torchrun", f"--nproc-per-node={nproc_for_batch(args.global_sequences)}", script, name, *arguments]),
        )
    )
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["envVars"] = [
        env for env in task.get("envVars", []) if env.get("name") not in {"GANTRY_USE_TORCHRUN", "PYTORCH_CUDA_ALLOC_CONF"}
    ]
    for env in task["envVars"]:
        if env.get("name") == "GIT_REF":
            env["value"] = MODELS[args.model]["revision"]
            break
    if args.model == "474m":
        task["envVars"].append({"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"})
    task["resources"] = {"gpuCount": nproc_for_batch(args.global_sequences), "sharedMemory": "10 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": "0s", "autoResume": False}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec.pop("description", None)
    return spec, name, output


def write_registry(args: argparse.Namespace, experiment: str, output: str) -> None:
    report = read_registry(args.model)
    nproc = nproc_for_batch(args.global_sequences)
    new_sweep = {
        "batchSequences": args.global_sequences,
        "globalBatchTokens": args.global_sequences * SEQUENCE_LENGTH,
        "contextLength": SEQUENCE_LENGTH,
        "lr": args.learning_rate,
        "wd": args.weight_decay,
        "warmupSteps": warmup_steps(args.global_sequences),
        "rankMicrobatchSequences": RANK_MICROBATCH_SEQUENCES,
        "gradientAccumulation": args.global_sequences
        // (nproc * RANK_MICROBATCH_SEQUENCES),
        "gpuCount": nproc,
        "status": "pending",
        "activeEpoch": args.target_epoch,
        "search": "small-model-adaptive-coordinate",
        "beaker": experiment,
        "output": output,
        "sourceCheckpoint": args.source_checkpoint,
        "retainedPreDecaySteps": retained_steps(
            args.target_epoch, args.global_sequences
        ),
        "results": {},
        "reason": "Adaptive small-model Step 1-1 coordinate search; exact-coordinate resumes, empirically tuned per-batch WD cap, and nondecreasing-WD frontier policy apply.",
    }
    if args.recover_experiment:
        recoverable = [
            sweep
            for sweep in report.get("batchSweeps", [])
            if sweep.get("beaker") == args.recover_experiment
            and sweep.get("batchSequences") == args.global_sequences
            and numeric(sweep.get("lr")) == numeric(args.learning_rate)
            and numeric(sweep.get("wd")) == numeric(args.weight_decay)
            and sweep.get("activeEpoch") == args.target_epoch
        ]
        if len(recoverable) != 1:
            raise RuntimeError("recovery target disappeared from the registry")
        sweep = recoverable[0]
        recovered_failure_class = str(sweep.get("failureClass", ""))
        attempt_history = list(sweep.get("attemptHistory", []))
        attempt_history.append(
            {
                key: sweep.get(key)
                for key in (
                    "beaker",
                    "job",
                    "status",
                    "failureClass",
                    "reason",
                    "resultDataset",
                    "output",
                    "sourceCheckpoint",
                    "activeWandb",
                    "progressPercent",
                    "progressStep",
                    "progressTotalSteps",
                )
                if sweep.get(key) is not None
            }
        )
        sweep.clear()
        sweep.update(new_sweep)
        sweep["attemptHistory"] = attempt_history
        sweep["recoveryOf"] = args.recover_experiment
        if recovered_failure_class.startswith("preflight-"):
            sweep["reason"] = (
                "Guarded replay of a deterministic no-step preflight failure from the "
                "verified exact-coordinate predecessor checkpoint."
            )
        elif recovered_failure_class.startswith("user-policy-pause"):
            sweep["reason"] = (
                "User-authorized replay of a previously paused exact tuple from the "
                "verified exact-coordinate predecessor checkpoint."
            )
        else:
            sweep["reason"] = (
                "Guarded recovery from a clear infrastructure-only failure; the newest "
                "surviving exact-coordinate pre-decay checkpoint is reused."
            )
    else:
        report["batchSweeps"].append(new_sweep)
    report["updated"] = "2026-08-09"
    registry = MODELS[args.model]["registry"]
    text = json.dumps(report, indent=2) + "\n"
    registry.write_text(text)
    registry.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    global SSH_HOST
    args = parse_args()
    SSH_HOST = args.ssh_host
    report = read_registry(args.model)
    audit_request(args, report)
    spec, name, output = build_spec(args, get_base_spec(MODELS[args.model]["base"]))
    audit_beaker_name(args.workspace, name)
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    result = subprocess.run(
        beaker_command("experiment", "create", "-", "--name", name, "--workspace", args.workspace),
        check=True,
        input=json.dumps(spec),
        text=True,
        stdout=subprocess.PIPE,
    )
    print(result.stdout, end="")
    ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", result.stdout)
    if args.register:
        if not ids:
            raise RuntimeError("submission succeeded but no experiment ID was found in Beaker output")
        write_registry(args, ids[0], output)


if __name__ == "__main__":
    main()
