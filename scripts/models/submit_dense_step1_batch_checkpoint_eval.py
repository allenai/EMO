#!/usr/bin/env python3
"""Submit an evaluation-only job for a retained Step 1 batch checkpoint.

This is intentionally narrower than the coordinate launcher: it never trains,
never resumes trainer state, and only evaluates a checkpoint that is already
registered as an incidental endpoint of a completed exact-coordinate run.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from submit_dense_step1_batch_coordinate import extract_training_command, numeric_value


DEFAULT_ATTEMPT_REGISTRY = Path("reports/0802/data/wsd_batch_size_1b.json")
MAX_LEARNING_RATE = Decimal("1e-3")
TOKENS_PER_SEQUENCE = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--global-sequences", type=int, choices=(16, 32, 64, 128, 256, 512), required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--learning-rate", choices=("1.25e-4", "2.5e-4", "5e-4", "1e-3"), required=True)
    parser.add_argument("--weight-decay", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_ATTEMPT_REGISTRY)
    parser.add_argument("--recovery-of")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.print_only and args.register:
        parser.error("--print-only and --register are mutually exclusive")
    if numeric_value(args.learning_rate) > MAX_LEARNING_RATE:
        parser.error("checkpoint evaluation is capped at LR1e-3 by policy")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", args.weight_decay):
        parser.error("unsupported weight-decay format")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.suffix):
        parser.error("--suffix must be a lowercase run-name component")
    if not args.checkpoint.endswith(f"/step{args.checkpoint_step}"):
        parser.error("--checkpoint must end at the exact --checkpoint-step")
    if args.global_sequences % (args.nproc * 4):
        parser.error("global batch must be divisible by nproc x rank microbatch (4)")
    return args


def get_base_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def argument_value(arguments: list[str], prefix: str) -> str | None:
    return next((argument[len(prefix) :] for argument in arguments if argument.startswith(prefix)), None)


def replace_argument(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    matches = [index for index, value in enumerate(arguments) if value.startswith(prefix)]
    if len(matches) > 1:
        raise ValueError(f"base command has duplicate {prefix} arguments")
    output = list(arguments)
    if matches:
        output[matches[0]] = replacement
    else:
        output.append(replacement)
    return output


def audit_registered_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    report = json.loads(args.registry.read_text())
    requested_lr = numeric_value(args.learning_rate)
    requested_wd = numeric_value(args.weight_decay)
    if args.recovery_of:
        matches = [
            sweep
            for sweep in report.get("batchSweeps", [])
            if sweep.get("beaker") == args.recovery_of
            and sweep.get("batchSequences") == args.global_sequences
            and numeric_value(sweep.get("lr")) == requested_lr
            and numeric_value(sweep.get("wd")) == requested_wd
            and sweep.get("activeEpoch") == args.epoch
        ]
        if len(matches) != 1:
            raise SystemExit(
                "Refusing endpoint recovery: expected exactly one registered canceled "
                f"attempt {args.recovery_of}; found {len(matches)}"
            )
        sweep = matches[0]
        if sweep.get("status") not in {"canceled", "failed"}:
            raise SystemExit("Endpoint recovery requires a terminal canceled/failed attempt")
        if not str(sweep.get("failureClass", "")).startswith("infrastructure-"):
            raise SystemExit("Endpoint recovery is restricted to infrastructure-only failures")
        partial = sweep.get("partialEndpoint", {})
        if not (
            partial.get("trainingCompleted") is True
            and partial.get("evaluationCompleted") is False
            and partial.get("checkpoint") == args.checkpoint
        ):
            raise SystemExit(
                "Endpoint recovery requires the exact registered completed-training checkpoint"
            )
        return sweep

    matches: list[dict[str, Any]] = []
    for sweep in report.get("batchSweeps", []):
        if sweep.get("batchSequences") != args.global_sequences:
            continue
        if numeric_value(sweep.get("lr")) != requested_lr or numeric_value(sweep.get("wd")) != requested_wd:
            continue
        result = sweep.get("results", {}).get(str(args.epoch))
        if result is not None:
            raise SystemExit(
                f"Refusing duplicate E{args.epoch} evaluation: result already recorded in "
                f"experiment {sweep.get('beaker', 'unknown')}"
            )
        if (
            sweep.get("activeEpoch") == args.epoch
            and str(sweep.get("status", "")).lower() in {"pending", "queued", "running", "complete", "failed", "canceled"}
        ):
            raise SystemExit(
                f"Refusing duplicate E{args.epoch} evaluation attempt in "
                f"experiment {sweep.get('beaker', 'unknown')}"
            )
        if args.epoch not in sweep.get("incidentalEvaluationEpochs", []):
            continue
        if args.checkpoint_step not in sweep.get("retainedPreDecaySteps", []):
            continue
        completed_results = [
            value
            for value in sweep.get("results", {}).values()
            if isinstance(value, dict) and value.get("status") == "complete" and value.get("output")
        ]
        if not any(args.checkpoint == f"{value['output']}/step{args.checkpoint_step}" for value in completed_results):
            continue
        matches.append(sweep)
    if len(matches) != 1:
        raise SystemExit(
            "Refusing checkpoint evaluation: expected exactly one completed registered "
            f"trajectory for BS{args.global_sequences} E{args.epoch} LR={args.learning_rate} "
            f"WD={args.weight_decay}; found {len(matches)}"
        )
    if report.get("healthAudit", {}).get("unhealthy", {}).get(matches[0].get("activeWandb")):
        raise SystemExit("Refusing checkpoint evaluation from an unhealthy exact trajectory")
    return matches[0]


def audit_base_tuple(
    args: argparse.Namespace,
    sweep: dict[str, Any],
    base_spec: dict[str, Any],
) -> tuple[str, str, list[str]]:
    if sweep.get("beaker") != args.base_experiment:
        raise SystemExit("--base-experiment must be the registered exact-coordinate trajectory")
    training_script, base_name, base_arguments = extract_training_command(base_spec["tasks"][0])
    expected_batch = str(args.global_sequences * TOKENS_PER_SEQUENCE)
    checks = {
        "learning rate": (argument_value(base_arguments, "--lr="), args.learning_rate),
        "weight decay": (argument_value(base_arguments, "--train_module.optim.weight_decay="), args.weight_decay),
        "global batch tokens": (argument_value(base_arguments, "--data_loader.global_batch_size="), expected_batch),
    }
    mismatches = [f"{label}: spec={actual!r}, requested={expected!r}" for label, (actual, expected) in checks.items() if numeric_value(actual) != numeric_value(expected)]
    if mismatches:
        raise SystemExit("Base-spec tuple mismatch:\n  - " + "\n  - ".join(mismatches))
    return training_script, base_name, base_arguments


def build_spec(
    args: argparse.Namespace,
    base_spec: dict[str, Any],
    training_script: str,
    base_name: str,
    base_arguments: list[str],
) -> tuple[dict[str, Any], str]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    stem = re.sub(r"_e\d+_", f"_e{args.epoch}_", base_name, count=1)
    run_name = f"{stem}_{args.suffix}"
    output = f"/weka/oe-training-default/sewonm/icsl/models/{run_name}"
    arguments = list(base_arguments)
    replacements = {
        "--save-folder=": f"--save-folder={output}",
        "--trainer.callbacks.wandb.name=": f"--trainer.callbacks.wandb.name={run_name}",
        "--trainer.callbacks.wandb.tags=": (
            "--trainer.callbacks.wandb.tags=[checkpoint-eval, step1, 0802, repeated-data, "
            f"dclm-validation, downstream-nine, batch-size-tune, bs{args.global_sequences}, e{args.epoch}]"
        ),
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=": "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
        "--load_path=": f"--load_path={args.checkpoint}",
        "--load_trainer_state=": "--load_trainer_state=false",
    }
    for prefix, replacement in replacements.items():
        arguments = replace_argument(arguments, prefix, replacement)
    arguments = replace_argument(
        arguments,
        "--trainer.callbacks.downstream_evaluator.eval_on_startup=",
        "--trainer.callbacks.downstream_evaluator.eval_on_startup=true",
    )
    arguments = replace_argument(
        arguments,
        "--trainer.callbacks.downstream_evaluator.cancel_after_first_eval=",
        "--trainer.callbacks.downstream_evaluator.cancel_after_first_eval=true",
    )
    heldout_prefix = "--trainer.callbacks.heldout_evaluator="
    heldout = argument_value(arguments, heldout_prefix)
    if heldout is None or "dclm_0802_validation.json" not in heldout:
        raise SystemExit("Base spec does not contain the required DCLM 0802 held-out evaluator")
    heldout = heldout.replace("eval_on_finish: true", "eval_on_finish: false, eval_on_startup: true")
    arguments = replace_argument(arguments, heldout_prefix, heldout_prefix + heldout)
    arguments = [value for value in arguments if not value.startswith("--trainer.callbacks.checkpointer.fixed_steps=")]
    arguments = replace_argument(
        arguments,
        "--trainer.callbacks.checkpointer.enabled=",
        "--trainer.callbacks.checkpointer.enabled=false",
    )
    commands = [
        "set -euo pipefail",
        shlex.join(["test", "-d", args.checkpoint]),
        shlex.join(["test", "!", "-e", output]),
        shlex.join(
            [
                "echo",
                f"CHECKPOINT_EVAL_PREFLIGHT_OK bs={args.global_sequences} epoch={args.epoch} "
                f"lr={args.learning_rate} wd={args.weight_decay} checkpoint={args.checkpoint}",
            ]
        ),
        shlex.join(["python", training_script, run_name, "--dry-run", *arguments]),
        shlex.join(["torchrun", f"--nproc-per-node={args.nproc}", training_script, run_name, *arguments]),
    ]
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    for env_var in task.get("envVars", []):
        if env_var.get("name") == "GIT_REF":
            env_var["value"] = args.revision
            break
    else:
        task.setdefault("envVars", []).append({"name": "GIT_REF", "value": args.revision})
    task["resources"] = {"gpuCount": args.nproc, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": "0s", "autoResume": False}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec.pop("description", None)
    return spec, output


def register_recovery(
    args: argparse.Namespace,
    experiment: str,
    evaluation_output: str,
) -> None:
    if not args.recovery_of:
        raise RuntimeError("--register currently requires --recovery-of")
    report = json.loads(args.registry.read_text())
    matches = [
        sweep
        for sweep in report.get("batchSweeps", [])
        if sweep.get("beaker") == args.recovery_of
        and sweep.get("batchSequences") == args.global_sequences
        and numeric_value(sweep.get("lr")) == numeric_value(args.learning_rate)
        and numeric_value(sweep.get("wd")) == numeric_value(args.weight_decay)
        and sweep.get("activeEpoch") == args.epoch
    ]
    if len(matches) != 1:
        raise RuntimeError("recovery target disappeared from the registry")
    sweep = matches[0]
    partial = copy.deepcopy(sweep.get("partialEndpoint", {}))
    history = list(sweep.get("attemptHistory", []))
    history.append(
        {
            key: copy.deepcopy(sweep.get(key))
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
                "partialEndpoint",
            )
            if sweep.get(key) is not None
        }
    )
    for key in (
        "job",
        "resultDataset",
        "failureClass",
        "progressPercent",
        "progressStep",
        "progressTotalSteps",
    ):
        sweep.pop(key, None)
    sweep.update(
        {
            "status": "pending",
            "beaker": experiment,
            "activeWandb": None,
            "attemptHistory": history,
            "recoveryOf": args.recovery_of,
            "evaluationOnly": True,
            "evaluationCheckpoint": args.checkpoint,
            "evaluationOutput": evaluation_output,
            "trainFallback": partial.get("train"),
            "reason": (
                "User-authorized evaluation-only recovery from the exact saved E"
                f"{args.epoch} endpoint checkpoint after an infrastructure-cordoned-node "
                "failure interrupted the original evaluation. No training is replayed."
            ),
        }
    )
    text = json.dumps(report, indent=2) + "\n"
    args.registry.write_text(text)
    args.registry.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    args = parse_args()
    sweep = audit_registered_checkpoint(args)
    base_spec = get_base_spec(args.base_experiment)
    training_script, base_name, base_arguments = audit_base_tuple(args, sweep, base_spec)
    spec, evaluation_output = build_spec(
        args, base_spec, training_script, base_name, base_arguments
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
    ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", result.stdout)
    if args.register:
        if not ids:
            raise RuntimeError("submission succeeded but no experiment ID was found")
        register_recovery(args, ids[0], evaluation_output)


if __name__ == "__main__":
    main()
