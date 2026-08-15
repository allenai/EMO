#!/usr/bin/env python3
"""Submit one selected-E1-coordinate fractional chain for a small dense model.

BS32, BS64, and BS128 evaluate E0.125, E0.25, and E0.5. BS256 evaluates E0.25
and E0.5.  Every later stage resumes the exact same-coordinate pre-decay
checkpoint from the preceding stage.  The launcher derives the coordinate
from the model report, uses the verified small-model revision and GPU topology,
audits report/Beaker duplicates, and can register the attempt in both mirrors.
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
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from submit_small_dense_step1_batch_coordinate import (
    DECAY_FRACTION,
    MODELS,
    RANK_MICROBATCH_SEQUENCES,
    SEQUENCE_LENGTH,
    TOKENS_PER_EPOCH,
    get_base_spec,
    nproc_for_batch,
    numeric,
    upsert,
    warmup_steps,
)


SEARCH = "small-model-selected-e1-fractional-chain"
ATTEMPTED_STATUSES = {"pending", "queued", "running", "complete", "failed", "canceled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--global-sequences", type=int, choices=(32, 64, 128, 256), required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.suffix):
        parser.error("--suffix must be a lowercase run-name component")
    return args


def targets_for_batch(batch: int) -> tuple[Fraction, ...]:
    if batch in (32, 64, 128):
        return (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2))
    if batch == 256:
        return (Fraction(1, 4), Fraction(1, 2))
    raise ValueError(f"unsupported fractional batch: {batch}")


def target_key(target: Fraction) -> str:
    return str(Decimal(target.numerator) / Decimal(target.denominator)).rstrip("0")


def target_slug(target: Fraction) -> str:
    return target_key(target).replace(".", "p")


def target_tokens(target: Fraction) -> int:
    numerator = target.numerator * TOKENS_PER_EPOCH
    if numerator % target.denominator:
        raise ValueError(f"E{target_key(target)} does not have an integer token count")
    return numerator // target.denominator


def end_step(target: Fraction, batch: int) -> int:
    return math.ceil(target_tokens(target) / (batch * SEQUENCE_LENGTH))


def stable_step(target: Fraction, batch: int) -> int:
    endpoint = end_step(target, batch)
    return endpoint - round(DECAY_FRACTION * endpoint) - 1


def read_report(model: str) -> dict[str, Any]:
    path: Path = MODELS[model]["registry"]
    return json.loads(path.read_text())


def selected_epoch_one(report: dict[str, Any], batch: int) -> dict[str, Any]:
    unhealthy = report.get("healthAudit", {}).get("unhealthy", {})
    max_lr = numeric(report.get("selectionPolicy", {}).get("maxLearningRate", "Infinity"))
    candidates: list[tuple[Decimal, Decimal, Decimal, dict[str, Any], dict[str, Any]]] = []
    for sweep in report.get("batchSweeps", []):
        if int(sweep.get("batchSequences", -1)) != batch:
            continue
        result = sweep.get("results", {}).get("1")
        if not isinstance(result, dict) or result.get("status") != "complete":
            continue
        if result.get("wandb") in unhealthy or result.get("validation") is None:
            continue
        lr = numeric(sweep["lr"])
        if lr > max_lr:
            continue
        candidates.append((numeric(result["validation"]), lr, numeric(sweep["wd"]), sweep, result))
    if not candidates:
        raise SystemExit(f"No healthy completed E1 coordinate for BS{batch}")
    overrides = report.get("selectionPolicy", {}).get(
        "selectedCoordinateOverrides", {}
    )
    override = overrides.get(str(batch), {}).get("1")
    if isinstance(override, dict):
        selected = [
            candidate
            for candidate in candidates
            if candidate[1] == numeric(override.get("lr"))
            and candidate[2] == numeric(override.get("wd"))
        ]
        if len(selected) != 1:
            raise SystemExit(
                f"Selected-coordinate override for BS{batch} E1 does not match "
                "exactly one healthy completed result"
            )
        validation, _, _, sweep, result = selected[0]
    else:
        validation, _, _, sweep, result = min(candidates, key=lambda item: item[:3])
    return {
        "lr": str(sweep["lr"]),
        "wd": str(sweep["wd"]),
        "validation": str(validation),
        "beaker": sweep.get("beaker"),
        "wandb": result.get("wandb"),
        "output": result.get("output"),
    }


def reject_report_duplicate(
    report: dict[str, Any], batch: int, lr: str, wd: str, targets: tuple[Fraction, ...]
) -> None:
    requested = {target_key(target) for target in targets}
    duplicates: list[str] = []
    for sweep in report.get("batchSweeps", []):
        if int(sweep.get("batchSequences", -1)) != batch:
            continue
        if numeric(sweep.get("lr")) != numeric(lr) or numeric(sweep.get("wd")) != numeric(wd):
            continue
        result_targets = requested.intersection(map(str, sweep.get("results", {})))
        if result_targets:
            duplicates.append(
                f"results {sorted(result_targets)} experiment={sweep.get('beaker', 'unknown')}"
            )
        if sweep.get("search") == SEARCH and str(sweep.get("status", "")).lower() in ATTEMPTED_STATUSES:
            duplicates.append(
                f"fractional chain status={sweep.get('status')} "
                f"experiment={sweep.get('beaker', 'unknown')}"
            )
    if duplicates:
        raise SystemExit("Refusing duplicate fractional work:\n  - " + "\n  - ".join(duplicates))


def run_name_prefix(
    args: argparse.Namespace, lr: str, wd: str, targets: tuple[Fraction, ...]
) -> str:
    target_part = "-".join(target_slug(target) for target in targets)
    return (
        f"dense_{args.model}_step1_0802_repeated_dclm1b_bs{args.global_sequences}_"
        f"frac_{target_part}_lr{lr}_wd{wd}_warmup{warmup_steps(args.global_sequences)}_"
    )


def run_name(args: argparse.Namespace, lr: str, wd: str, targets: tuple[Fraction, ...]) -> str:
    return run_name_prefix(args, lr, wd, targets) + args.suffix


def audit_beaker_tuple(workspace: str, prefix: str) -> None:
    result = subprocess.run(
        ["beaker", "workspace", "experiments", workspace, "--text", prefix, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)
    experiments = payload if isinstance(payload, list) else payload.get("experiments", [])
    duplicates = [
        item.get("name")
        for item in experiments
        if str(item.get("name", "")).startswith(prefix)
    ]
    if duplicates:
        raise SystemExit(
            "Refusing Beaker-attempted fractional tuple:\n  - "
            + "\n  - ".join(map(str, duplicates))
        )


def stage_arguments(
    base_arguments: list[str],
    *,
    args: argparse.Namespace,
    target: Fraction,
    name: str,
    output: str,
    lr: str,
    wd: str,
    load_path: str | None,
) -> list[str]:
    batch = args.global_sequences
    warmup = warmup_steps(batch)
    arguments = [
        value
        for value in base_arguments
        if not value.startswith(
            (
                "--load_path=",
                "--load_trainer_state=",
                "--trainer.callbacks.checkpointer.save_interval=",
                "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
            )
        )
    ]
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        ("--trainer.max_duration=", f"--trainer.max_duration={{value: {target_tokens(target)}, unit: tokens}}"),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
        (
            "--trainer.callbacks.wandb.tags=",
            "--trainer.callbacks.wandb.tags="
            + f"[pretraining, step1, 0802, dense-{args.model}, repeated-data, dclm-train-only, "
            + f"batch-size-tune, fractional-endpoint, wsd, bs{batch}, warmup{warmup}]",
        ),
        (
            "--trainer.callbacks.checkpointer.fixed_steps=",
            "--trainer.callbacks.checkpointer.fixed_steps="
            + json.dumps([stable_step(target, batch), end_step(target, batch)], separators=(",", ":")),
        ),
        ("--data_loader.global_batch_size=", f"--data_loader.global_batch_size={batch * SEQUENCE_LENGTH}"),
        (
            "--train_module.rank_microbatch_size=",
            f"--train_module.rank_microbatch_size={RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH}",
        ),
        (
            "--train_module.scheduler=",
            f"--train_module.scheduler={{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, "
            f"warmup: {warmup}, decay_fraction: {DECAY_FRACTION}}}",
        ),
        ("--train_module.optim.weight_decay=", f"--train_module.optim.weight_decay={wd}"),
        ("--lr=", f"--lr={lr}"),
    )
    for prefix, replacement in replacements:
        arguments = upsert(arguments, prefix, replacement)
    arguments.extend(
        (
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        )
    )
    if load_path:
        arguments.extend((f"--load_path={load_path}", "--load_trainer_state=true"))
    return arguments


def build_spec(
    base_spec: dict[str, Any],
    args: argparse.Namespace,
    lr: str,
    wd: str,
    targets: tuple[Fraction, ...],
    initial_load_path: str | None = None,
) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    original = task["arguments"]
    if original[:1] != ["python"]:
        raise ValueError("expected a single-endpoint Gantry Python task")
    script = original[1]
    base_arguments = original[3:]
    outer_name = run_name(args, lr, wd, targets)
    warmup = warmup_steps(args.global_sequences)
    commands = ["set -euo pipefail"]
    endpoint_metadata: dict[str, dict[str, Any]] = {}
    previous_output: str | None = None
    previous_target: Fraction | None = None
    for target in targets:
        key = target_key(target)
        endpoint_name = (
            f"dense_{args.model}_step1_0802_repeated_dclm1b_bs{args.global_sequences}_"
            f"e{target_slug(target)}_lr{lr}_wd{wd}_warmup{warmup}_{args.suffix}"
        )
        output = f"/weka/oe-training-default/sewonm/icsl/models/{endpoint_name}"
        load_path = initial_load_path
        if previous_output is not None and previous_target is not None:
            load_path = f"{previous_output}/step{stable_step(previous_target, args.global_sequences)}"
        endpoint_arguments = stage_arguments(
            base_arguments,
            args=args,
            target=target,
            name=endpoint_name,
            output=output,
            lr=lr,
            wd=wd,
            load_path=load_path,
        )
        if load_path:
            commands.append(shlex.join(["test", "-d", load_path]))
        commands.extend(
            (
                shlex.join(["test", "!", "-e", output]),
                shlex.join(
                    [
                        "echo",
                        f"FRACTIONAL_STAGE_BEGIN epoch={key} output={output} "
                        f"stable={stable_step(target, args.global_sequences)} "
                        f"end={end_step(target, args.global_sequences)}",
                    ]
                ),
                shlex.join(["python", script, endpoint_name, "--dry-run", *endpoint_arguments]),
                shlex.join(
                    [
                        "torchrun",
                        f"--nproc-per-node={nproc_for_batch(args.global_sequences)}",
                        script,
                        endpoint_name,
                        *endpoint_arguments,
                    ]
                ),
                shlex.join(["echo", f"FRACTIONAL_STAGE_END epoch={key}"]),
            )
        )
        endpoint_metadata[key] = {
            "stableStep": stable_step(target, args.global_sequences),
            "endStep": end_step(target, args.global_sequences),
            "output": output,
            "sourceCheckpoint": load_path,
        }
        previous_output = output
        previous_target = target

    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["envVars"] = [
        env
        for env in task.get("envVars", [])
        if env.get("name") not in {"GANTRY_USE_TORCHRUN", "PYTORCH_CUDA_ALLOC_CONF"}
    ]
    for env in task["envVars"]:
        if env.get("name") == "GIT_REF":
            env["value"] = MODELS[args.model]["revision"]
            break
    else:
        task["envVars"].append({"name": "GIT_REF", "value": MODELS[args.model]["revision"]})
    if args.model == "474m":
        task["envVars"].append(
            {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"}
        )
    task["resources"] = {
        "gpuCount": nproc_for_batch(args.global_sequences),
        "sharedMemory": "10 GiB",
    }
    task["context"] = {"priority": args.priority, "minRuntime": "0s", "autoResume": False}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec.pop("description", None)
    return spec, outer_name, endpoint_metadata


def write_registry(
    args: argparse.Namespace,
    experiment: str,
    selected: dict[str, Any],
    targets: tuple[Fraction, ...],
    endpoints: dict[str, dict[str, Any]],
) -> None:
    path: Path = MODELS[args.model]["registry"]
    report = json.loads(path.read_text())
    batch = args.global_sequences
    report["batchSweeps"].append(
        {
            "batchSequences": batch,
            "globalBatchTokens": batch * SEQUENCE_LENGTH,
            "contextLength": SEQUENCE_LENGTH,
            "lr": selected["lr"],
            "wd": selected["wd"],
            "warmupSteps": warmup_steps(batch),
            "rankMicrobatchSequences": RANK_MICROBATCH_SEQUENCES,
            "gradientAccumulation": batch // (nproc_for_batch(batch) * RANK_MICROBATCH_SEQUENCES),
            "gpuCount": nproc_for_batch(batch),
            "status": "pending",
            "activeEpoch": float(targets[-1]),
            "chainThrough": float(targets[-1]),
            "fractionalTargets": [float(target) for target in targets],
            "fractionalEndpoints": endpoints,
            "retainedPreDecaySteps": [
                endpoints[target_key(target)]["stableStep"] for target in targets
            ],
            "search": SEARCH,
            "beaker": experiment,
            "revision": MODELS[args.model]["revision"],
            "selectedEpochOne": selected,
            "results": {},
            "reason": (
                "User-authorized selected-E1-coordinate fractional chain; each stage "
                "resumes the exact preceding pre-decay checkpoint."
            ),
        }
    )
    report["updated"] = "2026-08-09"
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    args = parse_args()
    report = read_report(args.model)
    targets = targets_for_batch(args.global_sequences)
    selected = selected_epoch_one(report, args.global_sequences)
    reject_report_duplicate(report, args.global_sequences, selected["lr"], selected["wd"], targets)
    base_spec = get_base_spec(MODELS[args.model]["base"])
    initial_load_path = (
        f"{selected['output']}/step0" if args.global_sequences == 32 else None
    )
    spec, name, endpoints = build_spec(
        base_spec,
        args,
        selected["lr"],
        selected["wd"],
        targets,
        initial_load_path=initial_load_path,
    )
    audit_beaker_tuple(
        args.workspace,
        run_name_prefix(args, selected["lr"], selected["wd"], targets),
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
            name,
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
            raise RuntimeError("submission succeeded but no Beaker experiment ID was found")
        write_registry(args, ids[0], selected, targets, endpoints)


if __name__ == "__main__":
    main()
