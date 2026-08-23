#!/usr/bin/env python3
"""Submit and register a user-requested small-model POST finalizer."""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_small_dense_dr_wt_embedwd_chain as adaptive
import submit_small_dense_dr_wt_embedwd_chain as submit

POLICY = "locked_wd_requested_postdecay_finalizer_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(submit.MODELS), required=True)
    parser.add_argument("--global-sequences", type=int, choices=submit.BATCHES, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--resume-experiment", required=True)
    parser.add_argument("--epochs", default="")
    parser.add_argument("--stop-on-saturation", action="store_true")
    parser.add_argument("--preserve-existing-selection", action="store_true")
    parser.add_argument("--workspace", default=submit.WORKSPACE)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--name")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--register", action="store_true")
    return parser.parse_args()


def validate_revision(revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise SystemExit("--revision must be a full 40-character commit hash")
    submit.run(
        ["git", "merge-base", "--is-ancestor", revision, "origin/sewonm/icsl"]
    )


def requested_epochs(value: str) -> list[int]:
    if not value.strip():
        return []
    epochs = [int(item) for item in value.split(",")]
    if epochs != sorted(set(epochs)):
        raise ValueError("--epochs must be unique and strictly increasing")
    return epochs


def registered_record(args: argparse.Namespace) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    path = submit.report_path(args.model)
    report = json.loads(path.read_text())
    identifier = f"dense-{args.model}-bs{args.global_sequences}-dr-wt-embwd-adaptive"
    matches = [
        record
        for record in report.get("adaptiveDrWtEmbedWdChains", [])
        if record.get("id") == identifier
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one registered record for {identifier}")
    record = matches[0]
    if str(record.get("experiment")) != args.resume_experiment:
        raise RuntimeError(
            f"registered experiment {record.get('experiment')} does not match "
            f"{args.resume_experiment}"
        )
    if not record.get("lockedWd"):
        raise RuntimeError("requested POST finalization requires an already locked WD")
    return report, path, record


def manifest_path(args: argparse.Namespace) -> Path:
    path = submit.manifest_path(args.model, args.global_sequences, locked=True)
    if not path.is_file():
        raise FileNotFoundError(f"locked-WD manifest is missing: {path}")
    manifest = json.loads(path.read_text())
    if str(manifest.get("lockedWd")) == "None":
        raise RuntimeError("locked-WD manifest has no locked WD")
    return path


def build_spec(args: argparse.Namespace, manifest: Path) -> dict[str, Any]:
    base = submit.base_spec(submit.MODELS[args.model]["baseExperiment"])
    submit.audit_base_spec(args.model, base)
    spec = copy.deepcopy(base)
    task = spec["tasks"][0]
    arguments = [
        "python",
        "scripts/models/run_small_dense_requested_postdecay_finalizer.py",
        "--manifest",
        str(manifest),
        "--epochs",
        args.epochs,
    ]
    if args.stop_on_saturation:
        arguments.append("--stop-on-saturation")
    if args.preserve_existing_selection:
        arguments.append("--preserve-existing-selection")
    task["arguments"] = arguments
    task["envVars"] = [
        variable
        for variable in task.get("envVars", [])
        if variable.get("name") not in {"GANTRY_USE_TORCHRUN", "PYTORCH_CUDA_ALLOC_CONF"}
    ]
    submit.set_revision(task, args.revision)
    if args.model == "474m":
        task["envVars"].append(
            {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"}
        )
    task["resources"] = {
        "gpuCount": submit.nproc_for(args.global_sequences),
        "sharedMemory": "10 GiB",
    }
    task["context"] = {"priority": args.priority, "minRuntime": "0s", "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"Dense {args.model.upper()} BS{args.global_sequences} locked-WD requested POST "
        f"finalizer. It trains no new constant-LR frontier and evaluates only E{args.epochs}."
    )
    return spec


def register(
    args: argparse.Namespace,
    report: dict[str, Any],
    path: Path,
    record: dict[str, Any],
    experiment: str,
    epochs: list[int],
) -> None:
    previous_selection = record.get("postDecaySelection")
    record.setdefault("attemptHistory", []).append(
        {
            "beaker": args.resume_experiment,
            "status": "stopped-at-user-requested-frontier",
            "activePhase": record.get("activePhase"),
            "activeEpoch": record.get("activeEpoch"),
            "wandbHealth": record.get("wandbHealth"),
            "replacedBy": experiment,
        }
    )
    record.update(
        {
            "policy": POLICY,
            "status": "submitted",
            "beakerStatus": "submitted",
            "experiment": experiment,
            "beaker": experiment,
            "revision": args.revision,
            "recoveryOf": args.resume_experiment,
            "comparisonPolicy": "post_decay_only",
            "preDecayEvaluation": False,
            "wdTuningStopped": True,
            "activePhase": "requested_post_decay_finalization",
            "activeEpoch": epochs[0] if epochs else None,
            "activeWds": [str(record["lockedWd"])] if epochs else [],
            "requestedPostDecayEpochs": epochs,
            "stopOnPostDecaySaturation": bool(args.stop_on_saturation),
            "preserveExistingSelection": bool(args.preserve_existing_selection),
            "reason": (
                "Training stopped at the user-requested frontier. This replacement trains "
                "no new constant-LR checkpoint and evaluates only the explicitly requested "
                "post-decay endpoints."
            ),
        }
    )
    transition = dict(record.get("requestedPostDecayFinalization", {}))
    transition.update(
        {
            "status": "finalizer-submitted",
            "replacementExperiment": experiment,
            "requestedPostDecayEpochs": epochs,
        }
    )
    record["requestedPostDecayFinalization"] = transition
    if previous_selection is not None:
        record["priorPostDecaySelection"] = previous_selection
    if not args.preserve_existing_selection:
        record.pop("postDecaySelection", None)
        record.pop("selectedPostDecayEpoch", None)
        record.pop("selectedPostDecayValidationExact", None)
        record.pop("selectedCheckpoint", None)
        record.pop("saturatedEpoch", None)
        record.pop("stopReason", None)
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    args = parse_args()
    validate_revision(args.revision)
    epochs = requested_epochs(args.epochs)
    report, report_file, record = registered_record(args)
    manifest = manifest_path(args)
    manifest_value = json.loads(manifest.read_text())
    if str(manifest_value.get("lockedWd")) != str(record.get("lockedWd")):
        raise RuntimeError("manifest locked WD does not match the registered chain")
    for epoch in epochs:
        adaptive.targets_through(manifest_value, epoch)
    spec = build_spec(args, manifest)
    name = args.name or (
        f"dense-{args.model}-bs{args.global_sequences}-requested-post-finalizer-v1"
    )
    submit.audit_name(args.workspace, name)
    if args.print_only:
        print(json.dumps(spec, indent=2))
        return
    if str(record.get("status", "")).lower() not in {"complete", "stopped_by_user"}:
        submit.run(["beaker", "experiment", "stop", args.resume_experiment])
    output = submit.run(
        [
            "beaker",
            "experiment",
            "create",
            "-",
            "--name",
            name,
            "--workspace",
            args.workspace,
        ],
        input_text=json.dumps(spec),
    )
    print(output, end="")
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if args.register:
        if not identifiers:
            raise RuntimeError("submission succeeded without a parsed experiment ID")
        register(args, report, report_file, record, identifiers[0], epochs)


if __name__ == "__main__":
    main()
