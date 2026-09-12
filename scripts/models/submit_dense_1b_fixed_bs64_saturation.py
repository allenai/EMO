#!/usr/bin/env python3
"""Submit the reviewer-requested Dense-1B / DCLM-1B / BS64 fixed-order chain.

The first stage trains from scratch through the E4 WSD endpoint while retaining
the E1, E2, and E4 pre-decay checkpoints; only E4 is evaluated. Later stages
evaluate E8, E12, ... and stop at the first held-out CE non-improvement.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import submit_dense_1b_weight_tying_sequential as sequential
import submit_dense_step1_data_loader_coordinate as endpoint

REPORT_PATH = Path("reports/0802/data/wsd_data_loader_1b.json")
REPORT_JS_PATH = REPORT_PATH.with_suffix(".js")
MANIFEST_PATH = Path("scripts/models/manifests/dense-1b-fixed-bs64-lr1e-3-wd0.3.json")
SOURCE_EXPERIMENT = "01KZDCD9AYJ0REGZTE2YG18NHF"
OUTPUT = (
    "/weka/oe-training-default/sewonm/icsl/models/"
    "dense_1b_dclm1b_fixed_reviewer_v2/bs64_lr1e-3_wd0.3"
)
TARGETS = tuple(range(4, 65, 4))
RUN_ID = "fixed64-lr1e-3-wd0.3"
SEQUENTIAL_ID = "fixedseq-bs64-lr1e-3-wd0.3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", default="dense-1b-fixed-bs64-lr1e3-wd03-saturation")
    parser.add_argument("--workspace", default=endpoint.FLEX2_WORKSPACE)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.workspace != endpoint.FLEX2_WORKSPACE:
        parser.error(f"this chain submits only to {endpoint.FLEX2_WORKSPACE}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.name):
        parser.error("--name must be a lowercase Beaker name component")
    return args


def stage_args(epoch: int, source: str, revision: str, name: str, priority: str) -> Any:
    return SimpleNamespace(
        method="fixed_order",
        global_sequences=64,
        target_epoch=epoch,
        learning_rate="1e-3",
        weight_decay="0.3",
        source_experiment=SOURCE_EXPERIMENT,
        source_checkpoint=source,
        revision=revision,
        name=name,
        suffix="fixed-bs64-reviewer-v1",
        weight_tying=False,
        decay_embeddings=False,
        mlp_weight_decay=None,
        mlp_weight_decay_scope="all",
        workspace=endpoint.FLEX2_WORKSPACE,
        priority=priority,
        register=False,
        print_only=False,
        allow_pending_flex2_jobs=True,
    )


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old!r}")
    return text.replace(old, new, 1)


def replace_twice(text: str, old: str, new: str) -> str:
    if text.count(old) != 2:
        raise RuntimeError(f"expected exactly two occurrences of {old!r}")
    return text.replace(old, new)


def build_submission(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = endpoint.beaker_spec(SOURCE_EXPERIMENT)
    audit_args = stage_args(1, "fresh", args.revision, args.name, args.priority)
    script, base_arguments = endpoint.audit_source_spec(base, audit_args)
    base_arguments = endpoint.upsert(
        base_arguments, "--model.tie_embeddings=", "--model.tie_embeddings=false"
    )

    old_root = endpoint.MODEL_ROOT
    endpoint.MODEL_ROOT = OUTPUT.rsplit("/", 1)[0]
    stages: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    previous_epoch: int | None = None
    try:
        for epoch in TARGETS:
            source = "fresh" if previous_epoch is None else (
                f"{OUTPUT}/step{endpoint.stable_step(previous_epoch, 64)}"
            )
            # The generic endpoint builder intentionally treats E1 as the common
            # from-scratch case. Build that clean source-free command, then extend
            # the same WSD schedule to E4 and retain E1/E2/E4 along the way.
            build_epoch = 1 if previous_epoch is None else epoch
            current = stage_args(
                build_epoch, source, args.revision, args.name, args.priority
            )
            spec, generated_output = endpoint.build_spec(
                copy.deepcopy(base), current, script, base_arguments
            )
            if generated_output != OUTPUT:
                raise RuntimeError(f"unexpected canonical output {generated_output}")
            shell = spec["tasks"][0]["arguments"][2]
            if previous_epoch is None:
                shell = replace_twice(
                    shell,
                    shlex.quote(
                        "--trainer.max_duration={value: 1000000000, unit: tokens}"
                    ),
                    shlex.quote(
                        "--trainer.max_duration={value: 4000000000, unit: tokens}"
                    ),
                )
                shell = replace_twice(
                    shell,
                    shlex.quote(
                        "--trainer.callbacks.checkpointer.fixed_steps="
                        f"[{endpoint.stable_step(1, 64)}]"
                    ),
                    shlex.quote(
                        "--trainer.callbacks.checkpointer.fixed_steps="
                        f"[{endpoint.stable_step(1, 64)},"
                        f"{endpoint.stable_step(2, 64)},"
                        f"{endpoint.stable_step(4, 64)}]"
                    ),
                )
                shell = replace_twice(
                    shell,
                    "--model.tie_embeddings=false",
                    "--model.tie_embeddings=false --fixed-data-order",
                )
                shell = shell.replace("_fixed_e1_", "_fixed_e4_")
                shell = shell.replace(",e1,", ",e4,")
            shell, log_path = sequential.with_stage_log_capture(shell, OUTPUT, epoch)
            stage_shell = "\n".join(
                [
                    "set -euo pipefail",
                    f'echo "FIXED_BS64_STAGE_START epoch={epoch}"',
                    shell.removeprefix("set -euo pipefail\n"),
                    f'echo "FIXED_BS64_STAGE_COMPLETE epoch={epoch}"',
                    *sequential.saturation_gate(OUTPUT, epoch, previous_epoch, log_path),
                ]
            )
            stages.append(
                {
                    "epoch": epoch,
                    "shell": stage_shell,
                    "stopDecision": f"{OUTPUT}/.embwd_e{epoch}.decision",
                }
            )
            records.append(
                {
                    "epoch": epoch,
                    "status": "planned",
                    "sourceCheckpoint": source,
                    "preDecayCheckpoint": f"{OUTPUT}/step{endpoint.stable_step(epoch, 64)}",
                    "endpointCheckpoint": f"{OUTPUT}/step{endpoint.total_step(epoch, 64)}",
                }
            )
            previous_epoch = epoch
    finally:
        endpoint.MODEL_ROOT = old_root

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({"stages": stages}, indent=2) + "\n")
    submission = copy.deepcopy(spec)
    task = submission["tasks"][0]
    task["arguments"] = [
        "python",
        "scripts/models/run_dense_weight_tying_sequence.py",
        "--manifest",
        str(MANIFEST_PATH),
    ]
    for env in task.get("envVars", []):
        if env.get("name") == "GIT_REF":
            env["value"] = args.revision
            break
    else:
        raise RuntimeError("source task has no GIT_REF")
    task["replicas"] = 1
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": "8h", "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    task.pop("synchronizedStartTimeout", None)
    submission["retry"] = {"allowedTaskRetries": 8}
    submission["description"] = (
        "Dense 1B, repeated DCLM-1B, global BS64, LR1e-3, WD0.3, fixed order, "
        "untied embeddings, zero embedding WD. Train from scratch; retain E1/E2; "
        "evaluate E4/E8/E12/... and stop at first held-out non-improvement."
    )
    return submission, records


def register(experiment: str, revision: str, stages: list[dict[str, Any]]) -> None:
    report = json.loads(REPORT_PATH.read_text())
    if any(run.get("id") == RUN_ID for run in report.get("runs", [])):
        raise RuntimeError(f"duplicate registered run {RUN_ID}")
    report.setdefault("columns", []).insert(
        next(i for i, column in enumerate(report["columns"]) if column["key"] == "dr64"),
        {
            "key": "fixed64",
            "label": "BS64 · Fixed",
            "batchSequences": 64,
            "dataOrder": "fixed_order",
            "initialWd": "0.3",
            "color": "#d53e4f",
        },
    )
    report["runs"].append(
        {
            "id": RUN_ID,
            "method": "fixed64",
            "batchSequences": 64,
            "dataOrder": "fixed_order",
            "weightTying": False,
            "decayEmbeddings": False,
            "lr": "1e-3",
            "wd": "0.3",
            "status": "submitted",
            "activeEpoch": 4,
            "attemptedEpochs": list(TARGETS),
            "templateExperiment": SOURCE_EXPERIMENT,
            "sourceCheckpoint": "fresh",
            "gpuCount": 8,
            "nodeCount": 1,
            "rankMicrobatchSequences": 8,
            "gradientAccumulationSteps": 1,
            "plannedTargets": [1, 2, *TARGETS],
            "results": {},
            "retainedOnlyEpochs": [1, 2],
            "evaluatedEpochs": list(TARGETS),
            "experiment": experiment,
            "beaker": experiment,
            "revision": revision,
            "output": OUTPUT,
            "reason": (
                "Reviewer-requested fixed-order Original recipe trained from scratch. E1 and "
                "E2 are retained without evaluation; E4/E8/E12/... are WSD-decayed and "
                "evaluated until the first adjacent held-out non-improvement."
            ),
        }
    )
    report.setdefault("fixedOriginalSequentialRuns", []).append(
        {
            "id": SEQUENTIAL_ID,
            "coordinateRunId": RUN_ID,
            "status": "submitted",
            "currentEpoch": 4,
            "targets": list(TARGETS),
            "retainedOnlyEpochs": [1, 2],
            "experiment": experiment,
            "beaker": experiment,
            "revision": revision,
            "output": OUTPUT,
            "nodeCount": 1,
            "gpuCount": 8,
            "automaticTaskRetries": 8,
            "minRuntime": "8h",
            "stages": stages,
        }
    )
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS_PATH.write_text(
        "window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    args = parse_args()
    spec, stages = build_submission(args)
    if args.prepare_only:
        print(MANIFEST_PATH)
        return
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    result = subprocess.run(
        ["beaker", "experiment", "create", "-", "--name", args.name, "--workspace", args.workspace],
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
        register(ids[0], args.revision, stages)


if __name__ == "__main__":
    main()
