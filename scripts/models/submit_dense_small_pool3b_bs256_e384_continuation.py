#!/usr/bin/env python3
"""Guardedly submit the integrated 153M BS256 E256-E384 continuation."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_dense_small_pool3b_bs256_e384_continuation as runner
import submit_dense_small_pool3b_checkpoint_evaluator as evaluator_submitter

WORKSPACE = "ai2/flex2"
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
MANIFEST = runner.DEFAULT_MANIFEST
RUNNER = "scripts/models/run_dense_small_pool3b_bs256_e384_continuation.py"


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(arguments, input=input_text, text=True, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"command failed ({' '.join(arguments)}): {detail}")
    return result.stdout


def validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("--revision must be a full 40-character commit hash")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "origin/sewonm/icsl"],
        check=True,
    )


def set_env(task: dict[str, Any], name: str, value: str) -> None:
    for variable in task.get("envVars", []):
        if variable.get("name") == name:
            variable["value"] = value
            return
    task.setdefault("envVars", []).append({"name": name, "value": value})


def clean_task(spec: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("trusted base experiment must contain exactly one task")
    task = spec["tasks"][0]
    if "/weka/oe-training-default" not in {
        dataset.get("mountPath") for dataset in task.get("datasets", [])
    }:
        raise RuntimeError("trusted base experiment is missing the Weka mount")
    blocked = {
        "GANTRY_USE_TORCHRUN",
        "GANTRY_RDZV_ID",
        "GANTRY_RDZV_PORT",
        "NUM_NODES",
        "PYTORCH_CUDA_ALLOC_CONF",
    }
    task["envVars"] = [
        variable
        for variable in task.get("envVars", [])
        if variable.get("name") not in blocked
        and not (
            str(variable.get("name", "")).startswith("BEAKER_")
            and variable.get("name") != "BEAKER_TOKEN"
        )
    ]
    set_env(task, "GIT_REF", revision)
    task["name"] = "main"
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {
        "priority": priority,
        "autoResume": True,
    }
    task["hostNetworking"] = False
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    return task


def producer_record(report: dict[str, Any]) -> dict[str, Any]:
    matches = [
        record for record in report.get("producers", []) if record.get("id") == runner.EXPECTED_ID
    ]
    if len(matches) != 1:
        raise RuntimeError(f"report must contain exactly one {runner.EXPECTED_ID}")
    return matches[0]


def beaker_state(experiment: str) -> str:
    payload = json.loads(
        command(["beaker", "experiment", "inspect", experiment, "--format", "json"])
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"expected exactly one experiment for {experiment}")
    statuses = [job.get("status") or {} for job in payload[0].get("jobs") or []]
    terminal = {"exited", "finalized", "canceled", "cancelled"}
    if any(not terminal.intersection(status) for status in statuses):
        return "active"
    if any(status.get("exitCode") == 0 for status in statuses):
        return "complete"
    return "failed"


def ensure_producer_ready(report: dict[str, Any], item: dict[str, Any]) -> None:
    record = producer_record(report)
    if Path(str(record.get("output"))) != runner.EXPECTED_OUTPUT:
        raise RuntimeError("registered producer output does not match selected recovery output")
    if runner.SOURCE_EPOCH not in {
        int(epoch) for epoch in record.get("resolvedCheckpointEpochs", [])
    }:
        raise RuntimeError("exact E256 pre-decay checkpoint is not resolved")
    current_experiment = str(record.get("experiment"))
    if beaker_state(current_experiment) == "active":
        raise RuntimeError("current producer is active; refusing a second writer")
    if Path(str(item["sourceCheckpoint"])) != Path(str(record["output"])) / "step659179":
        raise RuntimeError("continuation source is not the registered exact E256 checkpoint")


def existing_named_experiment(name: str) -> str | None:
    payload = json.loads(
        command(
            [
                "beaker",
                "workspace",
                "experiments",
                WORKSPACE,
                "--text",
                name,
                "--format",
                "json",
            ]
        )
    )
    values = payload if isinstance(payload, list) else payload.get("experiments", [])
    matches = [value for value in values if value.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError(f"multiple experiments use guarded name {name}")
    return str(matches[0]["id"]) if matches else None


def producer_name() -> str:
    return f"{runner.EXPECTED_ID}-integrated-e256-e384-v3"


def evaluator_name(epoch: int) -> str:
    return f"{runner.EXPECTED_ID}-post-e{epoch}-v1"


def producer_spec(item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    spec = copy.deepcopy(
        json.loads(
            command(
                [
                    "beaker",
                    "experiment",
                    "spec",
                    str(item["baseExperiment"]),
                    "--format",
                    "json",
                ]
            )
        )
    )
    task = clean_task(spec, revision, priority)
    task["arguments"] = [
        "python",
        RUNNER,
        "--manifest",
        str(MANIFEST),
    ]
    spec["description"] = (
        "153M DCLM-3B BS256 LR2e-3 WD0.1 constant-LR continuation from exact "
        "selected-recovery pre-decay E256 step659179; checkpoint every 4 epochs, "
        "immediately run isolated 10% WSD decay plus heldout evaluation at E320 "
        "before any E320-E384 training, stop on strict non-improvement, and treat "
        "E384 as the hard ceiling; delete recovery-only checkpoints at the terminal "
        "gate; 8 GPUs, rank microbatch 16, "
        "gradient accumulation 2, auto-resume, eight retries; minRuntime omitted; "
        f"write only to {item['output']}."
    )
    return spec


def evaluator_spec(
    item: dict[str, Any], epoch: int, revision: str, priority: str
) -> dict[str, Any]:
    if epoch not in runner.EVALUATION_EPOCHS:
        raise ValueError(f"E{epoch} is not an authorized continuation evaluator")
    spec = copy.deepcopy(
        json.loads(
            command(
                [
                    "beaker",
                    "experiment",
                    "spec",
                    str(item["baseExperiment"]),
                    "--format",
                    "json",
                ]
            )
        )
    )
    task = clean_task(spec, revision, priority)
    task["arguments"] = [
        "python",
        RUNNER,
        "--manifest",
        str(MANIFEST),
        "--evaluate-epoch",
        str(epoch),
    ]
    spec["description"] = (
        f"153M DCLM-3B BS256 LR2e-3 WD0.1 independent E{epoch} uncapped 10% "
        "WSD decay and heldout evaluation from the exact selected-recovery "
        "pre-decay checkpoint; minRuntime omitted."
    )
    return spec


def create(name: str, spec: dict[str, Any]) -> str:
    existing = existing_named_experiment(name)
    if existing:
        return existing
    output = command(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        input_text=json.dumps(spec),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError(f"submission returned no experiment ID for {name}")
    return identifiers[0]


def register_producer(report: dict[str, Any], experiment: str, revision: str) -> None:
    record = producer_record(report)
    previous_experiment = str(record["experiment"])
    history = record.setdefault("experimentHistory", [])
    if not any(entry.get("experiment") == previous_experiment for entry in history):
        history.append(
            {
                "experiment": previous_experiment,
                "revision": record.get("revision"),
                "status": "canceled_after_next_checkpoint_for_integrated_workflow",
                "maxValidatedEpoch": max(record.get("resolvedCheckpointEpochs") or [256]),
                "output": record.get("output"),
                "recoveryAttempt": record.get("recoveryAttempt"),
                "stoppedAt": datetime.now(tz=UTC).isoformat(),
            }
        )
    record.update(
        {
            "experiment": experiment,
            "revision": revision,
            "policy": runner.POLICY,
            "status": "submitted",
            "beakerStatus": "submitted",
            "currentEpoch": runner.EVALUATION_EPOCHS[0],
            "currentPhase": "post_pending",
            "targetEpochs": [32, 64, 96, 128, 160, 192, 224, 256, *runner.CHECKPOINT_EPOCHS],
            "continuationSourceEpoch": 256,
            "continuationTargetEpoch": 384,
            "continuationCheckpointEpochs": list(runner.CHECKPOINT_EPOCHS),
            "evaluationEpochs": [320, 384],
            "checkpointIntervalEpochs": runner.CHECKPOINT_INTERVAL_EPOCHS,
            "checkpointCleanupKeepEpochs": [runner.SOURCE_EPOCH, *runner.EVALUATION_EPOCHS],
            "role": "integrated_checkpoint_producer_and_evaluator",
            "evaluationEnabled": True,
            "decayEnabled": True,
            "postBranchesIsolatedFromConstantFrontier": True,
            "standaloneEvaluatorSubmissionsAuthorized": False,
            "futureEvaluatorSubmissionsAuthorized": False,
            "submittedAt": datetime.now(tz=UTC).isoformat(),
        }
    )
    for key in ("job", "jobs", "wandbHealth", "needsAttention", "minRuntime"):
        record.pop(key, None)


def register_evaluator(item: dict[str, Any], epoch: int, experiment: str, revision: str) -> None:
    evaluator_submitter.register(
        item,
        epoch,
        experiment,
        revision,
        str(runner.EXPECTED_OUTPUT),
        MANIFEST,
    )


def write_report(report: dict[str, Any]) -> None:
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_CHECKPOINT_PRODUCER_GRID=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("producer", "evaluator"), required=True)
    parser.add_argument("--epoch", type=int, choices=runner.EVALUATION_EPOCHS)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-spec", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    if args.mode == "producer" and args.epoch is not None:
        raise SystemExit("--epoch is only valid for evaluator mode")
    if args.mode == "evaluator" and args.epoch is None:
        raise SystemExit("evaluator mode requires --epoch")
    if args.print_spec == args.submit_if_ready:
        raise SystemExit("select exactly one of --print-spec or --submit-if-ready")

    validate_revision(args.revision)
    config, item = runner.load(MANIFEST)
    report = json.loads(REPORT.read_text())
    if args.mode == "producer":
        ensure_producer_ready(report, item)
        spec = producer_spec(item, args.revision, args.priority)
        name = producer_name()
    else:
        record = producer_record(report)
        if record.get("standaloneEvaluatorSubmissionsAuthorized") is False:
            raise RuntimeError("standalone evaluator submissions are disabled for integrated workflow")
        if int(args.epoch) not in {
            int(value) for value in record.get("resolvedCheckpointEpochs", [])
        }:
            raise RuntimeError(f"E{args.epoch} pre-decay checkpoint is not resolved")
        spec = evaluator_spec(item, int(args.epoch), args.revision, args.priority)
        name = evaluator_name(int(args.epoch))
    if args.print_spec:
        print(json.dumps(spec, indent=2))
        return

    experiment = create(name, spec)
    if args.mode == "producer":
        register_producer(report, experiment, args.revision)
        write_report(report)
    else:
        register_evaluator(item, int(args.epoch), experiment, args.revision)
    print(f"{name}: {experiment}")


if __name__ == "__main__":
    main()
