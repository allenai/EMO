#!/usr/bin/env python3
"""Guardedly submit the integrated 1B Pool-3B BS128 continuation."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_1b_pool3b_bs128_e32_continuation as runner

WORKSPACE = "ai2/flex2"
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
MANIFEST = runner.DEFAULT_MANIFEST
RUNNER = "scripts/models/run_dense_1b_pool3b_bs128_e32_continuation.py"
EVALUATOR_ID = "dense-1b-dclm3b-bs128-lr1e-3-wd0.3-post-e8-e16"


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
    task["context"] = {"priority": priority, "autoResume": True}
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


def evaluator_record(report: dict[str, Any]) -> dict[str, Any]:
    matches = [
        record for record in report.get("evaluators", []) if record.get("id") == EVALUATOR_ID
    ]
    if len(matches) != 1:
        raise RuntimeError(f"report must contain exactly one {EVALUATOR_ID}")
    return matches[0]


def inspect(experiment: str) -> dict[str, Any]:
    payload = json.loads(
        command(["beaker", "experiment", "inspect", experiment, "--format", "json"])
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"expected exactly one experiment for {experiment}")
    return payload[0]


def beaker_state(payload: dict[str, Any]) -> str:
    jobs = payload.get("jobs") or []
    if not jobs:
        return "submitted"
    terminal = {"exited", "finalized", "canceled", "cancelled"}
    statuses = [job.get("status") or {} for job in jobs]
    if any(not terminal.intersection(status) for status in statuses):
        return "active"
    if any(status.get("exitCode") == 0 for status in statuses):
        return "complete"
    return "failed"


def completion_marker(payload: dict[str, Any], target_epoch: int) -> bool:
    marker = f"DENSE_CHECKPOINT_PRODUCER_COMPLETE id={runner.EXPECTED_ID} epoch={target_epoch}"
    if marker in str(payload.get("description") or ""):
        return True
    jobs = [job for job in payload.get("jobs") or [] if job.get("id")]
    if not jobs:
        return False
    logs = command(["beaker", "job", "logs", str(jobs[-1]["id"]), "--since", "24h"])
    return marker in logs


def result_value(report: dict[str, Any], epoch: int) -> Decimal:
    result = (evaluator_record(report).get("postDecayResults") or {}).get(str(epoch))
    if (
        not isinstance(result, dict)
        or result.get("status") != "complete"
        or result.get("comparisonGroup") != "post_decay"
        or int(result.get("epoch", -1)) != epoch
        or Path(str(result.get("sourcePreDecayCheckpoint")))
        != runner.EXPECTED_OUTPUT / f"step{runner.checkpoint_step(epoch)}"
    ):
        raise RuntimeError(f"missing healthy matched POST E{epoch} result")
    return Decimal(str(result["validationExact"]))


def producer_gate(report: dict[str, Any], config: dict[str, Any], target_epoch: int) -> None:
    record = producer_record(report)
    if Path(str(record.get("output"))) != runner.EXPECTED_OUTPUT:
        raise RuntimeError("registered producer output does not match the exact BS128 output")
    if runner.SOURCE_EPOCH not in {
        int(epoch) for epoch in record.get("resolvedCheckpointEpochs", [])
    }:
        raise RuntimeError("exact E32 pre-decay checkpoint is not resolved")
    current = inspect(str(record["experiment"]))
    if beaker_state(current) == "active":
        raise RuntimeError("current producer is active; refusing a second writer")
    baseline = Decimal(str(config["baselinePostDecay"]["validationExact"]))
    if target_epoch >= 64 and result_value(report, 48) > baseline:
        raise RuntimeError("E48 is worse than E32; E64 is not authorized")
    if target_epoch == 80 and result_value(report, 64) >= result_value(report, 48):
        raise RuntimeError("E64 is saturated relative to E48; E80 is not authorized")


def evaluator_gate(report: dict[str, Any], epoch: int) -> dict[str, Any]:
    record = producer_record(report)
    if int(record.get("continuationTargetEpoch", -1)) != epoch:
        raise RuntimeError(f"current producer stage does not target E{epoch}")
    payload = inspect(str(record["experiment"]))
    if beaker_state(payload) != "complete" or not completion_marker(payload, epoch):
        raise RuntimeError(f"E{epoch} producer stage is not marker-complete")
    return record


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


def experiment_name(mode: str, epoch: int | None) -> str:
    if mode == "integrated":
        return f"{runner.EXPECTED_ID}-integrated-e32-e80-v2"
    assert epoch is not None
    phase = "continuation" if mode == "producer" else "post"
    return f"{runner.EXPECTED_ID}-{phase}-e{epoch}-dense-v1"


def job_spec(
    item: dict[str, Any], mode: str, epoch: int | None, revision: str, priority: str
) -> dict[str, Any]:
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
        "--mode",
        mode,
    ]
    if epoch is not None:
        task["arguments"].extend(["--epoch", str(epoch)])
    if mode == "integrated":
        detail = (
            "integrated constant-LR continuation with immediate isolated 10% WSD "
            "decay and heldout evaluation at E48/E64/E80; each POST gate runs "
            "before any following training stage"
        )
    elif mode == "producer":
        detail = f"constant-LR continuation to E{epoch}, checkpointing every epoch"
    else:
        detail = f"uncapped 10% WSD decay plus heldout evaluation from exact PD E{epoch}"
    spec["description"] = (
        f"Dense-1B DCLM-3B BS128 LR1e-3 WD0.3 {detail}; 8 GPUs, rank "
        "microbatch 8, gradient accumulation 2, auto-resume, eight retries; "
        "minRuntime omitted; E80 hard ceiling."
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


def register_producer(
    report: dict[str, Any], experiment: str, revision: str, target_epoch: int
) -> None:
    record = producer_record(report)
    previous_experiment = str(record["experiment"])
    history = record.setdefault("experimentHistory", [])
    if not any(entry.get("experiment") == previous_experiment for entry in history):
        history.append(
            {
                "experiment": previous_experiment,
                "revision": record.get("revision"),
                "status": "canceled_after_next_checkpoint_for_integrated_workflow",
                "maxValidatedEpoch": max(record.get("resolvedCheckpointEpochs") or [32]),
                "output": record.get("output"),
                "stoppedAt": datetime.now(tz=UTC).isoformat(),
            }
        )
    resolved = sorted(
        {
            int(epoch)
            for epoch in record.get("resolvedCheckpointEpochs", [])
            if int(epoch) <= target_epoch
        }
    )
    record.update(
        {
            "policy": runner.POLICY,
            "experiment": experiment,
            "revision": revision,
            "status": "submitted",
            "beakerStatus": "submitted",
            "currentEpoch": max(resolved) + 1,
            "currentPhase": "repacked_shuffled_pool3b_constant_lr",
            "resolvedCheckpointEpochs": resolved,
            "targetEpochs": sorted({*resolved, *range(33, target_epoch + 1)}),
            "evaluationEpochs": list(runner.EVALUATION_EPOCHS),
            "checkpointIntervalEpochs": runner.CHECKPOINT_INTERVAL_EPOCHS,
            "checkpointCleanupKeepEpochs": [32, 48, 64, 80],
            "continuationSourceEpoch": runner.SOURCE_EPOCH,
            "continuationSourceCheckpoint": str(
                runner.EXPECTED_OUTPUT / f"step{runner.checkpoint_step(runner.SOURCE_EPOCH)}"
            ),
            "continuationTargetEpoch": target_epoch,
            "hardTerminalEpoch": runner.HARD_TERMINAL_EPOCH,
            "role": "integrated_checkpoint_producer_and_evaluator",
            "evaluationEnabled": True,
            "decayEnabled": True,
            "postBranchesIsolatedFromConstantFrontier": True,
            "standaloneEvaluatorSubmissionsAuthorized": False,
            "futureEvaluatorSubmissionsAuthorized": False,
            "stopAuthorized": False,
            "submittedAt": datetime.now(tz=UTC).isoformat(),
        }
    )
    for key in (
        "job",
        "jobs",
        "wandbHealth",
        "needsAttention",
        "minRuntime",
        "stopAfterEpoch",
        "stopAfterCheckpointStep",
        "stopDecision",
        "stopReason",
    ):
        record.pop(key, None)


def register_evaluator(report: dict[str, Any], experiment: str, revision: str, epoch: int) -> None:
    record = producer_record(report)
    record["resolvedCheckpointEpochs"] = sorted(
        {
            *[int(value) for value in record.get("resolvedCheckpointEpochs", [])],
            *range(33, epoch + 1),
        }
    )
    record["status"] = "pd_retained"
    record["beakerStatus"] = "complete"
    record["currentEpoch"] = epoch
    record["currentPhase"] = "post"
    evaluation = evaluator_record(report)
    additional = evaluation.setdefault("additionalExperiments", [])
    matches = [value for value in additional if int(value.get("epoch", -1)) == epoch]
    if matches and matches[0].get("experiment") != experiment:
        raise RuntimeError(f"a different E{epoch} evaluator is already registered")
    if not matches:
        additional.append(
            {
                "epoch": epoch,
                "experiment": experiment,
                "revision": revision,
                "status": "submitted",
            }
        )
    evaluation["status"] = "running"


def write_report(report: dict[str, Any]) -> None:
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_CHECKPOINT_PRODUCER_GRID=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("integrated", "producer", "evaluator"), required=True)
    parser.add_argument("--epoch", type=int, choices=runner.EVALUATION_EPOCHS)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-spec", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    if args.print_spec == args.submit_if_ready:
        raise SystemExit("select exactly one of --print-spec or --submit-if-ready")
    validate_revision(args.revision)
    config, item = runner.load(MANIFEST)
    report = json.loads(REPORT.read_text())
    if args.mode == "integrated" and args.epoch is not None:
        raise SystemExit("integrated mode does not accept --epoch")
    if args.mode != "integrated" and args.epoch is None:
        raise SystemExit("producer and evaluator modes require --epoch")
    epoch = int(args.epoch) if args.epoch is not None else None
    if args.mode == "integrated":
        producer_gate(report, config, 48)
    elif args.mode == "producer":
        assert epoch is not None
        producer_gate(report, config, epoch)
    else:
        raise RuntimeError(
            "standalone evaluator submissions are disabled for the integrated workflow"
        )
    spec = job_spec(item, args.mode, epoch, args.revision, args.priority)
    if args.print_spec:
        print(json.dumps(spec, indent=2))
        return
    name = experiment_name(args.mode, epoch)
    experiment = create(name, spec)
    if args.mode in {"integrated", "producer"}:
        register_producer(
            report,
            experiment,
            args.revision,
            runner.HARD_TERMINAL_EPOCH if args.mode == "integrated" else int(epoch),
        )
    else:
        register_evaluator(report, experiment, args.revision, epoch)
    write_report(report)
    print(f"{name}: {experiment}")


if __name__ == "__main__":
    main()
