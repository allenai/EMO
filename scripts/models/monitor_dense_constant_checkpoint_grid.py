#!/usr/bin/env python3
"""Refresh the shared checkpoint-producer report from Beaker logs.

The two Dense-1B producers retain the original v1 policy.  The eight small-model
records are the corrected Pool-3B v2 bridge/producers.  Older canceled
small-model Pool-1B producers are intentionally absent from the report.
The eight Pool-333M jobs are integrated producer/evaluators and are maintained
in a separate top-level collection so the historical Pool-3B manifest remains
unchanged.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_STEP = re.compile(r"\[step=([0-9,]+)/([0-9,]+),epoch=")
TRAIN_LOSS = re.compile(r"\btrain/CE loss=([^\s]+)")
WANDB = re.compile(r"https://wandb\.ai/[^\s]+/runs/([a-zA-Z0-9_-]+)")
EVAL_RESULT = re.compile(
    r"DENSE1B_CHECKPOINT_EVALUATOR_RESULT id=([^ ]+) epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
EVAL_DECISION = re.compile(
    r"DENSE1B_CHECKPOINT_EVALUATOR_COMPLETE id=([^ ]+) status=([^ ]+) json=(\{.*\})$",
    re.MULTILINE,
)
INTEGRATED_PD_RETAINED = re.compile(
    r"DENSE_DCLM333M_PD_RETAINED id=([^ ]+) epoch=([0-9]+) checkpoint=([^\s]+)$",
    re.MULTILINE,
)
INTEGRATED_POST_RESULT = re.compile(
    r"DENSE_DCLM333M_POST_RESULT id=([^ ]+) epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
INTEGRATED_STAGE_EVENT = re.compile(
    r"DENSE_DCLM333M_(PD_START|PD_RETAINED|POST_START|POST_COMPLETE) "
    r"id=([^ ]+) epoch=([0-9]+)",
)


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, text=True, capture_output=True).stdout


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
    statuses = [job.get("status") or {} for job in jobs]
    terminal = {"exited", "finalized", "canceled", "cancelled"}
    if any("started" in status and not terminal.intersection(status) for status in statuses):
        return "running"
    if any(
        "scheduled" in status and "started" not in status and not terminal.intersection(status)
        for status in statuses
    ):
        return "scheduled"
    if any("finalized" in status and status.get("exitCode") == 0 for status in statuses):
        return "complete"
    if all(terminal.intersection(status) for status in statuses):
        return "failed"
    return "submitted"


def write_report(report: dict[str, Any]) -> None:
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_CHECKPOINT_PRODUCER_GRID=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def health(logs: str, state: str) -> dict[str, Any]:
    steps = [
        (int(current.replace(",", "")), int(total.replace(",", "")))
        for current, total in TRAIN_STEP.findall(logs)
    ]
    critical: list[str] = []
    for raw in TRAIN_LOSS.findall(logs):
        try:
            if not math.isfinite(float(raw)):
                critical.append("nonfinite-training-loss")
                break
        except ValueError:
            critical.append("nonfinite-training-loss")
            break
    stage_markers = (
        "DENSE_CHECKPOINT_PRODUCER_START",
        "DENSE_SMALL_POOL3B_BRIDGE_START",
        "DENSE_SMALL_POOL3B_PRODUCER_START",
        "DENSE1B_CHECKPOINT_EVALUATOR_START",
    )
    if any(marker in logs for marker in stage_markers) and "Loading checkpoint from '" not in logs:
        critical.append("missing-exact-checkpoint-load")
    wandb = WANDB.findall(logs)
    return {
        "status": "critical" if critical else "healthy" if steps else "pending",
        "checkedAt": datetime.now(tz=UTC).isoformat(),
        "beakerState": state,
        "latestStep": steps[-1][0] if steps else None,
        "totalSteps": steps[-1][1] if steps else None,
        "run": wandb[-1] if wandb else None,
        "criticalSignals": critical,
    }


def experiment_logs(experiment: str, state: str, job: str | None = None) -> str:
    if state not in {"running", "complete", "failed"}:
        return ""
    arguments = (
        ["beaker", "job", "logs", job, "--tail", "50000"]
        if job
        else ["beaker", "experiment", "logs", experiment]
    )
    try:
        return ANSI.sub("", command(arguments))
    except subprocess.CalledProcessError:
        # A stale or oversized terminal log must not block refreshes for the
        # active grid.  Previously resolved report state remains authoritative.
        return ""


def refreshed_health(record: dict[str, Any], logs: str, state: str) -> dict[str, Any]:
    if logs:
        return health(logs, state)
    existing = record.get("wandbHealth")
    if isinstance(existing, dict):
        preserved = dict(existing)
        preserved["checkedAt"] = datetime.now(tz=UTC).isoformat()
        preserved["beakerState"] = state
        return preserved
    return health(logs, state)


def refresh_producer(record: dict[str, Any]) -> str:
    experiment = record.get("experiment")
    if not experiment:
        return "planned"
    payload = inspect(str(experiment))
    state = beaker_state(payload)
    jobs = [job for job in payload.get("jobs") or [] if job.get("id")]
    if jobs:
        record["jobs"] = [job["id"] for job in jobs]
        record["job"] = jobs[-1]["id"]
    job = jobs[-1]["id"] if jobs else None
    logs = experiment_logs(str(experiment), state, job)
    resolved = {int(epoch) for epoch in record.get("resolvedCheckpointEpochs", [])}
    pool3b_v2 = record.get("policy") == "dense_small_pool3b_checkpoint_producers_v2"
    if pool3b_v2 and f"DENSE_SMALL_POOL3B_BRIDGE_COMPLETE id={record['id']}" in logs:
        resolved.add(1)
    for epoch in record["targetEpochs"]:
        pool_tokens = 3_000_000_000 if record["pool"] == "dclm3b" else 1_000_000_000
        endpoint = -(-int(epoch) * pool_tokens // (int(record["batchSequences"]) * 4096))
        step = endpoint - round(0.1 * endpoint) - 1
        if re.search(rf"(?:/|\b)step{step}(?:\b|/)", logs):
            resolved.add(int(epoch))
    completion_markers = (
        f"DENSE_CHECKPOINT_PRODUCER_COMPLETE id={record['id']}",
        f"DENSE_SMALL_POOL3B_PRODUCER_COMPLETE id={record['id']}",
    )
    authorized_stop = (
        bool(record.get("stopAuthorized"))
        and int(record.get("stopAfterEpoch", -1)) in resolved
        and state == "failed"
    )
    if any(marker in logs for marker in completion_markers):
        if pool3b_v2:
            resolved.add(1)
        resolved.update(record["targetEpochs"])
        record["status"] = "complete"
    elif authorized_stop:
        record["status"] = "stopped_at_authorized_epoch"
        record.pop("needsAttention", None)
    elif state == "failed":
        record["status"] = "failed"
        record["needsAttention"] = True
    elif state == "complete":
        record["status"] = "complete_without_marker"
        record["needsAttention"] = True
    else:
        record["status"] = state
    record["beakerStatus"] = state
    record["resolvedCheckpointEpochs"] = sorted(resolved)
    record["currentEpoch"] = (
        1
        if pool3b_v2 and 1 not in resolved
        else next((epoch for epoch in record["targetEpochs"] if epoch not in resolved), None)
    )
    if pool3b_v2:
        record["currentPhase"] = (
            "fresh_2b_bridge_to_predecay_e1"
            if 1 not in resolved
            else "repacked_shuffled_pool3b_constant_lr"
            if record["currentEpoch"] is not None
            else "complete"
        )
    record["wandbHealth"] = refreshed_health(record, logs, state)
    return record["status"]


def refresh_evaluator(record: dict[str, Any]) -> str:
    experiment = record.get("experiment")
    if not experiment:
        return "planned"
    previous_status = record.get("status")
    previous_decision = record.get("decision")
    payload = inspect(str(experiment))
    state = beaker_state(payload)
    jobs = [job for job in payload.get("jobs") or [] if job.get("id")]
    if jobs:
        record["jobs"] = [job["id"] for job in jobs]
        record["job"] = jobs[-1]["id"]
    job = jobs[-1]["id"] if jobs else None
    logs = experiment_logs(str(experiment), state, job)
    results = record.setdefault("postDecayResults", {})
    for evaluator_id, epoch, raw in EVAL_RESULT.findall(logs):
        if evaluator_id == record["producerId"]:
            results[str(int(epoch))] = json.loads(raw)
    decisions = [
        (status, json.loads(raw))
        for evaluator_id, status, raw in EVAL_DECISION.findall(logs)
        if evaluator_id == record["producerId"]
    ]
    record["resolvedPostEpochs"] = sorted(int(epoch) for epoch in results)
    additional_states: list[str] = []
    additional_decisions: list[tuple[str, dict[str, Any]]] = []
    for additional in record.get("additionalExperiments", []):
        previous_additional_status = additional.get("status")
        previous_additional_decision = additional.get("decision")
        additional_experiment = str(additional["experiment"])
        additional_payload = inspect(additional_experiment)
        additional_state = beaker_state(additional_payload)
        additional_states.append(additional_state)
        additional_jobs = [job for job in additional_payload.get("jobs") or [] if job.get("id")]
        if additional_jobs:
            additional["jobs"] = [job["id"] for job in additional_jobs]
            additional["job"] = additional_jobs[-1]["id"]
        additional_job = additional_jobs[-1]["id"] if additional_jobs else None
        additional_logs = experiment_logs(additional_experiment, additional_state, additional_job)
        epoch = int(additional["epoch"])
        additional_results = [
            json.loads(raw)
            for evaluator_id, result_epoch, raw in EVAL_RESULT.findall(additional_logs)
            if evaluator_id == record["producerId"] and int(result_epoch) == epoch
        ]
        if additional_results:
            results[str(epoch)] = additional_results[-1]
            additional["postDecayResult"] = additional_results[-1]
        matched_decisions = [
            (status, json.loads(raw))
            for evaluator_id, status, raw in EVAL_DECISION.findall(additional_logs)
            if evaluator_id == record["producerId"]
        ]
        if matched_decisions:
            additional_decisions.append(matched_decisions[-1])
            additional["decision"] = matched_decisions[-1][1]
            additional["status"] = matched_decisions[-1][0]
            additional.pop("needsAttention", None)
        elif previous_additional_decision and additional_state == "complete":
            additional_decisions.append(
                (
                    str(
                        previous_additional_decision.get(
                            "status", previous_additional_status or "complete"
                        )
                    ),
                    previous_additional_decision,
                )
            )
            additional["decision"] = previous_additional_decision
            additional["status"] = previous_additional_decision.get(
                "status", previous_additional_status or "complete"
            )
            additional.pop("needsAttention", None)
        elif additional_state == "failed":
            additional["status"] = "failed"
            additional["needsAttention"] = True
        elif additional_state == "complete":
            additional["status"] = "complete_without_decision"
            additional["needsAttention"] = True
        else:
            additional["status"] = additional_state
        additional["beakerStatus"] = additional_state
        additional["wandbHealth"] = refreshed_health(additional, additional_logs, additional_state)

    record["resolvedPostEpochs"] = sorted(int(epoch) for epoch in results)
    active_additional = next(
        (value for value in additional_states if value in {"submitted", "scheduled", "running"}),
        None,
    )
    if additional_decisions:
        record["decision"] = additional_decisions[-1][1]
        record["status"] = additional_decisions[-1][0]
    elif active_additional:
        record["status"] = active_additional
    elif any(value == "failed" for value in additional_states):
        record["status"] = "failed"
        record["needsAttention"] = True
    elif decisions:
        record["decision"] = decisions[-1][1]
        record["status"] = decisions[-1][0]
    elif state == "failed":
        record["status"] = "failed"
        record["needsAttention"] = True
    elif state == "complete":
        if previous_decision:
            record["decision"] = previous_decision
            record["status"] = previous_decision.get("status", previous_status or "complete")
            record.pop("needsAttention", None)
        else:
            record["status"] = "complete_without_decision"
            record["needsAttention"] = True
    else:
        record["status"] = state
    record["beakerStatus"] = state
    record["wandbHealth"] = refreshed_health(record, logs, state)
    return record["status"]


def refresh_integrated_run(record: dict[str, Any]) -> str:
    experiment = str(record["experiment"])
    payload = inspect(experiment)
    state = beaker_state(payload)
    jobs = [job for job in payload.get("jobs") or [] if job.get("id")]
    if jobs:
        record["jobs"] = [job["id"] for job in jobs]
        record["job"] = jobs[-1]["id"]
    job = jobs[-1]["id"] if jobs else None
    logs = experiment_logs(experiment, state, job)
    resolved = {int(epoch) for epoch in record.get("resolvedCheckpointEpochs", [])}
    for producer_id, epoch, _checkpoint in INTEGRATED_PD_RETAINED.findall(logs):
        if producer_id == record["id"]:
            resolved.add(int(epoch))
    results = record.setdefault("postDecayResults", {})
    for producer_id, epoch, raw in INTEGRATED_POST_RESULT.findall(logs):
        if producer_id == record["id"]:
            results[str(int(epoch))] = json.loads(raw)
    record["resolvedCheckpointEpochs"] = sorted(resolved)
    record["resolvedPostEpochs"] = sorted(int(epoch) for epoch in results)

    retained = [int(epoch) for epoch in record["retainedCheckpointEpochs"]]
    evaluation_epochs = {int(epoch) for epoch in record["evaluationEpochs"]}
    events = [
        (match.start(), match.group(1), int(match.group(3)))
        for match in INTEGRATED_STAGE_EVENT.finditer(logs)
        if match.group(2) == record["id"]
    ]
    complete_marker = f"DENSE_DCLM333M_JOB_COMPLETE id={record['id']}" in logs
    record.pop("currentPostEpoch", None)
    if complete_marker:
        record["status"] = "complete"
        record["currentPhase"] = "complete"
        record["currentEpoch"] = retained[-1]
    else:
        last = events[-1] if events else None
        if last and last[1] == "POST_START" and last[2] not in record["resolvedPostEpochs"]:
            record["currentPhase"] = "post"
            record["currentPostEpoch"] = last[2]
            record["currentEpoch"] = last[2]
        elif last and last[1] == "PD_START" and last[2] not in resolved:
            record["currentPhase"] = "producer"
            record["currentEpoch"] = last[2]
        elif (
            last
            and last[1] == "PD_RETAINED"
            and last[2] in evaluation_epochs
            and last[2] not in record["resolvedPostEpochs"]
        ):
            record["currentPhase"] = "post_pending"
            record["currentEpoch"] = last[2]
        else:
            next_epoch = next((epoch for epoch in retained if epoch not in resolved), None)
            record["currentPhase"] = "producer" if next_epoch is not None else "finishing"
            record["currentEpoch"] = next_epoch if next_epoch is not None else retained[-1]
        if state == "failed":
            record["status"] = "failed"
            record["needsAttention"] = True
        elif state == "complete":
            record["status"] = "complete_without_marker"
            record["needsAttention"] = True
        else:
            record["status"] = state
            record.pop("needsAttention", None)
    record["beakerStatus"] = state
    record["wandbHealth"] = refreshed_health(record, logs, state)
    if "DENSE_DCLM333M_POST_START" in logs and "Loading checkpoint from '" not in logs:
        signals = record["wandbHealth"]["criticalSignals"]
        if "missing-exact-post-checkpoint-load" not in signals:
            signals.append("missing-exact-post-checkpoint-load")
        record["wandbHealth"]["status"] = "critical"
    return record["status"]


def main() -> None:
    report = json.loads(REPORT.read_text())
    if len(report.get("producers", [])) != 10 or len(report.get("evaluators", [])) != 2:
        raise RuntimeError("report must contain ten producers and two evaluators")
    if len(report.get("dclm333mIntegratedRuns", [])) != 8:
        raise RuntimeError("report must contain eight Pool-333M integrated runs")
    for record in report["producers"]:
        print(f"{record['id']}: {refresh_producer(record)}")
    for record in report["evaluators"]:
        print(f"{record['id']}: {refresh_evaluator(record)}")
    for record in report["dclm333mIntegratedRuns"]:
        print(f"{record['id']}: {refresh_integrated_run(record)}")
    write_report(report)


if __name__ == "__main__":
    main()
