#!/usr/bin/env python3
"""Refresh the isolated ten-producer/two-evaluator report from Beaker logs.

The two Dense-1B producers retain the original v1 policy.  The eight small-model
records are the corrected Pool-3B v2 bridge/producers.  Older canceled
small-model Pool-1B producers are intentionally absent from the report.
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
        "window.ICSL_CHECKPOINT_PRODUCER_GRID="
        + json.dumps(report, separators=(",", ":"))
        + ";\n"
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


def experiment_logs(experiment: str, state: str) -> str:
    if state not in {"running", "complete", "failed"}:
        return ""
    return ANSI.sub("", command(["beaker", "experiment", "logs", experiment]))


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
    logs = experiment_logs(str(experiment), state)
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
    record["wandbHealth"] = health(logs, state)
    return record["status"]


def refresh_evaluator(record: dict[str, Any]) -> str:
    experiment = record.get("experiment")
    if not experiment:
        return "planned"
    payload = inspect(str(experiment))
    state = beaker_state(payload)
    jobs = [job for job in payload.get("jobs") or [] if job.get("id")]
    if jobs:
        record["jobs"] = [job["id"] for job in jobs]
        record["job"] = jobs[-1]["id"]
    logs = experiment_logs(str(experiment), state)
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
        additional_experiment = str(additional["experiment"])
        additional_payload = inspect(additional_experiment)
        additional_state = beaker_state(additional_payload)
        additional_states.append(additional_state)
        additional_jobs = [
            job for job in additional_payload.get("jobs") or [] if job.get("id")
        ]
        if additional_jobs:
            additional["jobs"] = [job["id"] for job in additional_jobs]
            additional["job"] = additional_jobs[-1]["id"]
        additional_logs = experiment_logs(additional_experiment, additional_state)
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
        elif additional_state == "failed":
            additional["status"] = "failed"
            additional["needsAttention"] = True
        elif additional_state == "complete":
            additional["status"] = "complete_without_decision"
            additional["needsAttention"] = True
        else:
            additional["status"] = additional_state
        additional["beakerStatus"] = additional_state
        additional["wandbHealth"] = health(additional_logs, additional_state)

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
        record["status"] = "complete_without_decision"
        record["needsAttention"] = True
    else:
        record["status"] = state
    record["beakerStatus"] = state
    record["wandbHealth"] = health(logs, state)
    return record["status"]


def main() -> None:
    report = json.loads(REPORT.read_text())
    if len(report.get("producers", [])) != 10 or len(report.get("evaluators", [])) != 2:
        raise RuntimeError("report must contain ten producers and two evaluators")
    for record in report["producers"]:
        print(f"{record['id']}: {refresh_producer(record)}")
    for record in report["evaluators"]:
        print(f"{record['id']}: {refresh_evaluator(record)}")
    write_report(report)


if __name__ == "__main__":
    main()
