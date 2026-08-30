#!/usr/bin/env python3
"""Refresh the isolated ten-producer/two-evaluator report from Beaker logs."""

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
    if (
        ("DENSE_CHECKPOINT_PRODUCER_START" in logs or "DENSE1B_CHECKPOINT_EVALUATOR_START" in logs)
        and "Loading checkpoint from '" not in logs
    ):
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
    for epoch in record["targetEpochs"]:
        pool_tokens = 3_000_000_000 if record["model"] == "1b" else 1_000_000_000
        endpoint = -(-int(epoch) * pool_tokens // (int(record["batchSequences"]) * 4096))
        step = endpoint - round(0.1 * endpoint) - 1
        if re.search(rf"(?:/|\b)step{step}(?:\b|/)", logs):
            resolved.add(int(epoch))
    if f"DENSE_CHECKPOINT_PRODUCER_COMPLETE id={record['id']}" in logs:
        resolved.update(record["targetEpochs"])
        record["status"] = "complete"
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
    record["currentEpoch"] = next(
        (epoch for epoch in record["targetEpochs"] if epoch not in resolved), None
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
    if decisions:
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
