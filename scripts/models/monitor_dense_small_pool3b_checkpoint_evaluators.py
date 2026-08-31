#!/usr/bin/env python3
"""Refresh registered standalone small-model Pool-3B evaluators."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import monitor_dense_constant_checkpoint_grid as grid

REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
RESULT = re.compile(
    r"DENSE_SMALL_CHECKPOINT_EVALUATOR_RESULT id=([^ ]+) epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
COMPLETE = re.compile(
    r"DENSE_SMALL_CHECKPOINT_EVALUATOR_COMPLETE id=([^ ]+) epoch=([0-9]+)$",
    re.MULTILINE,
)


def refresh(record: dict[str, Any]) -> str:
    experiment = str(record["experiment"])
    payload = grid.inspect(experiment)
    state = grid.beaker_state(payload)
    jobs = [job for job in payload.get("jobs") or [] if job.get("id")]
    if jobs:
        record["jobs"] = [job["id"] for job in jobs]
        record["job"] = jobs[-1]["id"]
    logs = grid.experiment_logs(experiment, state)
    results = [
        json.loads(raw)
        for coordinate, epoch, raw in RESULT.findall(logs)
        if coordinate == record["producerId"] and int(epoch) == int(record["epoch"])
    ]
    complete = any(
        coordinate == record["producerId"] and int(epoch) == int(record["epoch"])
        for coordinate, epoch in COMPLETE.findall(logs)
    )
    if results:
        record["postDecayResult"] = results[-1]
        record["resolvedPostEpoch"] = int(record["epoch"])
    if complete and results:
        record["status"] = "complete"
    elif state == "failed":
        record["status"] = "failed"
        record["needsAttention"] = True
    elif state == "complete":
        record["status"] = "complete_without_result"
        record["needsAttention"] = True
    else:
        record["status"] = state
    health = grid.health(logs, state)
    if "DENSE_SMALL_CHECKPOINT_EVALUATOR_START" in logs and "Loading checkpoint from '" not in logs:
        health["status"] = "critical"
        health.setdefault("criticalSignals", []).append("missing-exact-checkpoint-load")
    record["beakerStatus"] = state
    record["wandbHealth"] = health
    return record["status"]


def main() -> None:
    report = json.loads(REPORT.read_text())
    records = report.get("smallEvaluators", [])
    for record in records:
        print(f"{record['id']}: {refresh(record)}")
    report["smallEvaluatorCount"] = len(records)
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_CHECKPOINT_PRODUCER_GRID=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


if __name__ == "__main__":
    main()
