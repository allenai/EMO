#!/usr/bin/env python3
"""Refresh the exact 13 retained Pool-333M evaluator jobs."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
REPORT_JS_PREFIX = "window.ICSL_CHECKPOINT_PRODUCER_GRID="
RESULT = re.compile(
    r"DENSE_DCLM333M_RETAINED_EVAL_RESULT id=([^ ]+) epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
COMPLETE = re.compile(
    r"DENSE_DCLM333M_RETAINED_EVAL_COMPLETE id=([^ ]+) epoch=([0-9]+)$",
    re.MULTILINE,
)
TERMINAL = {"succeeded", "failed", "canceled"}


def command(arguments: list[str]) -> str:
    result = subprocess.run(arguments, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout


def state(payload: dict[str, Any]) -> str:
    value = str(payload.get("status") or payload.get("state") or "unknown").lower()
    return {"completed": "succeeded", "complete": "succeeded"}.get(value, value)


def refresh(record: dict[str, Any]) -> None:
    job = str(record["job"])
    payload = json.loads(command(["beaker", "job", "inspect", job, "--format", "json"]))
    author = str(payload.get("author", {}).get("name") or payload.get("experiment", {}).get("author", {}).get("name"))
    if author != "sewonm":
        raise RuntimeError(f"owner mismatch for {job}: {author}")
    current = state(payload)
    logs = command(["beaker", "job", "logs", job, "--since", "70m"])
    matches = [
        json.loads(raw)
        for coordinate, epoch, raw in RESULT.findall(logs)
        if coordinate == record["producerId"] and int(epoch) == int(record["epoch"])
    ]
    complete = any(
        coordinate == record["producerId"] and int(epoch) == int(record["epoch"])
        for coordinate, epoch in COMPLETE.findall(logs)
    )
    if matches:
        record["postDecayResult"] = matches[-1]
        record["resolvedPostEpoch"] = int(record["epoch"])
    if complete and matches:
        record["status"] = "complete"
    elif current in TERMINAL:
        record["status"] = "failed" if current != "succeeded" else "complete_without_result"
        record["needsAttention"] = True
    else:
        record["status"] = current
    record["beakerStatus"] = current
    record["lastMonitoredAt"] = datetime.now(tz=UTC).isoformat()


def merge_results(report: dict[str, Any]) -> None:
    producers = {row["id"]: row for row in report["dclm333mIntegratedRuns"]}
    for record in report["dclm333mRetainedEvaluators"]:
        if record.get("status") != "complete" or not record.get("postDecayResult"):
            continue
        producer = producers[record["producerId"]]
        epoch = int(record["epoch"])
        key = str(epoch)
        existing = (producer.get("postDecayResults") or {}).get(key)
        if existing and existing != record["postDecayResult"]:
            raise RuntimeError(f"refusing to overwrite conflicting POST result {record['producerId']} E{epoch}")
        producer.setdefault("postDecayResults", {})[key] = record["postDecayResult"]
        producer["resolvedPostEpochs"] = sorted(
            {int(value) for value in producer.get("resolvedPostEpochs", [])} | {epoch}
        )


def main() -> None:
    report = json.loads(REPORT.read_text())
    records = report.get("dclm333mRetainedEvaluators", [])
    if len(records) != 13:
        raise RuntimeError("expected exactly 13 retained Pool-333M evaluators")
    for record in records:
        if record.get("status") not in {"complete", "failed", "canceled"}:
            refresh(record)
        print(f"{record['id']}: {record['status']}")
    merge_results(report)
    report["dclm333mRetainedEvaluatorCount"] = len(records)
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    rendered = json.dumps(report, indent=2) + "\n"
    REPORT.write_text(rendered)
    REPORT_JS.write_text(REPORT_JS_PREFIX + rendered.rstrip() + ";\n")


if __name__ == "__main__":
    main()
