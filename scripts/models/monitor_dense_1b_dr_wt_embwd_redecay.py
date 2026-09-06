#!/usr/bin/env python3
"""Refresh Dense-1B re-decay retry01 state in the data-loader report."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPORT = Path("reports/0802/data/wsd_data_loader_1b.json")
MANIFEST = Path("scripts/models/manifests/dense-1b-dr-wt-embwd-redecay-retry01.json")
POLICY = "dense_1b_dr_wt_embwd_redecay_retry01_v1"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
START = re.compile(r"DENSE1B_REDECAY_START id=([^ ]+).*$", re.MULTILINE)
RESULT = re.compile(r"DENSE1B_REDECAY_RESULT id=([^ ]+) json=(\{.*\})$", re.MULTILINE)


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout


def state(experiment: dict[str, Any]) -> str:
    jobs = experiment.get("jobs") or []
    if not jobs:
        return "submitted"
    statuses = [job.get("status") or {} for job in jobs]
    terminal = {"exited", "finalized", "canceled", "cancelled"}
    if any("started" in status and not terminal.intersection(status) for status in statuses):
        return "running"
    if any("scheduled" in status and "started" not in status and not terminal.intersection(status) for status in statuses):
        return "scheduled"
    if any("finalized" in status and status.get("exitCode") == 0 for status in statuses):
        return "complete"
    if all(terminal.intersection(status) for status in statuses):
        return "failed"
    return "submitted"


def main() -> None:
    report = json.loads(REPORT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    manifest_runs = {item["id"]: item for item in manifest["runs"]}
    retries = report.get("redecayRetries", [])
    if not retries:
        raise RuntimeError("no Dense-1B re-decay retries are registered")
    summary = []
    for record in retries:
        if record.get("policy") != POLICY:
            continue
        experiment_id = str(record["experiment"])
        payload = json.loads(command(["beaker", "experiment", "inspect", experiment_id, "--format", "json"]))
        if not isinstance(payload, list) or len(payload) != 1:
            raise RuntimeError(f"expected one experiment for {experiment_id}")
        experiment = payload[0]
        live = state(experiment)
        job = (experiment.get("jobs") or [{}])[-1]
        record.update({"beakerStatus": live, "job": job.get("id")})
        observed_seconds = int(manifest_runs[record["id"]]["expectedRuntimeSeconds"])
        record["expectedRuntimeSeconds"] = observed_seconds
        started = (job.get("status") or {}).get("started")
        if started:
            started_at = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
            record["startedAt"] = started_at.isoformat()
            record["expectedEta"] = (started_at + timedelta(seconds=observed_seconds)).isoformat()
        try:
            logs = ANSI.sub("", command(["beaker", "experiment", "logs", experiment_id]))
        except subprocess.CalledProcessError:
            logs = ""
        matches = [json.loads(raw) for identifier, raw in RESULT.findall(logs) if identifier == record["id"]]
        if matches:
            result = matches[-1]
            result.update({"experiment": experiment_id, "beaker": experiment_id, "job": job.get("id"), "revision": record.get("revision")})
            record.update({"status": "complete", "result": result, "validationExact": result["validationExact"], "checkpoint": result["checkpoint"]})
        elif live == "failed":
            record.update({"status": "failed", "reason": "Terminal without a healthy matched retry result marker."})
        else:
            started = any(identifier == record["id"] for identifier in START.findall(logs))
            record["status"] = "running" if started else live
        summary.append({"id": record["id"], "status": record["status"], "experiment": experiment_id})
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.with_suffix(".js").write_text("window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
