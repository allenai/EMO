#!/usr/bin/env python3
"""Synchronize registered small-dense Step 1-1 attempts with Beaker status."""

from __future__ import annotations

import json
import argparse
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path


REPORTS = (
    Path("reports/0802/data/wsd_batch_size_153m.json"),
    Path("reports/0802/data/wsd_batch_size_474m.json"),
)
TRACKED_SEARCHES = {
    "small-model-adaptive-coordinate",
    "small-model-selected-e1-fractional-chain",
}
WANDB = re.compile(r"wandb\.ai/[^\s]+/runs/([a-z0-9]+)")
PROGRESS = re.compile(r"\[(\d+)% complete, step ([0-9,]+)/([0-9,]+)")


def beaker_command(*args: str) -> list[str]:
    command = ["beaker", *args]
    if not ARGS.ssh_host:
        return command
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        ARGS.ssh_host,
        shlex.join(command),
    ]


def beaker_status(job: dict[str, object]) -> str:
    status = job.get("status", {})
    assert isinstance(status, dict)
    if "finalized" in status:
        if "canceled" in status or "canceledCode" in status:
            return "canceled"
        exit_code = status.get("exitCode")
        if exit_code == 0:
            return "complete"
        return "failed"
    if "started" in status:
        return "running"
    if "scheduled" in status:
        return "queued"
    return "pending"


def result_dataset(job: dict[str, object]) -> str | None:
    direct = job.get("result", {})
    if isinstance(direct, dict) and direct.get("beaker"):
        return str(direct["beaker"])
    execution = job.get("execution", {})
    if isinstance(execution, dict):
        nested = execution.get("result", {})
        if isinstance(nested, dict) and nested.get("beaker"):
            return str(nested["beaker"])
    return None


parser = argparse.ArgumentParser()
parser.add_argument("--ssh-host", help="Query Beaker through this authenticated SSH host.")
ARGS = parser.parse_args()

reports = {path: json.loads(path.read_text()) for path in REPORTS}
ids = [
    sweep["beaker"]
    for report in reports.values()
    for sweep in report.get("batchSweeps", [])
    if sweep.get("search") in TRACKED_SEARCHES and sweep.get("beaker")
]
if ids:
    result = subprocess.run(
        beaker_command("experiment", "get", *ids, "--format", "json"),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)
    experiments = payload if isinstance(payload, list) else [payload]
else:
    experiments = []
by_id = {experiment["id"]: experiment for experiment in experiments}

counts: Counter[str] = Counter()
for path, report in reports.items():
    for sweep in report.get("batchSweeps", []):
        experiment = by_id.get(sweep.get("beaker"))
        if experiment is None:
            continue
        jobs = experiment.get("jobs", [])
        if not jobs:
            continue
        job = jobs[-1]
        sweep["job"] = job["id"]
        sweep["status"] = beaker_status(job)
        counts[sweep["status"]] += 1
        job_status = job.get("status", {})
        if isinstance(job_status, dict) and sweep["status"] in {"canceled", "failed"}:
            active_wandb = str(sweep.get("activeWandb") or "")
            unhealthy = report.get("healthAudit", {}).get("unhealthy", {})
            if active_wandb and active_wandb in unhealthy:
                sweep["failureClass"] = "health-stop"
                sweep["reason"] = str(unhealthy[active_wandb]["reason"])
            elif sweep.get("pauseReason"):
                sweep["failureClass"] = "user-policy-pause"
                sweep["reason"] = str(sweep["pauseReason"])
            elif str(sweep.get("failureClass", "")).startswith(
                ("infrastructure-", "preflight-")
            ):
                # Preserve a more specific classification established from a
                # manual log audit (for example, an NCCL watchdog timeout or
                # a deterministic source-checkpoint preflight failure). The
                # Beaker job summary only exposes a generic non-zero exit.
                pass
            else:
                sweep["failureClass"] = (
                    "infrastructure-cordoned-node"
                    if "cordoned" in str(job_status.get("canceledFor", "")).lower()
                    else "job-failure"
                )
                sweep["reason"] = str(
                    job_status.get("canceledFor")
                    or job_status.get("message")
                    or "Beaker job failed"
                )
        if dataset := result_dataset(job):
            sweep["resultDataset"] = dataset
        description = str(experiment.get("description", ""))
        if matches := WANDB.findall(description):
            sweep["activeWandb"] = matches[-1]
        if matches := PROGRESS.findall(description):
            match = matches[-1]
            if not sweep.get("evaluationOnly"):
                sweep["progressPercent"] = int(match[0])
                sweep["progressStep"] = int(match[1].replace(",", ""))
                sweep["progressTotalSteps"] = int(match[2].replace(",", ""))
        if sweep.get("evaluationOnly") and sweep["status"] == "complete":
            sweep["progressPercent"] = 100
            sweep.pop("progressStep", None)
            sweep.pop("progressTotalSteps", None)
    text = json.dumps(report, indent=2) + "\n"
    path.write_text(text)
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )

print(json.dumps(dict(sorted(counts.items())), sort_keys=True))
