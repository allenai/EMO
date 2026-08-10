#!/usr/bin/env python3
"""Synchronize active nested-3B report rows with read-only Beaker metadata."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


REPORTS = {
    model: Path(f"reports/0802/data/wsd_batch_size_{model}_pool3b.json")
    for model in ("153m", "474m", "1b")
}
WANDB_RE = re.compile(r"wandb\.ai/ai2-llm/sewonm-icsl/runs/([a-z0-9]+)")


SSH_HOST: str | None = None


def experiment(experiment_id: str) -> dict:
    command = ["beaker", "experiment", "get", experiment_id, "--format", "json"]
    if SSH_HOST:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            SSH_HOST,
            shlex.join(command),
        ]
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    payload = json.loads(result.stdout)
    return payload[0] if isinstance(payload, list) else payload


def running(job: dict) -> bool:
    status = job.get("status", {})
    return bool(status.get("started")) and not any(
        status.get(field)
        for field in ("finalized", "succeeded", "failed", "canceled", "stopped")
    )


def completed_successfully(job: dict) -> bool:
    status = job.get("status", {})
    return bool(status.get("finalized")) and status.get("exitCode") == 0


def job_state(job: dict) -> dict:
    status = job.get("status", {})
    if completed_successfully(job):
        state = "succeeded"
    elif running(job):
        state = "running"
    elif status.get("finalized"):
        state = "canceled" if status.get("canceled") else "failed"
    elif status.get("started"):
        state = "running"
    else:
        state = "pending"
    execution = job.get("execution") or {}
    result = execution.get("result") or {}
    return {
        "job": job.get("id"),
        "state": state,
        "created": status.get("created"),
        "started": status.get("started"),
        "finalized": status.get("finalized"),
        "exitCode": status.get("exitCode"),
        "reason": status.get("canceledFor") or status.get("stoppedFor"),
        "resultDataset": result.get("beaker"),
    }


def experiment_state(jobs: list[dict], expected_replicas: int) -> str | None:
    """Classify actual jobs, including Beaker-created replacement attempts."""
    if not jobs:
        return None
    active = [job for job in jobs if not job.get("status", {}).get("finalized")]
    if active:
        return "running" if any(running(job) for job in active) else "pending"
    successes = [job for job in jobs if completed_successfully(job)]
    if expected_replicas == 1 and successes:
        return "complete"
    # Replicas from one synchronized attempt have the same creation timestamp.
    by_attempt: dict[str, list[dict]] = {}
    for job in successes:
        created = str(job.get("status", {}).get("created", ""))
        by_attempt.setdefault(created, []).append(job)
    if any(len(attempt) >= expected_replicas for attempt in by_attempt.values()):
        return "complete"
    return "failed"


def sync(path: Path) -> int:
    report = json.loads(path.read_text())
    changed = 0
    for sweep in report.get("batchSweeps", []):
        if not sweep.get("beaker") or sweep.get("status") in {"complete", "failed"}:
            continue
        payload = experiment(sweep["beaker"])
        jobs = payload.get("jobs", [])
        job_ids = [job["id"] for job in jobs if job.get("id")]
        results = [
            job.get("execution", {}).get("result", {}).get("beaker")
            for job in jobs
        ]
        results = [result for result in results if result]
        description = payload.get("description", "")
        match = WANDB_RE.search(description)
        updates = {
            "jobs": job_ids,
            "resultDatasets": results,
            "jobStates": [job_state(job) for job in jobs],
        }
        # Point the singular compatibility fields at the live replacement when
        # Beaker has canceled an earlier attempt.  Keep every attempt in the
        # plural provenance fields above.
        primary_job = (
            next((job for job in jobs if running(job)), None)
            or next(
                (
                    job
                    for job in jobs
                    if not job.get("status", {}).get("finalized")
                ),
                None,
            )
            or next((job for job in jobs if completed_successfully(job)), None)
            or (jobs[0] if jobs else None)
        )
        if primary_job and primary_job.get("id"):
            updates["job"] = primary_job["id"]
            primary_result = (
                primary_job.get("execution", {}).get("result", {}).get("beaker")
            )
            if primary_result:
                updates["resultDataset"] = primary_result
        elif results:
            updates["resultDataset"] = results[0]
        if match:
            updates["activeWandb"] = match.group(1)
        state = experiment_state(jobs, int(sweep.get("nodeCount", 1)))
        if state:
            updates["status"] = state
        if state == "failed":
            failures = [row for row in updates["jobStates"] if row["state"] != "succeeded"]
            updates["failureClass"] = sweep.get("failureClass", "infrastructure")
            updates["failureReason"] = "; ".join(
                f"{row['job']}: {row['reason'] or 'exit ' + str(row['exitCode'])}"
                for row in failures
            )
        for key, value in updates.items():
            if sweep.get(key) != value:
                sweep[key] = value
                changed += 1
    mirror = (
        "window.ICSL_REPORT_DATA="
        + json.dumps(report, separators=(",", ":"))
        + ";\n"
    )
    if changed:
        report["updated"] = "2026-08-09"
        path.write_text(json.dumps(report, indent=2) + "\n")
        mirror = (
            "window.ICSL_REPORT_DATA="
            + json.dumps(report, separators=(",", ":"))
            + ";\n"
        )
    if not path.with_suffix(".js").is_file() or path.with_suffix(".js").read_text() != mirror:
        path.with_suffix(".js").write_text(mirror)
        changed += 1
    return changed


def main() -> None:
    global SSH_HOST
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(REPORTS))
    parser.add_argument(
        "--ssh-host",
        help="Query Beaker through this authenticated SSH host.",
    )
    args = parser.parse_args()
    SSH_HOST = args.ssh_host
    models = (args.model,) if args.model else tuple(REPORTS)
    for model in models:
        print(f"{model}: {sync(REPORTS[model])} fields updated")


if __name__ == "__main__":
    main()
