#!/usr/bin/env python3
"""Synchronize active nested-3B report rows with read-only Beaker metadata."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


REPORTS = {
    model: Path(f"reports/0802/data/wsd_batch_size_{model}_pool3b.json")
    for model in ("153m", "474m", "1b")
}
WANDB_RE = re.compile(r"wandb\.ai/ai2-llm/sewonm-icsl/runs/([a-z0-9]+)")


def experiment(experiment_id: str) -> dict:
    result = subprocess.run(
        ["beaker", "experiment", "get", experiment_id, "--format", "json"],
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
        }
        if job_ids:
            updates["job"] = job_ids[0]
        if results:
            updates["resultDataset"] = results[0]
        if match:
            updates["activeWandb"] = match.group(1)
        if jobs and all(running(job) for job in jobs):
            updates["status"] = "running"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(REPORTS))
    args = parser.parse_args()
    models = (args.model,) if args.model else tuple(REPORTS)
    for model in models:
        print(f"{model}: {sync(REPORTS[model])} fields updated")


if __name__ == "__main__":
    main()
