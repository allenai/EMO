#!/usr/bin/env python3
"""Audit report W&B training health from the corresponding Beaker logs.

The W&B project is private, but every training job prints the same CE-loss and
total-gradient-norm series that it sends to W&B.  This script maps the W&B run
IDs recorded in the report data back to their Beaker jobs, splits persistent
job logs into individual W&B runs, and emits per-run tail-stability statistics.
It never mutates report data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPORTS = (
    Path("reports/0802/data/wsd_batch_size_1b.json"),
    Path("reports/0802/data/wsd_weight_decay_1b.json"),
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RUN_RE = re.compile(r"https://wandb\.ai/ai2-llm/sewonm-icsl/runs/([a-z0-9]+)")
NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
CE_RE = re.compile(rf"train/CE loss=({NUMBER_PATTERN})")
GRAD_RE = re.compile(rf"optim/total grad norm=({NUMBER_PATTERN})")
SSH_HOST: str | None = None


def beaker_command(*args: str) -> list[str]:
    command = ["beaker", *args]
    if SSH_HOST:
        return [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            SSH_HOST, shlex.join(command),
        ]
    return command


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def rolling_min_mean(values: list[float]) -> float:
    if not values:
        return math.nan
    width = min(len(values), max(10, len(values) // 20))
    current = sum(values[:width])
    best = current
    for index in range(width, len(values)):
        current += values[index] - values[index - width]
        best = min(best, current)
    return best / width


def series_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0}
    tail_count = min(len(values), max(20, math.ceil(len(values) * 0.1)))
    tail = values[-tail_count:]
    return {
        "n": len(values),
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "tailCount": tail_count,
        "tailMean": statistics.fmean(tail),
        "tailMedian": statistics.median(tail),
        "tailP95": percentile(tail, 0.95),
        "bestRollingMean": rolling_min_mean(values),
    }


def result_records(path: Path) -> Iterable[dict[str, Any]]:
    report = json.loads(path.read_text())
    if "batchSweeps" in report:
        for sweep in report["batchSweeps"]:
            default_job = sweep.get("job") or next(
                (result.get("job") for result in sweep.get("results", {}).values() if result.get("job")),
                None,
            )
            for epoch, result in sweep.get("results", {}).items():
                if result.get("wandb"):
                    yield {
                        "wandb": result["wandb"],
                        "job": result.get("job") or default_job,
                        "batch": sweep.get("batchSequences"),
                        "epoch": float(epoch) if "." in str(epoch) else int(epoch),
                        "lr": sweep.get("lr"),
                        "wd": sweep.get("wd"),
                        "status": result.get("status"),
                        "source": path.name,
                    }
            if sweep.get("activeWandb"):
                yield {
                    "wandb": sweep["activeWandb"],
                    "job": default_job,
                    "batch": sweep.get("batchSequences"),
                    "epoch": sweep.get("activeEpoch"),
                    "lr": sweep.get("lr"),
                    "wd": sweep.get("wd"),
                    "status": sweep.get("status"),
                    "source": path.name,
                }
    elif "runs" in report:
        for run in report.get("runs", []):
            if run.get("wandb") and run.get("job"):
                yield {
                    "wandb": run["wandb"],
                    "job": run["job"],
                    "batch": 1024,
                    "epoch": run.get("epoch"),
                    "lr": run.get("lr"),
                    "wd": run.get("wd"),
                    "status": run.get("status"),
                    "source": path.name,
                }
    else:
        phase_jobs = {
            wandb: chain.get("job")
            for chain in report.get("chainExperiments", [])
            for wandb in (chain.get("phaseWandb") or {}).values()
            if chain.get("job")
        }
        excluded_groups = {"chainExperiments", "dataGeneration"}
        for group, rows in report.items():
            if group in excluded_groups or not isinstance(rows, list):
                continue
            for run in rows:
                if not isinstance(run, dict) or not run.get("wandb"):
                    continue
                yield {
                    "wandb": run["wandb"],
                    "job": run.get("job") or phase_jobs.get(run["wandb"]),
                    "beaker": run.get("beaker"),
                    "batch": None,
                    "epoch": run.get("epoch"),
                    "lr": run.get("lr"),
                    "wd": run.get("wd"),
                    "status": run.get("status"),
                    "group": group,
                    "source": path.name,
                }


def fetch_experiment_job(experiment: str) -> tuple[str, str | None, str | None]:
    result = subprocess.run(
        beaker_command("experiment", "get", experiment, "--format", "json"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    if result.returncode:
        return experiment, None, f"beaker exited {result.returncode}: {result.stdout[-500:]}"
    try:
        payload = json.loads(result.stdout)
        experiment_data = payload[0] if isinstance(payload, list) else payload
        jobs = experiment_data.get("jobs") or []
        job = next((item.get("id") for item in jobs if item.get("id")), None)
        return experiment, job, None if job else "experiment has no jobs"
    except (json.JSONDecodeError, AttributeError, IndexError) as error:
        return experiment, None, f"could not parse experiment JSON: {error}"


def fetch_log(job: str, since: str | None = None) -> tuple[str, str | None]:
    command = beaker_command("job", "logs", job, "--no-timestamps")
    if since:
        if SSH_HOST:
            command[-1] = shlex.join(
                ["beaker", "job", "logs", job, "--no-timestamps", "--since", since]
            )
        else:
            command.extend(["--since", since])
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    if result.returncode:
        return job, f"beaker exited {result.returncode}: {result.stdout[-500:]}"
    return job, result.stdout


def parse_log(
    text: str,
    initial_run: str | None = None,
) -> dict[str, dict[str, list[float]]]:
    current = initial_run
    output: dict[str, dict[str, list[float]]] = {}
    if current is not None:
        output.setdefault(current, {"ce": [], "grad": []})
    for raw_line in ANSI_RE.sub("", text).splitlines():
        run_match = RUN_RE.search(raw_line)
        if run_match:
            current = run_match.group(1)
            output.setdefault(current, {"ce": [], "grad": []})
            continue
        if current is None:
            continue
        ce_match = CE_RE.search(raw_line)
        if ce_match:
            output[current]["ce"].append(float(ce_match.group(1)))
        grad_match = GRAD_RE.search(raw_line)
        if grad_match:
            output[current]["grad"].append(float(grad_match.group(1)))
    return output


def main() -> None:
    global SSH_HOST
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="*", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--since", help="Limit Beaker logs to a relative or absolute time window, e.g. 60m")
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument("--ssh-host", help="Query Beaker through this authenticated SSH host.")
    args = parser.parse_args()
    SSH_HOST = args.ssh_host

    records: dict[str, dict[str, Any]] = {}
    for path in args.reports:
        for record in result_records(path):
            records.setdefault(record["wandb"], record)
    if args.active_only:
        active_states = {"pending", "queued", "scheduled", "running"}
        records = {
            wandb: record
            for wandb, record in records.items()
            if record.get("status") in active_states
        }
    experiment_errors: dict[str, str] = {}
    unresolved_experiments = sorted(
        {record["beaker"] for record in records.values() if not record.get("job") and record.get("beaker")}
    )
    if unresolved_experiments:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_experiment_job, experiment): experiment for experiment in unresolved_experiments}
            resolved_jobs: dict[str, str] = {}
            for future in as_completed(futures):
                experiment, job, error = future.result()
                if job:
                    resolved_jobs[experiment] = job
                else:
                    experiment_errors[experiment] = error or "unknown experiment lookup error"
        for record in records.values():
            if not record.get("job") and record.get("beaker") in resolved_jobs:
                record["job"] = resolved_jobs[record["beaker"]]

    jobs = sorted({record["job"] for record in records.values() if record.get("job")})
    active_states = {"pending", "queued", "scheduled", "running"}
    active_wandb_by_job = {
        record["job"]: record["wandb"]
        for record in records.values()
        if record.get("job") and record.get("status") in active_states
    }

    telemetry: dict[str, dict[str, list[float]]] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_log, job, args.since): job for job in jobs}
        for future in as_completed(futures):
            job, result = future.result()
            if result is None or result.startswith("beaker exited "):
                errors[job] = result or "empty result"
            else:
                telemetry.update(parse_log(result, active_wandb_by_job.get(job)))

    audit = []
    for wandb, record in records.items():
        series = telemetry.get(wandb, {"ce": [], "grad": []})
        ce = series_stats(series["ce"])
        grad = series_stats(series["grad"])
        if ce.get("n"):
            ce["tailDeltaFromBest"] = ce["tailMean"] - ce["bestRollingMean"]
        audit.append({**record, "ce": ce, "grad": grad})
    audit.sort(key=lambda row: (row.get("batch") or 0, row.get("epoch") or 0, float(row.get("lr") or 0), float(row.get("wd") or 0)))
    print(
        json.dumps(
            {"runs": audit, "experimentErrors": experiment_errors, "logErrors": errors},
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
