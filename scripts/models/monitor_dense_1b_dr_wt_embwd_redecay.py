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
EXPECTED_BEAKER_AUTHOR = "sewonm"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
START = re.compile(
    r"^(\S+) DENSE1B_REDECAY_START id=([^ ]+).*$", re.MULTILINE
)
RESULT = re.compile(r"DENSE1B_REDECAY_RESULT id=([^ ]+) json=(\{.*\})$", re.MULTILINE)
STEP = re.compile(
    r"^(\S+).*\[step=([0-9]+)/([0-9]+).*?,eta=([^,\]]+)\]", re.MULTILINE
)


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout


def recent_log_arguments(job: dict[str, Any], live: str) -> list[str]:
    """Build a bounded log request that still includes terminal result markers.

    The Beaker HTTP API rejects ``job logs --tail`` even though the CLI exposes
    that flag.  For live jobs, a short relative window is enough for current
    telemetry.  For terminal jobs, anchor the window to the exit time so a
    later monitor pass can still recover the final result marker.
    """
    job_id = str(job["id"])
    status = job.get("status") or {}
    exited = status.get("exited") or status.get("finalized")
    if live in {"complete", "failed"} and exited:
        exited_at = datetime.fromisoformat(str(exited).replace("Z", "+00:00"))
        since = (exited_at - timedelta(minutes=15)).isoformat().replace("+00:00", "Z")
    else:
        since = "15m"
    return ["beaker", "job", "logs", job_id, "--since", since]


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


def duration_seconds(value: str) -> int:
    match = re.fullmatch(
        r"(?:(?P<days>[0-9]+)d)?(?:(?P<hours>[0-9]+)h)?"
        r"(?:(?P<minutes>[0-9]+)m)?(?:(?P<seconds>[0-9]+)s)?",
        value,
    )
    if match is None:
        raise ValueError(f"unrecognized trainer ETA {value}")
    parts = {key: int(raw or 0) for key, raw in match.groupdict().items()}
    return parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]


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
        needs_terminal_result_recovery = (
            record.get("beakerStatus") == "complete" and not record.get("result")
        )
        if (
            record.get("beakerStatus") not in {"submitted", "queued", "scheduled", "running"}
            and not needs_terminal_result_recovery
        ):
            summary.append(
                {
                    "id": record["id"],
                    "status": record["status"],
                    "experiment": record["experiment"],
                }
            )
            continue
        experiment_id = str(record["experiment"])
        payload = json.loads(command(["beaker", "experiment", "inspect", experiment_id, "--format", "json"]))
        if not isinstance(payload, list) or len(payload) != 1:
            raise RuntimeError(f"expected one experiment for {experiment_id}")
        experiment = payload[0]
        author = (experiment.get("author") or {}).get("name")
        if author != EXPECTED_BEAKER_AUTHOR:
            raise RuntimeError(
                f"refusing experiment {experiment_id} owned by {author!r}; "
                f"expected {EXPECTED_BEAKER_AUTHOR!r}"
            )
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
        job_id = job.get("id")
        try:
            logs = ANSI.sub(
                "",
                command(recent_log_arguments(job, live)),
            ) if job_id else ""
        except subprocess.CalledProcessError:
            logs = ""
        stage_starts = [
            timestamp for timestamp, identifier in START.findall(logs)
            if identifier == record["id"]
        ]
        if stage_starts:
            started_at = datetime.fromisoformat(stage_starts[-1].replace("Z", "+00:00"))
            record["startedAt"] = started_at.isoformat()
            historical_eta = started_at + timedelta(seconds=observed_seconds)
            record["historicalExpectedEta"] = historical_eta.isoformat()
            record["expectedEta"] = historical_eta.isoformat()
            record["etaBasis"] = "matched historical retry runtime"
        steps = STEP.findall(logs)
        if steps:
            timestamp, current_step, endpoint_step, trainer_eta = steps[-1]
            telemetry_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            # The matched runs spent roughly two minutes between their last train step
            # and the complete heldout/result marker.
            record["expectedEta"] = (
                telemetry_at + timedelta(seconds=duration_seconds(trainer_eta) + 120)
            ).isoformat()
            record["etaBasis"] = "live trainer ETA plus two-minute heldout/result tail"
            record["progress"] = {
                "currentStep": int(current_step),
                "endpointStep": int(endpoint_step),
                "trainerEta": trainer_eta,
                "observedAt": telemetry_at.isoformat(),
            }
        matches = [json.loads(raw) for identifier, raw in RESULT.findall(logs) if identifier == record["id"]]
        if matches:
            result = matches[-1]
            result.update({"experiment": experiment_id, "beaker": experiment_id, "job": job.get("id"), "revision": record.get("revision")})
            record.update({"status": "complete", "result": result, "validationExact": result["validationExact"], "checkpoint": result["checkpoint"]})
        elif live == "failed":
            record.update({"status": "failed", "reason": "Terminal without a healthy matched retry result marker."})
        else:
            record["status"] = "running" if stage_starts else live
        summary.append({"id": record["id"], "status": record["status"], "experiment": experiment_id})
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.with_suffix(".js").write_text("window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
