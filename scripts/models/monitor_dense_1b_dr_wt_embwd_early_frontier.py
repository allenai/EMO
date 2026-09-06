#!/usr/bin/env python3
"""Refresh Dense-1B E1/E2/E4 early-frontier evaluations in the report."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPORT = Path("reports/0802/data/wsd_data_loader_1b.json")
MANIFEST = Path("scripts/models/manifests/dense-1b-dr-wt-embwd-early-frontier-v1.json")
POLICY = "dense_1b_dr_wt_embwd_early_frontier_v1"
EXPECTED_AUTHOR = "sewonm"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
START = re.compile(r"^(\S+) DENSE1B_EARLY_FRONTIER_START id=([^ ]+).*$", re.MULTILINE)
RESULT = re.compile(r"DENSE1B_EARLY_FRONTIER_RESULT id=([^ ]+) json=(\{.*\})$", re.MULTILINE)
STEP = re.compile(r"^(\S+).*\[step=([0-9]+)/([0-9]+).*?,eta=([^,\]]+)\]", re.MULTILINE)


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout


def one_job(job_id: str) -> dict[str, Any]:
    payload = json.loads(command(["beaker", "job", "inspect", job_id, "--format", "json"]))
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"expected one exact job for {job_id}")
    job = payload[0]
    author = (job.get("author") or {}).get("name")
    if author != EXPECTED_AUTHOR:
        raise RuntimeError(f"refusing job {job_id} owned by {author!r}")
    return job


def live_state(job: dict[str, Any]) -> str:
    status = job.get("status") or {}
    if "canceled" in status or "cancelled" in status:
        return "canceled"
    if "finalized" in status or "exited" in status:
        return "complete" if status.get("exitCode") == 0 else "failed"
    if "started" in status:
        return "running"
    if "scheduled" in status:
        return "scheduled"
    return "submitted"


def duration_seconds(value: str) -> int:
    match = re.fullmatch(
        r"(?:(?P<days>[0-9]+)d)?(?:(?P<hours>[0-9]+)h)?(?:(?P<minutes>[0-9]+)m)?(?:(?P<seconds>[0-9]+)s)?",
        value,
    )
    if match is None:
        raise ValueError(f"unrecognized trainer ETA {value}")
    parts = {key: int(raw or 0) for key, raw in match.groupdict().items()}
    return parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]


def merge_result(report: dict[str, Any], record: dict[str, Any], result: dict[str, Any]) -> None:
    batch = int(record["batchSequences"])
    run_id = f"drwtembwd{batch}-lr1e-3-wd0.3"
    matches = [run for run in report.get("runs", []) if run.get("id") == run_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one report run {run_id}")
    run = matches[0]
    epoch = str(int(record["epoch"]))
    result = dict(result)
    result.update(
        {
            "experiment": record["experiment"],
            "beaker": record["experiment"],
            "job": record["job"],
            "revision": record["revision"],
        }
    )
    run.setdefault("earlyFrontierPostDecayResults", {})[epoch] = result
    predecay = run.get("preDecayResults", {}).get(epoch)
    run.setdefault("results", {})[epoch] = {
        "status": "complete",
        "preDecay": predecay,
        "postDecay": result,
        "validation": result.get("validation"),
        "validationExact": result.get("validationExact"),
        "train": result.get("train"),
        "gap": result.get("gap"),
        "wandb": result.get("wandb"),
    }
    record.update(
        {
            "status": "complete",
            "beakerStatus": "complete",
            "result": result,
            "validationExact": result["validationExact"],
            "checkpoint": result["checkpoint"],
        }
    )


def main() -> None:
    report = json.loads(REPORT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    expected = {item["id"]: item for item in manifest["runs"]}
    records = [record for record in report.get("earlyFrontierEvaluations", []) if record.get("policy") == POLICY]
    if not records:
        raise RuntimeError("no early-frontier evaluations are registered")
    by_job: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") in {"submitted", "scheduled", "queued", "running"}:
            if not record.get("job"):
                raise RuntimeError(f"active record {record['id']} has no exact allowlisted job ID")
            by_job.setdefault(str(record["job"]), []).append(record)
    for job_id, job_records in by_job.items():
        job = one_job(job_id)
        state = live_state(job)
        execution = job.get("execution") or {}
        experiment = str(execution.get("experiment") or "")
        if {str(record["experiment"]) for record in job_records} != {experiment}:
            raise RuntimeError(f"job {job_id} does not belong to its registered experiment")
        context = ((execution.get("spec") or {}).get("context") or {})
        if "minRuntime" in context:
            raise RuntimeError(f"job {job_id} has unauthorized minRuntime")
        logs = ""
        try:
            logs = ANSI.sub("", command(["beaker", "job", "logs", job_id, "--since", "70m"]))
        except subprocess.CalledProcessError:
            pass
        starts = START.findall(logs)
        results = {identifier: json.loads(raw) for identifier, raw in RESULT.findall(logs)}
        active_id = starts[-1][1] if starts else None
        steps = STEP.findall(logs)
        unresolved = [record for record in job_records if str(record["id"]) not in results]
        inferred_active_id = active_id
        if inferred_active_id is None and state == "running" and unresolved:
            inferred_active_id = str(min(unresolved, key=lambda item: int(item["epoch"]))["id"])
        for record in job_records:
            identifier = str(record["id"])
            record["beakerStatus"] = state
            if identifier in results:
                merge_result(report, record, results[identifier])
                continue
            if state in {"failed", "canceled"}:
                record.update({"status": state, "reason": "Terminal without a healthy matched POST result marker."})
                continue
            completed_before = sum(
                int(expected[item["id"]]["expectedRuntimeSeconds"])
                for item in job_records
                if int(item["epoch"]) < int(record["epoch"])
            )
            job_started = (job.get("status") or {}).get("started")
            if job_started:
                began = datetime.fromisoformat(str(job_started).replace("Z", "+00:00"))
                record["expectedEta"] = (
                    began + timedelta(seconds=completed_before + int(record["expectedRuntimeSeconds"]))
                ).isoformat()
                record["etaBasis"] = "matched historical stage durations"
            if identifier == inferred_active_id:
                record["status"] = "running"
            elif state in {"running", "scheduled", "submitted"}:
                record["status"] = "queued"
            else:
                record["status"] = state
            if identifier == active_id and steps:
                timestamp, current, endpoint, trainer_eta = steps[-1]
                observed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                record["expectedEta"] = (
                    observed + timedelta(seconds=duration_seconds(trainer_eta) + 120)
                ).isoformat()
                record["etaBasis"] = "live trainer ETA plus heldout/result tail"
                record["progress"] = {
                    "currentStep": int(current),
                    "endpointStep": int(endpoint),
                    "trainerEta": trainer_eta,
                    "observedAt": observed.isoformat(),
                }
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.with_suffix(".js").write_text("window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n")
    print(json.dumps([{"id": record["id"], "status": record["status"]} for record in records], indent=2))


if __name__ == "__main__":
    main()
