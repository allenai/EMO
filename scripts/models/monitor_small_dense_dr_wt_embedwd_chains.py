#!/usr/bin/env python3
"""Refresh the eight small-dense adaptive DR+WT+EmbedWD chains."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORTS = {
    "474m": Path("reports/0802/data/wsd_batch_size_474m.json"),
    "153m": Path("reports/0802/data/wsd_batch_size_153m.json"),
}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STAGE_RESULT = re.compile(
    r"SMALL_DRWTEMBWD_STAGE_RESULT model=(153m|474m) bs=([0-9]+) "
    r"epoch=([0-9]+) wd=([^ ]+) json=(\{.*\})$",
    re.MULTILINE,
)
FRONTIER_RESULT = re.compile(
    r"SMALL_DRWTEMBWD_FRONTIER_RESULT model=(153m|474m) bs=([0-9]+) "
    r"epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
FRONTIER_START = re.compile(
    r"SMALL_DRWTEMBWD_FRONTIER_START model=(153m|474m) bs=([0-9]+) "
    r"epoch=([0-9]+) candidates=([^\s]+)"
)
STAGE_START = re.compile(
    r"SMALL_DRWTEMBWD_STAGE_START model=(153m|474m) bs=([0-9]+) "
    r"epoch=([0-9]+) wd=([^ ]+)"
)
SATURATED = re.compile(
    r"SMALL_DRWTEMBWD_SATURATED model=(153m|474m) bs=([0-9]+) "
    r"epoch=([0-9]+) selected_wd=([^\s]+)"
)
TRAIN_STEP = re.compile(r"\[step=([0-9,]+)/([0-9,]+),epoch=")
TRAIN_LOSS = re.compile(r"\btrain/CE loss=([0-9]+(?:\.[0-9]+)?)")


def run(arguments: list[str]) -> str:
    completed = subprocess.run(arguments, check=True, capture_output=True, text=True)
    return completed.stdout


def inspect_experiment(experiment: str) -> dict[str, Any]:
    payload = json.loads(run(["beaker", "experiment", "inspect", experiment, "--format", "json"]))
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"expected one experiment for {experiment}")
    return payload[0]


def experiment_state(experiment: dict[str, Any]) -> str:
    jobs = experiment.get("jobs") or []
    if not jobs:
        return "submitted"
    statuses = [job.get("status") or {} for job in jobs]
    terminal = {"exited", "finalized", "canceled", "cancelled"}
    if any(
        "started" in status and not terminal.intersection(status)
        for status in statuses
    ):
        return "running"
    if any(
        "scheduled" in status and "started" not in status and not terminal.intersection(status)
        for status in statuses
    ):
        return "scheduled"
    if any("finalized" in status and status.get("exitCode") == 0 for status in statuses):
        return "complete"
    if all(
        "finalized" in status or "canceled" in status or "cancelled" in status
        for status in statuses
    ):
        return "failed"
    return "submitted"


def target_after(record: dict[str, Any], epoch: int) -> int:
    initial = [int(value) for value in record["initialTargets"]]
    if epoch in initial:
        index = initial.index(epoch)
        if index + 1 < len(initial):
            return initial[index + 1]
        return epoch + int(record["epochIncrement"])
    return epoch + int(record["epochIncrement"])


def parsed_stage_results(logs: str, model: str, batch: int) -> dict[tuple[int, str], dict[str, Any]]:
    parsed: dict[tuple[int, str], dict[str, Any]] = {}
    for parsed_model, parsed_batch, epoch, wd, payload in STAGE_RESULT.findall(logs):
        if parsed_model == model and int(parsed_batch) == batch:
            parsed[(int(epoch), wd)] = json.loads(payload)
    return parsed


def parsed_frontiers(logs: str, model: str, batch: int) -> dict[int, dict[str, Any]]:
    parsed: dict[int, dict[str, Any]] = {}
    for parsed_model, parsed_batch, epoch, payload in FRONTIER_RESULT.findall(logs):
        if parsed_model == model and int(parsed_batch) == batch:
            parsed[int(epoch)] = json.loads(payload)
    return parsed


def update_chain(model: str, record: dict[str, Any]) -> str:
    batch = int(record["batchSequences"])
    experiment_id = str(record["experiment"])
    inspected = inspect_experiment(experiment_id)
    state = experiment_state(inspected)
    jobs = [job for job in inspected.get("jobs") or [] if job.get("id")]
    record["status"] = state
    record["beakerStatus"] = state
    if jobs:
        record["job"] = jobs[-1]["id"]
        record["jobs"] = [job["id"] for job in jobs]

    logs = ""
    if state in {"running", "complete", "failed"}:
        logs = ANSI.sub("", run(["beaker", "experiment", "logs", experiment_id]))

    for (epoch, wd), result in parsed_stage_results(logs, model, batch).items():
        result.update(
            {
                "epoch": epoch,
                "wd": wd,
                "status": "complete",
                "experiment": experiment_id,
                "beaker": experiment_id,
                "job": record.get("job"),
                "revision": record.get("revision"),
            }
        )
        record.setdefault("results", {}).setdefault(str(epoch), {})[wd] = result

    frontiers = parsed_frontiers(logs, model, batch)
    for epoch, frontier in frontiers.items():
        frontier.update(
            {
                "epoch": epoch,
                "status": "complete",
                "experiment": experiment_id,
                "beaker": experiment_id,
                "job": record.get("job"),
            }
        )
        record.setdefault("frontiers", {})[str(epoch)] = frontier

    saturated = [
        (int(epoch), wd)
        for parsed_model, parsed_batch, epoch, wd in SATURATED.findall(logs)
        if parsed_model == model and int(parsed_batch) == batch
    ]
    if saturated:
        epoch, wd = saturated[-1]
        record.update(
            {
                "status": "complete",
                "beakerStatus": state,
                "activeEpoch": None,
                "activeWds": [],
                "saturatedEpoch": epoch,
                "selectedWd": wd,
                "stopReason": (
                    f"The selected held-out validation CE stopped strictly improving at E{epoch}."
                ),
                "reason": (
                    f"The persistent BS{batch} DR+WT+EmbedWD chain saturated at E{epoch}; "
                    "each fixed-WD trajectory remained in one canonical output directory."
                ),
            }
        )
    else:
        starts = [
            (int(epoch), candidates.split(","))
            for parsed_model, parsed_batch, epoch, candidates in FRONTIER_START.findall(logs)
            if parsed_model == model and int(parsed_batch) == batch
        ]
        if starts:
            current_epoch, candidates = starts[-1]
            completed = record.get("frontiers", {}).get(str(current_epoch))
            if completed and completed.get("decision") == "continue":
                current_epoch = target_after(record, current_epoch)
                selected = str(completed["selectedWd"])
                ladder = [str(value) for value in record["wdLadder"]]
                selected_index = ladder.index(selected)
                candidates = [selected]
                if selected_index + 1 < len(ladder):
                    candidates.append(ladder[selected_index + 1])
            record["activeEpoch"] = current_epoch
            record["activeWds"] = candidates
        stages = [
            (int(epoch), wd)
            for parsed_model, parsed_batch, epoch, wd in STAGE_START.findall(logs)
            if parsed_model == model and int(parsed_batch) == batch
        ]
        if stages:
            record["activeWd"] = stages[-1][1]
        if state == "failed":
            record["needsAttention"] = True
            record["reason"] = (
                "Beaker exhausted the configured automatic retries. Atomic stage and "
                "frontier files preserve all completed work for an exact continuation."
            )

    step_values = TRAIN_STEP.findall(logs)
    train_values = TRAIN_LOSS.findall(logs)
    if step_values and record.get("activeEpoch") is not None:
        step, total = step_values[-1]
        step_value = int(step.replace(",", ""))
        total_value = int(total.replace(",", ""))
        record["progress"] = {
            "step": step_value,
            "totalSteps": total_value,
            "percent": round(100 * step_value / total_value, 1),
        }
        if train_values:
            record["progress"]["latestTrain"] = float(train_values[-1])
    elif record.get("activeEpoch") is None:
        record.pop("progress", None)

    completed_frontiers = sorted(int(epoch) for epoch in record.get("frontiers", {}))
    detail = ""
    if record.get("progress"):
        progress = record["progress"]
        detail = f" {progress['percent']:g}% ({progress['step']}/{progress['totalSteps']})"
    return (
        f"{model} BS{batch} {record['status']}; frontiers={completed_frontiers or 'none'}; "
        f"current=E{record.get('activeEpoch') or 'done'} WD{record.get('activeWd', '—')}"
        f"{detail}; {experiment_id}"
    )


def refresh_model(model: str, path: Path) -> list[str]:
    report = json.loads(path.read_text())
    records = report.get("adaptiveDrWtEmbedWdChains", [])
    if len(records) != 4:
        raise RuntimeError(f"expected four {model} adaptive DR+WT+EmbedWD chains, found {len(records)}")
    summaries = [update_chain(model, record) for record in records]
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )
    return summaries


def main() -> None:
    summaries: list[str] = []
    for model, path in REPORTS.items():
        summaries.extend(refresh_model(model, path))
    print("\n".join(summaries))


if __name__ == "__main__":
    main()
