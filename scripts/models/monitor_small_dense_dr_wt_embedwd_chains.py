#!/usr/bin/env python3
"""Refresh the eight small-dense adaptive DR+WT+EmbedWD chains."""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise
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
STAGE_CONTEXT = re.compile(
    r"SMALL_DRWTEMBWD_STAGE_START model=(153m|474m) bs=([0-9]+) "
    r"epoch=([0-9]+) wd=([^ ]+) previous_epoch=([0-9]+) output=([^\s]+)"
)
WANDB_RUN = re.compile(r"wandb:\s+(?:setting up run |.*?/runs/)([a-z0-9]{8})")
SAVE_FOLDER = re.compile(r"save_folder='([^']+)'")
LOAD_CHECKPOINT = re.compile(r"Loading checkpoint from '([^']+)'", re.IGNORECASE)
INTERLEAVED_LOG_PREFIX = re.compile(
    r"\n[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.+-]+Z\s+"
)
METRIC_HEADER = re.compile(
    r"^(?P<timestamp>[0-9T:.+-]+Z)\s+.*console_logger:67.*"
    r"\[step=(?P<step>[0-9,]+)/(?P<total>[0-9,]+),epoch=",
)
LOSS_VALUE = re.compile(r"\btrain/CE loss=([^\s]+)")
MAX_WEIGHT_DECAY = Decimal("1.0")


def normalize_log_value(value: str | None) -> str | None:
    if value is None:
        return None
    return INTERLEAVED_LOG_PREFIX.sub("", value)


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


def parse_metric_samples(segment: str) -> tuple[list[int], list[float], list[str], str | None, int | None]:
    steps: list[int] = []
    losses: list[float] = []
    nonfinite: list[str] = []
    current_step: int | None = None
    last_timestamp: str | None = None
    total_steps: int | None = None
    for line in segment.splitlines():
        if "console_logger:" in line and "[step=" in line:
            header = METRIC_HEADER.search(line)
            if header is None:
                current_step = None
                continue
            current_step = int(header.group("step").replace(",", ""))
            total_steps = int(header.group("total").replace(",", ""))
            last_timestamp = header.group("timestamp")
            steps.append(current_step)
            continue
        if current_step is None:
            continue
        match = LOSS_VALUE.search(line)
        if match is None:
            continue
        raw = match.group(1)
        try:
            value = float(raw)
        except ValueError:
            nonfinite.append(raw)
            current_step = None
            continue
        if math.isfinite(value):
            losses.append(value)
        else:
            nonfinite.append(raw)
        current_step = None
    return steps, losses, nonfinite, last_timestamp, total_steps


def wandb_health(logs: str, record: dict[str, Any], state: str) -> dict[str, Any] | None:
    starts = list(STAGE_CONTEXT.finditer(logs))
    if not starts:
        return None
    start = starts[-1]
    model, batch, epoch, wd, previous_epoch, output = start.groups()
    segment = logs[start.start() :]
    run_ids = WANDB_RUN.findall(segment)
    run_id = run_ids[-1] if run_ids else None
    expected_output = str(record.get("outputByWd", {}).get(wd, ""))
    save_folders = SAVE_FOLDER.findall(segment)
    save_folder = normalize_log_value(save_folders[-1] if save_folders else None)
    load_paths = LOAD_CHECKPOINT.findall(segment)
    load_path = normalize_log_value(load_paths[-1] if load_paths else None)
    steps, losses, nonfinite, last_timestamp, total_steps = parse_metric_samples(segment)

    critical: list[str] = []
    warnings: list[str] = []
    if expected_output and output != expected_output:
        critical.append("stage-output-path-mismatch")
    if save_folder and save_folder != output:
        critical.append("trainer-save-path-mismatch")
    if load_path and not load_path.startswith(output.rstrip("/") + "/"):
        critical.append("checkpoint-load-path-mismatch")
    if int(previous_epoch) > 0 and steps and load_path is None:
        critical.append("missing-resume-checkpoint-load")

    regressions = [(left, right) for left, right in pairwise(steps) if right < left]
    if regressions:
        critical.append("wandb-step-regression")
    if nonfinite:
        critical.append("nonfinite-training-loss")

    recent_median: float | None = None
    baseline_median: float | None = None
    maximum: float | None = None
    spike_count = 0
    if losses:
        maximum = max(losses)
        recent = losses[-50:]
        baseline_pool = losses[-250:-50] or losses[:-50] or losses
        recent_median = statistics.median(recent)
        baseline_median = statistics.median(baseline_pool)
        spike_count = sum(value > baseline_median + 0.75 for value in losses)
        if len(recent) >= 25 and recent_median > baseline_median + 0.5:
            critical.append("sustained-training-loss-shift")
        elif spike_count:
            warnings.append("isolated-finite-loss-spikes")

    stalled_minutes: float | None = None
    if last_timestamp and state == "running" and steps and total_steps and steps[-1] < total_steps:
        last_metric = datetime.fromisoformat(last_timestamp.replace("Z", "+00:00"))
        stalled_minutes = (datetime.now(tz=UTC) - last_metric).total_seconds() / 60
        if stalled_minutes > 45:
            critical.append("training-metrics-stalled")

    if steps and run_id is None:
        warnings.append("wandb-run-id-not-found-in-job-log")
    status = "critical" if critical else "warning" if warnings else "healthy"
    if not steps and not critical:
        status = "pending"
    path_recovery_ready = bool(
        expected_output
        and expected_output != output
        and {
            "stage-output-path-mismatch",
            "trainer-save-path-mismatch",
            "checkpoint-load-path-mismatch",
        }.intersection(critical)
    )
    return {
        "status": status,
        "checkedAt": datetime.now(tz=UTC).isoformat(),
        "run": run_id,
        "url": (
            f"https://wandb.ai/ai2-llm/sewonm-icsl/runs/{run_id}"
            if run_id
            else None
        ),
        "model": model,
        "batchSequences": int(batch),
        "epoch": int(epoch),
        "wd": wd,
        "expectedOutput": expected_output or None,
        "stageOutput": output,
        "saveFolder": save_folder,
        "loadPath": load_path,
        "samples": len(losses),
        "firstStep": steps[0] if steps else None,
        "latestStep": steps[-1] if steps else None,
        "totalSteps": total_steps,
        "recentLossMedian": round(recent_median, 6) if recent_median is not None else None,
        "baselineLossMedian": round(baseline_median, 6) if baseline_median is not None else None,
        "maxLoss": maximum,
        "isolatedSpikeCount": spike_count,
        "stepRegressions": regressions[:5],
        "nonfiniteLosses": nonfinite[:5],
        "stalledMinutes": round(stalled_minutes, 1) if stalled_minutes is not None else None,
        "warnings": warnings,
        "criticalSignals": critical,
        "shouldRecover": bool(critical),
        "automaticPathRecoveryReady": path_recovery_ready,
        "recoveryOutput": expected_output if path_recovery_ready else None,
    }


def update_chain(model: str, record: dict[str, Any]) -> str:
    batch = int(record["batchSequences"])
    record["wdLadder"] = [
        str(value)
        for value in record["wdLadder"]
        if Decimal(str(value)) <= MAX_WEIGHT_DECAY
    ]
    record["outputByWd"] = {
        str(wd): output
        for wd, output in record.get("outputByWd", {}).items()
        if Decimal(str(wd)) <= MAX_WEIGHT_DECAY
    }
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

    health = wandb_health(logs, record, state)
    if health is not None:
        record["wandbHealth"] = health
        if health["shouldRecover"]:
            record["needsAttention"] = True

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
                ladder = [
                    str(value)
                    for value in record["wdLadder"]
                    if Decimal(str(value)) <= MAX_WEIGHT_DECAY
                ]
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
    health_detail = ""
    if record.get("wandbHealth"):
        health = record["wandbHealth"]
        health_detail = f"; wandb={health.get('run') or 'pending'} {health['status']}"
    return (
        f"{model} BS{batch} {record['status']}; frontiers={completed_frontiers or 'none'}; "
        f"current=E{record.get('activeEpoch') or 'done'} WD{record.get('activeWd', '—')}"
        f"{detail}{health_detail}; {experiment_id}"
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
