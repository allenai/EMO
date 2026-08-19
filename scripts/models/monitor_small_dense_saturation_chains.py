#!/usr/bin/env python3
"""Refresh the two persistent small-dense BS32 saturation chains."""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

REPORTS = {
    "474m": Path("reports/0802/data/wsd_batch_size_474m.json"),
    "153m": Path("reports/0802/data/wsd_batch_size_153m.json"),
}
SEQUENCE_LENGTH = 4096
TOKENS_PER_EPOCH = 1_000_000_000
DECAY_FRACTION = 0.1
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RESULT = re.compile(
    r"SMALL_SATURATION_STAGE_RESULT model=(153m|474m) epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
START = re.compile(r"SMALL_SATURATION_STAGE_START model=(153m|474m) epoch=([0-9]+)")
SATURATED = re.compile(r"SMALL_SATURATION_SATURATED model=(153m|474m) epoch=([0-9]+)")
TRAIN_STEP = re.compile(r"\[step=([0-9,]+)/([0-9,]+),epoch=")
TRAIN_LOSS = re.compile(r"\btrain/CE loss=([0-9]+(?:\.[0-9]+)?)")
START_CONTEXT = re.compile(
    r"SMALL_SATURATION_STAGE_START model=(153m|474m) epoch=([0-9]+) "
    r"previous_epoch=([0-9]+) previous_validation=([^ ]+) output=([^\s]+)"
)
WANDB_RUN = re.compile(r"wandb:\s+(?:setting up run |.*?/runs/)([a-z0-9]{8})")
SAVE_FOLDER = re.compile(r"save_folder='([^']+)'")
LOAD_CHECKPOINT = re.compile(r"Loading checkpoint from '([^']+)'", re.IGNORECASE)
METRIC_HEADER = re.compile(
    r"^(?P<timestamp>[0-9T:.+-]+Z)\s+.*console_logger:67.*"
    r"\[step=(?P<step>[0-9,]+)/(?P<total>[0-9,]+),epoch=",
)
LOSS_VALUE = re.compile(r"\btrain/CE loss=([^\s]+)")


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


def total_step(epoch: int) -> int:
    return math.ceil(epoch * TOKENS_PER_EPOCH / (32 * SEQUENCE_LENGTH))


def stable_step(epoch: int) -> int:
    end = total_step(epoch)
    return end - round(DECAY_FRACTION * end) - 1


def latest_job(experiment: dict[str, Any]) -> dict[str, Any]:
    jobs = experiment.get("jobs") or []
    return jobs[-1] if jobs else {}


def completed_results(logs: str, model: str) -> dict[int, dict[str, Any]]:
    parsed: dict[int, dict[str, Any]] = {}
    for parsed_model, epoch, payload in RESULT.findall(ANSI.sub("", logs)):
        if parsed_model == model:
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


def wandb_health(logs: str, sweep: dict[str, Any], state: str) -> dict[str, Any] | None:
    starts = list(START_CONTEXT.finditer(logs))
    if not starts:
        return None
    start = starts[-1]
    model, epoch, previous_epoch, _, output = start.groups()
    segment = logs[start.start() :]
    run_ids = WANDB_RUN.findall(segment)
    run_id = run_ids[-1] if run_ids else None
    expected_output = str(sweep.get("canonicalOutput") or sweep.get("output") or "")
    save_folders = SAVE_FOLDER.findall(segment)
    save_folder = save_folders[-1] if save_folders else None
    load_paths = LOAD_CHECKPOINT.findall(segment)
    load_path = load_paths[-1] if load_paths else None
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
    if "dynamic_repacking=True" in segment:
        critical.append("unexpected-dynamic-repacking")
    if "tie_embeddings=True" in segment:
        critical.append("unexpected-weight-tying")

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
        "batchSequences": 32,
        "epoch": int(epoch),
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


def update_result(
    sweep: dict[str, Any], epoch: int, result: dict[str, Any], job: dict[str, Any]
) -> None:
    previous_epoch = epoch - int(sweep["epochIncrement"])
    output = str(sweep["output"])
    result.update(
        {
            "epoch": epoch,
            "status": "complete",
            "beaker": sweep["experiment"],
            "experiment": sweep["experiment"],
            "job": job.get("id"),
            "revision": sweep.get("revision"),
            "output": output,
            "sourceCheckpoint": f"{output}/step{stable_step(previous_epoch)}",
            "resumeCheckpoint": f"{output}/step{stable_step(epoch)}",
            "preDecayCheckpoint": f"{output}/step{stable_step(epoch)}",
            "endpointCheckpoint": f"{output}/step{total_step(epoch)}",
            "retainedPreDecaySteps": [
                stable_step(value) for value in range(previous_epoch + 1, epoch + 1)
            ],
            "sequential": True,
        }
    )
    previous_validation = float(result["previousValidationExact"])
    validation = float(result["validationExact"])
    if result.get("saturationDecision") == "stop":
        result["reason"] = (
            f"Held-out validation CE{validation:.3f} did not strictly improve the "
            f"preceding CE{previous_validation:.3f}; retained as the first saturation "
            "endpoint and stopped the persistent chain."
        )
    else:
        result["reason"] = (
            f"Held-out validation CE{validation:.3f} strictly improved the preceding "
            f"CE{previous_validation:.3f}; the persistent job continued from this "
            "endpoint's exact pre-decay checkpoint in the same output directory."
        )
    sweep.setdefault("results", {})[str(epoch)] = result


def refresh_model(model: str, path: Path) -> str:
    report = json.loads(path.read_text())
    matches = [
        sweep for sweep in report.get("batchSweeps", []) if sweep.get("saturationChain") is True
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {model} saturation chain, found {len(matches)}")
    sweep = matches[0]
    experiment = inspect_experiment(str(sweep["experiment"]))
    state = experiment_state(experiment)
    job = latest_job(experiment)
    jobs = [
        candidate.get("id") for candidate in experiment.get("jobs") or [] if candidate.get("id")
    ]
    sweep["status"] = state
    sweep["beakerStatus"] = state
    if jobs:
        sweep["job"] = jobs[-1]
        sweep["jobs"] = jobs

    logs = ""
    if state in {"running", "complete", "failed"}:
        logs = run(["beaker", "experiment", "logs", str(sweep["experiment"])])
    parsed_results = completed_results(logs, model)
    for epoch, result in parsed_results.items():
        update_result(sweep, epoch, result, job)

    clean = ANSI.sub("", logs)
    health = wandb_health(clean, sweep, state)
    if health is not None:
        sweep["wandbHealth"] = health
        if health["shouldRecover"]:
            sweep["needsAttention"] = True
    saturated_epochs = [
        int(epoch) for parsed_model, epoch in SATURATED.findall(clean) if parsed_model == model
    ]
    starts = [int(epoch) for parsed_model, epoch in START.findall(clean) if parsed_model == model]
    completed_epochs = sorted(int(epoch) for epoch in sweep.get("results", {}))
    if saturated_epochs:
        saturated_epoch = saturated_epochs[-1]
        sweep["status"] = "complete"
        sweep["activeEpoch"] = None
        sweep["saturatedEpoch"] = saturated_epoch
        sweep["stopReason"] = (
            f"Held-out validation stopped strictly improving at E{saturated_epoch}."
        )
        sweep["reason"] = (
            f"Persistent BS32 chain saturated at E{saturated_epoch}; all completed "
            "endpoints used the same canonical output directory."
        )
    elif starts:
        current = starts[-1]
        if current in completed_epochs:
            current += int(sweep["epochIncrement"])
        sweep["activeEpoch"] = current
    elif state == "failed":
        sweep["activeEpoch"] = None
        sweep["needsAttention"] = True
        sweep["reason"] = (
            "Beaker exhausted the configured automatic task retries before saturation; "
            "the completed atomic stages remain valid."
        )

    step_values = TRAIN_STEP.findall(clean)
    train_values = TRAIN_LOSS.findall(clean)
    if step_values and sweep.get("activeEpoch") is not None:
        step, total = step_values[-1]
        step_value = int(step.replace(",", ""))
        total_value = int(total.replace(",", ""))
        sweep["progress"] = {
            "step": step_value,
            "totalSteps": total_value,
            "percent": round(100 * step_value / total_value, 1),
        }
        if train_values:
            sweep["progress"]["latestTrain"] = float(train_values[-1])
    elif sweep.get("activeEpoch") is None:
        sweep.pop("progress", None)

    targets = {
        int(float(epoch)) if float(epoch).is_integer() else float(epoch)
        for epoch in report.get("targetEpochs", [])
    }
    targets.update(completed_epochs)
    report["targetEpochs"] = sorted(targets)
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )

    detail = ""
    if sweep.get("progress"):
        progress = sweep["progress"]
        detail = f" {progress['percent']:g}% ({progress['step']}/{progress['totalSteps']})"
        if "latestTrain" in progress:
            detail += f" train={progress['latestTrain']:.3f}"
    health_detail = ""
    if sweep.get("wandbHealth"):
        health = sweep["wandbHealth"]
        health_detail = f"; wandb={health.get('run') or 'pending'} {health['status']}"
    return (
        f"{model} {sweep['status']}; completed={completed_epochs or 'none'}; "
        f"current=E{sweep.get('activeEpoch') or 'done'}{detail}{health_detail}; "
        f"{sweep['experiment']}"
    )


def main() -> None:
    print("\n".join(refresh_model(model, path) for model, path in REPORTS.items()))


if __name__ == "__main__":
    main()
