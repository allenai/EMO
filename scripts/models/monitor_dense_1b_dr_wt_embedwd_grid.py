#!/usr/bin/env python3
"""Refresh the deferred Dense-1B BS128/256 DR+WT+EmbedWD grid chains."""

from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_PATH = Path("reports/0802/data/wsd_data_loader_1b.json")
REPORT_JS_PATH = REPORT_PATH.with_suffix(".js")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STAGE_RESULT = re.compile(
    r"DENSE1B_DRWTEMBWD_GRID_STAGE_RESULT bs=([0-9]+) epoch=([0-9]+) "
    r"lr=([^ ]+) wd=([^ ]+) json=(\{.*\})$",
    re.MULTILINE,
)
FRONTIER_RESULT = re.compile(
    r"DENSE1B_DRWTEMBWD_GRID_FRONTIER_RESULT bs=([0-9]+) epoch=([0-9]+) "
    r"json=(\{.*\})$",
    re.MULTILINE,
)
FRONTIER_START = re.compile(
    r"DENSE1B_DRWTEMBWD_GRID_FRONTIER_START bs=([0-9]+) epoch=([0-9]+) candidates=([^\s]+)"
)
STAGE_START = re.compile(
    r"DENSE1B_DRWTEMBWD_GRID_STAGE_START bs=([0-9]+) epoch=([0-9]+) "
    r"lr=([^ ]+) wd=([^ ]+) previous_epoch=([0-9]+) output=([^\s]+)"
)
SATURATED = re.compile(
    r"DENSE1B_DRWTEMBWD_GRID_SATURATED bs=([0-9]+) epoch=([0-9]+) "
    r"lr=([^ ]+) wd=([^ ]+) reason=([^\s]+)"
)
TRAIN_STEP = re.compile(r"\[step=([0-9,]+)/([0-9,]+),epoch=")
TRAIN_LOSS = re.compile(r"\btrain/CE loss=([^\s]+)")
WANDB_RUN = re.compile(r"wandb:\s+(?:setting up run |.*?/runs/)([a-z0-9]{8})")


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
    if any("started" in status and not terminal.intersection(status) for status in statuses):
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


def write_report(report: dict[str, Any]) -> None:
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS_PATH.write_text(
        "window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def coordinate_run(report: dict[str, Any], batch: int, lr: str, wd: str) -> dict[str, Any]:
    identifier = f"drwtembwd{batch}-lr{lr}-wd{wd}"
    matches = [record for record in report.get("runs", []) if record.get("id") == identifier]
    if len(matches) != 1:
        raise RuntimeError(f"expected one registered coordinate {identifier}, found {len(matches)}")
    return matches[0]


def active_health(logs: str, chain: dict[str, Any], state: str) -> dict[str, Any] | None:
    starts = list(STAGE_START.finditer(logs))
    if not starts:
        return None
    batch, epoch, lr, wd, previous_epoch, output = starts[-1].groups()
    segment = logs[starts[-1].start() :]
    steps = [
        (int(step.replace(",", "")), int(total.replace(",", "")))
        for step, total in TRAIN_STEP.findall(segment)
    ]
    raw_losses = TRAIN_LOSS.findall(segment)
    finite_losses: list[float] = []
    nonfinite: list[str] = []
    for raw in raw_losses:
        try:
            value = float(raw)
        except ValueError:
            nonfinite.append(raw)
            continue
        if math.isfinite(value):
            finite_losses.append(value)
        else:
            nonfinite.append(raw)
    run_ids = WANDB_RUN.findall(segment)
    critical = ["nonfinite-training-loss"] if nonfinite else []
    expected = f"/bs{batch}_dr_wt_embwd_lr{lr}_wd{wd}"
    if not output.endswith(expected):
        critical.append("stage-output-path-mismatch")
    if int(previous_epoch) > 0 and "Loading checkpoint from '" not in segment:
        critical.append("missing-resume-checkpoint-load")
    return {
        "status": "critical" if critical else "healthy" if steps else "pending",
        "checkedAt": datetime.now(tz=UTC).isoformat(),
        "run": run_ids[-1] if run_ids else None,
        "url": (f"https://wandb.ai/ai2-llm/sewonm-icsl/runs/{run_ids[-1]}" if run_ids else None),
        "batchSequences": int(batch),
        "epoch": int(epoch),
        "lr": lr,
        "wd": wd,
        "output": output,
        "latestStep": steps[-1][0] if steps else None,
        "totalSteps": steps[-1][1] if steps else None,
        "latestTrain": finite_losses[-1] if finite_losses else None,
        "nonfiniteLosses": nonfinite[:5],
        "criticalSignals": critical,
        "shouldRecover": bool(critical),
        "beakerState": state,
    }


def refresh_chain(report: dict[str, Any], chain: dict[str, Any]) -> str:
    experiment_id = chain.get("experiment")
    if not experiment_id:
        return (
            f"1b BS{chain['batchSequences']} {chain.get('status', 'planned')}; trigger="
            f"{chain.get('completedSmallChainsAtLastCheck', 0)}/{chain.get('triggerThreshold', 5)}"
        )
    batch = int(chain["batchSequences"])
    inspected = inspect_experiment(str(experiment_id))
    state = experiment_state(inspected)
    jobs = [job for job in inspected.get("jobs") or [] if job.get("id")]
    chain["status"] = state
    chain["beakerStatus"] = state
    if jobs:
        chain["job"] = jobs[-1]["id"]
        chain["jobs"] = [job["id"] for job in jobs]
    logs = ""
    if state in {"running", "complete", "failed"}:
        logs = ANSI.sub("", run(["beaker", "experiment", "logs", str(experiment_id)]))
    for parsed_batch, epoch, lr, wd, payload in STAGE_RESULT.findall(logs):
        if int(parsed_batch) != batch:
            continue
        result = json.loads(payload)
        result.update(
            {
                "status": "complete",
                "epoch": int(epoch),
                "lr": lr,
                "wd": wd,
                "experiment": experiment_id,
                "beaker": experiment_id,
                "job": chain.get("job"),
                "revision": chain.get("revision"),
            }
        )
        coordinate = coordinate_run(report, batch, lr, wd)
        coordinate.setdefault("results", {})[str(int(epoch))] = result
        coordinate.setdefault("attemptedEpochs", [])
        if int(epoch) not in coordinate["attemptedEpochs"]:
            coordinate["attemptedEpochs"].append(int(epoch))
    for parsed_batch, epoch, payload in FRONTIER_RESULT.findall(logs):
        if int(parsed_batch) != batch:
            continue
        frontier = json.loads(payload)
        frontier.update(
            {
                "status": "complete",
                "experiment": experiment_id,
                "beaker": experiment_id,
                "job": chain.get("job"),
            }
        )
        chain.setdefault("frontiers", {})[str(int(epoch))] = frontier
        targets = {int(value) for value in report.get("targetEpochs", [])}
        targets.add(int(epoch))
        report["targetEpochs"] = sorted(targets)
    starts = [
        (int(epoch), candidates)
        for parsed_batch, epoch, candidates in FRONTIER_START.findall(logs)
        if int(parsed_batch) == batch
    ]
    stages = [
        (int(epoch), lr, wd)
        for parsed_batch, epoch, lr, wd, _, _ in STAGE_START.findall(logs)
        if int(parsed_batch) == batch
    ]
    if starts:
        chain["activeEpoch"] = starts[-1][0]
    if stages:
        chain["activeCoordinate"] = {"lr": stages[-1][1], "wd": stages[-1][2]}
        coordinate = coordinate_run(report, batch, stages[-1][1], stages[-1][2])
        coordinate["activeEpoch"] = stages[-1][0]
        coordinate["status"] = state
        run_ids = WANDB_RUN.findall(logs[logs.rfind("DENSE1B_DRWTEMBWD_GRID_STAGE_START") :])
        if run_ids:
            coordinate["wandb"] = run_ids[-1]
    health = active_health(logs, chain, state)
    if health:
        chain["wandbHealth"] = health
        if health["shouldRecover"]:
            chain["needsAttention"] = True
    saturated = [
        (int(epoch), lr, wd, reason)
        for parsed_batch, epoch, lr, wd, reason in SATURATED.findall(logs)
        if int(parsed_batch) == batch
    ]
    if saturated:
        epoch, lr, wd, reason = saturated[-1]
        chain.update(
            {
                "status": "complete",
                "beakerStatus": state,
                "activeEpoch": None,
                "activeCoordinate": None,
                "saturatedEpoch": epoch,
                "selectedLr": lr,
                "selectedWd": wd,
                "stopReason": reason,
                "reason": (
                    f"The persistent BS{batch} grid stopped at E{epoch}; every requested "
                    "coordinate retained its own canonical output directory."
                ),
            }
        )
        for item in chain["coordinates"]:
            coordinate = coordinate_run(report, batch, str(item["lr"]), str(item["wd"]))
            coordinate["status"] = "complete"
            coordinate["activeEpoch"] = None
    elif state == "failed":
        chain["needsAttention"] = True
        chain["reason"] = (
            "Beaker exhausted eight automatic task retries. Atomic stage/frontier files "
            "preserve completed work for exact recovery."
        )
    step_values = TRAIN_STEP.findall(logs)
    if step_values and chain.get("activeEpoch") is not None:
        step, total = step_values[-1]
        step_value = int(step.replace(",", ""))
        total_value = int(total.replace(",", ""))
        chain["progress"] = {
            "step": step_value,
            "totalSteps": total_value,
            "percent": round(100 * step_value / total_value, 1),
        }
    elif chain.get("activeEpoch") is None:
        chain.pop("progress", None)
    progress = chain.get("progress") or {}
    detail = (
        f" {progress['percent']:g}% ({progress['step']}/{progress['totalSteps']})"
        if progress
        else ""
    )
    active = chain.get("activeCoordinate") or {}
    health_detail = chain.get("wandbHealth", {}).get("status", "pending")
    return (
        f"1b BS{batch} {chain['status']}; frontiers="
        f"{sorted(int(value) for value in chain.get('frontiers', {})) or 'none'}; "
        f"current=E{chain.get('activeEpoch') or 'done'} LR{active.get('lr', '—')} "
        f"WD{active.get('wd', '—')}{detail}; wandb={health_detail}; {experiment_id}"
    )


def main() -> None:
    report = json.loads(REPORT_PATH.read_text())
    chains = report.get("drWtEmbedWdGridChains", [])
    if len(chains) != 2:
        raise RuntimeError(f"expected two registered 1B LR/WD grid chains, found {len(chains)}")
    summaries = [refresh_chain(report, chain) for chain in chains]
    write_report(report)
    print("\n".join(summaries))


if __name__ == "__main__":
    main()
