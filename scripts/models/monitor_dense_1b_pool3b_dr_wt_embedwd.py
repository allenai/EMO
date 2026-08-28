#!/usr/bin/env python3
"""Refresh the isolated Dense-1B Pool-3B report from Beaker logs."""

from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT = Path("reports/0802/data/wsd_data_loader_1b_pool3b_drwtembwd.json")
REPORT_JS = REPORT.with_suffix(".js")
POLICY = "dense_1b_pool3b_dr_wt_embedwd_postdecay_saturation_v1"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RESULT = re.compile(
    r"DENSE1B_PDPOST_RESULT bs=([0-9]+) lr=([^ ]+) wd=([^ ]+) "
    r"phase=(pre_decay|post_decay) epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
STAGE = re.compile(
    r"DENSE1B_PDPOST_STAGE_START bs=([0-9]+) lr=([^ ]+) wd=([^ ]+) "
    r"phase=([^ ]+) epoch=([0-9]+)(?: previous_epoch=([0-9]+))? "
    r"source=([^ ]+) output=([^\s]+)"
)
TERMINAL = re.compile(
    r"DENSE1B_PDPOST_SELECTED bs=([0-9]+) lr=([^ ]+) wd=([^ ]+) "
    r"trigger_epoch=([0-9]+) selected_epoch=([0-9]+) validation=([^ ]+) "
    r"json=(\{.*\})$",
    re.MULTILINE,
)
TRAIN_STEP = re.compile(r"\[step=([0-9,]+)/([0-9,]+),epoch=")
TRAIN_LOSS = re.compile(r"\btrain/CE loss=([^\s]+)")
WANDB = re.compile(r"https://wandb\.ai/[^\s]+/runs/([a-zA-Z0-9_-]+)")


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, text=True, capture_output=True).stdout


def write_report(report: dict[str, Any]) -> None:
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_POOL3B_DATA="
        + json.dumps(report, separators=(",", ":"))
        + ";\n"
    )


def inspect(experiment: str) -> dict[str, Any]:
    payload = json.loads(
        command(["beaker", "experiment", "inspect", experiment, "--format", "json"])
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"expected exactly one experiment for {experiment}")
    return payload[0]


def state(payload: dict[str, Any]) -> str:
    jobs = payload.get("jobs") or []
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
    if all(terminal.intersection(status) for status in statuses):
        return "failed"
    return "submitted"


def health(run: dict[str, Any], logs: str, beaker_state: str) -> dict[str, Any]:
    starts = list(STAGE.finditer(logs))
    if not starts:
        return {
            "status": "pending",
            "checkedAt": datetime.now(tz=UTC).isoformat(),
            "beakerState": beaker_state,
            "criticalSignals": [],
        }
    batch, lr, wd, phase, epoch, previous, source, output = starts[-1].groups()
    segment = logs[starts[-1].start() :]
    steps = [
        (int(current.replace(",", "")), int(total.replace(",", "")))
        for current, total in TRAIN_STEP.findall(segment)
    ]
    nonfinite: list[str] = []
    for raw in TRAIN_LOSS.findall(segment):
        try:
            if not math.isfinite(float(raw)):
                nonfinite.append(raw)
        except ValueError:
            nonfinite.append(raw)
    critical: list[str] = []
    if nonfinite:
        critical.append("nonfinite-training-loss")
    if "eval" not in phase and not output.startswith(str(run["output"])):
        critical.append("stage-output-path-mismatch")
    if previous and int(previous) > 0 and "Loading checkpoint from '" not in segment:
        critical.append("missing-resume-checkpoint-load")
    wandb = WANDB.findall(segment)
    return {
        "status": "critical" if critical else "healthy" if steps else "pending",
        "checkedAt": datetime.now(tz=UTC).isoformat(),
        "beakerState": beaker_state,
        "batchSequences": int(batch),
        "lr": lr,
        "wd": wd,
        "phase": phase,
        "epoch": int(epoch),
        "source": source,
        "output": output,
        "latestStep": steps[-1][0] if steps else None,
        "totalSteps": steps[-1][1] if steps else None,
        "run": wandb[-1] if wandb else None,
        "criticalSignals": critical,
    }


def refresh(run: dict[str, Any]) -> str:
    experiment = run.get("experiment")
    if not experiment:
        return f"BS{run['batchSequences']}: planned"
    payload = inspect(str(experiment))
    beaker_state = state(payload)
    jobs = [job for job in payload.get("jobs") or [] if job.get("id")]
    run["beakerStatus"] = beaker_state
    if jobs:
        run["job"] = jobs[-1]["id"]
        run["jobs"] = [job["id"] for job in jobs]
    logs = ""
    if beaker_state in {"running", "complete", "failed"}:
        logs = ANSI.sub("", command(["beaker", "experiment", "logs", str(experiment)]))
    for batch, lr, wd, phase, epoch, raw in RESULT.findall(logs):
        if int(batch) != int(run["batchSequences"]):
            continue
        result = json.loads(raw)
        result.update(
            {
                "epoch": int(epoch),
                "lr": lr,
                "wd": wd,
                "experiment": experiment,
                "job": run.get("job"),
                "revision": run.get("revision"),
            }
        )
        target = run.setdefault(
            "preDecayResults" if phase == "pre_decay" else "postDecayResults", {}
        )
        target[str(int(epoch))] = result
        run.setdefault("results", {})[str(int(epoch))] = {
            "preDecay": run.get("preDecayResults", {}).get(str(int(epoch))),
            "postDecay": run.get("postDecayResults", {}).get(str(int(epoch))),
        }
    starts = [
        match.groups()
        for match in STAGE.finditer(logs)
        if int(match.group(1)) == int(run["batchSequences"])
    ]
    if starts:
        _, _, _, phase, epoch, _, source, output = starts[-1]
        run.update(
            {
                "activePhase": phase,
                "activeEpoch": int(epoch),
                "activeSource": source,
                "activeOutput": output,
            }
        )
    terminal = [
        match.groups()
        for match in TERMINAL.finditer(logs)
        if int(match.group(1)) == int(run["batchSequences"])
    ]
    if terminal:
        _, _, _, trigger, selected, _, raw = terminal[-1]
        run.update(
            {
                "status": "complete",
                "activePhase": None,
                "activeEpoch": None,
                "postDecaySelection": json.loads(raw),
                "selectedPostDecayEpoch": int(selected),
                "saturationTriggerEpoch": int(trigger),
            }
        )
    elif beaker_state == "failed":
        run["status"] = "failed"
        run["needsAttention"] = True
    elif beaker_state in {"running", "scheduled", "submitted"}:
        run["status"] = beaker_state
    elif beaker_state == "complete" and run.get("status") != "complete":
        run["status"] = "complete_without_selection"
        run["needsAttention"] = True
    run["wandbHealth"] = health(run, logs, beaker_state)
    return (
        f"BS{run['batchSequences']}: {run['status']} "
        f"phase={run.get('activePhase') or 'terminal'} "
        f"E{run.get('activeEpoch') or run.get('saturationTriggerEpoch') or '-'}"
    )


def main() -> None:
    report = json.loads(REPORT.read_text())
    runs = [run for run in report.get("runs", []) if run.get("policy") == POLICY]
    if len(runs) != 2:
        raise RuntimeError("expected exactly two Pool-3B coordinates")
    for run in runs:
        print(refresh(run))
    write_report(report)


if __name__ == "__main__":
    main()
