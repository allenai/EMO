#!/usr/bin/env python3
"""Refresh the Dense-153M BS32 LR/WD hybrid in its report JSON/JS."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT = Path("reports/0802/data/wsd_batch_size_153m.json")
POLICY = "dense_153m_bs32_lr_wd_hybrid_v1"
EXPECTED_BEAKER_AUTHOR = "sewonm"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RESULT = re.compile(
    r"DENSE153M_BS32_HYBRID_POST_RESULT phase=(probe|followup) lr=([^ ]+) wd=([^ ]+) "
    r"epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
PD_START = re.compile(
    r"DENSE153M_BS32_HYBRID_PD_START phase=(probe|followup) lr=([^ ]+) wd=([^ ]+) epoch=([0-9]+)"
)
POST_START = re.compile(
    r"DENSE153M_BS32_HYBRID_POST_START phase=(probe|followup) lr=([^ ]+) wd=([^ ]+) epoch=([0-9]+)"
)
FOLLOWUP_START = re.compile(
    r"DENSE153M_BS32_HYBRID_FOLLOWUP_START lr=([^ ]+) wd=([^ ]+) output=([^ ]+)"
)
DECISION = re.compile(r"DENSE153M_BS32_HYBRID_LR_DECISION json=(\{.*\})$", re.MULTILINE)
COMPLETE = re.compile(
    r"DENSE153M_BS32_HYBRID_WORKFLOW_COMPLETE selectedLearningRate=([^ ]+) followupWeightDecay=([^ ]+)"
)
E20_GATE_RESULT = re.compile(
    r"DENSE153M_BS32_E20_GATE_RESULT label=(probe|baseline) json=(\{.*\})$", re.MULTILINE
)
E20_GATE_START = re.compile(
    r"DENSE153M_BS32_E20_GATE_POST_START label=(probe|baseline) source=([^ ]+) output=([^ ]+)"
)
E20_GATE_HOLD = re.compile(r"DENSE153M_BS32_E20_GATE_COMPLETE action=hold_no_followup")


def run(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout


def inspect_experiment(experiment: str) -> dict[str, Any]:
    payload = json.loads(run(["beaker", "experiment", "inspect", experiment, "--format", "json"]))
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"expected one experiment for {experiment}")
    record = payload[0]
    author = (record.get("author") or {}).get("name")
    if author != EXPECTED_BEAKER_AUTHOR:
        raise RuntimeError(
            f"refusing experiment {experiment} owned by {author!r}; "
            f"expected {EXPECTED_BEAKER_AUTHOR!r}"
        )
    return record


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
    if all(terminal.intersection(status) for status in statuses):
        return "failed"
    return "submitted"


def find_sweep(report: dict[str, Any], identifier: str) -> dict[str, Any] | None:
    return next(
        (sweep for sweep in report.get("batchSweeps", []) if sweep.get("id") == identifier), None
    )


def latest_job(experiment: dict[str, Any]) -> dict[str, Any]:
    return (experiment.get("jobs") or [{}])[-1]


def update_result(
    sweep: dict[str, Any], result: dict[str, Any], experiment_id: str, job: dict[str, Any]
) -> None:
    result = dict(result)
    result.update(
        {
            "status": "complete",
            "beaker": experiment_id,
            "experiment": experiment_id,
            "job": job.get("id"),
            "revision": sweep.get("revision"),
            "reason": (
                "Completed isolated 10% WSD decay from the exact retained pre-decay "
                "checkpoint, followed by matched held-out and all nine downstream evaluations."
            ),
        }
    )
    sweep.setdefault("results", {})[str(result["epoch"])] = result


def followup_sweep(
    report: dict[str, Any], primary: dict[str, Any], lr: str, wd: str, output: str
) -> dict[str, Any]:
    identifier = f"dense-153m-bs32-lr{lr}-wd0.1-hybrid-v1"
    existing = find_sweep(report, identifier)
    if existing is not None:
        return existing
    sweep = {
        "id": identifier,
        "policy": POLICY,
        "batchSequences": 32,
        "globalBatchTokens": 131072,
        "contextLength": 4096,
        "lr": lr,
        "wd": wd,
        "warmupSteps": 768,
        "rankMicrobatchSequences": 16,
        "gradientAccumulation": 1,
        "gpuCount": 2,
        "status": "running",
        "activeEpoch": 40,
        "activePhase": "followup_producer",
        "search": "small-model-bs32-lr-wd-hybrid",
        "hybridWorkflow": True,
        "hybridFollowup": True,
        "beaker": primary["experiment"],
        "experiment": primary["experiment"],
        "revision": primary.get("revision"),
        "output": output,
        "constantOutput": output + "/constant_lr",
        "saveEveryEpochs": 4,
        "evaluationEpochs": [40, 48, 56, 64, 72, 80],
        "dynamicRepacking": False,
        "weightTying": False,
        "embeddingWeightDecay": "zero",
        "automaticTaskRetries": 8,
        "minRuntimeOmitted": True,
        "results": {},
        "reason": "The same guarded task started the fresh WD0.1 trajectory at the selected LR.",
    }
    report.setdefault("batchSweeps", []).append(sweep)
    return sweep


def refresh() -> dict[str, Any]:
    report = json.loads(REPORT.read_text())
    primary = find_sweep(report, "dense-153m-bs32-lr-wd-hybrid-v1")
    if primary is None:
        raise RuntimeError("Dense-153M BS32 hybrid is not registered")
    experiment_id = str(primary["experiment"])
    experiment = inspect_experiment(experiment_id)
    live_state = experiment_state(experiment)
    job = latest_job(experiment)
    job_id = job.get("id")
    try:
        logs = ANSI.sub(
            "",
            run(["beaker", "job", "logs", str(job_id), "--since", "70m"]),
        ) if job_id else ""
    except subprocess.CalledProcessError:
        logs = ""

    decision_matches = DECISION.findall(logs)
    decision = json.loads(decision_matches[-1]) if decision_matches else None
    workflow = report.setdefault("bs32LrWdHybridWorkflow", {})
    workflow.update(
        {
            "policy": POLICY,
            "experiment": experiment_id,
            "revision": primary.get("revision"),
            "beakerStatus": live_state,
            "job": job.get("id"),
        }
    )
    if decision is not None:
        workflow["lrDecision"] = decision
        workflow["selectedLearningRate"] = decision["selectedLearningRate"]
        primary["lrDecision"] = decision

    for label, raw_result in E20_GATE_RESULT.findall(logs):
        result = json.loads(raw_result)
        result.update(
            {
                "status": "complete",
                "beaker": experiment_id,
                "experiment": experiment_id,
                "job": job.get("id"),
                "revision": primary.get("revision"),
            }
        )
        if label == "probe":
            primary.setdefault("results", {})["20"] = result
        else:
            workflow.setdefault("matchedBaselineResults", {})["20"] = result

    followup_matches = FOLLOWUP_START.findall(logs)
    followup: dict[str, Any] | None = None
    if followup_matches:
        lr, wd, output = followup_matches[-1]
        followup = followup_sweep(report, primary, lr, wd, output)
        primary["status"] = "complete"
        primary["activeEpoch"] = max([int(epoch) for epoch in primary.get("results", {})] or [40])
        primary["activePhase"] = "lr_decided"
        workflow["status"] = "followup_running"

    for phase, lr, wd, raw_epoch, raw_result in RESULT.findall(logs):
        epoch = int(raw_epoch)
        result = json.loads(raw_result)
        if phase == "probe":
            target = primary
        else:
            if followup is None:
                output = str(result["output"])
                followup = followup_sweep(report, primary, lr, wd, output)
            target = followup
        update_result(target, result, experiment_id, job)

    starts: list[tuple[int, str, str, str, int]] = []
    starts.extend(
        (match.start(), "producer", *match.groups()[:3], int(match.group(4)))
        for match in PD_START.finditer(logs)
    )
    starts.extend(
        (match.start(), "post", match.group(1), "5e-4" if match.group(1) == "probe" else "1e-3", "0.033", 20)
        for match in E20_GATE_START.finditer(logs)
    )
    starts.extend(
        (match.start(), "post", *match.groups()[:3], int(match.group(4)))
        for match in POST_START.finditer(logs)
    )
    if starts:
        _, stage, phase, lr, wd, epoch = max(starts, key=lambda item: item[0])
        target = primary if phase == "probe" else followup
        if target is not None and live_state not in {"failed", "complete"}:
            target["status"] = live_state
            target["activeEpoch"] = epoch
            target["activePhase"] = f"{phase}_{stage}"
        workflow.update(
            {
                "status": live_state,
                "activePhase": f"{phase}_{stage}",
                "activeEpoch": epoch,
                "activeLearningRate": lr,
                "activeWeightDecay": wd,
            }
        )
    elif live_state not in {"failed", "complete"}:
        primary["status"] = live_state
        workflow["status"] = live_state

    complete_matches = COMPLETE.findall(logs)
    if complete_matches:
        selected_lr, followup_wd = complete_matches[-1]
        if followup is not None:
            followup["status"] = "complete"
            followup["activeEpoch"] = 80
            followup["activePhase"] = "complete"
        primary["status"] = "complete"
        workflow.update(
            {
                "status": "complete",
                "activePhase": "complete",
                "activeEpoch": 80,
                "selectedLearningRate": selected_lr,
                "followupWeightDecay": followup_wd,
            }
        )
    elif live_state == "failed":
        active = followup if followup is not None else primary
        active["status"] = "failed"
        active["reason"] = (
            "The integrated hybrid task is terminal without a complete workflow marker."
        )
        workflow["status"] = "failed"

    if E20_GATE_HOLD.search(logs):
        primary["status"] = "complete"
        primary["activeEpoch"] = 20
        primary["activePhase"] = "lr_decided_no_followup"
        workflow.update({"status": "complete", "activePhase": "lr_decided_no_followup", "activeEpoch": 20})

    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )
    return {
        "experiment": experiment_id,
        "beakerStatus": live_state,
        "workflowStatus": workflow.get("status"),
        "primaryResults": sorted(primary.get("results", {}), key=int),
        "followupResults": sorted((followup or {}).get("results", {}), key=int),
        "decision": decision,
    }


def main() -> None:
    print(json.dumps(refresh(), indent=2))


if __name__ == "__main__":
    main()
