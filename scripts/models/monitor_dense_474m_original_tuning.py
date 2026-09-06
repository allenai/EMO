#!/usr/bin/env python3
"""Refresh Dense-474M original-model tuning results in report JSON/JS."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT = Path("reports/0802/data/wsd_batch_size_474m.json")
POLICY = "dense_474m_original_tuning_v1"
MODES = ("bs32-probes", "bs128-e52", "bs32-lr5e4-e32")
MODE_PATTERN = "(?:" + "|".join(MODES) + ")"
EXPECTED_BEAKER_AUTHOR = "sewonm"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PD_START = re.compile(
    rf"DENSE474M_ORIGINAL_PD_START mode=({MODE_PATTERN}) coordinate=([^ ]+) epoch=([0-9]+)"
)
PD_RETAINED = re.compile(
    rf"DENSE474M_ORIGINAL_PD_RETAINED mode=({MODE_PATTERN}) coordinate=([^ ]+) "
    r"epoch=([0-9]+) checkpoint=([^ ]+) retained=(\[[^\]]*\])"
)
POST_START = re.compile(
    rf"DENSE474M_ORIGINAL_POST_START mode=({MODE_PATTERN}) coordinate=([^ ]+) epoch=([0-9]+)"
)
RESULT = re.compile(
    rf"DENSE474M_ORIGINAL_POST_RESULT mode=({MODE_PATTERN}) coordinate=([^ ]+) "
    r"epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
BS32_DECISION = re.compile(r"DENSE474M_ORIGINAL_BS32_DECISION json=(\{.*\})$", re.MULTILINE)
BS128_DECISION = re.compile(r"DENSE474M_ORIGINAL_BS128_DECISION json=(\{.*\})$", re.MULTILINE)
COMPLETE = re.compile(rf"DENSE474M_ORIGINAL_WORKFLOW_COMPLETE mode=({MODE_PATTERN})")
SWEEP_BY_COORDINATE = {
    "lr5e-4-wd0.1": "dense-474m-bs32-lr5e-4-wd0.1-e4-probe-v1",
    "lr2e-3-wd0.1": "dense-474m-bs32-lr2e-3-wd0.1-e4-probe-v1",
    "bs128-lr2e-3-wd0.3-e52": "dense-474m-bs128-lr2e-3-wd0.3-e52-extension-v1",
}


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


def find_sweep(report: dict[str, Any], identifier: str) -> dict[str, Any]:
    matches = [sweep for sweep in report.get("batchSweeps", []) if sweep.get("id") == identifier]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one report sweep {identifier}, found {len(matches)}")
    return matches[0]


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
    sweep.update(
        {
            "activeEpoch": result["epoch"],
            "activePhase": "post_complete",
            "reason": result["reason"],
        }
    )


def refresh_mode(report: dict[str, Any], mode: str, experiment_id: str) -> dict[str, Any]:
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

    coordinates = {
        "bs32-probes": ["lr5e-4-wd0.1", "lr2e-3-wd0.1"],
        "bs128-e52": ["bs128-lr2e-3-wd0.3-e52"],
        "bs32-lr5e4-e32": ["lr5e-4-wd0.1"],
    }[mode]
    sweeps = {
        coordinate: find_sweep(report, SWEEP_BY_COORDINATE[coordinate])
        for coordinate in coordinates
    }
    for sweep in sweeps.values():
        sweep["job"] = job.get("id")
        if live_state not in {"complete", "failed"}:
            sweep["status"] = live_state
    if mode == "bs32-lr5e4-e32" and live_state not in {"complete", "failed"}:
        extension_sweep = sweeps["lr5e-4-wd0.1"]
        if int(extension_sweep.get("activeEpoch", 0)) < 16:
            extension_sweep.update(
                {
                    "activeEpoch": 16,
                    "activePhase": "producer_pending",
                    "reason": (
                        "Continuing the selected exact E4 PD frontier through E32, with "
                        "isolated POST evaluations at E16, E24, and E32."
                    ),
                }
            )

    for parsed_mode, coordinate, raw_epoch, _checkpoint, raw_retained in PD_RETAINED.findall(logs):
        if parsed_mode != mode or coordinate not in sweeps:
            continue
        sweep = sweeps[coordinate]
        retained = [int(epoch) for epoch in json.loads(raw_retained)]
        sweep["retainedPreDecayEpochs"] = retained
        sweep["activeEpoch"] = int(raw_epoch)

    for parsed_mode, coordinate, raw_epoch, raw_result in RESULT.findall(logs):
        if parsed_mode == mode and coordinate in sweeps:
            update_result(sweeps[coordinate], json.loads(raw_result), experiment_id, job)

    starts: list[tuple[int, str, str, int]] = []
    starts.extend(
        (match.start(), "producer", match.group(2), int(match.group(3)))
        for match in PD_START.finditer(logs)
        if match.group(1) == mode
    )
    starts.extend(
        (match.start(), "post", match.group(2), int(match.group(3)))
        for match in POST_START.finditer(logs)
        if match.group(1) == mode
    )
    if starts and live_state not in {"complete", "failed"}:
        _, phase, coordinate, epoch = max(starts, key=lambda item: item[0])
        sweep = sweeps[coordinate]
        sweep.update({"status": live_state, "activeEpoch": epoch, "activePhase": phase})

    complete_modes = COMPLETE.findall(logs)
    workflow_complete = mode in complete_modes
    if workflow_complete:
        for sweep in sweeps.values():
            sweep["status"] = "complete"
    elif live_state == "complete":
        expected_epochs = {
            "bs32-probes": {
                "lr5e-4-wd0.1": {"4"},
                "lr2e-3-wd0.1": {"4"},
            },
            "bs128-e52": {"bs128-lr2e-3-wd0.3-e52": {"52"}},
            "bs32-lr5e4-e32": {"lr5e-4-wd0.1": {"16", "24", "32"}},
        }[mode]
        for coordinate, sweep in sweeps.items():
            if expected_epochs[coordinate].issubset(set(sweep.get("results", {}))):
                sweep["status"] = "complete"
            else:
                sweep.update(
                    {
                        "status": "failed",
                        "reason": "The integrated workflow exited before all expected POST results resolved.",
                    }
                )
    elif live_state == "failed":
        unfinished = [sweep for sweep in sweeps.values() if not sweep.get("results")]
        for sweep in unfinished:
            sweep.update(
                {
                    "status": "failed",
                    "reason": "The integrated workflow is terminal without a complete result marker.",
                }
            )

    decision_match = (
        BS32_DECISION.findall(logs)
        if mode == "bs32-probes"
        else BS128_DECISION.findall(logs) if mode == "bs128-e52" else []
    )
    decision = json.loads(decision_match[-1]) if decision_match else None
    return {
        "experiment": experiment_id,
        "job": job.get("id"),
        "beakerStatus": live_state,
        "status": "complete" if workflow_complete else live_state,
        "decision": decision,
        "results": {
            coordinate: sorted(sweep.get("results", {}), key=float)
            for coordinate, sweep in sweeps.items()
        },
    }


def refresh() -> dict[str, Any]:
    report = json.loads(REPORT.read_text())
    workflow = report.get("originalTuningWorkflow")
    if not isinstance(workflow, dict) or workflow.get("policy") != POLICY:
        raise RuntimeError("Dense-474M original tuning workflow is not registered")
    experiments = workflow.get("experiments", {})
    if not experiments:
        raise RuntimeError("Dense-474M original tuning workflow has no experiments")
    prior_mode_summaries = workflow.get("modes", {})
    mode_summaries: dict[str, Any] = {}
    for mode in MODES:
        experiment_id = experiments.get(mode)
        if experiment_id:
            mode_summaries[mode] = refresh_mode(report, mode, str(experiment_id))
            if mode_summaries[mode].get("decision") is None:
                prior_decision = (prior_mode_summaries.get(mode) or {}).get("decision")
                if prior_decision is None:
                    prior_decision = (workflow.get("decisions", {}) or {}).get(mode)
                if prior_decision is not None:
                    mode_summaries[mode]["decision"] = prior_decision
            decision = mode_summaries[mode].get("decision")
            if decision is not None:
                workflow.setdefault("decisions", {})[mode] = decision
    statuses = [summary["status"] for summary in mode_summaries.values()]
    if statuses and all(status == "complete" for status in statuses):
        workflow["status"] = "complete"
    elif any(status == "failed" for status in statuses):
        workflow["status"] = "failed"
    elif any(status == "running" for status in statuses):
        workflow["status"] = "running"
    elif any(status == "scheduled" for status in statuses):
        workflow["status"] = "scheduled"
    else:
        workflow["status"] = "submitted"
    workflow["modes"] = mode_summaries
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )
    return {"status": workflow["status"], "modes": mode_summaries}


def main() -> None:
    print(json.dumps(refresh(), indent=2))


if __name__ == "__main__":
    main()
