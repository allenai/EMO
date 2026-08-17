#!/usr/bin/env python3
"""Stop the waiting duplicate once a Jupiter/Saturn downstream race starts."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluate_dense_1b_missing_downstream import REPORT_PATH, load_report, write_report


def inspect(experiment: str) -> dict[str, Any]:
    payload = json.loads(
        subprocess.check_output(
            ["beaker", "experiment", "inspect", experiment, "--format", "json"],
            text=True,
        )
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"expected exactly one experiment for {experiment}")
    jobs = payload[0].get("jobs") or []
    if len(jobs) != 1:
        raise RuntimeError(f"race experiment {experiment} does not contain exactly one job")
    return jobs[0]


def state(job: dict[str, Any]) -> str:
    status = job.get("status") or {}
    if "finalized" in status:
        return "complete" if status.get("exitCode") == 0 else "failed"
    if "started" in status:
        return "running"
    if "scheduled" in status:
        return "scheduled"
    return "submitted"


def started_at(job: dict[str, Any]) -> datetime:
    value = (job.get("status") or {}).get("started")
    if not value:
        return datetime.max
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def evaluation_started(job: dict[str, Any]) -> bool:
    job_state = state(job)
    if job_state == "complete":
        return True
    if job_state != "running":
        return False
    logs = subprocess.run(
        ["beaker", "job", "logs", job["id"], "--no-timestamps"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    return "DOWNSTREAM_DISCOVERY_COMPLETE" in logs


def choose_winner(
    first: tuple[dict[str, Any], dict[str, Any]],
    second: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    first_campaign, first_job = first
    second_campaign, second_job = second
    first_state, second_state = state(first_job), state(second_job)
    first_started = evaluation_started(first_job)
    second_started = evaluation_started(second_job)
    stoppable = {"submitted", "scheduled", "running"}
    if first_state == "complete" and second_state in stoppable:
        return first_campaign, second_campaign
    if second_state == "complete" and first_state in stoppable:
        return second_campaign, first_campaign
    if first_started and not second_started and second_state in stoppable:
        return first_campaign, second_campaign
    if second_started and not first_started and first_state in stoppable:
        return second_campaign, first_campaign
    if first_started and second_started and first_state == second_state == "running":
        if started_at(first_job) <= started_at(second_job):
            return first_campaign, second_campaign
        return second_campaign, first_campaign
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Stop selected losing experiments.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = load_report(args.report)
    campaigns = report.get("downstreamEvaluationCampaigns", [])
    by_experiment = {campaign.get("experiment"): campaign for campaign in campaigns}
    mirrors = [campaign for campaign in campaigns if campaign.get("mirrorOf")]
    changed = False
    for mirror in mirrors:
        original = by_experiment.get(mirror["mirrorOf"])
        if original is None:
            raise RuntimeError(f"missing original race experiment {mirror['mirrorOf']}")
        if mirror.get("status") == "race_cancelled" or original.get("status") == "race_cancelled":
            continue
        original_job = inspect(original["experiment"])
        mirror_job = inspect(mirror["experiment"])
        decision = choose_winner((original, original_job), (mirror, mirror_job))
        group = mirror.get("raceGroup") or original.get("raceGroup")
        print(
            f"{group}: jupiter={state(original_job)} saturn={state(mirror_job)}",
            end="",
        )
        if decision is None:
            print()
            continue
        winner, loser = decision
        print(f" winner={winner['experiment']} loser={loser['experiment']}")
        if not args.apply:
            continue
        subprocess.run(
            ["beaker", "experiment", "stop", loser["experiment"]], check=True
        )
        loser["status"] = "race_cancelled"
        loser["raceWinner"] = winner["experiment"]
        winner["raceLoser"] = loser["experiment"]
        for task in loser.get("tasks", []):
            if task.get("status") not in {"complete", "unavailable"}:
                task["status"] = "race_cancelled"
        changed = True
    if changed:
        write_report(report, args.report)


if __name__ == "__main__":
    main()
