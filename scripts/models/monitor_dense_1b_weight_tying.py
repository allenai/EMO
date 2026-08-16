#!/usr/bin/env python3
"""Monitor and advance the registered Dense-1B DR weight-tying study.

The script is intentionally idempotent so a Codex heartbeat can run it every 30
minutes. It refreshes active Beaker experiments, ingests completed endpoint losses,
advances only strictly improving coordinates, updates the HTML report's data, and
submits at most one next endpoint while the flex2 pending-job gate is clear.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_PATH = Path("reports/0802/data/wsd_data_loader_1b.json")
REPORT_JS_PATH = REPORT_PATH.with_suffix(".js")
LAUNCHER = Path("scripts/models/submit_dense_step1_data_loader_coordinate.py")
FLEX2_WORKSPACE_ID = "01K1DJ083DM56DQTHPR4JWS7DD"
BEAKER_AUTHOR_ID = "01J4HQ6ZE87JNS4EAHE4SNHSD1"
SEQUENCE_LENGTH = 4096
TOKENS_PER_EPOCH = 1_000_000_000
DECAY_FRACTION = 0.1
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_LOSS = re.compile(r"\btrain/CE loss=([0-9]+(?:\.[0-9]+)?)")
VALIDATION_LOSS = re.compile(
    r"\bdclm-validation-0802/CE loss=([0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE,
)
WANDB_VALIDATION_LOSS = re.compile(
    r"wandb:\s+eval/heldout/dclm-validation-0802/CE loss\s+([0-9]+(?:\.[0-9]+)?)"
)
WANDB_RUN = re.compile(r"https://wandb\.ai/[^\s]+/runs/([a-zA-Z0-9_-]+)")
TRAIN_STEP = re.compile(r"\[step=([0-9,]+)/([0-9,]+),epoch=")
DESCRIPTION_PROGRESS = re.compile(r"([0-9]+(?:\.[0-9]+)?)% complete, step ([0-9,]+)/([0-9,]+)")
ACTIVE_STATUSES = {"submitted", "scheduled", "running"}
REVISION_ALIASES = {
    "2c6b1fb98": "sewonm/icsl",
    "2c6b1fb982091a3202348b39883255a05075f946": "sewonm/icsl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit-next",
        action="store_true",
        help="Submit at most one planned endpoint if the shared pending-job gate is clear.",
    )
    parser.add_argument(
        "--revision",
        help="Pushed git revision to use for new submissions; required with --submit-next.",
    )
    parser.add_argument(
        "--allow-pending-flex2-jobs",
        action="store_true",
        help="Use the user's explicit override of the shared pending-job gate.",
    )
    parser.add_argument(
        "--retry-git-checkout-failures",
        action="store_true",
        help="Reactivate registered E1 attempts that failed before startup on Git checkout.",
    )
    args = parser.parse_args()
    if args.submit_next and not args.revision:
        parser.error("--revision is required with --submit-next")
    return args


def run_command(arguments: list[str]) -> str:
    result = subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def load_report() -> dict[str, Any]:
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"run from the repository root; missing {REPORT_PATH}")
    return json.loads(REPORT_PATH.read_text())


def write_report(report: dict[str, Any]) -> None:
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS_PATH.write_text(
        "window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def tied_runs(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [run for run in report.get("runs", []) if run.get("weightTying") is True]


def inspect_experiment(experiment: str) -> dict[str, Any]:
    payload = json.loads(
        run_command(["beaker", "experiment", "inspect", experiment, "--format", "json"])
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"expected one inspected experiment for {experiment}")
    return payload[0]


def experiment_state(experiment: dict[str, Any]) -> str:
    jobs = experiment.get("jobs") or []
    if not jobs:
        return "submitted"
    statuses = [job.get("status") or {} for job in jobs]
    if any(
        "canceled" in status
        or "cancelled" in status
        or ("finalized" in status and status.get("exitCode") not in {None, 0})
        for status in statuses
    ):
        return "failed"
    if all("finalized" in status and status.get("exitCode") == 0 for status in statuses):
        return "complete"
    if any("started" in status for status in statuses):
        return "running"
    if any("scheduled" in status for status in statuses):
        return "scheduled"
    return "submitted"


def parse_progress(experiment: dict[str, Any], logs: str | None = None) -> dict[str, Any]:
    progress: dict[str, Any] = {}
    description = str(experiment.get("description") or "")
    match = DESCRIPTION_PROGRESS.search(description)
    if match:
        progress.update(
            {
                "percent": float(match.group(1)),
                "step": int(match.group(2).replace(",", "")),
                "totalSteps": int(match.group(3).replace(",", "")),
            }
        )
    if logs:
        cleaned = ANSI.sub("", logs)
        step_matches = TRAIN_STEP.findall(cleaned)
        if step_matches:
            step, total = step_matches[-1]
            progress.update(
                {
                    "step": int(step.replace(",", "")),
                    "totalSteps": int(total.replace(",", "")),
                }
            )
            progress["percent"] = round(100 * progress["step"] / progress["totalSteps"], 1)
        train_matches = TRAIN_LOSS.findall(cleaned)
        if train_matches:
            progress["latestTrain"] = float(train_matches[-1])
    return progress


def experiment_logs(experiment: str) -> str:
    return run_command(["beaker", "experiment", "logs", experiment])


def total_step(epoch: int, global_sequences: int) -> int:
    return math.ceil(epoch * TOKENS_PER_EPOCH / (global_sequences * SEQUENCE_LENGTH))


def stable_step(epoch: int, global_sequences: int) -> int:
    end = total_step(epoch, global_sequences)
    return end - round(DECAY_FRACTION * end) - 1


def completed_result(
    run: dict[str, Any], epoch: int, experiment: dict[str, Any], logs: str
) -> dict[str, Any]:
    cleaned = ANSI.sub("", logs)
    train_values = TRAIN_LOSS.findall(cleaned)
    validation_values = WANDB_VALIDATION_LOSS.findall(cleaned) or VALIDATION_LOSS.findall(cleaned)
    wandb_values = WANDB_RUN.findall(cleaned)
    if not train_values or not validation_values:
        raise RuntimeError(
            f"completed experiment {run['experiment']} is missing train or held-out CE"
        )
    train = float(train_values[-1])
    validation_exact = float(validation_values[-1])
    jobs = experiment.get("jobs") or []
    leader = jobs[0] if jobs else {}
    result_id = (leader.get("execution") or {}).get("result", {}).get("beaker") or (
        leader.get("result") or {}
    ).get("beaker")
    output = run["output"]
    result: dict[str, Any] = {
        "status": "complete",
        "beaker": run["experiment"],
        "experiment": run["experiment"],
        "job": leader.get("id"),
        "result": result_id,
        "revision": run.get("revision"),
        "output": output,
        "resumeCheckpoint": run.get("sourceCheckpoint"),
        "preDecayCheckpoint": f"{output}/step{stable_step(epoch, int(run['batchSequences']))}",
        "endpointCheckpoint": f"{output}/step{total_step(epoch, int(run['batchSequences']))}",
        "train": train,
        "validation": round(validation_exact, 3),
        "validationExact": validation_exact,
        "gap": round(validation_exact - train, 6),
    }
    result["retainedCheckpoints"] = [
        result["preDecayCheckpoint"],
        result["endpointCheckpoint"],
    ]
    if wandb_values:
        result["wandb"] = wandb_values[-1]
    if leader.get("node"):
        result["node"] = leader["node"]
    return result


def advance_or_stop(run: dict[str, Any], epoch: int, result: dict[str, Any]) -> str:
    targets = [int(value) for value in run.get("plannedTargets", [])]
    target_index = targets.index(epoch)
    previous = None
    if target_index:
        previous = run.get("results", {}).get(str(targets[target_index - 1]))
    improved = previous is None or float(result["validationExact"]) < float(
        previous.get("validationExact", previous["validation"])
    )
    if not improved:
        previous_validation = float(previous.get("validationExact", previous["validation"]))
        result["reason"] = (
            f"Held-out CE{result['validationExact']:.5f} did not strictly improve the "
            f"preceding CE{previous_validation:.5f}; retained as the first overfitting "
            "endpoint and stopped this coordinate."
        )
        run["status"] = "complete"
        run["activeEpoch"] = None
        run["stopReason"] = result["reason"]
        return "stopped after first non-improving endpoint"
    if target_index == len(targets) - 1:
        result["reason"] = "Completed the final planned E28 endpoint with a strict improvement."
        run["status"] = "complete"
        run["activeEpoch"] = None
        run["stopReason"] = "Reached the final planned E28 endpoint."
        return "finished at E28"
    next_epoch = targets[target_index + 1]
    if previous is None:
        result["reason"] = (
            f"Completed the tied E1 bootstrap at held-out CE{result['validationExact']:.5f}; "
            f"unlocked E{next_epoch} from the retained exact pre-decay checkpoint."
        )
    else:
        previous_validation = float(previous.get("validationExact", previous["validation"]))
        result["reason"] = (
            f"Held-out CE{result['validationExact']:.5f} strictly improved the preceding "
            f"CE{previous_validation:.5f}; unlocked E{next_epoch} from the retained exact "
            "pre-decay checkpoint."
        )
    run["status"] = "planned"
    run["activeEpoch"] = next_epoch
    run["sourceExperiment"] = run["experiment"]
    run["sourceCheckpoint"] = result["preDecayCheckpoint"]
    run.pop("progress", None)
    return f"unlocked E{next_epoch}"


def refresh_run(run: dict[str, Any]) -> str | None:
    if str(run.get("status", "")).lower() not in ACTIVE_STATUSES:
        return None
    experiment_id = str(run.get("experiment") or run.get("beaker"))
    epoch = int(run["activeEpoch"])
    experiment = inspect_experiment(experiment_id)
    state = experiment_state(experiment)
    run["status"] = state
    run["beakerStatus"] = state
    job_ids = [job["id"] for job in experiment.get("jobs") or [] if job.get("id")]
    if job_ids:
        run["job"] = job_ids[0]
        run["jobs"] = job_ids
    attempted = run.setdefault("attemptedEpochs", [])
    if epoch not in attempted:
        attempted.append(epoch)
    if state == "failed":
        run.setdefault("results", {})[str(epoch)] = {
            "status": "failed",
            "beaker": experiment_id,
            "experiment": experiment_id,
            "reason": "Beaker experiment finalized unsuccessfully; inspect its logs before retrying.",
        }
        run["activeEpoch"] = None
        run["stopReason"] = "Beaker endpoint failed."
        return f"{run['id']} E{epoch} failed ({experiment_id})"
    logs = None
    if state in {"running", "complete"}:
        logs = experiment_logs(experiment_id)
    progress = parse_progress(experiment, logs)
    if progress:
        run["progress"] = progress
    if state != "complete":
        detail = ""
        if progress:
            detail = f" {progress.get('percent', 0):g}% ({progress.get('step', 0)}/{progress.get('totalSteps', 0)})"
        return f"{run['id']} E{epoch} {state}{detail} ({experiment_id})"
    assert logs is not None
    result = completed_result(run, epoch, experiment, logs)
    run.setdefault("results", {})[str(epoch)] = result
    action = advance_or_stop(run, epoch, result)
    return (
        f"{run['id']} E{epoch} complete: train={result['train']:.3f}, "
        f"val={result['validationExact']:.5f}, gap={result['gap']:+.5f}; {action}"
    )


def pending_flex2_jobs() -> list[dict[str, Any]]:
    endpoint = f"jobs?author={BEAKER_AUTHOR_ID}&scheduled=false&finalized=false"
    payload = json.loads(run_command(["beaker", "api", endpoint, "--format", "json"]))
    jobs = (payload.get("data") or []) if isinstance(payload, dict) else (payload or [])
    return [
        job
        for job in jobs
        if job.get("workspace") == FLEX2_WORKSPACE_ID
        and "scheduled" not in (job.get("status") or {})
        and "finalized" not in (job.get("status") or {})
    ]


def slug_number(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def fetchable_revision(revision: str) -> str:
    return REVISION_ALIASES.get(revision, revision)


def reactivate_git_checkout_failures(report: dict[str, Any]) -> list[str]:
    reactivated = []
    for run in tied_runs(report):
        if run.get("status") != "failed" or run.get("activeEpoch") is not None:
            continue
        failed_result = run.get("results", {}).get("1")
        if not isinstance(failed_result, dict) or failed_result.get("status") != "failed":
            continue
        experiment = run.get("experiment") or run.get("beaker")
        if not experiment:
            continue
        attempts = run.setdefault("attempts", [])
        if not any(attempt.get("experiment") == experiment for attempt in attempts):
            attempts.append(
                {
                    "epoch": 1,
                    "status": "failed",
                    "failureClass": "git_checkout_unfetchable_short_revision",
                    "experiment": experiment,
                    "beaker": experiment,
                    "job": run.get("job"),
                    "jobs": run.get("jobs", []),
                    "revision": run.get("revision"),
                    "reason": (
                        "Gantry failed before repository startup because the short commit "
                        "SHA was not a fetchable remote ref; no training artifact was created."
                    ),
                }
            )
        del run["results"]["1"]
        run["status"] = "planned"
        run["activeEpoch"] = 1
        run["reason"] = (
            "User-authorized E1 retry after a pre-start Git-ref failure; use a fetchable "
            "remote branch while preserving all model and duplicate guards."
        )
        run.pop("stopReason", None)
        for key in (
            "beaker",
            "beakerStatus",
            "experiment",
            "job",
            "jobs",
            "output",
            "progress",
            "revision",
        ):
            run.pop(key, None)
        reactivated.append(run["id"])
    return reactivated


def submit_next(report: dict[str, Any], revision: str, allow_pending_flex2_jobs: bool) -> str:
    if not allow_pending_flex2_jobs:
        pending = pending_flex2_jobs()
        if pending:
            identifiers = ", ".join(str(job.get("id")) for job in pending)
            return (
                f"No submission: flex2 has {len(pending)} unscheduled job(s) under the "
                f"shared safety gate: {identifiers}."
            )
    planned = [
        run
        for run in tied_runs(report)
        if str(run.get("status", "")).lower() == "planned" and run.get("activeEpoch") is not None
    ]
    if not planned:
        active = [
            run
            for run in tied_runs(report)
            if str(run.get("status", "")).lower() in ACTIVE_STATUSES
        ]
        if active:
            labels = ", ".join(
                f"{run['id']} E{run['activeEpoch']} ({run['status']})" for run in active
            )
            return f"No submission: current weight-tying endpoints remain active: {labels}."
        return "No submission: all registered weight-tying coordinates are finished."
    run = planned[0]
    epoch = int(run["activeEpoch"])
    batch = int(run["batchSequences"])
    attempt = len(run.get("attempts", [])) + 1
    attempt_tag = f"r{attempt:02d}"
    suffix = f"wt-bs{batch}-e{epoch}-wd{slug_number(run['wd'])}-{attempt_tag}"
    name = f"dense-1b-dr-wt-bs{batch}-e{epoch}-wd{slug_number(run['wd'])}-{attempt_tag}"
    command = [
        str(Path(".venv/bin/python")),
        str(LAUNCHER),
        "--method",
        "dynamic_repacking",
        "--global-sequences",
        str(batch),
        "--target-epoch",
        str(epoch),
        "--learning-rate",
        "1e-3",
        "--weight-decay",
        str(run["wd"]),
        "--source-experiment",
        str(run["sourceExperiment"]),
        "--source-checkpoint",
        str(run["sourceCheckpoint"]),
        "--revision",
        revision,
        "--name",
        name,
        "--suffix",
        suffix,
        "--weight-tying",
        "--register",
    ]
    if allow_pending_flex2_jobs:
        command.append("--allow-pending-flex2-jobs")
    output = run_command(command)
    # The launcher writes the registry. Reload it so this invocation never overwrites
    # the newly recorded experiment ID with the pre-submission in-memory report.
    submitted_report = load_report()
    submitted = next(item for item in tied_runs(submitted_report) if item["id"] == run["id"])
    experiment = submitted.get("experiment") or submitted.get("beaker")
    parsed_ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not experiment and parsed_ids:
        experiment = parsed_ids[0]
    return f"Submitted {run['id']} E{epoch}: {experiment}."


def main() -> None:
    args = parse_args()
    report = load_report()
    messages = []
    changed = False
    if args.retry_git_checkout_failures:
        reactivated = reactivate_git_checkout_failures(report)
        if reactivated:
            messages.append("Reactivated Git-checkout retries: " + ", ".join(reactivated))
            changed = True
    for run in tied_runs(report):
        message = refresh_run(run)
        if message:
            messages.append(message)
            changed = True
    if changed:
        write_report(report)
    if args.submit_next:
        messages.append(
            submit_next(
                report,
                fetchable_revision(args.revision),
                args.allow_pending_flex2_jobs,
            )
        )
    if not messages:
        messages.append("No active weight-tying endpoints; registry unchanged.")
    print("\n".join(messages))


if __name__ == "__main__":
    main()
