#!/usr/bin/env python3
"""Ingest successfully completed nested-3B endpoints from Beaker logs."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
from pathlib import Path


REPORTS = {
    model: Path(f"reports/0802/data/wsd_batch_size_{model}_pool3b.json")
    for model in ("153m", "474m", "1b")
}
SEQ = 4096
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_CE = re.compile(r"train/CE loss=([0-9]+(?:\.[0-9]+)?)")
VALIDATION_CE = re.compile(r"dclm-validation-0802/CE loss=([0-9]+(?:\.[0-9]+)?)")
WANDB = re.compile(r"wandb\.ai/[^\s]+/runs/([a-z0-9]+)")
DOWNSTREAM = re.compile(
    r"(?:^|\s)(arc_challenge|arc_easy|boolq|csqa_val_rc_5shot|hellaswag|"
    r"openbookqa_test_rc_5shot|piqa|socialiqa_val_rc_5shot|winogrande) "
    r"\((length-normalized accuracy|accuracy|BPB)\)=([0-9]+(?:\.[0-9]+)?)"
)
TASK_NAMES = {
    "csqa_val_rc_5shot": "csqa",
    "openbookqa_test_rc_5shot": "openbookqa",
    "socialiqa_val_rc_5shot": "socialiqa",
}
AVG8 = (
    "arc_challenge", "arc_easy", "csqa", "hellaswag", "openbookqa",
    "piqa", "socialiqa", "winogrande",
)
SSH_HOST: str | None = None


def beaker_command(*args: str) -> list[str]:
    command = ["beaker", *args]
    if SSH_HOST:
        return [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            SSH_HOST, shlex.join(command),
        ]
    return command


def clean(text: str) -> str:
    return "\n".join(
        re.sub(r"^\d{4}-\d\d-\d\dT\S+Z\s+", "", line)
        for line in ANSI.sub("", text).splitlines()
    )


def parse(text: str) -> dict:
    text = clean(text)
    train = [float(value) for value in TRAIN_CE.findall(text)]
    validation = [float(value) for value in VALIDATION_CE.findall(text)]
    wandb = WANDB.findall(text)
    if not train or not validation or not wandb:
        raise ValueError("missing train CE, DCLM validation CE, or W&B provenance")
    accuracy: dict[str, float] = {}
    bpb: dict[str, float] = {}
    for raw_task, metric, raw_value in DOWNSTREAM.findall(text):
        task = TASK_NAMES.get(raw_task, raw_task)
        if metric == "BPB":
            bpb[task] = float(raw_value)
        else:
            accuracy[task] = 100 * float(raw_value)
    expected = set(AVG8) | {"boolq"}
    if expected - accuracy.keys() or expected - bpb.keys():
        raise ValueError(
            f"missing downstream metrics: accuracy={sorted(expected-accuracy.keys())}, "
            f"BPB={sorted(expected-bpb.keys())}"
        )
    return {
        "train": train[-1],
        "validation": validation[-1],
        "c4": validation[-1],
        "wandb": wandb[-1],
        "acc": accuracy["hellaswag"],
        "bpb": bpb["hellaswag"],
        "avg8Bpb": sum(bpb[task] for task in AVG8) / len(AVG8),
        "downstream": accuracy,
        "downstreamBpb": bpb,
    }


def beaker_payload(experiment: str) -> dict:
    result = subprocess.run(
        beaker_command("experiment", "get", experiment, "--format", "json"),
        check=True, text=True, stdout=subprocess.PIPE,
    )
    payload = json.loads(result.stdout)
    return payload[0] if isinstance(payload, list) else payload


def successful(payload: dict, expected_replicas: int) -> bool:
    jobs = payload.get("jobs", [])
    successes = [
        job for job in jobs
        if job.get("status", {}).get("finalized")
        and job.get("status", {}).get("exitCode") == 0
    ]
    if expected_replicas == 1:
        return bool(successes)
    by_attempt: dict[str, list[dict]] = {}
    for job in successes:
        created = str(job.get("status", {}).get("created", ""))
        by_attempt.setdefault(created, []).append(job)
    return any(len(attempt) >= expected_replicas for attempt in by_attempt.values())


def logs(experiment: str) -> str:
    result = subprocess.run(
        beaker_command("experiment", "logs", experiment),
        check=True, text=True, stdout=subprocess.PIPE,
    )
    return result.stdout


def write(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA="
        + json.dumps(report, separators=(",", ":"))
        + ";\n"
    )


def ingest(
    path: Path,
    only_experiment: str | None = None,
    metrics_override: dict | None = None,
) -> list[dict]:
    report = json.loads(path.read_text())
    ingested = []
    changed = False
    for sweep in report.get("batchSweeps", []):
        if only_experiment and sweep.get("beaker") != only_experiment:
            continue
        if not sweep.get("beaker") or sweep.get("status") in {"failed", "canceled"}:
            continue
        epoch = str(sweep["activeEpoch"])
        successful_state = next(
            (
                row
                for row in sweep.get("jobStates", [])
                if row.get("state") == "succeeded"
            ),
            None,
        )
        if successful_state:
            primary_job = successful_state.get("job")
            primary_result = successful_state.get("resultDataset")
            endpoint = sweep.get("results", {}).get(epoch, {})
            if primary_job and sweep.get("job") != primary_job:
                sweep["job"] = primary_job
                changed = True
            if primary_job and endpoint.get("job") != primary_job:
                endpoint["job"] = primary_job
                changed = True
            if primary_result and sweep.get("resultDataset") != primary_result:
                sweep["resultDataset"] = primary_result
                changed = True
            if primary_result and endpoint.get("resultDataset") != primary_result:
                endpoint["resultDataset"] = primary_result
                changed = True
        if sweep.get("results", {}).get(epoch, {}).get("status") == "complete":
            continue
        synchronized_success = (
            sweep.get("status") == "complete"
            and sweep.get("jobs")
            and sweep.get("resultDatasets")
            and any(row.get("state") == "succeeded" for row in sweep.get("jobStates", []))
        )
        if synchronized_success:
            jobs = list(sweep["jobs"])
            result_datasets = list(sweep["resultDatasets"])
        else:
            payload = beaker_payload(sweep["beaker"])
            if not successful(payload, int(sweep.get("nodeCount", 1))):
                continue
            jobs = [job["id"] for job in payload.get("jobs", []) if job.get("id")]
            result_datasets = [
                job.get("execution", {}).get("result", {}).get("beaker")
                for job in payload.get("jobs", [])
            ]
            result_datasets = [dataset for dataset in result_datasets if dataset]
        metrics = metrics_override or parse(logs(sweep["beaker"]))
        retained = [int(step) for step in sweep.get("retainedPreDecaySteps", [])]
        if not retained:
            raise ValueError(f"{sweep['beaker']} has no retained pre-decay checkpoint")
        target_tokens = int(sweep["actualTargetTokens"])
        target_steps = math.ceil(target_tokens / int(sweep["globalBatchTokens"]))
        endpoint = {
            "epoch": sweep["activeEpoch"],
            "lr": sweep["lr"],
            "wd": sweep["wd"],
            "status": "complete",
            "job": jobs[0],
            "jobs": jobs,
            "beaker": sweep["beaker"],
            "resultDataset": result_datasets[0],
            "resultDatasets": result_datasets,
            "output": sweep["output"],
            "sourceCheckpoint": sweep["sourceCheckpoint"],
            # A successful recovery writes newly retained checkpoints under its
            # own output directory.  Do not inherit a target path recorded on
            # an older failed attempt: doing so makes the next frontier point
            # at a directory that was never materialized.
            "resumeCheckpoint": (
                sweep["sourceCheckpoint"]
                if (
                    sweep.get("recoverySourceStep") is not None
                    and int(sweep["recoverySourceStep"]) >= retained[-1]
                )
                else f"{sweep['output']}/step{retained[-1]}"
            ),
            "retainedPreDecaySteps": retained,
            "dataManifest": sweep.get("dataManifest"),
            "dataLoaderReset": sweep.get("dataLoaderReset"),
            "dataLoaderSeed": sweep.get("dataLoaderSeed", 0),
            "actualTargetTokens": target_tokens,
            "actualTargetSteps": target_steps,
            **metrics,
            "reason": (
                "Completed successfully with full DCLM validation and all nine "
                "downstream evaluations; exact nested-pool, loader-reset, multi-replica, "
                "and retained pre-decay checkpoint provenance recorded."
            ),
        }
        sweep.setdefault("results", {})[epoch] = endpoint
        sweep.update({
            "status": "complete",
            "job": jobs[0],
            "jobs": jobs,
            "resultDataset": result_datasets[0],
            "resultDatasets": result_datasets,
            "activeWandb": metrics["wandb"],
            "reason": endpoint["reason"],
        })
        ingested.append({
            "model": path.stem,
            "batch": sweep["batchSequences"],
            "epoch": sweep["activeEpoch"],
            "lr": sweep["lr"],
            "wd": sweep["wd"],
            "validation": metrics["validation"],
        })
        changed = True
    if changed:
        report["updated"] = "2026-08-09"
        write(path, report)
    return ingested


def main() -> None:
    global SSH_HOST
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(REPORTS))
    parser.add_argument(
        "--ssh-host",
        help="Query Beaker through this authenticated SSH host.",
    )
    parser.add_argument("--experiment", help="Ingest only this experiment ID.")
    parser.add_argument(
        "--metrics-json",
        help="Use these already-parsed endpoint metrics for --experiment.",
    )
    args = parser.parse_args()
    SSH_HOST = args.ssh_host
    metrics_override = json.loads(args.metrics_json) if args.metrics_json else None
    if metrics_override and not args.experiment:
        parser.error("--metrics-json requires --experiment")
    models = (args.model,) if args.model else tuple(REPORTS)
    rows = [
        row
        for model in models
        for row in ingest(REPORTS[model], args.experiment, metrics_override)
    ]
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
