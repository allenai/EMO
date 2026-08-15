#!/usr/bin/env python3
"""Ingest completed small-dense Step 1-1 endpoints from Beaker logs."""

from __future__ import annotations

import json
import argparse
import re
import shlex
import subprocess
from pathlib import Path


REPORTS = (
    Path("reports/0802/data/wsd_batch_size_153m.json"),
    Path("reports/0802/data/wsd_batch_size_474m.json"),
)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_CE = re.compile(r"train/CE loss=([0-9]+(?:\.[0-9]+)?)")
VALIDATION_CE = re.compile(
    r"dclm-validation-0802/CE loss=([0-9]+(?:\.[0-9]+)?)"
)
WANDB = re.compile(r"wandb\.ai/[^\s]+/runs/([a-z0-9]+)")
DOWNSTREAM = re.compile(
    r"(?:^|\s)(arc_challenge|arc_easy|boolq|csqa_val_rc_5shot|hellaswag|"
    r"openbookqa_test_rc_5shot|piqa|socialiqa_val_rc_5shot|winogrande) "
    r"\((length-normalized accuracy|accuracy|BPB)\)=([0-9]+(?:\.[0-9]+)?)",
)
TASK_NAMES = {
    "csqa_val_rc_5shot": "csqa",
    "openbookqa_test_rc_5shot": "openbookqa",
    "socialiqa_val_rc_5shot": "socialiqa",
}
AVG8_TASKS = (
    "arc_challenge",
    "arc_easy",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
)
ADAPTIVE_SEARCH = "small-model-adaptive-coordinate"
FRACTIONAL_SEARCH = "small-model-selected-e1-fractional-chain"
FRACTIONAL_STAGE = re.compile(
    r"FRACTIONAL_STAGE_BEGIN epoch=([0-9.]+) output=(\S+) stable=(\d+) end=(\d+)"
    r"(.*?)FRACTIONAL_STAGE_END epoch=\1",
    re.DOTALL,
)


def beaker_command(*args: str) -> list[str]:
    command = ["beaker", *args]
    if not ARGS.ssh_host:
        return command
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        ARGS.ssh_host,
        shlex.join(command),
    ]


def clean_log(text: str) -> str:
    cleaned = ANSI.sub("", text)
    return "\n".join(
        re.sub(r"^\d{4}-\d\d-\d\dT\S+Z\s+", "", line)
        for line in cleaned.splitlines()
    )


def parse_log(text: str, train_fallback: float | None = None) -> dict[str, object]:
    text = clean_log(text)
    train_values = [float(value) for value in TRAIN_CE.findall(text)]
    validation_values = [float(value) for value in VALIDATION_CE.findall(text)]
    wandb_values = WANDB.findall(text)
    if (not train_values and train_fallback is None) or not validation_values or not wandb_values:
        raise ValueError("missing train CE/fallback, validation CE, or W&B provenance")

    accuracy: dict[str, float] = {}
    bpb: dict[str, float] = {}
    for raw_task, metric, raw_value in DOWNSTREAM.findall(text):
        task = TASK_NAMES.get(raw_task, raw_task)
        value = float(raw_value)
        if metric == "BPB":
            bpb[task] = value
        else:
            accuracy[task] = value * 100

    expected = set(AVG8_TASKS) | {"boolq"}
    missing_accuracy = expected - accuracy.keys()
    missing_bpb = expected - bpb.keys()
    if missing_accuracy or missing_bpb:
        raise ValueError(
            f"missing downstream metrics: accuracy={sorted(missing_accuracy)}, "
            f"BPB={sorted(missing_bpb)}"
        )
    return {
        "train": train_values[-1] if train_values else train_fallback,
        "validation": validation_values[-1],
        "wandb": wandb_values[-1],
        "downstream": accuracy,
        "downstreamBpb": bpb,
        "acc": accuracy["hellaswag"],
        "bpb": bpb["hellaswag"],
        "avg8Bpb": sum(bpb[task] for task in AVG8_TASKS) / len(AVG8_TASKS),
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA="
        + json.dumps(report, separators=(",", ":"))
        + ";\n"
    )


parser = argparse.ArgumentParser()
parser.add_argument("--ssh-host", help="Query Beaker through this authenticated SSH host.")
ARGS = parser.parse_args()

ingested: list[dict[str, object]] = []
for path in REPORTS:
    report = json.loads(path.read_text())
    model = "153M" if "153m" in path.name else "474M"
    changed = False
    for sweep in report.get("batchSweeps", []):
        search = sweep.get("search")
        if search not in {ADAPTIVE_SEARCH, FRACTIONAL_SEARCH}:
            continue
        if not sweep.get("beaker"):
            continue
        if search == FRACTIONAL_SEARCH:
            if sweep.get("status") not in {"running", "complete", "failed", "canceled"}:
                continue
            result = subprocess.run(
                beaker_command("experiment", "logs", str(sweep["beaker"])),
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            for match in FRACTIONAL_STAGE.finditer(clean_log(result.stdout)):
                epoch, output, stable_step, end_step, segment = match.groups()
                if epoch in sweep.get("results", {}):
                    continue
                metrics = parse_log(segment)
                endpoint_metadata = sweep.get("fractionalEndpoints", {}).get(epoch, {})
                if int(endpoint_metadata.get("stableStep", -1)) != int(stable_step):
                    raise ValueError(f"{sweep['beaker']} E{epoch} stable-step marker mismatch")
                if int(endpoint_metadata.get("endStep", -1)) != int(end_step):
                    raise ValueError(f"{sweep['beaker']} E{epoch} end-step marker mismatch")
                endpoint = {
                    "epoch": float(epoch),
                    "lr": sweep["lr"],
                    "wd": sweep["wd"],
                    "status": "complete",
                    "job": sweep.get("job"),
                    "beaker": sweep.get("beaker"),
                    "resultDataset": sweep.get("resultDataset"),
                    "output": output,
                    "sourceCheckpoint": endpoint_metadata.get("sourceCheckpoint"),
                    "resumeCheckpoint": f"{output}/step{stable_step}",
                    "retainedPreDecaySteps": [int(stable_step), int(end_step)],
                    **metrics,
                    "c4": metrics["validation"],
                    "reason": (
                        f"Completed selected-E1-coordinate E{epoch} with full held-out "
                        "DCLM validation and all nine downstream evaluations; exact "
                        "pre-decay and endpoint checkpoints are retained."
                    ),
                }
                sweep.setdefault("results", {})[epoch] = endpoint
                sweep["activeWandb"] = metrics["wandb"]
                sweep["activeStage"] = float(epoch)
                changed = True
                ingested.append(
                    {
                        "model": model,
                        "batch": sweep["batchSequences"],
                        "epoch": float(epoch),
                        "lr": sweep["lr"],
                        "wd": sweep["wd"],
                        "validation": metrics["validation"],
                    }
                )
            if sweep.get("status") == "complete" and len(sweep.get("results", {})) == len(
                sweep.get("fractionalTargets", [])
            ):
                sweep["reason"] = (
                    "All selected-E1-coordinate fractional endpoints completed with "
                    "exact same-coordinate pre-decay resumes and full evaluation."
                )
                changed = True
            continue
        if sweep.get("status") != "complete":
            continue
        epoch = str(sweep["activeEpoch"])
        if epoch in sweep.get("results", {}):
            continue
        result = subprocess.run(
            beaker_command("experiment", "logs", str(sweep["beaker"])),
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        metrics = parse_log(result.stdout, sweep.get("trainFallback"))
        retained_steps = [int(step) for step in sweep.get("retainedPreDecaySteps", [])]
        if not retained_steps:
            raise ValueError(f"{sweep['beaker']} has no retained pre-decay step")
        evaluation_checkpoint = sweep.get("evaluationCheckpoint")
        endpoint = {
            "epoch": sweep["activeEpoch"],
            "lr": sweep["lr"],
            "wd": sweep["wd"],
            "status": "complete",
            "job": sweep.get("job"),
            "beaker": sweep.get("beaker"),
            "resultDataset": sweep.get("resultDataset"),
            "output": sweep.get("output"),
            "sourceCheckpoint": sweep.get("sourceCheckpoint"),
            "resumeCheckpoint": evaluation_checkpoint
            or f"{sweep['output']}/step{retained_steps[-1]}",
            "evaluationCheckpoint": evaluation_checkpoint,
            "retainedPreDecaySteps": retained_steps,
            **metrics,
            "c4": metrics["validation"],
            "reason": (
                "Completed evaluation-only infrastructure recovery from the exact saved "
                f"endpoint checkpoint {evaluation_checkpoint} with full held-out DCLM "
                "validation and all nine downstream evaluations; no training was replayed."
                if sweep.get("evaluationOnly")
                else "Completed successfully with full held-out DCLM validation and all "
                "nine downstream evaluations; the exact pre-decay checkpoint is retained."
            ),
        }
        sweep.setdefault("results", {})[epoch] = endpoint
        sweep["activeWandb"] = metrics["wandb"]
        sweep["reason"] = endpoint["reason"]
        if sweep.get("evaluationOnly"):
            # The canceled partial endpoint remains preserved in attemptHistory.
            sweep.pop("partialEndpoint", None)
            sweep["progressPercent"] = 100
            sweep.pop("progressStep", None)
            sweep.pop("progressTotalSteps", None)
        changed = True
        ingested.append(
            {
                "model": model,
                "batch": sweep["batchSequences"],
                "epoch": sweep["activeEpoch"],
                "lr": sweep["lr"],
                "wd": sweep["wd"],
                "validation": metrics["validation"],
            }
        )
    if changed:
        write_report(path, report)

print(json.dumps(ingested, sort_keys=True))
