#!/usr/bin/env python3
"""Refresh the three persistent Dense-1B DR+WT E1->E24 chains."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import monitor_dense_1b_weight_tying as common

REPORT_PATH = Path("reports/0802/data/wsd_data_loader_1b.json")
REPORT_JS_PATH = REPORT_PATH.with_suffix(".js")
STAGE_START = re.compile(r"SEQUENTIAL_WT_STAGE_START epoch=(1|2|4|8|12|16|20|24)")
STAGE_COMPLETE = re.compile(r"SEQUENTIAL_WT_STAGE_COMPLETE epoch=(1|2|4|8|12|16|20|24)")
SATURATED = re.compile(r"SEQUENTIAL_EMBWD_SATURATED epoch=(1|2|4|8|12|16|20|24)")


def load_report() -> dict[str, Any]:
    return json.loads(REPORT_PATH.read_text())


def write_report(report: dict[str, Any]) -> None:
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS_PATH.write_text(
        "window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def stage_sections(logs: str) -> dict[int, str]:
    cleaned = common.ANSI.sub("", logs)
    matches = list(STAGE_START.finditer(cleaned))
    return {
        int(match.group(1)): cleaned[
            match.start() : matches[index + 1].start() if index + 1 < len(matches) else None
        ]
        for index, match in enumerate(matches)
    }


def parse_stage(epoch: int, section: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    step_values = common.TRAIN_STEP.findall(section)
    if step_values:
        step, total = step_values[-1]
        result["progress"] = {
            "step": int(step.replace(",", "")),
            "totalSteps": int(total.replace(",", "")),
            "percent": round(100 * int(step.replace(",", "")) / int(total.replace(",", "")), 1),
        }
    train_values = common.TRAIN_LOSS.findall(section)
    if train_values:
        result.setdefault("progress", {})["latestTrain"] = float(train_values[-1])
    if STAGE_COMPLETE.search(section):
        validation_values = common.WANDB_VALIDATION_LOSS.findall(
            section
        ) or common.VALIDATION_LOSS.findall(section)
        if not train_values or not validation_values:
            raise RuntimeError(f"completed sequential E{epoch} is missing train or validation CE")
        train = float(train_values[-1])
        validation = float(validation_values[-1])
        wandb_values = common.WANDB_RUN.findall(section)
        result.update(
            {
                "status": "complete",
                "train": train,
                "validation": round(validation, 3),
                "validationExact": validation,
                "gap": round(validation - train, 6),
            }
        )
        if wandb_values:
            result["wandb"] = wandb_values[-1]
    else:
        result["status"] = "running"
    return result


def refresh(report: dict[str, Any], sequential: dict[str, Any]) -> str:
    mode = "_dr_wt_embwd" if sequential.get("decayEmbeddings") else "_dr_wt"
    canonical_output = (
        "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b/"
        f"bs{sequential['batchSequences']}{mode}_lr1e-3_wd{sequential['wd']}"
    )
    sequential["canonicalOutput"] = canonical_output
    sequential["mergeAfterCompletion"] = sequential["output"] != canonical_output
    experiment_id = sequential["experiment"]
    inspected = common.inspect_experiment(experiment_id)
    state = common.experiment_state(inspected)
    jobs = inspected.get("jobs") or []
    job_ids = [job["id"] for job in jobs if job.get("id")]
    sequential["status"] = state
    sequential["beakerStatus"] = state
    if job_ids:
        sequential["job"] = job_ids[0]
        sequential["jobs"] = job_ids

    sections: dict[int, str] = {}
    logs = ""
    if state in {"running", "complete", "failed"}:
        logs = common.experiment_logs(experiment_id)
        sections = stage_sections(logs)
    stages_by_epoch = {int(stage["epoch"]): stage for stage in sequential["stages"]}
    completed_epochs = []
    coordinate = next(run for run in report["runs"] if run["id"] == sequential["coordinateRunId"])
    for epoch, section in sections.items():
        parsed = parse_stage(epoch, section)
        stage = stages_by_epoch[epoch]
        stage.update(parsed)
        if parsed["status"] != "complete":
            continue
        completed_epochs.append(epoch)
        coordinate.setdefault("results", {})[str(epoch)] = {
            key: value for key, value in parsed.items() if key != "progress"
        } | {
            "experiment": experiment_id,
            "beaker": experiment_id,
            "job": sequential.get("job"),
            "revision": sequential["revision"],
            "output": sequential["output"],
            "preDecayCheckpoint": stage["preDecayCheckpoint"],
            "endpointCheckpoint": stage["endpointCheckpoint"],
            "sequential": True,
            "reason": (
                "Completed inside the persistent E1->E24 DR+WT chain; the next stage "
                "resumes this stage's exact pre-decay checkpoint."
            ),
        }

    targets = [int(value) for value in sequential["targets"]]
    saturated_values = SATURATED.findall(common.ANSI.sub("", logs)) if logs else []
    saturated_epoch = int(saturated_values[-1]) if saturated_values else None
    remaining = (
        [] if saturated_epoch is not None else [epoch for epoch in targets if epoch not in completed_epochs]
    )
    current = remaining[0] if remaining else None
    sequential["currentEpoch"] = current
    coordinate["status"] = state
    coordinate["activeEpoch"] = current
    coordinate["experiment"] = experiment_id
    coordinate["beaker"] = experiment_id
    coordinate["job"] = sequential.get("job")
    coordinate["jobs"] = sequential.get("jobs", [])
    coordinate["output"] = sequential["output"]
    coordinate["revision"] = sequential["revision"]
    if saturated_epoch is not None:
        sequential["stoppingReason"] = "heldout_validation_saturated"
        sequential["saturatedEpoch"] = saturated_epoch
        sequential["reason"] = (
            f"Persistent chain stopped after E{saturated_epoch} because held-out validation "
            "CE did not strictly improve."
        )
    elif state == "failed":
        sequential[
            "reason"
        ] = f"Persistent chain failed during E{current}; completed earlier stages remain valid."
    elif state == "complete":
        sequential["reason"] = "Persistent E1->E24 chain completed all eight evaluated stages."

    detail = ""
    if current is not None and stages_by_epoch[current].get("progress"):
        progress = stages_by_epoch[current]["progress"]
        detail = (
            f" E{current} {progress.get('percent', 0):g}% "
            f"({progress.get('step', 0)}/{progress.get('totalSteps', 0)})"
        )
        if "latestTrain" in progress:
            detail += f" train={progress['latestTrain']:.3f}"
    return (
        f"{sequential['id']} {state}; completed={completed_epochs or 'none'};"
        f" current=E{current if current is not None else 'done'}{detail}; {experiment_id}"
    )


def main() -> None:
    report = load_report()
    records = report.get("weightTyingSequentialRuns") or []
    if len(records) != 3:
        raise RuntimeError(
            f"expected exactly three registered sequential chains, found {len(records)}"
        )
    embedding_decay_records = report.get("weightTyingEmbeddingDecaySequentialRuns") or []
    if embedding_decay_records and len(embedding_decay_records) != 3:
        raise RuntimeError(
            "expected either zero or three registered embedding-decay chains, found "
            f"{len(embedding_decay_records)}"
        )
    messages = [refresh(report, record) for record in records + embedding_decay_records]
    write_report(report)
    print("\n".join(messages))


if __name__ == "__main__":
    main()
