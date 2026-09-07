#!/usr/bin/env python3
"""Guardedly submit the 13 retained Pool-333M decay/evaluation jobs."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_dense_dclm333m_checkpoint_producer as integrated
import run_dense_dclm333m_retained_checkpoint_evaluator as evaluator


WORKSPACE = "ai2/flex2"
MANIFEST = Path("scripts/models/manifests/dense-dclm333m-checkpoint-producers-v1.json")
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
REPORT_JS_PREFIX = "window.ICSL_CHECKPOINT_PRODUCER_GRID="
RUNNER = "scripts/models/run_dense_dclm333m_retained_checkpoint_evaluator.py"


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(arguments, input=input_text, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout


def validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("--revision must be a full 40-character commit hash")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "origin/sewonm/icsl"],
        check=True,
    )


def set_env(task: dict[str, Any], name: str, value: str) -> None:
    for variable in task.get("envVars", []):
        if variable.get("name") == name:
            variable["value"] = value
            return
    task.setdefault("envVars", []).append({"name": name, "value": value})


def guarded_name(item: dict[str, Any], epoch: int) -> str:
    return f"{item['id']}-retained-post-e{epoch}-v1"


def existing_named_experiment(name: str) -> str | None:
    payload = json.loads(
        command(
            [
                "beaker",
                "workspace",
                "experiments",
                WORKSPACE,
                "--text",
                name,
                "--format",
                "json",
            ]
        )
    )
    values = payload if isinstance(payload, list) else payload.get("experiments", [])
    matches = [row for row in values if row.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError(f"multiple experiments use guarded name {name}")
    return str(matches[0]["id"]) if matches else None


def spec_for(item: dict[str, Any], epoch: int, revision: str, priority: str) -> dict[str, Any]:
    spec = copy.deepcopy(
        json.loads(
            command(
                ["beaker", "experiment", "spec", str(item["baseExperiment"]), "--format", "json"]
            )
        )
    )
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("trusted base experiment must contain exactly one task")
    task = spec["tasks"][0]
    if "/weka/oe-training-default" not in {
        dataset.get("mountPath") for dataset in task.get("datasets", [])
    }:
        raise RuntimeError("trusted base experiment is missing the Weka mount")
    task["name"] = "main"
    task["arguments"] = [
        "python",
        RUNNER,
        "--manifest",
        str(MANIFEST),
        "--coordinate",
        str(item["id"]),
        "--epoch",
        str(epoch),
    ]
    blocked = {
        "GANTRY_USE_TORCHRUN",
        "GANTRY_RDZV_ID",
        "GANTRY_RDZV_PORT",
        "NUM_NODES",
        "PYTORCH_CUDA_ALLOC_CONF",
    }
    task["envVars"] = [
        variable
        for variable in task.get("envVars", [])
        if variable.get("name") not in blocked
        and not (
            str(variable.get("name", "")).startswith("BEAKER_")
            and variable.get("name") != "BEAKER_TOKEN"
        )
    ]
    set_env(task, "GIT_REF", revision)
    set_env(task, "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    task["resources"] = {"gpuCount": integrated.gpu_count(item), "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
    task["hostNetworking"] = False
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"Evaluation-only decay and heldout evaluation of retained {item['model']} "
        f"Pool-333M BS{item['batchSequences']} LR{item['learningRate']} "
        f"WD{item['weightDecay']} E{epoch}; isolated POST branch; one 8-GPU node; "
        "auto-resume; eight retries; minRuntime omitted; no producer authorization."
    )
    return spec


def create(item: dict[str, Any], epoch: int, revision: str, priority: str) -> str:
    existing = existing_named_experiment(guarded_name(item, epoch))
    if existing:
        return existing
    output = command(
        [
            "beaker",
            "experiment",
            "create",
            "-",
            "--name",
            guarded_name(item, epoch),
            "--workspace",
            WORKSPACE,
        ],
        input_text=json.dumps(spec_for(item, epoch, revision, priority)),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError(f"submission returned no experiment ID for {item['id']} E{epoch}")
    return identifiers[0]


def inspect_experiment(experiment: str) -> tuple[str, str]:
    payload = json.loads(command(["beaker", "experiment", "inspect", experiment, "--format", "json"]))
    if str(payload.get("author", {}).get("name")) != "sewonm":
        raise RuntimeError(f"owner mismatch for {experiment}")
    jobs = [job for job in payload.get("jobs", []) if job.get("id")]
    if len(jobs) != 1:
        raise RuntimeError(f"expected exactly one initial job for {experiment}")
    return str(jobs[0]["id"]), str(jobs[0].get("status", "submitted"))


def write_report(records: list[dict[str, Any]]) -> None:
    report = json.loads(REPORT.read_text())
    if report.get("dclm333mRetainedEvaluators"):
        raise RuntimeError("retained Pool-333M evaluators are already registered")
    report["dclm333mRetainedEvaluatorPolicy"] = evaluator.POLICY
    report["dclm333mRetainedEvaluators"] = records
    report["dclm333mRetainedEvaluatorCount"] = len(records)
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    rendered = json.dumps(report, indent=2) + "\n"
    REPORT.write_text(rendered)
    REPORT_JS.write_text(REPORT_JS_PREFIX + rendered.rstrip() + ";\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-specs", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    validate_revision(args.revision)
    manifest = integrated.load_manifest(MANIFEST)
    targets = [
        (integrated.coordinate(manifest, coordinate_id), epoch)
        for coordinate_id, epochs in evaluator.TARGETS.items()
        for epoch in epochs
    ]
    if len(targets) != 13:
        raise RuntimeError("retained Pool-333M evaluator target count must remain 13")
    for item, epoch in targets:
        evaluator.load(MANIFEST, str(item["id"]), epoch)
    if args.print_specs:
        for item, epoch in targets:
            print(json.dumps(spec_for(item, epoch, args.revision, args.priority), indent=2))
        return
    if not args.submit_if_ready:
        for item, epoch in targets:
            print(f"{guarded_name(item, epoch)}: ready")
        return
    records = []
    for item, epoch in targets:
        experiment = create(item, epoch, args.revision, args.priority)
        job, state = inspect_experiment(experiment)
        record = {
            "id": guarded_name(item, epoch),
            "role": "evaluation_only_retained_checkpoint_wsd_and_heldout",
            "policy": evaluator.POLICY,
            "producerId": str(item["id"]),
            "model": str(item["model"]),
            "pool": "dclm333m",
            "batchSequences": int(item["batchSequences"]),
            "learningRate": str(item["learningRate"]),
            "weightDecay": str(item["weightDecay"]),
            "epoch": epoch,
            "sourceCheckpoint": str(evaluator.source_checkpoint(item, epoch)),
            "output": str(integrated.state_dir(item) / "post_decay_runs" / f"e{epoch}"),
            "status": "submitted",
            "beakerStatus": state,
            "experiment": experiment,
            "job": job,
            "jobs": [job],
            "revision": args.revision,
            "submittedAt": datetime.now(tz=UTC).isoformat(),
            "author": "sewonm",
            "futureProducerStagesAuthorized": False,
        }
        records.append(record)
        print(f"{record['id']}: experiment={experiment} job={job}")
    write_report(records)


if __name__ == "__main__":
    main()
