#!/usr/bin/env python3
"""Guardedly submit the authorized standalone Dense-1B BS64 E24 evaluator."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_constant_checkpoint_producer as producer

WORKSPACE = "ai2/flex2"
MANIFEST = Path("scripts/models/manifests/dense-constant-checkpoint-producers-v1.json")
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
RUNNER = "scripts/models/run_dense_1b_bs64_e24_evaluator.py"
PRODUCER_ID = "dense-1b-dclm3b-bs64-lr1e-3-wd0.3"
EPOCH = 24
REPORT_EVALUATOR_ID = "dense-1b-dclm3b-bs64-lr1e-3-wd0.3-post-e8-e16"
STANDALONE_ID = "dense-1b-dclm3b-bs64-lr1e-3-wd0.3-post-e24"
GUARDED_NAME = f"{STANDALONE_ID}-v1"
MIN_RUNTIME = "6h"


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(
        arguments,
        check=True,
        input=input_text,
        text=True,
        capture_output=True,
    ).stdout


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


def load_item() -> dict[str, Any]:
    manifest = producer.load_manifest(MANIFEST)
    item = producer.coordinate(manifest, PRODUCER_ID)
    if (
        item["model"] != "1b"
        or item["pool"] != "dclm3b"
        or int(item["batchSequences"]) != 64
        or Decimal(str(item["learningRate"])) != Decimal("1e-3")
        or Decimal(str(item["weightDecay"])) != Decimal("0.3")
    ):
        raise ValueError("standalone E24 producer recipe mismatch")
    source_step = producer.stable_step(EPOCH, int(item["poolTokens"]), 64)
    if source_step != 247192:
        raise RuntimeError(f"unexpected E24 source step {source_step}")
    source = Path(str(item["output"])) / f"step{source_step}"
    # Submission commonly runs from a laptop without the Weka mount.  When the
    # coordinate directory is locally visible, retain the full checkpoint
    # validation here; otherwise the report gate below proves that the exact
    # checkpoint was externally validated, and the evaluator validates the
    # source again inside its Weka-mounted Beaker job before loading it.
    if source.parent.is_dir():
        producer.validate_source_checkpoint(
            {**item, "sourceEpoch": EPOCH, "sourceCheckpoint": str(source)}
        )
    return item


def existing_experiment() -> str | None:
    payload = json.loads(
        command(
            [
                "beaker",
                "workspace",
                "experiments",
                WORKSPACE,
                "--text",
                GUARDED_NAME,
                "--format",
                "json",
            ]
        )
    )
    values = payload if isinstance(payload, list) else payload.get("experiments", [])
    matches = [value for value in values if value.get("name") == GUARDED_NAME]
    if len(matches) > 1:
        raise RuntimeError(f"multiple experiments use guarded name {GUARDED_NAME}")
    return str(matches[0]["id"]) if matches else None


def spec_for(item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    spec = copy.deepcopy(
        json.loads(
            command(
                ["beaker", "experiment", "spec", str(item["baseExperiment"]), "--format", "json"]
            )
        )
    )
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("trusted base experiment must contain one task")
    task = spec["tasks"][0]
    if "/weka/oe-training-default" not in {
        dataset.get("mountPath") for dataset in task.get("datasets", [])
    }:
        raise RuntimeError("trusted base experiment is missing the Weka mount")
    task["name"] = "main"
    task["arguments"] = ["python", RUNNER, "--manifest", str(MANIFEST)]
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
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {
        "priority": priority,
        "minRuntime": MIN_RUNTIME,
        "autoResume": True,
    }
    task["hostNetworking"] = False
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        "Dense-1B DCLM-3B BS64 independent uncapped 10% WSD decay and heldout "
        f"evaluation from exact E24 PD step247192; buffered minRuntime={MIN_RUNTIME}."
    )
    return spec


def create(spec: dict[str, Any]) -> str:
    existing = existing_experiment()
    if existing:
        return existing
    output = command(
        ["beaker", "experiment", "create", "-", "--name", GUARDED_NAME, "--workspace", WORKSPACE],
        input_text=json.dumps(spec),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError("submission returned no experiment ID")
    return identifiers[0]


def write_report(report: dict) -> None:
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_CHECKPOINT_PRODUCER_GRID="
        + json.dumps(report, separators=(",", ":"))
        + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--submit-if-ready", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    validate_revision(args.revision)
    item = load_item()
    report = json.loads(REPORT.read_text())
    producer_record = next(value for value in report["producers"] if value["id"] == PRODUCER_ID)
    if producer_record.get("stopAfterEpoch") != EPOCH or not producer_record.get("stopAuthorized"):
        raise RuntimeError("report does not authorize the BS64 E24 stop/evaluator gate")
    if EPOCH not in {int(value) for value in producer_record.get("resolvedCheckpointEpochs", [])}:
        raise RuntimeError("report has not resolved the exact BS64 E24 checkpoint")
    report_evaluator = next(
        value for value in report["evaluators"] if value["id"] == REPORT_EVALUATOR_ID
    )
    spec = spec_for(item, args.revision, args.priority)
    if args.print_only:
        print(json.dumps(spec, indent=2))
        return
    if not args.submit_if_ready:
        print(f"{STANDALONE_ID}: ready with minRuntime={MIN_RUNTIME}; pass --submit-if-ready")
        return
    experiment = create(spec)
    additional = report_evaluator.setdefault("additionalExperiments", [])
    matches = [value for value in additional if int(value.get("epoch", -1)) == EPOCH]
    if matches and matches[0].get("experiment") != experiment:
        raise RuntimeError("a different E24 evaluator is already registered")
    if not matches:
        additional.append(
            {
                "epoch": EPOCH,
                "experiment": experiment,
                "revision": args.revision,
                "status": "submitted",
                "minRuntime": MIN_RUNTIME,
            }
        )
    report_evaluator["plannedPostEpochs"] = [
        value for value in report_evaluator.get("plannedPostEpochs", []) if int(value) != EPOCH
    ]
    report_evaluator["status"] = "running"
    write_report(report)
    print(f"{STANDALONE_ID}: {experiment} minRuntime={MIN_RUNTIME}")


if __name__ == "__main__":
    main()
