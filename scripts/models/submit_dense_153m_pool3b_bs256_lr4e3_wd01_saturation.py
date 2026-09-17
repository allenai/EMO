#!/usr/bin/env python3
"""Submit the protected two-node 153M Pool-3B LR4e-3 probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_dense_153m_pool3b_bs256_lr4e3_wd01_saturation as runner
import submit_dense_small_pool3b_bs256_e512_continuation as base

WORKSPACE = "ai2/flex2"
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
NAME = "dense-153m-dclm3b-bs256-lr4e-3-wd0.1-saturation-protected-two-node-v4"
MIN_RUNTIME = "8h"


def build_spec(item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    spec = copy.deepcopy(
        json.loads(
            base.command(
                ["beaker", "experiment", "spec", str(item["baseExperiment"]), "--format", "json"]
            )
        )
    )
    tasks = spec.get("tasks", [])
    if len(tasks) != 1:
        raise RuntimeError("trusted base experiment must contain exactly one task")
    task = tasks[0]
    if "/weka/oe-training-default" not in {x.get("mountPath") for x in task.get("datasets", [])}:
        raise RuntimeError("trusted base experiment is missing Weka")
    blocked = {"GANTRY_USE_TORCHRUN", "GANTRY_RDZV_ID", "GANTRY_RDZV_PORT", "NUM_NODES"}
    task["envVars"] = [
        x
        for x in task.get("envVars", [])
        if x.get("name") not in blocked
        and not (str(x.get("name", "")).startswith("BEAKER_") and x.get("name") != "BEAKER_TOKEN")
    ]
    rendezvous = hashlib.sha256(NAME.encode()).hexdigest()
    base.set_env(task, "GIT_REF", revision)
    base.set_env(task, "GIT_BRANCH", "sewonm/icsl")
    base.set_env(task, "NUM_NODES", "2")
    base.set_env(task, "GANTRY_RDZV_ID", rendezvous[:12])
    base.set_env(task, "GANTRY_RDZV_PORT", str(29000 + int(rendezvous[:8], 16) % 1000))
    task["name"] = "main"
    task["arguments"] = [
        "python",
        "scripts/models/run_dense_153m_pool3b_bs256_lr4e3_wd01_saturation.py",
    ]
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "minRuntime": MIN_RUNTIME, "autoResume": True}
    task.update(
        replicas=2,
        leaderSelection=True,
        hostNetworking=True,
        propagateFailure=True,
        propagatePreemption=True,
        synchronizedStartTimeout="90m",
    )
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        "153M DCLM-3B BS256 DR+WT+EmbedWD LR4e-3 WD0.1 matched saturation probe. "
        "Bootstrap the exact Pool-1B E1 source from scratch, bridge the sealed disjoint 2B "
        "extension, then train the canonical Pool-3B trajectory on two synchronized 8-GPU "
        "nodes. Save one rolling recovery checkpoint about every epoch and permanent PD "
        "every 32 epochs through E384; evaluate isolated uncapped 10% WSD "
        "POST at E64/E128/E192/E256/E320/E384; stop on first adjacent "
        "non-improvement. Protected minRuntime=8h, auto-resume, eight retries."
    )
    return spec


def validate_spec(spec: dict[str, Any], revision: str) -> None:
    task = spec["tasks"][0]
    env = {x["name"]: x.get("value") for x in task.get("envVars", [])}
    assert task["replicas"] == 2 and task["resources"]["gpuCount"] == 8
    assert task["leaderSelection"] and task["hostNetworking"]
    assert task["propagateFailure"] and task["propagatePreemption"]
    assert task["context"]["minRuntime"] == MIN_RUNTIME and task["context"]["autoResume"]
    assert spec["retry"]["allowedTaskRetries"] == 8
    assert env["GIT_REF"] == revision and env["NUM_NODES"] == "2"


def register(experiment: str, revision: str, *, replace_existing: bool) -> None:
    report = json.loads(REPORT.read_text())
    records = report.setdefault("pool3bLearningRateProbes", [])
    existing = next((record for record in records if record.get("id") == runner.EXPECTED_ID), None)
    if existing is not None and not replace_existing:
        raise RuntimeError("Pool-3B LR4e-3 probe is already registered")
    history = list((existing or {}).get("experimentHistory", []))
    if existing is not None:
        history.append(
            {
                "experiment": existing.get("experiment"),
                "jobs": existing.get("jobs", []),
                "revision": existing.get("revision"),
                "status": (
                    "failed_resume_guard_replaced"
                    if existing.get("status") == "failed"
                    else "canceled_to_enable_per_epoch_recovery_checkpointing"
                ),
                "replacedAt": datetime.now(tz=UTC).isoformat(),
            }
        )
        records.remove(existing)
    record = {
            **copy.deepcopy(existing or {}),
            "id": runner.EXPECTED_ID,
            "role": "integrated_checkpoint_producer_and_evaluator",
            "policy": runner.POLICY,
            "model": "153m",
            "pool": "dclm3b",
            "batchSequences": 256,
            "learningRate": "4e-3",
            "weightDecay": "0.1",
            "nodeCount": 2,
            "gpusPerNode": 8,
            "gpuCount": 16,
            "rankMicrobatchSequences": 16,
            "gradientAccumulationSteps": 1,
            "resolvedCheckpointEpochs": list((existing or {}).get("resolvedCheckpointEpochs", [])),
            "resolvedPostEpochs": list((existing or {}).get("resolvedPostEpochs", [])),
            "postDecayResults": copy.deepcopy((existing or {}).get("postDecayResults", {})),
            "targetEpochs": list(runner.CHECKPOINT_EPOCHS),
            "evaluationEpochs": list(runner.EVALUATION_EPOCHS),
            "currentEpoch": 384,
            "currentPhase": "repacked_shuffled_pool3b_constant_lr",
            "status": "submitted",
            "beakerStatus": "submitted",
            "experiment": experiment,
            "jobs": [],
            "revision": revision,
            "bootstrapOutput": str(runner.BOOTSTRAP_OUTPUT),
            "output": str(runner.OUTPUT),
            "minRuntime": MIN_RUNTIME,
            "stopOnAdjacentPostNonImprovement": True,
            "submittedAt": datetime.now(tz=UTC).isoformat(),
        }
    record.pop("job", None)
    record.pop("wandbHealth", None)
    if history:
        record["experimentHistory"] = history
    records.append(record)
    report["pool3bLearningRateProbeCount"] = len(records)
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    rendered = json.dumps(report, indent=2) + "\n"
    REPORT.write_text(rendered)
    REPORT_JS.write_text(
        "window.ICSL_CHECKPOINT_PRODUCER_GRID=" + json.dumps(report, indent=2) + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-spec", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    if args.print_spec == args.submit_if_ready:
        raise SystemExit("choose exactly one of --print-spec or --submit-if-ready")
    base.validate_revision(args.revision)
    _, item = runner.load(runner.DEFAULT_MANIFEST)
    report = json.loads(REPORT.read_text())
    if (
        any(record.get("id") == runner.EXPECTED_ID for record in report.get("pool3bLearningRateProbes", []))
        and not args.replace_existing
    ):
        raise RuntimeError("Pool-3B LR4e-3 probe is already registered")
    spec = build_spec(item, args.revision, args.priority)
    validate_spec(spec, args.revision)
    if args.print_spec:
        print(json.dumps(spec, indent=2))
        return
    output = base.command(
        ["beaker", "experiment", "create", "-", "--name", NAME, "--workspace", WORKSPACE],
        input_text=json.dumps(spec),
    )
    ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not ids:
        raise RuntimeError("submission returned no experiment ID")
    register(ids[0], args.revision, replace_existing=args.replace_existing)
    print(f"{NAME}: {ids[0]}")


if __name__ == "__main__":
    main()
