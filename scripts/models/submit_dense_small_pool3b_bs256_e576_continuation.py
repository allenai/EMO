#!/usr/bin/env python3
"""Submit the authorized two-node 153M Pool-3B E512-E576 continuation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_dense_small_pool3b_bs256_e576_continuation as runner
import submit_dense_small_pool3b_bs256_e512_continuation as base

WORKSPACE = "ai2/flex2"
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
NAME = "dense-153m-dclm3b-bs256-lr2e-3-wd0.1-integrated-e568-e576-protected-two-node-v2"
MIN_RUNTIME = "8h"


def producer_record(report: dict[str, Any]) -> dict[str, Any]:
    matches = [x for x in report["producers"] if x.get("id") == runner.EXPECTED_ID]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one registered 153M Pool-3B coordinate")
    return matches[0]


def verify_ready(record: dict[str, Any]) -> None:
    if 512 not in {int(x) for x in record.get("resolvedCheckpointEpochs", [])}:
        raise RuntimeError("E512 PD is not registered as retained")
    if "512" not in record.get("postDecayResults", {}):
        raise RuntimeError("matched E512 POST result is unresolved")
    if record.get("beakerStatus") in {"submitted", "queued", "scheduled", "running"}:
        raise RuntimeError("registered coordinate already has an active writer")
    if Path(str(record.get("output"))) != runner.EXPECTED_OUTPUT:
        raise RuntimeError("registered output differs from the exact E512 lineage")


def build_spec(item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    spec = copy.deepcopy(json.loads(base.command(["beaker", "experiment", "spec", str(item["baseExperiment"]), "--format", "json"])))
    tasks = spec.get("tasks", [])
    if len(tasks) not in {1, 2}:
        raise RuntimeError("base experiment must contain one logical task or two materialized replicas")
    if len(tasks) == 2:
        if {task.get("name") for task in tasks} != {"main-replica-0", "main-replica-1"}:
            raise RuntimeError("unexpected materialized replica names")
        spec["tasks"] = [copy.deepcopy(tasks[0])]
    task = spec["tasks"][0]
    if "/weka/oe-training-default" not in {x.get("mountPath") for x in task.get("datasets", [])}:
        raise RuntimeError("base experiment is missing Weka")
    blocked = {"GANTRY_USE_TORCHRUN", "GANTRY_RDZV_ID", "GANTRY_RDZV_PORT", "NUM_NODES"}
    task["envVars"] = [
        x for x in task.get("envVars", [])
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
    task["arguments"] = ["python", "scripts/models/run_dense_small_pool3b_bs256_e576_continuation.py"]
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {
        "priority": priority,
        "minRuntime": MIN_RUNTIME,
        "autoResume": True,
    }
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
        "Authorized 153M DCLM-3B DR+WT+EmbedWD BS256 LR2e-3 WD0.1 continuation "
        "from exact retained E512 PD step1318359 to exact E576 PD step1483154, then "
        "immediate isolated uncapped 10% WSD decay and heldout/downstream evaluation. "
        "Two synchronized 8-GPU nodes, rank microbatch 16, gradient accumulation 1, "
        "global batch 256, one logical writer, protected minRuntime=8h, auto-resume, "
        "eight retries; save PD every four epochs and resumable WSD checkpoints at "
        "25/50/75/100%, then stop at E576."
    )
    return spec


def validate_spec(spec: dict[str, Any], revision: str) -> None:
    task = spec["tasks"][0]
    env = {x["name"]: x.get("value") for x in task.get("envVars", [])}
    assert task["replicas"] == 2 and task["resources"]["gpuCount"] == 8
    assert task["leaderSelection"] and task["hostNetworking"]
    assert task["propagateFailure"] and task["propagatePreemption"]
    assert task["context"] == {
        "priority": task["context"]["priority"],
        "minRuntime": MIN_RUNTIME,
        "autoResume": True,
    }
    assert spec["retry"]["allowedTaskRetries"] == 8
    assert env["GIT_REF"] == revision and env["NUM_NODES"] == "2"
    assert "GANTRY_RDZV_ID" in env and "GANTRY_RDZV_PORT" in env


def register(report: dict[str, Any], experiment: str, revision: str) -> None:
    record = producer_record(report)
    record.setdefault("experimentHistory", []).append({
        "experiment": record.get("experiment"), "job": record.get("job"),
        "revision": record.get("revision"), "status": record.get("status"),
        "maxValidatedEpoch": 512, "output": record.get("output"),
        "stoppedAt": datetime.now(tz=UTC).isoformat(),
    })
    record.update({
        "experiment": experiment, "revision": revision, "policy": runner.POLICY,
        "status": "submitted", "beakerStatus": "submitted", "currentEpoch": 516,
        "currentPhase": "repacked_shuffled_pool3b_constant_lr",
        "targetEpochs": sorted({*[int(x) for x in record.get("targetEpochs", [])], *runner.CHECKPOINT_EPOCHS}),
        "continuationSourceEpoch": 512, "continuationTargetEpoch": 576,
        "continuationCheckpointEpochs": list(runner.CHECKPOINT_EPOCHS),
        "evaluationEpochs": [576], "checkpointIntervalEpochs": 4,
        "checkpointCleanupKeepEpochs": [256, 320, 384, 448, 512, 576],
        "role": "integrated_checkpoint_producer_and_evaluator",
        "evaluationEnabled": True, "decayEnabled": True,
        "postBranchesIsolatedFromConstantFrontier": True,
        "standaloneEvaluatorSubmissionsAuthorized": False,
        "futureEvaluatorSubmissionsAuthorized": False,
        "nodeCount": 2, "gpusPerNode": 8, "gpuCount": 16,
        "rankMicrobatchSequences": 16, "gradientAccumulationSteps": 1,
        "minRuntime": MIN_RUNTIME, "minRuntimeOmitted": False,
        "postDecayRecoverySteps": list(runner.POST_RECOVERY_STEPS),
        "runtimeEstimate": {"producerHours": [8, 10], "postHours": [8, 9], "evaluationAndOverheadHours": [0.1, 0.5], "totalHours": [16, 20]},
        "submittedAt": datetime.now(tz=UTC).isoformat(),
    })
    for key in ("job", "jobs", "wandbHealth", "needsAttention", "decision", "lastDecisionEpoch", "stopAuthorized", "stopAfterEpoch"):
        record.pop(key, None)
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text("window.ICSL_CHECKPOINT_PRODUCER_GRID=" + json.dumps(report, separators=(",", ":")) + ";\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-spec", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    if args.print_spec == args.submit_if_ready:
        raise SystemExit("choose exactly one of --print-spec or --submit-if-ready")
    base.validate_revision(args.revision)
    runner.configure_base()
    config, item = base.runner.load(runner.DEFAULT_MANIFEST)
    report = json.loads(REPORT.read_text())
    verify_ready(producer_record(report))
    spec = build_spec(item, args.revision, args.priority)
    validate_spec(spec, args.revision)
    if args.print_spec:
        print(json.dumps(spec, indent=2))
        return
    output = base.command(["beaker", "experiment", "create", "-", "--name", NAME, "--workspace", WORKSPACE], input_text=json.dumps(spec))
    ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not ids:
        raise RuntimeError("submission returned no experiment ID")
    register(report, ids[0], args.revision)
    print(f"{NAME}: {ids[0]}")


if __name__ == "__main__":
    main()
