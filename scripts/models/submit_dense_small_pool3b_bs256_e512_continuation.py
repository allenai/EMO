#!/usr/bin/env python3
"""Submit the authorized two-node 153M Pool-3B E448-E512 continuation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_dense_small_pool3b_bs256_e512_continuation as runner

WORKSPACE = "ai2/flex2"
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
NAME = "dense-153m-dclm3b-bs256-lr2e-3-wd0.1-integrated-e448-e512-two-node-v1"


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(arguments, input=input_text, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout


def validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full commit hash")
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, "origin/sewonm/icsl"], check=True)


def set_env(task: dict[str, Any], name: str, value: str) -> None:
    for variable in task.get("envVars", []):
        if variable.get("name") == name:
            variable["value"] = value
            return
    task.setdefault("envVars", []).append({"name": name, "value": value})


def producer_record(report: dict[str, Any]) -> dict[str, Any]:
    matches = [x for x in report["producers"] if x.get("id") == runner.EXPECTED_ID]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one registered 153M Pool-3B coordinate")
    return matches[0]


def verify_ready(record: dict[str, Any]) -> None:
    if 448 not in {int(x) for x in record.get("resolvedCheckpointEpochs", [])}:
        raise RuntimeError("E448 PD is not registered as retained")
    if "448" not in record.get("postDecayResults", {}):
        raise RuntimeError("matched E448 POST result is unresolved")
    if record.get("beakerStatus") in {"submitted", "queued", "scheduled", "running"}:
        raise RuntimeError("registered coordinate already has an active writer")
    if Path(str(record.get("output"))) != runner.EXPECTED_OUTPUT:
        raise RuntimeError("registered output differs from the exact E448 lineage")


def build_spec(item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    spec = copy.deepcopy(json.loads(command(["beaker", "experiment", "spec", str(item["baseExperiment"]), "--format", "json"])))
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("base experiment must contain exactly one task")
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
    set_env(task, "GIT_REF", revision)
    set_env(task, "GIT_BRANCH", "sewonm/icsl")
    set_env(task, "NUM_NODES", "2")
    set_env(task, "GANTRY_RDZV_ID", rendezvous[:12])
    set_env(task, "GANTRY_RDZV_PORT", str(29000 + int(rendezvous[:8], 16) % 1000))
    task["name"] = "main"
    task["arguments"] = ["python", "scripts/models/run_dense_small_pool3b_bs256_e512_continuation.py"]
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
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
        "from exact retained E448 PD step1153564 to exact E512 PD step1318359, then "
        "immediate isolated uncapped 10% WSD decay and heldout/downstream evaluation. "
        "Two synchronized 8-GPU nodes, rank microbatch 16, gradient accumulation 1, "
        "global batch 256, one logical writer, no minRuntime, auto-resume, eight retries; "
        "save every four epochs and stop at E512."
    )
    return spec


def validate_spec(spec: dict[str, Any], revision: str) -> None:
    task = spec["tasks"][0]
    env = {x["name"]: x.get("value") for x in task.get("envVars", [])}
    assert task["replicas"] == 2 and task["resources"]["gpuCount"] == 8
    assert task["leaderSelection"] and task["hostNetworking"]
    assert task["context"] == {"priority": task["context"]["priority"], "autoResume": True}
    assert "minRuntime" not in task["context"]
    assert spec["retry"]["allowedTaskRetries"] == 8
    assert env["GIT_REF"] == revision and env["NUM_NODES"] == "2"
    assert "GANTRY_RDZV_ID" in env and "GANTRY_RDZV_PORT" in env


def register(report: dict[str, Any], experiment: str, revision: str) -> None:
    record = producer_record(report)
    record.setdefault("experimentHistory", []).append({
        "experiment": record.get("experiment"), "job": record.get("job"),
        "revision": record.get("revision"), "status": record.get("status"),
        "maxValidatedEpoch": 448, "output": record.get("output"),
        "stoppedAt": datetime.now(tz=UTC).isoformat(),
    })
    record.update({
        "experiment": experiment, "revision": revision, "policy": runner.POLICY,
        "status": "submitted", "beakerStatus": "submitted", "currentEpoch": 452,
        "currentPhase": "repacked_shuffled_pool3b_constant_lr",
        "targetEpochs": sorted({*[int(x) for x in record.get("targetEpochs", [])], *runner.CHECKPOINT_EPOCHS}),
        "continuationSourceEpoch": 448, "continuationTargetEpoch": 512,
        "continuationCheckpointEpochs": list(runner.CHECKPOINT_EPOCHS),
        "evaluationEpochs": [512], "checkpointIntervalEpochs": 4,
        "checkpointCleanupKeepEpochs": [256, 320, 384, 448, 512],
        "role": "integrated_checkpoint_producer_and_evaluator",
        "evaluationEnabled": True, "decayEnabled": True,
        "postBranchesIsolatedFromConstantFrontier": True,
        "standaloneEvaluatorSubmissionsAuthorized": False,
        "futureEvaluatorSubmissionsAuthorized": False,
        "nodeCount": 2, "gpusPerNode": 8, "gpuCount": 16,
        "rankMicrobatchSequences": 16, "gradientAccumulationSteps": 1,
        "minRuntimeOmitted": True,
        "runtimeEstimate": {"producerHours": [8, 10], "postHours": [7, 8], "evaluationAndOverheadHours": [0.1, 0.5], "totalHours": [15, 18]},
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
    validate_revision(args.revision)
    config, item = runner.load(runner.DEFAULT_MANIFEST)
    report = json.loads(REPORT.read_text())
    verify_ready(producer_record(report))
    spec = build_spec(item, args.revision, args.priority)
    validate_spec(spec, args.revision)
    if args.print_spec:
        print(json.dumps(spec, indent=2))
        return
    output = command(["beaker", "experiment", "create", "-", "--name", NAME, "--workspace", WORKSPACE], input_text=json.dumps(spec))
    ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not ids:
        raise RuntimeError("submission returned no experiment ID")
    register(report, ids[0], args.revision)
    print(f"{NAME}: {ids[0]}")


if __name__ == "__main__":
    main()
