#!/usr/bin/env python3
"""Submit the exact 474M Pool-3B BS256 WD0.3 E32-to-E64 continuation."""

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

import run_dense_474m_pool3b_bs256_wd03_e64_two_node as runner

WORKSPACE = "ai2/flex2"
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
NAME = "dense-474m-dclm3b-bs256-lr2e-3-wd0.3-e32-e64-two-node-v1"
PREDECESSOR_EXPERIMENT = "01M1WMWN6DWG4W9Z6JSTQBY6W5"
PREDECESSOR_JOB = "01M1WTXF2VER11M5RWBYFKW053"


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


def coordinate(report: dict[str, Any]) -> dict[str, Any]:
    group = report.get("weightDecayProbes474m") or {}
    matches = [x for x in group.get("coordinates", []) if x.get("id") == runner.EXPECTED_ID]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one registered WD0.3 coordinate")
    return matches[0]


def predecessor_terminal() -> bool:
    payload = json.loads(command(["beaker", "job", "inspect", PREDECESSOR_JOB, "--format", "json"]))
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError("unexpected predecessor job inspection")
    job = payload[0]
    if (job.get("author") or {}).get("name") != "sewonm":
        raise RuntimeError("predecessor owner mismatch")
    status = job.get("status") or {}
    return any(key in status for key in ("exited", "finalized", "canceled", "cancelled"))


def verify_ready(record: dict[str, Any]) -> None:
    if record.get("experiment") != PREDECESSOR_EXPERIMENT or record.get("job") != PREDECESSOR_JOB:
        raise RuntimeError("registry does not point to the completed exact WD0.3 probe")
    if not predecessor_terminal():
        raise RuntimeError("predecessor is still active; refusing a simultaneous writer")
    if record.get("status") != "complete" or "32" not in record.get("postDecayResults", {}):
        raise RuntimeError("healthy matched E32 PD+POST is unresolved")
    if Path(str(record.get("output"))) != runner.EXPECTED_OUTPUT:
        raise RuntimeError("registry output differs from the canonical WD0.3 lineage")


def build_spec(item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    spec = copy.deepcopy(json.loads(command([
        "beaker", "experiment", "spec", str(item["baseExperiment"]), "--format", "json"
    ])))
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
    set_env(task, "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    task["name"] = "main"
    task["arguments"] = ["python", "scripts/models/run_dense_474m_pool3b_bs256_wd03_e64_two_node.py"]
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
        "Authorized 474M DCLM-3B DR+WT+EmbedWD BS256 LR2e-3 WD0.3 continuation "
        "from exact retained E32 PD step82397 through exact E64 PD step164794, then "
        "immediate isolated uncapped 10% WSD decay and heldout/downstream evaluation. "
        "Two synchronized 8-GPU nodes, rank microbatch 16, accumulation 1, global BS256, "
        "one logical writer, no minRuntime, auto-resume, eight retries, recovery every four epochs."
    )
    return spec


def validate_spec(spec: dict[str, Any], revision: str) -> None:
    task = spec["tasks"][0]
    env = {x["name"]: x.get("value") for x in task.get("envVars", [])}
    assert task["replicas"] == 2 and task["resources"]["gpuCount"] == 8
    assert task["leaderSelection"] and task["hostNetworking"]
    assert task["propagateFailure"] and task["propagatePreemption"]
    assert task["context"] == {"priority": task["context"]["priority"], "autoResume": True}
    assert "minRuntime" not in task["context"]
    assert spec["retry"]["allowedTaskRetries"] == 8
    assert env["GIT_REF"] == revision and env["NUM_NODES"] == "2"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert "GANTRY_RDZV_ID" in env and "GANTRY_RDZV_PORT" in env


def register(report: dict[str, Any], experiment: str, revision: str) -> None:
    record = coordinate(report)
    record.setdefault("experimentHistory", []).append({
        "experiment": PREDECESSOR_EXPERIMENT,
        "job": PREDECESSOR_JOB,
        "revision": record.get("revision"),
        "status": "complete_e16_e32",
        "maxValidatedEpoch": 32,
        "output": record.get("output"),
        "stoppedAt": datetime.now(tz=UTC).isoformat(),
    })
    record.update({
        "experiment": experiment,
        "revision": revision,
        "policy": runner.POLICY,
        "status": "submitted",
        "beakerStatus": "submitted",
        "currentEpoch": 36,
        "currentPhase": "repacked_shuffled_pool3b_constant_lr",
        "targetEpochs": [64],
        "continuationSourceEpoch": 32,
        "continuationSourceCheckpoint": str(runner.EXPECTED_OUTPUT / f"step{runner.checkpoint_step(32)}"),
        "continuationTargetEpoch": 64,
        "continuationCheckpointEpochs": list(runner.CHECKPOINT_EPOCHS),
        "evaluationEpochs": [16, 32, 64],
        "checkpointIntervalEpochs": 4,
        "checkpointCleanupKeepEpochs": [16, 32, 64],
        "role": "integrated_checkpoint_producer_and_evaluator",
        "evaluationEnabled": True,
        "decayEnabled": True,
        "postBranchesIsolatedFromConstantFrontier": True,
        "standaloneEvaluatorSubmissionsAuthorized": False,
        "futureEvaluatorSubmissionsAuthorized": False,
        "nodeCount": 2,
        "gpusPerNode": 8,
        "gpuCount": 16,
        "rankMicrobatchSequences": 16,
        "gradientAccumulationSteps": 1,
        "minRuntimeOmitted": True,
        "runtimeEstimate": {
            "producerHours": [10, 12],
            "postHours": [2, 3],
            "totalHours": [12, 15],
        },
        "submittedAt": datetime.now(tz=UTC).isoformat(),
    })
    for key in ("job", "jobs", "wandbHealth", "needsAttention", "decision", "lastDecisionEpoch"):
        record.pop(key, None)
    report["weightDecayProbes474m"]["status"] = "running"
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
    verify_ready(coordinate(report))
    spec = build_spec({**item, "baseExperiment": config["producerCoordinate"]["baseExperiment"]}, args.revision, args.priority)
    validate_spec(spec, args.revision)
    if args.print_spec:
        print(json.dumps(spec, indent=2))
        return
    output = command(
        ["beaker", "experiment", "create", "-", "--name", NAME, "--workspace", WORKSPACE],
        input_text=json.dumps(spec),
    )
    ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not ids:
        raise RuntimeError("submission returned no experiment ID")
    register(report, ids[0], args.revision)
    print(f"{NAME}: {ids[0]}")


if __name__ == "__main__":
    main()
