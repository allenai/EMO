#!/usr/bin/env python3
"""Submit the exact two-node 1B Pool-3B E80-to-E96 continuation."""

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

import run_dense_1b_pool3b_bs128_e80_e96_two_node as runner

WORKSPACE = "ai2/flex2"
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
NAME = "dense-1b-dclm3b-bs128-lr1e-3-wd0.3-integrated-e80-e96-two-node-v1"
PREVIOUS_EXPERIMENT = "01M1YAWGKHT9EX1PBN33Z5X446"
PREVIOUS_JOBS = ["01M1ZP3ED04PJ5SXX1JCK6VM79", "01M1ZP3EHXHX7TDMWSBYXR4WSF"]


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
    matches = [record for record in report["producers"] if record.get("id") == runner.EXPECTED_ID]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {runner.EXPECTED_ID} record")
    return matches[0]


def terminal_job(job_id: str) -> bool:
    payload = json.loads(command(["beaker", "job", "inspect", job_id, "--format", "json"]))
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"unexpected inspect result for {job_id}")
    status = payload[0].get("status") or {}
    return any(key in status for key in ("exited", "finalized", "canceled", "cancelled"))


def verify_ready(record: dict[str, Any]) -> None:
    if str(record.get("experiment")) != PREVIOUS_EXPERIMENT or record.get("status") != "terminal_e80":
        raise RuntimeError("registered coordinate is not the resolved E80 trajectory")
    if not all(terminal_job(job) for job in PREVIOUS_JOBS):
        raise RuntimeError("previous two-node writer is still active")
    if Path(str(record.get("output"))) != runner.EXPECTED_OUTPUT:
        raise RuntimeError("registered output differs from the canonical BS128 lineage")
    result = record.get("postDecayResults", {}).get("80", {})
    if result.get("status") != "complete" or not isinstance(result.get("validationExact"), (int, float)):
        raise RuntimeError("healthy matched E80 POST result is unresolved")


def build_spec(item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    spec = copy.deepcopy(json.loads(command(["beaker", "experiment", "spec", str(item["baseExperiment"]), "--format", "json"])))
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("base experiment must contain exactly one task")
    task = spec["tasks"][0]
    if "/weka/oe-training-default" not in {value.get("mountPath") for value in task.get("datasets", [])}:
        raise RuntimeError("base experiment is missing Weka")
    blocked = {"GANTRY_USE_TORCHRUN", "GANTRY_RDZV_ID", "GANTRY_RDZV_PORT", "NUM_NODES"}
    task["envVars"] = [
        value for value in task.get("envVars", [])
        if value.get("name") not in blocked
        and not (str(value.get("name", "")).startswith("BEAKER_") and value.get("name") != "BEAKER_TOKEN")
    ]
    rendezvous = hashlib.sha256(NAME.encode()).hexdigest()
    set_env(task, "GIT_REF", revision)
    set_env(task, "GIT_BRANCH", "sewonm/icsl")
    set_env(task, "NUM_NODES", "2")
    set_env(task, "GANTRY_RDZV_ID", rendezvous[:12])
    set_env(task, "GANTRY_RDZV_PORT", str(29000 + int(rendezvous[:8], 16) % 1000))
    task["name"] = "main"
    task["arguments"] = ["python", "scripts/models/run_dense_1b_pool3b_bs128_e80_e96_two_node.py"]
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
    task.update(replicas=2, leaderSelection=True, hostNetworking=True, propagateFailure=True, propagatePreemption=True, synchronizedStartTimeout="90m")
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        "Dense-1B DCLM-3B DR+WT+EmbedWD BS128 LR1e-3 WD0.3 continuation. Resume exact "
        "retained E80 PD step411987, save complete 16-rank recovery checkpoints every epoch "
        "through E96 step494384, then run isolated uncapped 10% WSD plus heldout/downstream "
        "evaluation at E96 and stop. Two synchronized 8-GPU nodes, rank microbatch 8, GA1, "
        "global BS128, no minRuntime, auto-resume, eight retries."
    )
    return spec


def validate_spec(spec: dict[str, Any], revision: str) -> None:
    task = spec["tasks"][0]
    env = {value["name"]: value.get("value") for value in task.get("envVars", [])}
    assert task["replicas"] == 2 and task["resources"]["gpuCount"] == 8
    assert task["leaderSelection"] and task["hostNetworking"]
    assert task["propagateFailure"] and task["propagatePreemption"]
    assert task["context"] == {"priority": task["context"]["priority"], "autoResume": True}
    assert "minRuntime" not in task["context"]
    assert spec["retry"]["allowedTaskRetries"] == 8
    assert env["GIT_REF"] == revision and env["NUM_NODES"] == "2"
    assert "GANTRY_RDZV_ID" in env and "GANTRY_RDZV_PORT" in env


def register(report: dict[str, Any], experiment: str, revision: str) -> None:
    record = producer_record(report)
    history = record.setdefault("experimentHistory", [])
    if not any(value.get("experiment") == PREVIOUS_EXPERIMENT for value in history):
        history.append({
            "experiment": PREVIOUS_EXPERIMENT, "jobs": PREVIOUS_JOBS,
            "revision": record.get("revision"), "status": "complete_e80",
            "maxValidatedEpoch": 80, "output": record.get("output"),
            "stoppedAt": datetime.now(tz=UTC).isoformat(),
        })
    record.update({
        "experiment": experiment, "revision": revision, "policy": runner.POLICY,
        "status": "submitted", "beakerStatus": "submitted", "currentEpoch": 81,
        "currentPhase": "repacked_shuffled_pool3b_constant_lr",
        "resolvedCheckpointEpochs": sorted({*[int(value) for value in record.get("resolvedCheckpointEpochs", [])], runner.SOURCE_EPOCH}),
        "targetEpochs": sorted({*[int(value) for value in record.get("targetEpochs", [])], *runner.CHECKPOINT_EPOCHS}),
        "continuationSourceEpoch": runner.SOURCE_EPOCH,
        "continuationSourceCheckpoint": str(runner.EXPECTED_OUTPUT / f"step{runner.checkpoint_step(runner.SOURCE_EPOCH)}"),
        "continuationTargetEpoch": runner.TARGET_EPOCH,
        "continuationCheckpointEpochs": list(runner.CHECKPOINT_EPOCHS),
        "evaluationEpochs": [runner.TARGET_EPOCH], "checkpointIntervalEpochs": 1,
        "checkpointCleanupKeepEpochs": [32, 48, 64, 80, 96],
        "role": "integrated_checkpoint_producer_and_evaluator",
        "evaluationEnabled": True, "decayEnabled": True,
        "postBranchesIsolatedFromConstantFrontier": True,
        "standaloneEvaluatorSubmissionsAuthorized": False,
        "futureEvaluatorSubmissionsAuthorized": False,
        "nodeCount": 2, "gpusPerNode": 8, "gpuCount": 16,
        "rankMicrobatchSequences": 8, "gradientAccumulationSteps": 1,
        "minRuntimeOmitted": True,
        "runtimeEstimate": {"producerHours": [13, 15], "postHours": [8, 10], "totalHours": [21, 25]},
        "hardTerminalEpoch": 96, "submittedAt": datetime.now(tz=UTC).isoformat(),
    })
    for key in ("job", "jobs", "wandbHealth", "needsAttention", "stopAuthorized", "stopAfterEpoch", "stopAfterCheckpointStep", "stopDecision", "stopReason", "decision", "lastDecisionEpoch"):
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
    _, item = runner.load(runner.DEFAULT_MANIFEST)
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
    register(report, ids[0], revision=args.revision)
    print(f"{NAME}: {ids[0]}")


if __name__ == "__main__":
    main()
