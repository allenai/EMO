#!/usr/bin/env python3
"""Submit and register the guarded Dense-153M BS32 E16 early evaluator."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKSPACE = "ai2/flex2"
BASE_EXPERIMENT = "01KZ6Q4DJ8J994A6SQ39MEGTZ2"
NAME = "dense-153m-bs32-lr5e-4-wd0.033-e16-early-eval-v1"
POLICY = "dense_153m_bs32_e16_early_eval_v1"
RUNNER = Path("scripts/models/run_dense_153m_bs32_e16_early_eval.py")
MANIFEST = Path("scripts/models/manifests/dense-153m-bs32-lr5e-4-wd0.033-e16-early-eval-v1.json")
REPORT = Path("reports/0802/data/wsd_batch_size_153m.json")


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(arguments, check=True, input=input_text, capture_output=True, text=True).stdout


def validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full commit hash")
    subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], check=True)
    upstream = command(["git", "rev-parse", "@{upstream}"]).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", revision, upstream], check=False).returncode:
        raise RuntimeError(f"revision {revision} is not reachable from pushed upstream {upstream}")


def build_spec(revision: str, priority: str) -> dict[str, Any]:
    subprocess.run([sys.executable, str(RUNNER), "--manifest", str(MANIFEST), "--validate-only"], check=True)
    spec = copy.deepcopy(json.loads(command(["beaker", "experiment", "spec", BASE_EXPERIMENT, "--format", "json"])))
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("trusted source experiment must have exactly one task")
    task = spec["tasks"][0]
    task["arguments"] = ["python", str(RUNNER), "--manifest", str(MANIFEST)]
    blocked = {"GANTRY_USE_TORCHRUN", "GANTRY_RDZV_ID", "GANTRY_RDZV_PORT", "NUM_NODES", "PYTORCH_CUDA_ALLOC_CONF"}
    task["envVars"] = [
        variable for variable in task.get("envVars", [])
        if variable.get("name") not in blocked
        and not (str(variable.get("name", "")).startswith("BEAKER_") and variable.get("name") != "BEAKER_TOKEN")
    ]
    for variable in task["envVars"]:
        if variable.get("name") == "GIT_REF":
            variable["value"] = revision
            break
    else:
        raise RuntimeError("trusted source task has no GIT_REF")
    task["resources"] = {"gpuCount": 2, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "hostNetworking", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        "Guarded Dense-153M Original BS32 LR5e-4/WD0.033 E16 diagnostic. Wait for exact "
        "E16 PD step109863, then run isolated uncapped 10% WSD plus heldout/downstream "
        "evaluation to e16-early-v1. One node/two GPUs; minRuntime omitted."
    )
    if "minRuntime" in task["context"]:
        raise AssertionError("early E16 task context must omit minRuntime")
    return spec


def write_report(experiment: str, revision: str) -> None:
    report = json.loads(REPORT.read_text())
    evaluations = report.setdefault("earlyDiagnosticEvaluations", [])
    if any(item.get("id") == NAME for item in evaluations):
        raise RuntimeError(f"{NAME} is already registered")
    manifest = json.loads(MANIFEST.read_text())
    evaluations.append(
        {
            "id": NAME,
            "policy": POLICY,
            "status": "submitted_waiting_for_source",
            "batchSequences": 32,
            "lr": manifest["learningRate"],
            "wd": manifest["weightDecay"],
            "epoch": 16,
            "sourceCheckpoint": manifest["sourceCheckpoint"],
            "output": manifest["output"],
            "endpointCheckpoint": manifest["endpointCheckpoint"],
            "experiment": experiment,
            "beaker": experiment,
            "revision": revision,
            "gpuCount": 2,
            "nodeCount": 1,
            "automaticTaskRetries": 8,
            "minRuntimeOmitted": True,
            "decayFraction": 0.1,
            "diagnosticOnly": True,
            "reason": "Waits for exact retained E16 PD and then runs one isolated early POST diagnostic without changing the E40 gate.",
        }
    )
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.with_suffix(".js").write_text("window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.print_only and args.register:
        raise SystemExit("--print-only cannot be combined with --register")
    validate_revision(args.revision)
    spec = build_spec(args.revision, args.priority)
    if args.print_only:
        print(json.dumps(spec, indent=2))
        return
    output = command(
        ["beaker", "experiment", "create", "-", "--name", NAME, "--workspace", WORKSPACE],
        input_text=json.dumps(spec),
    )
    print(output, end="")
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError("submission succeeded without a parsed experiment ID")
    if args.register:
        write_report(identifiers[0], args.revision)


if __name__ == "__main__":
    main()
