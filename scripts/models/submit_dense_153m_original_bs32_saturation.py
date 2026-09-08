#!/usr/bin/env python3
"""Submit the integrated 153M Original BS32 continuation from exact E80."""

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
BASE_EXPERIMENT = "01M1WTNSWNSK87R0KD9HQP5WK9"
NAME = "dense-153m-original-bs32-lr1e-3-wd0.1-e80-saturation-v1"
RUNNER = Path("scripts/models/run_dense_153m_original_bs32_saturation.py")
MANIFEST = Path("scripts/models/manifests/dense-153m-original-bs32-e80-saturation-v1.json")
REPORT = Path("reports/0802/data/wsd_batch_size_153m.json")
POLICY = "dense_153m_original_bs32_e80_post_saturation_v1"


def run(arguments: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(
        arguments, check=True, input=input_text, capture_output=True, text=True
    ).stdout


def validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full 40-character commit")
    subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], check=True)
    upstream = run(["git", "rev-parse", "@{upstream}"]).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, upstream], check=False
    ).returncode:
        raise RuntimeError(f"revision {revision} is not reachable from pushed upstream")


def set_revision(task: dict[str, Any], revision: str) -> None:
    for variable in task.get("envVars", []):
        if variable.get("name") == "GIT_REF":
            variable["value"] = revision
            return
    raise ValueError("trusted source task has no GIT_REF")


def build_spec(revision: str, priority: str) -> dict[str, Any]:
    subprocess.run([sys.executable, str(RUNNER), "--manifest", str(MANIFEST), "--validate-only"], check=True)
    spec = json.loads(run(["beaker", "experiment", "spec", BASE_EXPERIMENT, "--format", "json"]))
    if len(spec.get("tasks", [])) != 1:
        raise ValueError("trusted predecessor must contain exactly one task")
    task = copy.deepcopy(spec["tasks"][0])
    task["arguments"] = ["python", str(RUNNER), "--manifest", str(MANIFEST)]
    task["envVars"] = [
        variable
        for variable in task.get("envVars", [])
        if variable.get("name") not in {"GANTRY_USE_TORCHRUN", "NUM_NODES", "PYTORCH_CUDA_ALLOC_CONF"}
    ]
    set_revision(task, revision)
    task["resources"] = {"gpuCount": 2, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    if "minRuntime" in task["context"]:
        raise AssertionError("continuation must omit minRuntime")
    return {
        **{key: value for key, value in spec.items() if key not in {"tasks", "retry", "description"}},
        "tasks": [task],
        "retry": {"allowedTaskRetries": 8},
        "description": (
            "Continue exact Dense-153M Original Pool-1B BS32 LR1e-3/WD0.1 E80 PD. "
            "Save complete recovery checkpoints every four epochs, evaluate isolated 10% WSD "
            "POST at E88, E96, ... and stop at the first adjacent strict non-improvement. "
            "One canonical writer; minRuntime omitted."
        ),
    }


def write_report(experiment: str, revision: str) -> None:
    report = json.loads(REPORT.read_text())
    if any(sweep.get("id") == "dense-153m-original-bs32-e80-saturation-v1" for sweep in report.get("batchSweeps", [])):
        raise RuntimeError("continuation is already registered")
    manifest = json.loads(MANIFEST.read_text())
    sweep = {
        "id": manifest["id"],
        "policy": POLICY,
        "batchSequences": 32,
        "globalBatchTokens": 131072,
        "contextLength": 4096,
        "lr": "1e-3",
        "wd": "0.1",
        "warmupSteps": 768,
        "rankMicrobatchSequences": 16,
        "gradientAccumulation": 1,
        "gpuCount": 2,
        "status": "submitted",
        "activeEpoch": 88,
        "activePhase": "producer",
        "search": "small-model-bs32-post-saturation-continuation",
        "saturationChain": True,
        "beaker": experiment,
        "experiment": experiment,
        "revision": revision,
        "output": manifest["output"],
        "constantOutput": manifest["constantOutput"],
        "sourceCheckpoint": manifest["initialSourceCheckpoint"],
        "frontierEpoch": 80,
        "frontierValidationExact": 3.351,
        "epochIncrement": 8,
        "saveEveryEpochs": 4,
        "evaluationEpochs": "E88,E96,... until strict POST non-improvement",
        "stopOnNonImprovement": True,
        "comparisonMetric": "healthy_matched_post_validationExact",
        "dynamicRepacking": False,
        "weightTying": False,
        "embeddingWeightDecay": "zero",
        "automaticTaskRetries": 8,
        "minRuntimeOmitted": True,
        "results": {},
        "reason": (
            "Submitted one persistent integrated continuation from exact retained E80 PD. "
            "It saves recovery checkpoints every four epochs, evaluates E88, E96, ... inline, "
            "and stops at the first adjacent POST validation non-improvement."
        ),
    }
    report.setdefault("batchSweeps", []).append(sweep)
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    validate_revision(args.revision)
    spec = build_spec(args.revision, args.priority)
    if args.print_only:
        print(json.dumps(spec, indent=2))
        return
    output = run(
        ["beaker", "experiment", "create", "-", "--name", args.name, "--workspace", args.workspace],
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
