#!/usr/bin/env python3
"""Submit and register the Dense-153M BS32 LR/WD hybrid workflow."""

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
DEFAULT_NAME = "dense-153m-bs32-lr-wd-hybrid-v1"
RUNNER = Path("scripts/models/run_dense_153m_bs32_lr_wd_hybrid.py")
MANIFEST = Path("scripts/models/manifests/dense-153m-bs32-lr-wd-hybrid-v1.json")
REPORT = Path("reports/0802/data/wsd_batch_size_153m.json")
POLICY = "dense_153m_bs32_lr_wd_hybrid_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--register", action="store_true")
    return parser.parse_args()


def run(arguments: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        arguments, check=True, input=input_text, capture_output=True, text=True
    )
    return completed.stdout


def validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full 40-character commit")
    subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], check=True)
    upstream = run(["git", "rev-parse", "@{upstream}"]).strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, upstream], check=False
    )
    if ancestor.returncode:
        raise RuntimeError(f"revision {revision} is not reachable from pushed upstream {upstream}")


def base_spec() -> dict[str, Any]:
    return json.loads(run(["beaker", "experiment", "spec", BASE_EXPERIMENT, "--format", "json"]))


def set_revision(task: dict[str, Any], revision: str) -> None:
    for environment_variable in task.get("envVars", []):
        if environment_variable.get("name") == "GIT_REF":
            environment_variable["value"] = revision
            return
    raise ValueError("trusted source task has no GIT_REF")


def build_spec(revision: str, priority: str) -> dict[str, Any]:
    subprocess.run(
        [sys.executable, str(RUNNER), "--manifest", str(MANIFEST), "--validate-only"],
        check=True,
    )
    spec = copy.deepcopy(base_spec())
    if len(spec.get("tasks", [])) != 1:
        raise ValueError("trusted base experiment must contain exactly one task")
    task = spec["tasks"][0]
    task["arguments"] = ["python", str(RUNNER), "--manifest", str(MANIFEST)]
    task["envVars"] = [
        variable
        for variable in task.get("envVars", [])
        if variable.get("name")
        not in {"GANTRY_USE_TORCHRUN", "NUM_NODES", "PYTORCH_CUDA_ALLOC_CONF"}
    ]
    set_revision(task, revision)
    task["resources"] = {"gpuCount": 2, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        "Dense 153M Pool-1B BS32 integrated LR/WD hybrid. Fresh LR5e-4/WD0.033 "
        "constant frontier saves every four epochs, gates at POST E40 with an E32 "
        "negative sanity check, otherwise evaluates E48..E80, selects against the "
        "matched LR1e-3 baseline, then trains selected-LR/WD0.1 through E80. Every "
        "WSD+heldout+downstream branch is isolated; minRuntime is omitted."
    )
    if "minRuntime" in task["context"]:
        raise AssertionError("hybrid task context must omit minRuntime")
    return spec


def audit_name(workspace: str, name: str) -> None:
    payload = json.loads(
        run(["beaker", "workspace", "experiments", workspace, "--text", name, "--format", "json"])
    )
    experiments = payload if isinstance(payload, list) else payload.get("experiments", [])
    if any(experiment.get("name") == name for experiment in experiments):
        raise RuntimeError(f"refusing duplicate Beaker experiment name {name}")


def write_report(experiment: str, revision: str) -> None:
    report = json.loads(REPORT.read_text())
    matches = [
        sweep
        for sweep in report.get("batchSweeps", [])
        if sweep.get("id") == "dense-153m-bs32-lr-wd-hybrid-v1"
    ]
    if matches:
        raise RuntimeError("hybrid workflow is already registered in the 153M report")
    manifest = json.loads(MANIFEST.read_text())
    probe = manifest["probe"]
    sweep = {
        "id": manifest["id"],
        "policy": POLICY,
        "batchSequences": 32,
        "globalBatchTokens": 131072,
        "contextLength": 4096,
        "lr": probe["learningRate"],
        "wd": probe["weightDecay"],
        "warmupSteps": 768,
        "rankMicrobatchSequences": 16,
        "gradientAccumulation": 1,
        "gpuCount": 2,
        "status": "submitted",
        "activeEpoch": 40,
        "activePhase": "probe_producer",
        "search": "small-model-bs32-lr-wd-hybrid",
        "hybridWorkflow": True,
        "beaker": experiment,
        "experiment": experiment,
        "revision": revision,
        "output": probe["output"],
        "constantOutput": probe["output"] + "/constant_lr",
        "saveEveryEpochs": 4,
        "firstEvaluationEpoch": 40,
        "negativeSanityEpoch": 32,
        "lateEvaluationEpochs": probe["lateEvaluationEpochs"],
        "followupWeightDecay": "0.1",
        "followupEvaluationEpochs": manifest["followup"]["evaluationEpochs"],
        "comparisonMetric": manifest["comparisonMetric"],
        "dynamicRepacking": False,
        "weightTying": False,
        "embeddingWeightDecay": "zero",
        "automaticTaskRetries": 8,
        "minRuntimeOmitted": True,
        "results": {},
        "reason": (
            "Submitted the fresh LR5e-4/WD0.033 hybrid probe. It saves exact pre-decay "
            "checkpoints every four epochs, evaluates E40 first, uses E32 only to confirm "
            "a negative E40 comparison, and otherwise compares best matched POST results "
            "through E80 before starting the selected-LR/WD0.1 follow-up in the same task."
        ),
    }
    report.setdefault("batchSweeps", []).append(sweep)
    report["bs32LrWdHybridWorkflow"] = {
        "policy": POLICY,
        "status": "submitted",
        "experiment": experiment,
        "revision": revision,
        "probeLearningRate": probe["learningRate"],
        "baselineLearningRate": manifest["baseline"]["learningRate"],
        "probeWeightDecay": probe["weightDecay"],
        "followupWeightDecay": manifest["followup"]["weightDecay"],
        "saveEveryEpochs": 4,
        "firstEvaluationEpoch": 40,
        "negativeSanityEpoch": 32,
        "lateEvaluationEpochs": probe["lateEvaluationEpochs"],
        "minRuntimeOmitted": True,
    }
    note = (
        " BS32 now includes a guarded LR5e-4/WD0.033 hybrid probe: POST E40 is the "
        "first gate, E32 is evaluated only to confirm a negative E40 result, and a "
        "promising or inconclusive probe continues through E80 before the selected LR "
        "is tested from scratch with WD0.1."
    )
    if note.strip() not in str(report.get("selection", "")):
        report["selection"] = str(report.get("selection", "")).rstrip() + note
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    args = parse_args()
    if args.register and args.print_only:
        raise SystemExit("--register cannot be combined with --print-only")
    validate_revision(args.revision)
    audit_name(args.workspace, args.name)
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
