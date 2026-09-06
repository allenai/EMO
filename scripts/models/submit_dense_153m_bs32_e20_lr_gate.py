#!/usr/bin/env python3
"""Submit the matched E20 LR gate and conditional LR1e-3/WD0.1 follow-up."""

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
NAME = "dense-153m-bs32-e20-lr-gate-v2"
RUNNER = Path("scripts/models/run_dense_153m_bs32_e20_lr_gate.py")
MANIFEST = Path("scripts/models/manifests/dense-153m-bs32-e20-lr-gate-v1.json")
REPORT = Path("reports/0802/data/wsd_batch_size_153m.json")


def command(args: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(args, check=True, input=input_text, capture_output=True, text=True).stdout


def build_spec(revision: str, priority: str) -> dict[str, Any]:
    subprocess.run([sys.executable, str(RUNNER), "--manifest", str(MANIFEST), "--validate-only"], check=True)
    spec = copy.deepcopy(json.loads(command(["beaker", "experiment", "spec", BASE_EXPERIMENT, "--format", "json"])))
    task = spec["tasks"][0]
    task["arguments"] = ["python", str(RUNNER), "--manifest", str(MANIFEST)]
    task["envVars"] = [v for v in task.get("envVars", []) if v.get("name") not in {"GANTRY_USE_TORCHRUN", "NUM_NODES", "PYTORCH_CUDA_ALLOC_CONF"}]
    for variable in task["envVars"]:
        if variable.get("name") == "GIT_REF":
            variable["value"] = revision
            break
    else:
        raise RuntimeError("trusted task has no GIT_REF")
    task["resources"] = {"gpuCount": 2, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "hostNetworking", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        "Dense-153M Original BS32 E20 LR gate. Reuse the completed isolated LR5e-4 E20 "
        "POST and compare it with the completed LR1e-3 E16 reference. If LR5e-4 is strictly worse, "
        "immediately start fresh LR1e-3/WD0.1 through E80 with inline POST gates. "
        "One node/two GPUs; minRuntime omitted."
    )
    if "minRuntime" in task["context"]:
        raise AssertionError("E20 gate must omit minRuntime")
    return spec


def register(experiment: str, revision: str) -> None:
    report = json.loads(REPORT.read_text())
    primary = next(s for s in report["batchSweeps"] if s.get("id") == "dense-153m-bs32-lr-wd-hybrid-v1")
    prior = {k: primary.get(k) for k in ("experiment", "job", "revision", "status", "activeEpoch")}
    primary.setdefault("priorAttempts", []).append(prior)
    primary.update({
        "status": "submitted", "activeEpoch": 20, "activePhase": "matched_e20_post",
        "experiment": experiment, "beaker": experiment, "revision": revision,
        "firstEvaluationEpoch": 20, "baselineReferenceEpoch": 16,
        "reason": "E20 POST retained; submitted its LR decision against the completed LR1e-3 E16 reference and conditional fresh LR1e-3/WD0.1 follow-up.",
    })
    workflow = report.setdefault("bs32LrWdHybridWorkflow", {})
    workflow.update({
        "policy": "dense_153m_bs32_e20_lr_gate_v1", "status": "submitted",
        "experiment": experiment, "revision": revision, "activeEpoch": 20,
        "activePhase": "matched_e20_post", "firstEvaluationEpoch": 20,
        "conditionalFollowupLearningRate": "1e-3", "followupWeightDecay": "0.1",
        "minRuntimeOmitted": True,
    })
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.with_suffix(".js").write_text("window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("revision must be a full commit hash")
    subprocess.run(["git", "cat-file", "-e", f"{args.revision}^{{commit}}"], check=True)
    upstream = command(["git", "rev-parse", "@{upstream}"]).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", args.revision, upstream], check=False).returncode:
        raise RuntimeError("revision is not pushed")
    spec = build_spec(args.revision, args.priority)
    if args.print_only:
        print(json.dumps(spec, indent=2)); return
    output = command(["beaker", "experiment", "create", "-", "--name", NAME, "--workspace", WORKSPACE], input_text=json.dumps(spec))
    print(output, end="")
    ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not ids:
        raise RuntimeError("could not parse experiment ID")
    if args.register:
        register(ids[0], args.revision)


if __name__ == "__main__":
    main()
