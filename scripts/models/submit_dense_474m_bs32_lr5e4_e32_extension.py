#!/usr/bin/env python3
"""Submit and register the Dense-474M Original BS32 LR5e-4 E4-to-E32 extension."""

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

import submit_dense_474m_original_tuning as original

WORKSPACE = "ai2/flex2"
NAME = "dense-474m-bs32-lr5e-4-wd0.1-e32-extension-v1"
MODE = "bs32-lr5e4-e32"
POLICY = "dense_474m_original_lr5e4_e32_v1"
RUNNER = Path("scripts/models/run_dense_474m_bs32_lr5e4_e32_extension.py")
MANIFEST = Path("scripts/models/manifests/dense-474m-original-tuning-v1.json")
REPORT = Path("reports/0802/data/wsd_batch_size_474m.json")
SWEEP_ID = "dense-474m-bs32-lr5e-4-wd0.1-e4-probe-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--replace-experiment")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--register", action="store_true")
    return parser.parse_args()


def run(arguments: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        arguments, check=True, input=input_text, capture_output=True, text=True
    )
    return completed.stdout


def build_spec(revision: str, priority: str) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--manifest",
            str(MANIFEST),
            "--validate-only",
        ],
        check=True,
    )
    spec = copy.deepcopy(original.base_spec())
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
    task["envVars"].append(
        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"}
    )
    original.set_revision(task, revision)
    task["resources"] = {"gpuCount": 2, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        "Dense 474M Pool-1B Original BS32 LR5e-4/WD0.1 continuation from exact E4 PD "
        "through E32, with isolated 10% WSD plus heldout/downstream evaluations at "
        "E16, E24, and E32. One node, two GPUs, auto-resume, eight retries, expandable "
        "CUDA segments, and minRuntime omitted."
    )
    if "minRuntime" in task["context"]:
        raise AssertionError("extension task context must omit minRuntime")
    return spec


def write_report(
    experiment: str, revision: str, *, replace_experiment: str | None = None
) -> None:
    report = json.loads(REPORT.read_text())
    workflow = report.get("originalTuningWorkflow")
    if not isinstance(workflow, dict):
        raise RuntimeError("474M original tuning workflow is not registered")
    experiments = workflow.setdefault("experiments", {})
    if MODE in experiments and experiments[MODE] != replace_experiment:
        raise RuntimeError(f"474M original tuning mode {MODE} is already registered")
    if replace_experiment:
        workflow.setdefault("experimentHistory", []).append(
            {
                "mode": MODE,
                "experiment": replace_experiment,
                "status": "canceled",
                "reason": "Canceled after the isolated state-directory ownership guard failed before training.",
            }
        )
    experiments[MODE] = experiment
    sweeps = [sweep for sweep in report.get("batchSweeps", []) if sweep.get("id") == SWEEP_ID]
    if len(sweeps) != 1:
        raise RuntimeError(f"expected exactly one sweep {SWEEP_ID}, found {len(sweeps)}")
    sweep = sweeps[0]
    previous_experiment = sweep.get("experiment") or sweep.get("beaker")
    history = sweep.setdefault("continuations", [])
    if previous_experiment and not any(item.get("experiment") == previous_experiment for item in history):
        history.append(
            {
                "experiment": previous_experiment,
                "status": "complete",
                "sourceEpoch": 1,
                "targetEpoch": 4,
                "reason": "Completed the LR5e-4 E4 probe and established the winning E4 source.",
            }
        )
    source = (
        "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm1b/"
        "bs32_lr5e-4_wd0.1/step27465"
    )
    output = source.rsplit("/", 1)[0]
    sweep.update(
        {
            "policy": POLICY,
            "status": "submitted",
            "activeEpoch": 16,
            "activePhase": "producer",
            "search": "small-model-original-lr-selected-extension",
            "beaker": experiment,
            "experiment": experiment,
            "revision": revision,
            "sourceEpoch": 4,
            "sourceCheckpoint": source,
            "output": output,
            "constantOutput": output,
            "retainedPreDecayEpochs": sorted(
                set(int(epoch) for epoch in [*sweep.get("retainedPreDecayEpochs", []), 4])
            ),
            "targetEpoch": 32,
            "evaluationEpochs": [16, 24, 32],
            "automaticTaskRetries": 8,
            "minRuntimeOmitted": True,
            "reason": (
                "LR5e-4 strictly improved on LR1e-3 at E4; continuing the exact E4 PD "
                "frontier through E32 with inline isolated POST evaluations at E16/E24/E32."
            ),
        }
    )
    workflow.update({"status": "submitted", "revision": revision, "minRuntimeOmitted": True})
    selection_note = (
        " Because LR5e-4/WD0.1 strictly beat LR1e-3/WD0.1 at E4, its exact E4 PD "
        "frontier continues through E32 with POST evaluations at E16, E24, and E32."
    )
    if selection_note.strip() not in str(report.get("selection", "")):
        report["selection"] = str(report.get("selection", "")).rstrip() + selection_note
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    args = parse_args()
    if args.register and args.print_only:
        raise SystemExit("--register cannot be combined with --print-only")
    original.validate_revision(args.revision)
    original.audit_name(args.workspace, args.name)
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
        write_report(
            identifiers[0], args.revision, replace_experiment=args.replace_experiment
        )


if __name__ == "__main__":
    main()
