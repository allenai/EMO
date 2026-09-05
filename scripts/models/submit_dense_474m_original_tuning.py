#!/usr/bin/env python3
"""Submit and register the Dense-474M original-model tuning workflows."""

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
BASE_EXPERIMENT = "01KZ7307CK7ZZQ1XCJ2QQ08KD4"
NAMES = {
    "bs32-probes": "dense-474m-bs32-lr-e4-probes-v1",
    "bs128-e52": "dense-474m-bs128-e52-extension-v1",
}
RUNNER = Path("scripts/models/run_dense_474m_original_tuning.py")
MANIFEST = Path("scripts/models/manifests/dense-474m-original-tuning-v1.json")
REPORT = Path("reports/0802/data/wsd_batch_size_474m.json")
POLICY = "dense_474m_original_tuning_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(NAMES), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--name")
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
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, upstream], check=False
    ).returncode:
        raise RuntimeError(f"revision {revision} is not reachable from pushed upstream {upstream}")


def base_spec() -> dict[str, Any]:
    return json.loads(run(["beaker", "experiment", "spec", BASE_EXPERIMENT, "--format", "json"]))


def set_revision(task: dict[str, Any], revision: str) -> None:
    for variable in task.get("envVars", []):
        if variable.get("name") == "GIT_REF":
            variable["value"] = revision
            return
    raise ValueError("trusted source task has no GIT_REF")


def build_spec(mode: str, revision: str, priority: str) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--manifest",
            str(MANIFEST),
            "--mode",
            mode,
            "--validate-only",
        ],
        check=True,
    )
    spec = copy.deepcopy(base_spec())
    if len(spec.get("tasks", [])) != 1:
        raise ValueError("trusted base experiment must contain exactly one task")
    task = spec["tasks"][0]
    task["arguments"] = [
        "python",
        str(RUNNER),
        "--manifest",
        str(MANIFEST),
        "--mode",
        mode,
    ]
    task["envVars"] = [
        variable
        for variable in task.get("envVars", [])
        if variable.get("name")
        not in {"GANTRY_USE_TORCHRUN", "NUM_NODES", "PYTORCH_CUDA_ALLOC_CONF"}
    ]
    set_revision(task, revision)
    gpu_count = 2 if mode == "bs32-probes" else 8
    task["resources"] = {"gpuCount": gpu_count, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec["retry"] = {"allowedTaskRetries": 8}
    if mode == "bs32-probes":
        detail = (
            "continues exact LR5e-4/WD0.1 and LR2e-3/WD0.1 E1 PD checkpoints "
            "to E4 and immediately performs isolated WSD plus heldout/downstream evaluation"
        )
    else:
        detail = (
            "continues exact LR2e-3/WD0.3 E48 PD to E52 and immediately performs "
            "isolated WSD plus heldout/downstream evaluation"
        )
    spec["description"] = (
        f"Dense 474M Pool-1B original-model tuning: {detail}. One node, {gpu_count} GPUs, "
        "auto-resume, eight retries, and minRuntime omitted."
    )
    if "minRuntime" in task["context"]:
        raise AssertionError("original tuning task context must omit minRuntime")
    return spec


def audit_name(workspace: str, name: str) -> None:
    payload = json.loads(
        run(["beaker", "workspace", "experiments", workspace, "--text", name, "--format", "json"])
    )
    experiments = payload if isinstance(payload, list) else payload.get("experiments", [])
    if any(experiment.get("name") == name for experiment in experiments):
        raise RuntimeError(f"refusing duplicate Beaker experiment name {name}")


def write_report(mode: str, experiment: str, revision: str) -> None:
    report = json.loads(REPORT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    workflow = report.setdefault(
        "originalTuningWorkflow",
        {
            "policy": POLICY,
            "status": "submitted",
            "revision": revision,
            "minRuntimeOmitted": True,
            "experiments": {},
        },
    )
    experiments = workflow.setdefault("experiments", {})
    if mode in experiments:
        raise RuntimeError(f"474M original tuning mode {mode} is already registered")
    experiments[mode] = experiment
    workflow.update({"status": "submitted", "revision": revision, "minRuntimeOmitted": True})

    if mode == "bs32-probes":
        probes = manifest["bs32Probes"]
        for coordinate in probes["coordinates"]:
            identifier = f"dense-474m-bs32-{coordinate['id']}-e4-probe-v1"
            if any(sweep.get("id") == identifier for sweep in report.get("batchSweeps", [])):
                raise RuntimeError(f"474M tuning sweep {identifier} is already registered")
            report.setdefault("batchSweeps", []).append(
                {
                    "id": identifier,
                    "policy": POLICY,
                    "batchSequences": 32,
                    "globalBatchTokens": 131072,
                    "contextLength": 4096,
                    "lr": coordinate["learningRate"],
                    "wd": coordinate["weightDecay"],
                    "warmupSteps": 768,
                    "rankMicrobatchSequences": 16,
                    "gradientAccumulation": 1,
                    "gpuCount": 2,
                    "status": "submitted",
                    "activeEpoch": 4,
                    "activePhase": "producer",
                    "search": "small-model-original-lr-e4-probe",
                    "beaker": experiment,
                    "experiment": experiment,
                    "revision": revision,
                    "sourceEpoch": 1,
                    "sourceCheckpoint": coordinate["sourceCheckpoint"],
                    "output": coordinate["output"],
                    "constantOutput": coordinate["output"] + "/constant_lr",
                    "retainedPreDecayEpochs": [],
                    "comparisonEpoch": 4,
                    "baselineLearningRate": probes["baseline"]["learningRate"],
                    "baselineValidationExact": probes["baseline"]["validationExact"],
                    "dynamicRepacking": False,
                    "weightTying": False,
                    "embeddingWeightDecay": "zero",
                    "automaticTaskRetries": 8,
                    "minRuntimeOmitted": True,
                    "results": {},
                    "reason": (
                        "Submitted an exact E1-to-E4 constant-LR continuation followed by "
                        "isolated 10% WSD decay and matched evaluation in the same task."
                    ),
                }
            )
        workflow["bs32Baseline"] = probes["baseline"]
    else:
        extension = manifest["bs128Extension"]
        identifier = "dense-474m-bs128-lr2e-3-wd0.3-e52-extension-v1"
        if any(sweep.get("id") == identifier for sweep in report.get("batchSweeps", [])):
            raise RuntimeError(f"474M tuning sweep {identifier} is already registered")
        report.setdefault("batchSweeps", []).append(
            {
                "id": identifier,
                "policy": POLICY,
                "batchSequences": 128,
                "globalBatchTokens": 524288,
                "contextLength": 4096,
                "lr": extension["learningRate"],
                "wd": extension["weightDecay"],
                "warmupSteps": 192,
                "rankMicrobatchSequences": 16,
                "gradientAccumulation": 1,
                "gpuCount": 8,
                "status": "submitted",
                "activeEpoch": 52,
                "activePhase": "producer",
                "search": "small-model-original-e52-extension",
                "beaker": experiment,
                "experiment": experiment,
                "revision": revision,
                "sourceEpoch": 48,
                "sourceCheckpoint": extension["sourceCheckpoint"],
                "output": extension["output"],
                "constantOutput": extension["output"] + "/constant_lr",
                "retainedPreDecayEpochs": [],
                "previousValidationExact": extension["previousValidationExact"],
                "dynamicRepacking": False,
                "weightTying": False,
                "embeddingWeightDecay": "zero",
                "automaticTaskRetries": 8,
                "minRuntimeOmitted": True,
                "results": {},
                "reason": (
                    "Submitted the user-authorized E48-to-E52 constant-LR extension followed "
                    "by isolated 10% WSD decay and matched evaluation in the same task."
                ),
            }
        )
        targets = report.setdefault("batchTargetEpochs", {}).setdefault("128", [])
        if 52 not in targets:
            targets.append(52)
            targets.sort(key=float)

    note = (
        " Original-model tuning now includes BS32 LR5e-4 and LR2e-3 WD0.1 POST E4 "
        "probes against LR1e-3/WD0.1, plus a user-authorized BS128 LR2e-3/WD0.3 E52 "
        "extension after the rounded E40/E48 tie."
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
    name = args.name or NAMES[args.mode]
    validate_revision(args.revision)
    audit_name(args.workspace, name)
    spec = build_spec(args.mode, args.revision, args.priority)
    if args.print_only:
        print(json.dumps(spec, indent=2))
        return
    output = run(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", args.workspace],
        input_text=json.dumps(spec),
    )
    print(output, end="")
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError("submission succeeded without a parsed experiment ID")
    if args.register:
        write_report(args.mode, identifiers[0], args.revision)


if __name__ == "__main__":
    main()
