#!/usr/bin/env python3
"""Submit one adaptive DR+WT+EmbedWD chain for a small Dense model/batch."""

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
SEQUENCE_LENGTH = 4096
RANK_MICROBATCH_SEQUENCES = 16
REFERENCE_WARMUP_SEQUENCE_STEPS = 24 * 1024
LEARNING_RATE = "2e-3"
MANIFEST_DIR = Path("scripts/models/manifests")
REPORT_DIR = Path("reports/0802/data")
OUTPUT_ROOT = "/weka/oe-training-default/sewonm/icsl/models"
MODELS: dict[str, dict[str, Any]] = {
    "474m": {
        "baseExperiment": "01KZ7307CK7ZZQ1XCJ2QQ08KD4",
        "initialTargets": [1, 2, 4, 8, 16, 24],
        "epochIncrement": 8,
        "baseline": {
            64: {"wd": "0.1", "epoch": 32, "validation": 3.093},
            128: {"wd": "0.3", "epoch": 40, "validation": 3.101},
            256: {"wd": "0.333", "epoch": 32, "validation": 3.102},
            512: {"wd": "0.3", "epoch": 24, "validation": 3.136},
        },
    },
    "153m": {
        "baseExperiment": "01KZ6Q4DJ8J994A6SQ39MEGTZ2",
        "initialTargets": [1, 2, 4, 8, 16, 32, 48],
        "epochIncrement": 16,
        "baseline": {
            64: {"wd": "0.1", "epoch": 112, "validation": 3.332},
            128: {"wd": "0.1", "epoch": 72, "validation": 3.341},
            256: {"wd": "0.1", "epoch": 48, "validation": 3.360},
            512: {"wd": "0.1", "epoch": 40, "validation": 3.416},
        },
    },
}
BATCHES = (64, 128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--global-sequences", type=int, choices=BATCHES, required=True)
    parser.add_argument("--revision", required=True, help="Pushed, fetchable git revision")
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--name")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--register", action="store_true")
    return parser.parse_args()


def run(arguments: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        arguments,
        check=True,
        input=input_text,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def nproc_for(batch: int) -> int:
    return 4 if batch == 64 else 8


def warmup_steps(batch: int) -> int:
    if REFERENCE_WARMUP_SEQUENCE_STEPS % batch:
        raise ValueError(f"BS{batch} does not preserve token-matched warmup")
    return REFERENCE_WARMUP_SEQUENCE_STEPS // batch


def wd_ladder(batch: int) -> list[str]:
    middle = "0.333" if batch == 256 else "0.3"
    return ["0.01", "0.033", "0.1", middle, "1.0", "3.0"]


def manifest_path(model: str, batch: int) -> Path:
    return MANIFEST_DIR / f"small-dense-{model}-bs{batch}-dr-wt-embwd-adaptive.json"


def output_root(model: str) -> str:
    return f"{OUTPUT_ROOT}/dense_{model}_dclm1b"


def output_for(model: str, batch: int, wd: str) -> str:
    return f"{output_root(model)}/bs{batch}_dr_wt_embwd_lr{LEARNING_RATE}_wd{wd}"


def build_manifest(model: str, batch: int) -> tuple[dict[str, Any], Path]:
    settings = MODELS[model]
    baseline = settings["baseline"][batch]
    ladder = wd_ladder(batch)
    center_index = ladder.index(str(baseline["wd"]))
    initial = ladder[center_index - 1 : center_index + 2]
    if len(initial) != 3:
        raise ValueError(f"baseline WD {baseline['wd']} lacks two WD neighbors")
    manifest = {
        "model": model,
        "baseExperiment": settings["baseExperiment"],
        "globalSequences": batch,
        "nprocPerNode": nproc_for(batch),
        "rankMicrobatchSequences": RANK_MICROBATCH_SEQUENCES,
        "gradientAccumulation": batch // (nproc_for(batch) * RANK_MICROBATCH_SEQUENCES),
        "warmupSteps": warmup_steps(batch),
        "learningRate": LEARNING_RATE,
        "wdLadder": ladder,
        "baselineOptimalWd": baseline["wd"],
        "baselineEvidenceEpoch": baseline["epoch"],
        "baselineValidation": baseline["validation"],
        "initialWds": initial,
        "initialTargets": settings["initialTargets"],
        "epochIncrement": settings["epochIncrement"],
        "outputRoot": output_root(model),
        "runSuffix": "sm0818-drwtembwd-adaptive",
        "variant": "DR+WT+EmbedWD",
        "dataOrder": "dynamic_repacking_from_e2",
        "weightTying": True,
        "decayEmbeddings": True,
    }
    path = manifest_path(model, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, path


def base_spec(experiment: str) -> dict[str, Any]:
    return json.loads(run(["beaker", "experiment", "spec", experiment, "--format", "json"]))


def audit_base_spec(model: str, spec: dict[str, Any]) -> None:
    tasks = spec.get("tasks", [])
    if len(tasks) != 1:
        raise ValueError("trusted small-model base experiment must contain one task")
    arguments = tasks[0].get("arguments", [])
    if arguments[:2] != ["python", "src/scripts/train/olmo2-1B.py"]:
        raise ValueError("trusted base does not use the expected training entrypoint")
    expected_model = "--model-size=153M" if model == "153m" else "--model.d_model=1024"
    if expected_model not in arguments:
        raise ValueError(f"trusted base is missing {expected_model}")
    manifest_values = [
        value
        for value in arguments
        if value.startswith("--dataset.subset_manifest=")
    ]
    if manifest_values != [
        "--dataset.subset_manifest=src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json"
    ]:
        raise ValueError("trusted base uses the wrong repeated-data manifest")


def set_revision(task: dict[str, Any], revision: str) -> None:
    for variable in task.get("envVars", []):
        if variable.get("name") == "GIT_REF":
            variable["value"] = revision
            return
    raise ValueError("source task has no GIT_REF")


def build_submission(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest, path = build_manifest(args.model, args.global_sequences)
    spec = copy.deepcopy(base_spec(str(manifest["baseExperiment"])))
    audit_base_spec(args.model, spec)
    task = spec["tasks"][0]
    task["arguments"] = [
        "python",
        "scripts/models/run_small_dense_dr_wt_embedwd_chain.py",
        "--manifest",
        str(path),
    ]
    task["envVars"] = [
        variable
        for variable in task.get("envVars", [])
        if variable.get("name") not in {"GANTRY_USE_TORCHRUN", "PYTORCH_CUDA_ALLOC_CONF"}
    ]
    set_revision(task, args.revision)
    if args.model == "474m":
        task["envVars"].append(
            {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"}
        )
    task["resources"] = {
        "gpuCount": int(manifest["nprocPerNode"]),
        "sharedMemory": "10 GiB",
    }
    task["context"] = {
        "priority": args.priority,
        "minRuntime": "0s",
        "autoResume": True,
    }
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"Dense {args.model.upper()} BS{args.global_sequences} DR+WT+EmbedWD adaptive "
        "WD saturation chain. E1 is the shared bootstrap and DR begins at E2; each "
        "fixed-WD trajectory stays in one canonical output directory."
    )
    return spec, manifest, path


def experiment_name(model: str, batch: int) -> str:
    return f"dense-{model}-bs{batch}-dr-wt-embwd-adaptive-chain"


def audit_name(workspace: str, name: str) -> None:
    payload = json.loads(
        run(["beaker", "workspace", "experiments", workspace, "--text", name, "--format", "json"])
    )
    experiments = payload if isinstance(payload, list) else payload.get("experiments", [])
    if any(experiment.get("name") == name for experiment in experiments):
        raise SystemExit(f"refusing duplicate Beaker experiment name {name}")


def report_path(model: str) -> Path:
    return REPORT_DIR / f"wsd_batch_size_{model}.json"


def write_report(
    model: str,
    batch: int,
    experiment: str,
    revision: str,
    manifest: dict[str, Any],
) -> None:
    path = report_path(model)
    report = json.loads(path.read_text())
    records = report.setdefault("adaptiveDrWtEmbedWdChains", [])
    record_id = f"dense-{model}-bs{batch}-dr-wt-embwd-adaptive"
    if any(record.get("id") == record_id for record in records):
        raise RuntimeError(f"adaptive chain {record_id} is already registered")
    record = {
        "id": record_id,
        "label": f"BS{batch} · DR+WT+EmbedWD",
        "variant": "DR+WT+EmbedWD",
        "batchSequences": batch,
        "globalBatchTokens": batch * SEQUENCE_LENGTH,
        "contextLength": SEQUENCE_LENGTH,
        "lr": LEARNING_RATE,
        "baselineOptimalWd": manifest["baselineOptimalWd"],
        "baselineEvidenceEpoch": manifest["baselineEvidenceEpoch"],
        "baselineValidation": manifest["baselineValidation"],
        "initialWds": manifest["initialWds"],
        "wdLadder": manifest["wdLadder"],
        "initialTargets": manifest["initialTargets"],
        "epochIncrement": manifest["epochIncrement"],
        "rankMicrobatchSequences": RANK_MICROBATCH_SEQUENCES,
        "gpuCount": nproc_for(batch),
        "gradientAccumulation": manifest["gradientAccumulation"],
        "dataOrder": "dynamic_repacking_from_e2",
        "dynamicRepacking": True,
        "weightTying": True,
        "decayEmbeddings": True,
        "embeddingWeightDecay": "global",
        "status": "submitted",
        "beakerStatus": "submitted",
        "activeEpoch": 1,
        "activeWds": manifest["initialWds"],
        "experiment": experiment,
        "beaker": experiment,
        "revision": revision,
        "automaticTaskRetries": 8,
        "stopOnNonImprovement": True,
        "outputByWd": {
            wd: output_for(model, batch, wd) for wd in manifest["wdLadder"]
        },
        "results": {},
        "frontiers": {},
        "reason": (
            "Submitted as one persistent adaptive WD job. Initial E1 candidates are the "
            "trusted ordinary-run optimum plus its lower and higher neighbors. Every later "
            "frontier evaluates the preceding winner and one WD level higher; fixed-WD "
            "trajectories keep one canonical directory and the chain stops at the first "
            "selected held-out validation non-improvement."
        ),
    }
    records.append(record)
    note = (
        f" BS{batch} also has a persistent DR+WT+EmbedWD adaptive-WD saturation chain "
        f"at LR{LEARNING_RATE}, starting from baseline WD{manifest['baselineOptimalWd']} "
        "and its immediate neighbors."
    )
    if note.strip() not in str(report.get("selection", "")):
        report["selection"] = str(report.get("selection", "")).rstrip() + note
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    args = parse_args()
    name = args.name or experiment_name(args.model, args.global_sequences)
    if args.manifest_only:
        _, path = build_manifest(args.model, args.global_sequences)
        print(path)
        return
    spec, manifest, path = build_submission(args)
    audit_name(args.workspace, name)
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    output = run(
        [
            "beaker",
            "experiment",
            "create",
            "-",
            "--name",
            name,
            "--workspace",
            args.workspace,
        ],
        input_text=json.dumps(spec),
    )
    print(output, end="")
    ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if args.register:
        if not ids:
            raise RuntimeError("submission succeeded without a parsed experiment ID")
        write_report(args.model, args.global_sequences, ids[0], args.revision, manifest)


if __name__ == "__main__":
    main()
