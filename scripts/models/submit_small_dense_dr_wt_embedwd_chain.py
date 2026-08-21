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
MAX_WEIGHT_DECAY = "1.0"
HISTORICAL_PREDECAY_START_EPOCH = 8
ALLOWED_OUTPUT_PREFIX = "/weka/oe-training-default/sewonm/icsl/models/"
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
    parser.add_argument(
        "--resume-experiment",
        help="Replace this registered experiment after a guarded path recovery",
    )
    parser.add_argument(
        "--stop-existing",
        action="store_true",
        help="Stop --resume-experiment immediately before creating its replacement",
    )
    parser.add_argument(
        "--output-override",
        action="append",
        default=[],
        metavar="WD=/ABSOLUTE/PATH",
        help="Reroute a fixed-WD trajectory in the replacement task",
    )
    parser.add_argument(
        "--policy-replacement",
        action="store_true",
        help="Replace a registered chain in place after a policy-only manifest update",
    )
    parser.add_argument(
        "--predecay-policy-replacement",
        action="store_true",
        help=(
            "Stop adaptive WD tuning and replace the chain with the locked-WD "
            "pre-decay saturation controller"
        ),
    )
    parser.add_argument(
        "--locked-wd",
        help="Previously selected WD to retain for --predecay-policy-replacement",
    )
    parser.add_argument(
        "--historical-predecay-through-epoch",
        type=int,
        help="Last epoch whose pre-decay checkpoint belongs to the historical WD output",
    )
    parser.add_argument(
        "--checkpoint-override",
        action="append",
        default=[],
        metavar="WD:EPOCH=/ABSOLUTE/PATH",
        help="Restart a failed frontier from its preceding clean checkpoint",
    )
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
    return ["0.01", "0.033", "0.1", middle, MAX_WEIGHT_DECAY]


def manifest_path(model: str, batch: int, *, locked: bool = False) -> Path:
    policy = "locked-wd-predecay" if locked else "adaptive"
    return MANIFEST_DIR / f"small-dense-{model}-bs{batch}-dr-wt-embwd-{policy}.json"


def output_root(model: str) -> str:
    return f"{OUTPUT_ROOT}/dense_{model}_dclm1b"


def output_for(model: str, batch: int, wd: str) -> str:
    return f"{output_root(model)}/bs{batch}_dr_wt_embwd_lr{LEARNING_RATE}_wd{wd}"


def parse_output_overrides(
    values: list[str],
    ladder: list[str],
    *,
    model: str | None = None,
    batch: int | None = None,
) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        wd, separator, raw_path = value.partition("=")
        if not separator or not wd or not raw_path:
            raise ValueError(f"output override must have the form WD=/absolute/path: {value}")
        path = Path(raw_path)
        if wd not in ladder:
            raise ValueError(f"output override WD{wd} is outside the configured ladder")
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"output override must be a normalized absolute path: {raw_path}")
        normalized = str(path)
        if not normalized.startswith(ALLOWED_OUTPUT_PREFIX):
            raise ValueError(f"output override is outside the approved model root: {normalized}")
        if model is not None and not normalized.startswith(output_root(model).rstrip("/") + "/"):
            raise ValueError(f"output override is outside the {model} output root: {normalized}")
        if wd in overrides:
            raise ValueError(f"duplicate output override for WD{wd}")
        overrides[wd] = normalized
    if len(set(overrides.values())) != len(overrides):
        raise ValueError("output overrides must use distinct directories")
    if model is not None and batch is not None:
        resolved = [overrides.get(wd, output_for(model, batch, wd)) for wd in ladder]
        if len(set(resolved)) != len(resolved):
            raise ValueError("an output override collides with another fixed-WD trajectory")
    return overrides


def parse_checkpoint_overrides(
    values: list[str], ladder: list[str], *, model: str, batch: int
) -> dict[str, dict[str, str]]:
    overrides: dict[str, dict[str, str]] = {}
    for value in values:
        coordinate, separator, raw_path = value.partition("=")
        wd, epoch_separator, raw_epoch = coordinate.partition(":")
        if not separator or not epoch_separator or not wd or not raw_epoch or not raw_path:
            raise ValueError(
                f"checkpoint override must have the form WD:EPOCH=/absolute/path: {value}"
            )
        if wd not in ladder:
            raise ValueError(f"checkpoint override WD{wd} is outside the configured ladder")
        epoch = str(int(raw_epoch))
        path = Path(raw_path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"checkpoint override must be a normalized absolute path: {raw_path}")
        normalized = str(path)
        expected_parent = Path(output_for(model, batch, wd))
        if path.parent != expected_parent:
            raise ValueError(
                f"checkpoint override must remain in the canonical WD{wd} directory {expected_parent}"
            )
        if epoch in overrides.setdefault(wd, {}):
            raise ValueError(f"duplicate checkpoint override for WD{wd} E{epoch}")
        overrides[wd][epoch] = normalized
    return overrides


def build_manifest(
    model: str,
    batch: int,
    *,
    locked_wd: str | None = None,
    historical_predecay_through_epoch: int | None = None,
) -> tuple[dict[str, Any], Path]:
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
    if locked_wd is not None:
        if historical_predecay_through_epoch is None:
            raise ValueError("locked-WD manifest requires a historical pre-decay boundary")
        if locked_wd not in ladder:
            raise ValueError(f"locked WD {locked_wd} is outside the historical ladder")
        if float(locked_wd) > float(MAX_WEIGHT_DECAY):
            raise ValueError("locked WD must not exceed 1.0")
        manifest.update(
            {
                "policy": "locked_wd_predecay_saturation_v1",
                "lockedWd": locked_wd,
                "postDecaySourceCount": 3,
                "comparisonPolicy": "within_phase_only",
                "preDecaySaturationCriterion": "strict_non_improvement",
                "historicalPreDecayStartEpoch": HISTORICAL_PREDECAY_START_EPOCH,
                "historicalPreDecayThroughEpoch": historical_predecay_through_epoch,
                "runSuffix": "sm0821-lockedwd-predecay",
            }
        )
    path = manifest_path(model, batch, locked=locked_wd is not None)
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
        value for value in arguments if value.startswith("--dataset.subset_manifest=")
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
    locked_wd = args.locked_wd if args.predecay_policy_replacement else None
    manifest, path = build_manifest(
        args.model,
        args.global_sequences,
        locked_wd=locked_wd,
        historical_predecay_through_epoch=args.historical_predecay_through_epoch,
    )
    overrides = parse_output_overrides(
        args.output_override,
        manifest["wdLadder"],
        model=args.model,
        batch=args.global_sequences,
    )
    spec = copy.deepcopy(base_spec(str(manifest["baseExperiment"])))
    audit_base_spec(args.model, spec)
    task = spec["tasks"][0]
    runner = (
        "scripts/models/run_small_dense_locked_wd_predecay_chain.py"
        if args.predecay_policy_replacement
        else "scripts/models/run_small_dense_dr_wt_embedwd_chain.py"
    )
    task["arguments"] = ["python", runner, "--manifest", str(path)]
    for wd, output in overrides.items():
        task["arguments"].extend(("--output-override", f"{wd}={output}"))
    for value in args.checkpoint_override:
        task["arguments"].extend(("--checkpoint-override", value))
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
    if args.predecay_policy_replacement:
        spec["description"] = (
            f"Dense {args.model.upper()} BS{args.global_sequences} DR+WT+EmbedWD "
            f"locked-WD{args.locked_wd} pre-decay saturation chain. Pre-decay and "
            "post-decay results are isolated comparison groups."
        )
    else:
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


def validate_recovery_target(args: argparse.Namespace) -> None:
    if not args.resume_experiment:
        return
    report = json.loads(report_path(args.model).read_text())
    record_id = f"dense-{args.model}-bs{args.global_sequences}-dr-wt-embwd-adaptive"
    record = next(
        (
            candidate
            for candidate in report.get("adaptiveDrWtEmbedWdChains", [])
            if candidate.get("id") == record_id
        ),
        None,
    )
    if record is None:
        raise RuntimeError(f"adaptive chain {record_id} is not registered")
    if str(record.get("experiment")) != args.resume_experiment:
        raise RuntimeError(
            f"registered experiment {record.get('experiment')} does not match "
            f"recovery target {args.resume_experiment}"
        )
    if args.predecay_policy_replacement:
        frontiers = record.get("frontiers", {})
        if not frontiers:
            raise RuntimeError("locked-WD transition requires a resolved WD frontier")
        latest_epoch = max(int(epoch) for epoch in frontiers)
        selected = str(frontiers[str(latest_epoch)]["selectedWd"])
        if str(args.locked_wd) != selected:
            raise RuntimeError(
                f"locked WD must equal the last resolved selection WD{selected}, "
                f"got WD{args.locked_wd}"
            )
        transition = record.get("policyTransition") or {}
        awaited = transition.get("awaitStage") or {}
        expected_epoch = awaited.get("epoch")
        if expected_epoch is None:
            raise RuntimeError("pre-decay transition is missing its requested stage boundary")
        if int(args.historical_predecay_through_epoch) != int(expected_epoch):
            raise RuntimeError(
                "historical pre-decay boundary must equal the completed transition stage "
                f"E{expected_epoch}"
            )
        return
    if args.policy_replacement:
        return
    if args.checkpoint_override:
        overrides = parse_checkpoint_overrides(
            args.checkpoint_override,
            list(record["wdLadder"]),
            model=args.model,
            batch=args.global_sequences,
        )
        active_wd = str(record.get("activeWd"))
        active_epoch = str(record.get("activeEpoch"))
        if active_epoch not in overrides.get(active_wd, {}):
            raise RuntimeError(
                f"checkpoint recovery must override active WD{active_wd} E{active_epoch}"
            )
        return
    overrides = parse_output_overrides(
        args.output_override,
        list(record["wdLadder"]),
        model=args.model,
        batch=args.global_sequences,
    )
    active_wd = record.get("activeWd")
    if active_wd and str(active_wd) not in overrides:
        raise RuntimeError(
            f"recovery must override the active WD{active_wd} trajectory; got {sorted(overrides)}"
        )


def write_report(
    model: str,
    batch: int,
    experiment: str,
    revision: str,
    manifest: dict[str, Any],
    *,
    resume_experiment: str | None = None,
    output_overrides: dict[str, str] | None = None,
    checkpoint_overrides: dict[str, dict[str, str]] | None = None,
    policy_replacement: bool = False,
    predecay_policy_replacement: bool = False,
    locked_wd: str | None = None,
) -> None:
    path = report_path(model)
    report = json.loads(path.read_text())
    records = report.setdefault("adaptiveDrWtEmbedWdChains", [])
    record_id = f"dense-{model}-bs{batch}-dr-wt-embwd-adaptive"
    existing = next((record for record in records if record.get("id") == record_id), None)
    if existing is not None and resume_experiment is None:
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
        "outputByWd": {wd: output_for(model, batch, wd) for wd in manifest["wdLadder"]},
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
    if resume_experiment is None:
        records.append(record)
    else:
        if existing is None:
            raise RuntimeError(f"adaptive chain {record_id} is not registered")
        if str(existing.get("experiment")) != resume_experiment:
            raise RuntimeError(
                f"registered experiment {existing.get('experiment')} does not match "
                f"recovery target {resume_experiment}"
            )
        overrides = output_overrides or {}
        history = list(existing.get("attemptHistory", []))
        history.append(
            {
                "beaker": resume_experiment,
                "status": (
                    "stopped-for-predecay-policy-replacement"
                    if predecay_policy_replacement
                    else "stopped-for-policy-replacement"
                    if policy_replacement
                    else "stopped-for-checkpoint-recovery"
                    if checkpoint_overrides
                    else "stopped-for-path-recovery"
                ),
                "activeEpoch": existing.get("activeEpoch"),
                "activeWd": existing.get("activeWd"),
                "wandbHealth": existing.get("wandbHealth"),
                "outputByWd": dict(existing.get("outputByWd", {})),
            }
        )
        preserved = {
            "results": existing.get("results", {}),
            "frontiers": existing.get("frontiers", {}),
            "activeEpoch": existing.get("activeEpoch", 1),
            "activeWds": existing.get("activeWds", manifest["initialWds"]),
            "activeWd": existing.get("activeWd"),
            "policyTransition": existing.get("policyTransition", {}),
            "preDecayResults": existing.get("preDecayResults", {}),
            "postDecayResults": existing.get("postDecayResults", {}),
        }
        existing.clear()
        existing.update(record)
        existing.update(preserved)
        existing["attemptHistory"] = history
        existing["recoveryOf"] = resume_experiment
        existing["outputByWd"].update(overrides)
        if overrides:
            existing["pathOverrides"] = overrides
        if checkpoint_overrides:
            existing["checkpointOverrides"] = checkpoint_overrides
        if predecay_policy_replacement:
            existing.update(
                {
                    "policy": "locked_wd_predecay_saturation_v1",
                    "lockedWd": str(locked_wd),
                    "wdTuningStopped": True,
                    "comparisonPolicy": "within_phase_only",
                    "preDecaySaturationCriterion": "strict_non_improvement",
                    "postDecaySourceCount": 3,
                    "historicalPreDecayStartEpoch": manifest[
                        "historicalPreDecayStartEpoch"
                    ],
                    "historicalPreDecayThroughEpoch": manifest["historicalPreDecayThroughEpoch"],
                    "activePhase": "backfill_pre_decay_evaluations",
                    "activeEpoch": manifest["historicalPreDecayStartEpoch"],
                    "activeWds": [str(locked_wd)],
                    "activeWd": str(locked_wd),
                    "preDecayResults": existing.get("preDecayResults", {}),
                    "postDecayResults": existing.get("postDecayResults", {}),
                }
            )
            transition = dict(existing.get("policyTransition", {}))
            transition.update(
                {
                    "status": "replacement-submitted",
                    "replacementExperiment": experiment,
                    "lockedWd": str(locked_wd),
                }
            )
            existing["policyTransition"] = transition
            existing["reason"] = (
                f"WD tuning stopped at the last resolved selection WD{locked_wd}. "
                "Starting at E8, the replacement evaluates exact pre-decay checkpoints "
                "only at configured legacy WSD frontiers, continues with constant LR one "
                "frontier at a time until pre-decay non-improvement, then decays the last "
                "three frontier checkpoints and selects only among post-decay results."
            )
        elif policy_replacement:
            existing["reason"] = (
                "Policy replacement stopped the prior Beaker experiment and resumed the "
                "same canonical trajectories with a hard WD<=1.0 cap. Completed stages "
                "and frontier evidence were retained."
            )
        elif checkpoint_overrides:
            selected = next(iter(checkpoint_overrides.values()))
            checkpoint = next(iter(selected.values()))
            existing["reason"] = (
                "Numerical recovery replaced the retry-exhausted Beaker experiment and "
                f"forced the failed frontier to restart from clean checkpoint {checkpoint}. "
                "Canonical fixed-WD output paths, completed stages, and frontier evidence "
                "were retained."
            )
        else:
            existing["reason"] = (
                "Guarded recovery replaced a W&B-suspicious task after stopping its prior "
                "Beaker experiment. Completed frontier evidence was retained and the affected "
                "fixed-WD trajectory was rerouted to the explicitly supplied directory."
            )
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
    if args.resume_experiment and not args.register:
        raise SystemExit("--resume-experiment requires --register")
    if args.resume_experiment and not args.stop_existing:
        raise SystemExit("--resume-experiment requires --stop-existing")
    if (
        args.resume_experiment
        and not args.output_override
        and not args.policy_replacement
        and not args.predecay_policy_replacement
        and not args.checkpoint_override
    ):
        raise SystemExit(
            "--resume-experiment requires a policy replacement, --checkpoint-override, "
            "or at least one --output-override"
        )
    if args.policy_replacement and args.predecay_policy_replacement:
        raise SystemExit("choose exactly one policy replacement mode")
    if args.policy_replacement and not args.resume_experiment:
        raise SystemExit("--policy-replacement requires --resume-experiment")
    if args.policy_replacement and args.output_override:
        raise SystemExit("--policy-replacement must preserve canonical output paths")
    if args.policy_replacement and args.checkpoint_override:
        raise SystemExit("--policy-replacement cannot also be a checkpoint recovery")
    if args.predecay_policy_replacement and not args.resume_experiment:
        raise SystemExit("--predecay-policy-replacement requires --resume-experiment")
    if args.predecay_policy_replacement and not args.locked_wd:
        raise SystemExit("--predecay-policy-replacement requires --locked-wd")
    if args.predecay_policy_replacement and args.historical_predecay_through_epoch is None:
        raise SystemExit(
            "--predecay-policy-replacement requires --historical-predecay-through-epoch"
        )
    if args.locked_wd and not args.predecay_policy_replacement:
        raise SystemExit("--locked-wd is scoped to --predecay-policy-replacement")
    if args.historical_predecay_through_epoch is not None and not args.predecay_policy_replacement:
        raise SystemExit(
            "--historical-predecay-through-epoch is scoped to the pre-decay replacement"
        )
    if args.predecay_policy_replacement and (args.output_override or args.checkpoint_override):
        raise SystemExit(
            "pre-decay policy replacement must preserve canonical paths and exact lineage"
        )
    if args.output_override and args.checkpoint_override:
        raise SystemExit("path recovery and checkpoint recovery must be separate submissions")
    if args.stop_existing and not args.resume_experiment:
        raise SystemExit("--stop-existing requires --resume-experiment")
    validate_recovery_target(args)
    default_name = experiment_name(args.model, args.global_sequences)
    if args.resume_experiment:
        mode = "predecay" if args.predecay_policy_replacement else "recovery"
        default_name += f"-{mode}-" + datetime.now(tz=UTC).strftime("%Y%m%d-%H%M")
    name = args.name or default_name
    if args.manifest_only:
        _, path = build_manifest(
            args.model,
            args.global_sequences,
            locked_wd=args.locked_wd if args.predecay_policy_replacement else None,
            historical_predecay_through_epoch=args.historical_predecay_through_epoch,
        )
        print(path)
        return
    spec, manifest, path = build_submission(args)
    audit_name(args.workspace, name)
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    if args.resume_experiment:
        run(["beaker", "experiment", "stop", args.resume_experiment])
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
        if args.resume_experiment:
            overrides = parse_output_overrides(
                args.output_override,
                manifest["wdLadder"],
                model=args.model,
                batch=args.global_sequences,
            )
            checkpoint_overrides = parse_checkpoint_overrides(
                args.checkpoint_override,
                manifest["wdLadder"],
                model=args.model,
                batch=args.global_sequences,
            )
            write_report(
                args.model,
                args.global_sequences,
                ids[0],
                args.revision,
                manifest,
                resume_experiment=args.resume_experiment,
                output_overrides=overrides,
                checkpoint_overrides=checkpoint_overrides,
                policy_replacement=args.policy_replacement,
                predecay_policy_replacement=args.predecay_policy_replacement,
                locked_wd=args.locked_wd,
            )
        else:
            write_report(args.model, args.global_sequences, ids[0], args.revision, manifest)


if __name__ == "__main__":
    main()
