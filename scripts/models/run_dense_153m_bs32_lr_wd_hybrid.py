#!/usr/bin/env python3
"""Run the guarded Dense-153M BS32 LR/WD hybrid study in one Beaker task.

The constant-LR checkpoints and every WSD branch are isolated. The first
phase probes LR5e-4/WD0.033 at E40, uses E32 only as a negative-result sanity
check, and otherwise compares the best matched POST result through E80 with
the existing LR1e-3/WD0.033 trajectory. The second phase starts a fresh
WD0.1 trajectory at the selected LR and evaluates E40,E48,...,E80.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_small_dense_dr_wt_embedwd_chain as small
import run_small_dense_saturation_chain as common

POLICY = "dense_153m_bs32_lr_wd_hybrid_v1"
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"
ALLOWED_OUTPUT_PREFIX = "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm1b/"
STATE_DIRECTORY = ".dense_153m_bs32_lr_wd_hybrid_v1"
DECAY_FRACTION = 0.1
SEQUENCE_LENGTH = 4096
POOL_TOKENS = 1_000_000_000
GLOBAL_SEQUENCES = 32
WORLD_SIZE = 2
RANK_MICROBATCH_SEQUENCES = 16
WARMUP_STEPS = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    return parser.parse_args()


def total_step(epoch: int) -> int:
    return math.ceil(epoch * POOL_TOKENS / (GLOBAL_SEQUENCES * SEQUENCE_LENGTH))


def stable_step(epoch: int) -> int:
    endpoint = total_step(epoch)
    return endpoint - round(DECAY_FRACTION * endpoint) - 1


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    validate_manifest(value)
    return value


def validate_manifest(value: dict[str, Any]) -> None:
    required = {
        "policy",
        "id",
        "model",
        "pool",
        "poolTokens",
        "sequenceLength",
        "globalSequences",
        "nprocPerNode",
        "rankMicrobatchSequences",
        "gradientAccumulationSteps",
        "warmupSteps",
        "saveEveryEpochs",
        "probe",
        "baseline",
        "followup",
        "comparisonMetric",
        "tiePolicy",
        "dynamicRepacking",
        "weightTying",
        "embeddingWeightDecay",
        "runSuffix",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"manifest is missing keys: {missing}")
    fixed = {
        "policy": POLICY,
        "model": "153m",
        "pool": "dclm1b",
        "poolTokens": POOL_TOKENS,
        "sequenceLength": SEQUENCE_LENGTH,
        "globalSequences": GLOBAL_SEQUENCES,
        "nprocPerNode": WORLD_SIZE,
        "rankMicrobatchSequences": RANK_MICROBATCH_SEQUENCES,
        "gradientAccumulationSteps": 1,
        "warmupSteps": WARMUP_STEPS,
        "saveEveryEpochs": 4,
        "comparisonMetric": "healthy_matched_post_validationExact",
        "dynamicRepacking": False,
        "weightTying": False,
        "embeddingWeightDecay": "zero",
    }
    mismatches = [key for key, expected in fixed.items() if value.get(key) != expected]
    if mismatches:
        raise ValueError(f"manifest fixed-policy mismatch: {mismatches}")

    probe = value["probe"]
    baseline = value["baseline"]
    followup = value["followup"]
    if Decimal(str(probe.get("learningRate"))) != Decimal("0.0005"):
        raise ValueError("probe LR must be 5e-4")
    if Decimal(str(probe.get("weightDecay"))) != Decimal("0.033"):
        raise ValueError("probe WD must be 0.033")
    if int(probe.get("firstEvaluationEpoch", 0)) != 40:
        raise ValueError("probe must evaluate E40 first")
    if int(probe.get("sanityEpoch", 0)) != 32:
        raise ValueError("negative E40 probe must use E32 as its sanity check")
    if [int(epoch) for epoch in probe.get("lateEvaluationEpochs", [])] != [48, 56, 64, 72, 80]:
        raise ValueError("probe late evaluation ladder must be E48,E56,...,E80")
    if Decimal(str(baseline.get("learningRate"))) != Decimal("0.001"):
        raise ValueError("baseline LR must be 1e-3")
    if Decimal(str(baseline.get("weightDecay"))) != Decimal("0.033"):
        raise ValueError("baseline WD must be 0.033")
    expected_baseline_epochs = {"32", "40", "48", "56", "64", "72", "80"}
    if set(baseline.get("validationExactByEpoch", {})) != expected_baseline_epochs:
        raise ValueError("baseline must contain every matched E32-E80 POST result")
    if Decimal(str(followup.get("weightDecay"))) != Decimal("0.1"):
        raise ValueError("follow-up WD must be 0.1")
    if [int(epoch) for epoch in followup.get("evaluationEpochs", [])] != [40, 48, 56, 64, 72, 80]:
        raise ValueError("follow-up must evaluate E40,E48,...,E80")
    if set(followup.get("outputsByLearningRate", {})) != {"5e-4", "1e-3"}:
        raise ValueError("follow-up must define isolated outputs for either selected LR")
    outputs = [probe["output"], *followup["outputsByLearningRate"].values()]
    if len(outputs) != len(set(outputs)):
        raise ValueError("every possible fixed LR/WD trajectory needs a distinct output")
    for raw_output in outputs:
        output = Path(str(raw_output))
        if not output.is_absolute() or ".." in output.parts:
            raise ValueError(f"output is not a normalized absolute path: {output}")
        if not str(output).startswith(ALLOWED_OUTPUT_PREFIX):
            raise ValueError(f"output is outside the approved Dense-153M root: {output}")


def plan(value: dict[str, Any]) -> dict[str, Any]:
    probe = value["probe"]
    return {
        "probe": {
            "coordinate": f"LR{probe['learningRate']}/WD{probe['weightDecay']}",
            "retainedEpochs": list(range(4, 81, 4)),
            "firstEvaluationEpoch": 40,
            "negativeSanityEpoch": 32,
            "lateEvaluationEpochs": probe["lateEvaluationEpochs"],
        },
        "lrDecision": (
            "Select LR1e-3 early only when LR5e-4 is strictly worse at both E40 and E32; "
            "otherwise compare the best matched POST through E80, retaining LR1e-3 on a tie."
        ),
        "followup": {
            "coordinate": f"selected LR/WD{value['followup']['weightDecay']}",
            "retainedEpochs": list(range(4, 81, 4)),
            "evaluationEpochs": value["followup"]["evaluationEpochs"],
        },
    }


def checkpoint_complete(path: Path) -> bool:
    model_and_optim = path / "model_and_optim"
    train = path / "train"
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (model_and_optim / ".metadata").is_file()
        and any(model_and_optim.glob("*.distcp"))
        and all((train / f"rank{rank}.pt").is_file() for rank in range(WORLD_SIZE))
    )


def state_root(value: dict[str, Any]) -> Path:
    return Path(str(value["probe"]["output"])) / STATE_DIRECTORY


def coordinate_output(value: dict[str, Any], phase: str, lr: str | None = None) -> Path:
    if phase == "probe":
        return Path(str(value["probe"]["output"]))
    if phase == "followup" and lr is not None:
        return Path(str(value["followup"]["outputsByLearningRate"][lr]))
    raise ValueError(f"invalid coordinate request phase={phase!r} lr={lr!r}")


def claim_output(value: dict[str, Any], phase: str, lr: str, wd: str) -> Path:
    root = coordinate_output(value, phase, lr)
    marker = root / STATE_DIRECTORY / "owner.json"
    expected = {
        "policy": POLICY,
        "workflowId": value["id"],
        "phase": phase,
        "learningRate": lr,
        "weightDecay": wd,
        "constantOutput": str(root / "constant_lr"),
    }
    if marker.is_file():
        current = json.loads(marker.read_text())
        if current != expected:
            raise RuntimeError(f"output ownership mismatch at {marker}")
        return root
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing nonempty unowned output {root}")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, expected)
    return root


def base_arguments(lr: str, wd: str, *, evaluation: bool) -> list[str]:
    arguments = [*small.COMMON_ARGUMENTS, *small.MODEL_ARGUMENTS["153m"]]
    if not evaluation:
        arguments = common.upsert(
            arguments,
            "--trainer.callbacks.downstream_evaluator.tasks=",
            "--trainer.callbacks.downstream_evaluator.tasks=[]",
        )
        arguments = common.upsert(
            arguments,
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=",
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
        )
        heldout = small.HELDOUT_EVALUATOR.replace("eval_on_finish: true", "eval_on_finish: false")
        arguments = common.upsert(
            arguments,
            "--trainer.callbacks.heldout_evaluator=",
            f"--trainer.callbacks.heldout_evaluator={heldout}",
        )
    return [
        *arguments,
        f"--data_loader.global_batch_size={GLOBAL_SEQUENCES * SEQUENCE_LENGTH}",
        f"--train_module.rank_microbatch_size={RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH}",
        f"--train_module.optim.weight_decay={wd}",
        f"--lr={lr}",
        "--model.tie_embeddings=false",
    ]


def validate_predecay(path: Path, lr: str, wd: str) -> Path:
    if not checkpoint_complete(path):
        raise FileNotFoundError(f"incomplete exact pre-decay checkpoint {path}")
    config = json.loads((path / "config.json").read_text())
    mismatches: list[str] = []
    if int(config["data_loader"]["global_batch_size"]) != GLOBAL_SEQUENCES * SEQUENCE_LENGTH:
        mismatches.append("global_batch_size")
    if bool(config["model"]["tie_embeddings"]):
        mismatches.append("weight_tying")
    if bool(config.get("dataset", {}).get("dynamic_repacking", False)):
        mismatches.append("dynamic_repacking")
    optim = config["train_module"]["optim"]
    if Decimal(str(optim["lr"])) != Decimal(lr):
        mismatches.append("learning_rate")
    if Decimal(str(optim["weight_decay"])) != Decimal(wd):
        mismatches.append("weight_decay")
    if mismatches:
        raise RuntimeError(f"pre-decay recipe mismatch {mismatches}: {path}")
    return path


def run_torch(name: str, arguments: list[str], log_path: Path) -> None:
    subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        common.stream_command(
            ["torchrun", f"--nproc-per-node={WORLD_SIZE}", TRAINING_SCRIPT, name, *arguments],
            log_file,
        )


def constant_arguments(
    value: dict[str, Any],
    phase: str,
    lr: str,
    wd: str,
    root: Path,
    target_epoch: int,
    source: Path | None,
) -> tuple[str, list[str]]:
    checkpoint_epochs = list(range(4, target_epoch + 1, 4))
    if source is not None:
        source_step = int(source.name.removeprefix("step"))
        checkpoint_epochs = [
            epoch for epoch in checkpoint_epochs if stable_step(epoch) > source_step
        ]
    fixed_steps = [stable_step(epoch) for epoch in checkpoint_epochs]
    name = (
        f"dense_153m_step1_0802_repeated_dclm1b_bs32_{phase}_pd_e{target_epoch}_"
        f"lr{lr}_wd{wd}_{value['runSuffix']}"
    )
    arguments = [
        *base_arguments(lr, wd, evaluation=False),
        f"--save-folder={root / 'constant_lr'}",
        f"--trainer.max_duration={{value: {stable_step(target_epoch)}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dense-153m,"
            f"dclm1b,bs32,{phase},constant-lr,pre-decay,e{target_epoch},lr{lr},wd{wd}]"
        ),
        "--trainer.callbacks.checkpointer.fixed_steps="
        + json.dumps(fixed_steps, separators=(",", ":")),
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=1000",
        "--data_loader.restore_data_order_from_state=true",
        "--data_loader.ignore_fingerprint_mismatch=false",
        (
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.ConstantWithWarmup, "
            f"warmup: {WARMUP_STEPS}}}"
        ),
    ]
    if source is not None:
        arguments.extend(
            [
                "--force_exact_trainer_load_path=true",
                f"--trainer.load_path={source}",
                "--trainer.load_trainer_state=true",
                "--trainer.load_optim_state=true",
                "--trainer.reset_data_loader_state_on_load_path=false",
                "--train_module.validate_optimizer_hyperparameters_on_load=true",
            ]
        )
    return name, arguments


def latest_predecay(root: Path, through_epoch: int, lr: str, wd: str) -> tuple[int, Path] | None:
    for epoch in range(through_epoch, 0, -4):
        candidate = root / "constant_lr" / f"step{stable_step(epoch)}"
        if checkpoint_complete(candidate):
            return epoch, validate_predecay(candidate, lr, wd)
    return None


def ensure_predecay(
    value: dict[str, Any], phase: str, lr: str, wd: str, target_epoch: int, state: dict[str, Any]
) -> Path:
    root = claim_output(value, phase, lr, wd)
    target = root / "constant_lr" / f"step{stable_step(target_epoch)}"
    if not checkpoint_complete(target):
        latest = latest_predecay(root, target_epoch - 4, lr, wd)
        source_epoch, source = latest if latest is not None else (None, None)
        name, arguments = constant_arguments(value, phase, lr, wd, root, target_epoch, source)
        state.update(
            {
                "status": "producer_running",
                "phase": phase,
                "learningRate": lr,
                "weightDecay": wd,
                "currentEpoch": target_epoch,
                "sourceEpoch": source_epoch,
            }
        )
        atomic_json(state_root(value) / "workflow.json", state)
        print(
            f"DENSE153M_BS32_HYBRID_PD_START phase={phase} lr={lr} wd={wd} "
            f"epoch={target_epoch} sourceEpoch={source_epoch} source={source} output={root / 'constant_lr'}",
            flush=True,
        )
        run_torch(name, arguments, state_root(value) / "logs" / f"{phase}_pd_e{target_epoch}.log")
    target = validate_predecay(target, lr, wd)
    retained = [
        epoch
        for epoch in range(4, target_epoch + 1, 4)
        if checkpoint_complete(root / "constant_lr" / f"step{stable_step(epoch)}")
    ]
    print(
        f"DENSE153M_BS32_HYBRID_PD_RETAINED phase={phase} lr={lr} wd={wd} "
        f"epoch={target_epoch} checkpoint={target} retained={retained}",
        flush=True,
    )
    return target


def post_arguments(
    value: dict[str, Any], phase: str, lr: str, wd: str, epoch: int, source: Path, output: Path
) -> tuple[str, list[str]]:
    name = (
        f"dense_153m_step1_0802_repeated_dclm1b_bs32_{phase}_post_e{epoch}_"
        f"lr{lr}_wd{wd}_{value['runSuffix']}"
    )
    arguments = common.upsert(
        base_arguments(lr, wd, evaluation=True),
        "--train_module.scheduler=",
        (
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, "
            f"warmup: {WARMUP_STEPS}, decay_fraction: {DECAY_FRACTION}}}"
        ),
    )
    return name, [
        *arguments,
        f"--save-folder={output}",
        f"--trainer.max_duration={{value: {total_step(epoch)}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dense-153m,dclm1b,"
            f"bs32,{phase},post-decay,heldout-eval,downstream-eval,e{epoch},lr{lr},wd{wd}]"
        ),
        f"--trainer.callbacks.checkpointer.fixed_steps=[{total_step(epoch)}]",
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        "--data_loader.restore_data_order_from_state=false",
        "--data_loader.ignore_fingerprint_mismatch=true",
        "--force_exact_trainer_load_path=true",
        f"--trainer.load_path={source}",
        "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=true",
        "--trainer.reset_data_loader_state_on_load_path=false",
        "--train_module.validate_optimizer_hyperparameters_on_load=true",
    ]


def recovery_eval_arguments(
    lr: str, wd: str, checkpoint: Path, output: Path, name: str
) -> list[str]:
    heldout = small.HELDOUT_EVALUATOR.replace(
        "eval_on_finish: true", "eval_on_finish: false, eval_on_startup: true"
    )
    arguments = common.upsert(
        base_arguments(lr, wd, evaluation=True),
        "--trainer.callbacks.heldout_evaluator=",
        f"--trainer.callbacks.heldout_evaluator={heldout}",
    )
    arguments = common.upsert(
        arguments,
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=",
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
    )
    arguments = common.upsert(
        arguments,
        "--trainer.callbacks.downstream_evaluator.eval_on_startup=",
        "--trainer.callbacks.downstream_evaluator.eval_on_startup=true",
    )
    return [
        *arguments,
        f"--save-folder={output}",
        f"--trainer.callbacks.wandb.name={name}",
        "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dense-153m,bs32,recovered-post-eval]",
        "--trainer.max_duration={value: 1000000000000, unit: steps}",
        "--trainer.callbacks.checkpointer.enabled=false",
        f"--load_path={checkpoint}",
        "--load_trainer_state=false",
    ]


def quarantine_partial(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(path.name + f".partial.{timestamp}")
    counter = 1
    while destination.exists():
        destination = path.with_name(path.name + f".partial.{timestamp}.{counter}")
        counter += 1
    shutil.move(str(path), str(destination))
    print(
        f"DENSE153M_BS32_HYBRID_POST_QUARANTINE source={path} destination={destination}", flush=True
    )


def parse_result(log_paths: list[Path], epoch: int) -> dict[str, Any]:
    clean = common.ANSI.sub("", "\n".join(path.read_text() for path in log_paths if path.is_file()))
    validation_values = common.WANDB_VALIDATION_LOSS.findall(
        clean
    ) or common.VALIDATION_LOSS.findall(clean)
    if not validation_values:
        raise RuntimeError(f"E{epoch} completed without held-out validationExact")
    accuracy: dict[str, float] = {}
    bpb: dict[str, float] = {}
    for task, raw_value in common.ACCURACY.findall(clean):
        accuracy[common.normalize_task(task)] = 100 * float(raw_value)
    for task, raw_value in common.BPB.findall(clean):
        bpb[common.normalize_task(task)] = float(raw_value)
    missing = [task for task in common.REPORT_TASKS if task not in accuracy or task not in bpb]
    if missing:
        raise RuntimeError(f"E{epoch} completed without downstream metrics for {missing}")
    validation = float(validation_values[-1])
    train_values = common.TRAIN_LOSS.findall(clean)
    train = float(train_values[-1]) if train_values else None
    result: dict[str, Any] = {
        "epoch": epoch,
        "status": "complete",
        "validation": round(validation, 3),
        "validationExact": validation,
        "c4": round(validation, 3),
        "downstream": accuracy,
        "downstreamBpb": bpb,
        "acc": accuracy["hellaswag"],
        "bpb": bpb["hellaswag"],
        "avg8Accuracy": sum(accuracy[task] for task in common.AVERAGE_TASKS)
        / len(common.AVERAGE_TASKS),
        "avg8Bpb": sum(bpb[task] for task in common.AVERAGE_TASKS) / len(common.AVERAGE_TASKS),
    }
    if train is not None:
        result["train"] = train
        result["gap"] = round(validation - train, 6)
    wandb_values = common.WANDB_RUN.findall(clean)
    if wandb_values:
        result["wandb"] = wandb_values[-1]
    return result


def evaluate(
    value: dict[str, Any], phase: str, lr: str, wd: str, epoch: int, state: dict[str, Any]
) -> dict[str, Any]:
    result_path = state_root(value) / "results" / phase / f"e{epoch}.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        emit_result(phase, lr, wd, epoch, result)
        return result
    root = claim_output(value, phase, lr, wd)
    source = validate_predecay(root / "constant_lr" / f"step{stable_step(epoch)}", lr, wd)
    output = root / "post_decay_runs" / f"e{epoch}"
    endpoint = output / f"step{total_step(epoch)}"
    log_path = state_root(value) / "logs" / f"{phase}_post_e{epoch}.log"
    recovery_log = state_root(value) / "logs" / f"{phase}_post_e{epoch}_recovered_eval.log"
    state.update(
        {
            "status": "post_running",
            "phase": phase,
            "learningRate": lr,
            "weightDecay": wd,
            "currentEpoch": epoch,
        }
    )
    atomic_json(state_root(value) / "workflow.json", state)
    if not checkpoint_complete(endpoint):
        if output.exists():
            quarantine_partial(output)
        name, arguments = post_arguments(value, phase, lr, wd, epoch, source, output)
        print(
            f"DENSE153M_BS32_HYBRID_POST_START phase={phase} lr={lr} wd={wd} "
            f"epoch={epoch} source={source} output={output}",
            flush=True,
        )
        run_torch(name, arguments, log_path)
    if not checkpoint_complete(endpoint):
        raise RuntimeError(f"E{epoch} decay exited without complete endpoint {endpoint}")
    try:
        result = parse_result([log_path], epoch)
    except RuntimeError:
        recovered_name = (
            f"dense_153m_bs32_{phase}_post_e{epoch}_recovered_eval_{value['runSuffix']}"
        )
        run_torch(
            recovered_name,
            recovery_eval_arguments(lr, wd, endpoint, output / "recovered_eval", recovered_name),
            recovery_log,
        )
        result = parse_result([log_path, recovery_log], epoch)
    result.update(
        {
            "policy": POLICY,
            "phase": phase,
            "comparisonGroup": "post_decay",
            "lr": lr,
            "wd": wd,
            "output": str(root),
            "constantOutput": str(root / "constant_lr"),
            "preDecayCheckpoint": str(source),
            "sourcePreDecayCheckpoint": str(source),
            "endpointCheckpoint": str(endpoint),
            "postOutput": str(output),
            "dynamicRepacking": False,
            "weightTying": False,
            "embeddingWeightDecay": "zero",
            "source": "integrated_isolated_wsd_decay_heldout_and_downstream_eval",
        }
    )
    atomic_json(result_path, result)
    emit_result(phase, lr, wd, epoch, result)
    return result


def emit_result(phase: str, lr: str, wd: str, epoch: int, result: dict[str, Any]) -> None:
    payload = json.dumps(result, separators=(",", ":"), sort_keys=True)
    print(
        f"DENSE153M_BS32_HYBRID_POST_RESULT phase={phase} lr={lr} wd={wd} "
        f"epoch={epoch} json={payload}",
        flush=True,
    )


def decide_probe_route(
    e40: float, baseline_e40: float, e32: float | None = None, baseline_e32: float | None = None
) -> str:
    if e40 <= baseline_e40:
        return "continue_to_e80"
    if e32 is None or baseline_e32 is None:
        return "evaluate_e32_sanity"
    return "select_1e-3" if e32 > baseline_e32 else "continue_to_e80"


def select_learning_rate(
    probe_results: dict[int, dict[str, Any]], baseline: dict[str, float]
) -> tuple[str, dict[str, Any]]:
    matched_epochs = sorted(epoch for epoch in (40, 48, 56, 64, 72, 80) if epoch in probe_results)
    if matched_epochs != [40, 48, 56, 64, 72, 80]:
        raise ValueError(f"late LR decision is missing matched epochs: {matched_epochs}")
    probe_best_epoch = min(
        matched_epochs, key=lambda epoch: float(probe_results[epoch]["validationExact"])
    )
    baseline_best_epoch = min(matched_epochs, key=lambda epoch: float(baseline[str(epoch)]))
    probe_best = float(probe_results[probe_best_epoch]["validationExact"])
    baseline_best = float(baseline[str(baseline_best_epoch)])
    selected = "5e-4" if probe_best < baseline_best else "1e-3"
    return selected, {
        "decision": selected,
        "reason": "strict_best_matched_post" if selected == "5e-4" else "baseline_better_or_tied",
        "probeBestEpoch": probe_best_epoch,
        "probeBestValidationExact": probe_best,
        "baselineBestEpoch": baseline_best_epoch,
        "baselineBestValidationExact": baseline_best,
        "matchedEpochs": matched_epochs,
    }


def emit_decision(decision: dict[str, Any]) -> None:
    print(
        "DENSE153M_BS32_HYBRID_LR_DECISION json="
        + json.dumps(decision, separators=(",", ":"), sort_keys=True),
        flush=True,
    )


def load_results(value: dict[str, Any], phase: str) -> dict[int, dict[str, Any]]:
    directory = state_root(value) / "results" / phase
    results: dict[int, dict[str, Any]] = {}
    if directory.is_dir():
        for path in directory.glob("e*.json"):
            result = json.loads(path.read_text())
            results[int(result["epoch"])] = result
    return results


def run(value: dict[str, Any]) -> None:
    probe = value["probe"]
    baseline = {
        str(epoch): float(metric)
        for epoch, metric in value["baseline"]["validationExactByEpoch"].items()
    }
    state_path = state_root(value) / "workflow.json"
    claim_output(value, "probe", str(probe["learningRate"]), str(probe["weightDecay"]))
    state = (
        json.loads(state_path.read_text())
        if state_path.is_file()
        else {
            "policy": POLICY,
            "id": value["id"],
            "status": "starting",
            "phase": "probe",
            "minRuntimeOmitted": True,
        }
    )
    if state.get("policy") != POLICY or state.get("id") != value["id"]:
        raise RuntimeError("workflow state ownership mismatch")
    atomic_json(state_path, state)

    probe_lr = str(probe["learningRate"])
    probe_wd = str(probe["weightDecay"])
    ensure_predecay(value, "probe", probe_lr, probe_wd, 40, state)
    e40 = evaluate(value, "probe", probe_lr, probe_wd, 40, state)
    route = decide_probe_route(float(e40["validationExact"]), baseline["40"])
    if route == "evaluate_e32_sanity":
        e32 = evaluate(value, "probe", probe_lr, probe_wd, 32, state)
        route = decide_probe_route(
            float(e40["validationExact"]),
            baseline["40"],
            float(e32["validationExact"]),
            baseline["32"],
        )

    if route == "select_1e-3":
        decision = {
            "decision": "1e-3",
            "reason": "probe_strictly_worse_at_e40_and_e32",
            "e40ProbeValidationExact": float(e40["validationExact"]),
            "e40BaselineValidationExact": baseline["40"],
            "e32ProbeValidationExact": float(load_results(value, "probe")[32]["validationExact"]),
            "e32BaselineValidationExact": baseline["32"],
            "matchedEpochs": [32, 40],
        }
        selected_lr = "1e-3"
    else:
        for epoch in [48, 56, 64, 72, 80]:
            ensure_predecay(value, "probe", probe_lr, probe_wd, epoch, state)
            evaluate(value, "probe", probe_lr, probe_wd, epoch, state)
        selected_lr, decision = select_learning_rate(load_results(value, "probe"), baseline)
    decision.update(
        {"policy": POLICY, "selectedLearningRate": selected_lr, "selectedWeightDecay": "0.033"}
    )
    atomic_json(state_root(value) / "lr_decision.json", decision)
    emit_decision(decision)

    followup_wd = str(value["followup"]["weightDecay"])
    claim_output(value, "followup", selected_lr, followup_wd)
    print(
        f"DENSE153M_BS32_HYBRID_FOLLOWUP_START lr={selected_lr} wd={followup_wd} "
        f"output={coordinate_output(value, 'followup', selected_lr)}",
        flush=True,
    )
    for epoch in [40, 48, 56, 64, 72, 80]:
        ensure_predecay(value, "followup", selected_lr, followup_wd, epoch, state)
        evaluate(value, "followup", selected_lr, followup_wd, epoch, state)

    state.update(
        {
            "status": "complete",
            "phase": "complete",
            "currentEpoch": 80,
            "selectedLearningRate": selected_lr,
            "selectedWeightDecay": followup_wd,
        }
    )
    atomic_json(state_path, state)
    print(
        f"DENSE153M_BS32_HYBRID_WORKFLOW_COMPLETE selectedLearningRate={selected_lr} "
        f"followupWeightDecay={followup_wd}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    value = load_manifest(args.manifest)
    if args.validate_only:
        print(f"validated {args.manifest}")
        return
    if args.print_plan:
        print(json.dumps(plan(value), indent=2))
        return
    run(value)


if __name__ == "__main__":
    main()
