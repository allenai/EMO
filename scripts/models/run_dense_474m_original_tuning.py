#!/usr/bin/env python3
"""Run guarded Dense-474M original-model BS32 probes or the BS128 E52 extension.

Each constant-LR frontier and WSD branch has an isolated output. The BS32 mode
continues exact E1 PD checkpoints for LR5e-4 and LR2e-3 through E4, evaluates
both, and compares them with the existing LR1e-3/WD0.1 E4 result. The BS128
mode continues exact E48 PD through E52, then immediately decays and evaluates.
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

POLICY = "dense_474m_original_tuning_v1"
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"
ALLOWED_OUTPUT_PREFIX = "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm1b/"
STATE_DIRECTORY = ".dense_474m_original_tuning_v1"
DECAY_FRACTION = 0.1
POOL_TOKENS = 1_000_000_000
SEQUENCE_LENGTH = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("bs32-probes", "bs128-e52"), required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    return parser.parse_args()


def total_step(epoch: int, batch: int) -> int:
    return math.ceil(epoch * POOL_TOKENS / (batch * SEQUENCE_LENGTH))


def stable_step(epoch: int, batch: int) -> int:
    endpoint = total_step(epoch, batch)
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
        "comparisonMetric",
        "dynamicRepacking",
        "weightTying",
        "embeddingWeightDecay",
        "bs32Probes",
        "bs128Extension",
        "runSuffix",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"manifest is missing keys: {missing}")
    fixed = {
        "policy": POLICY,
        "model": "474m",
        "pool": "dclm1b",
        "poolTokens": POOL_TOKENS,
        "sequenceLength": SEQUENCE_LENGTH,
        "comparisonMetric": "healthy_matched_post_validationExact",
        "dynamicRepacking": False,
        "weightTying": False,
        "embeddingWeightDecay": "zero",
    }
    mismatches = [key for key, expected in fixed.items() if value.get(key) != expected]
    if mismatches:
        raise ValueError(f"manifest fixed-policy mismatch: {mismatches}")

    probes = value["bs32Probes"]
    extension = value["bs128Extension"]
    validate_shape(probes, batch=32, nproc=2, warmup=768, source_epoch=1, target_epoch=4)
    validate_shape(extension, batch=128, nproc=8, warmup=192, source_epoch=48, target_epoch=52)
    coordinates = probes.get("coordinates", [])
    if len(coordinates) != 2:
        raise ValueError("BS32 must define exactly two LR probes")
    if {Decimal(str(item.get("learningRate"))) for item in coordinates} != {
        Decimal("0.0005"),
        Decimal("0.002"),
    }:
        raise ValueError("BS32 probes must be LR5e-4 and LR2e-3")
    if any(Decimal(str(item.get("weightDecay"))) != Decimal("0.1") for item in coordinates):
        raise ValueError("BS32 probes must use WD0.1")
    baseline = probes.get("baseline", {})
    if (
        Decimal(str(baseline.get("learningRate"))) != Decimal("0.001")
        or Decimal(str(baseline.get("weightDecay"))) != Decimal("0.1")
        or int(baseline.get("epoch", 0)) != 4
    ):
        raise ValueError("BS32 baseline must be LR1e-3/WD0.1 POST E4")
    if Decimal(str(extension.get("learningRate"))) != Decimal("0.002"):
        raise ValueError("BS128 extension must use LR2e-3")
    if Decimal(str(extension.get("weightDecay"))) != Decimal("0.3"):
        raise ValueError("BS128 extension must use WD0.3")
    all_items = [*coordinates, extension]
    outputs = [str(item["output"]) for item in all_items]
    if len(outputs) != len(set(outputs)):
        raise ValueError("every coordinate must have a distinct output")
    for item in all_items:
        output = Path(str(item["output"]))
        source = Path(str(item["sourceCheckpoint"]))
        if not output.is_absolute() or ".." in output.parts:
            raise ValueError(f"invalid output path: {output}")
        if not str(output).startswith(ALLOWED_OUTPUT_PREFIX):
            raise ValueError(f"output is outside the approved Dense-474M root: {output}")
        expected = stable_step(
            int(item.get("sourceEpoch", probes["sourceEpoch"])),
            int(item.get("globalSequences", probes["globalSequences"])),
        )
        if source.name != f"step{expected}":
            raise ValueError(
                f"source is not the exact E{item.get('sourceEpoch', probes['sourceEpoch'])} PD checkpoint: {source}"
            )


def validate_shape(
    value: dict[str, Any],
    *,
    batch: int,
    nproc: int,
    warmup: int,
    source_epoch: int,
    target_epoch: int,
) -> None:
    expected = {
        "globalSequences": batch,
        "nprocPerNode": nproc,
        "rankMicrobatchSequences": 16,
        "gradientAccumulationSteps": 1,
        "warmupSteps": warmup,
        "sourceEpoch": source_epoch,
        "targetEpoch": target_epoch,
    }
    mismatches = [
        key for key, expected_value in expected.items() if value.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(f"coordinate shape mismatch: {mismatches}")


def plan(value: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "bs32-probes":
        probes = value["bs32Probes"]
        return {
            "mode": mode,
            "coordinates": [item["id"] for item in probes["coordinates"]],
            "sourceEpoch": 1,
            "retainedEpochs": [2, 3, 4],
            "evaluationEpoch": 4,
            "baseline": probes["baseline"],
        }
    extension = value["bs128Extension"]
    return {
        "mode": mode,
        "coordinate": "LR2e-3/WD0.3",
        "sourceEpoch": 48,
        "retainedEpochs": [49, 50, 51, 52],
        "evaluationEpoch": 52,
        "previousValidationExact": extension["previousValidationExact"],
    }


def coordinate_values(value: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    if mode == "bs32-probes":
        probes = value["bs32Probes"]
        inherited = {
            key: probes[key]
            for key in (
                "globalSequences",
                "nprocPerNode",
                "rankMicrobatchSequences",
                "warmupSteps",
                "sourceEpoch",
                "targetEpoch",
            )
        }
        return [{**item, **inherited} for item in probes["coordinates"]]
    return [{"id": "bs128-lr2e-3-wd0.3-e52", **value["bs128Extension"]}]


def checkpoint_complete(path: Path, world_size: int) -> bool:
    model_and_optim = path / "model_and_optim"
    train = path / "train"
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (model_and_optim / ".metadata").is_file()
        and any(model_and_optim.glob("*.distcp"))
        and all((train / f"rank{rank}.pt").is_file() for rank in range(world_size))
    )


def state_root(coordinate: dict[str, Any]) -> Path:
    return Path(str(coordinate["output"])) / STATE_DIRECTORY


def claim_output(value: dict[str, Any], mode: str, coordinate: dict[str, Any]) -> Path:
    root = Path(str(coordinate["output"]))
    marker = root / STATE_DIRECTORY / "owner.json"
    expected = {
        "policy": POLICY,
        "workflowId": value["id"],
        "mode": mode,
        "coordinate": coordinate["id"],
        "constantOutput": str(root / "constant_lr"),
    }
    if marker.is_file():
        if json.loads(marker.read_text()) != expected:
            raise RuntimeError(f"output ownership mismatch at {marker}")
        return root
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing nonempty unowned output {root}")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, expected)
    return root


def base_arguments(coordinate: dict[str, Any], *, evaluation: bool) -> list[str]:
    batch = int(coordinate["globalSequences"])
    lr = str(coordinate["learningRate"])
    wd = str(coordinate["weightDecay"])
    arguments = [*small.COMMON_ARGUMENTS, *small.MODEL_ARGUMENTS["474m"]]
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
        f"--data_loader.global_batch_size={batch * SEQUENCE_LENGTH}",
        f"--train_module.rank_microbatch_size={int(coordinate['rankMicrobatchSequences']) * SEQUENCE_LENGTH}",
        f"--train_module.optim.weight_decay={wd}",
        f"--lr={lr}",
        "--model.tie_embeddings=false",
    ]


def validate_predecay(path: Path, coordinate: dict[str, Any]) -> Path:
    if not checkpoint_complete(path, int(coordinate["nprocPerNode"])):
        raise FileNotFoundError(f"incomplete exact pre-decay checkpoint {path}")
    config = json.loads((path / "config.json").read_text())
    batch = int(coordinate["globalSequences"])
    mismatches: list[str] = []
    if int(config["data_loader"]["global_batch_size"]) != batch * SEQUENCE_LENGTH:
        mismatches.append("global_batch_size")
    # Refactored canonical OLMo2 checkpoints omit this legacy key; omission is
    # the untied default, while an explicit true value remains a mismatch.
    if bool(config["model"].get("tie_embeddings", False)):
        mismatches.append("weight_tying")
    if bool(config.get("dataset", {}).get("dynamic_repacking", False)):
        mismatches.append("dynamic_repacking")
    optim = config["train_module"]["optim"]
    if Decimal(str(optim["lr"])) != Decimal(str(coordinate["learningRate"])):
        mismatches.append("learning_rate")
    if Decimal(str(optim["weight_decay"])) != Decimal(str(coordinate["weightDecay"])):
        mismatches.append("weight_decay")
    if mismatches:
        raise RuntimeError(f"pre-decay recipe mismatch {mismatches}: {path}")
    return path


def run_torch(name: str, arguments: list[str], log_path: Path, world_size: int) -> None:
    subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        common.stream_command(
            ["torchrun", f"--nproc-per-node={world_size}", TRAINING_SCRIPT, name, *arguments],
            log_file,
        )


def constant_arguments(
    value: dict[str, Any], coordinate: dict[str, Any], source: Path, root: Path
) -> tuple[str, list[str]]:
    batch = int(coordinate["globalSequences"])
    source_epoch = int(coordinate["sourceEpoch"])
    target_epoch = int(coordinate["targetEpoch"])
    lr = str(coordinate["learningRate"])
    wd = str(coordinate["weightDecay"])
    fixed_steps = [stable_step(epoch, batch) for epoch in range(source_epoch + 1, target_epoch + 1)]
    name = (
        f"dense_474m_step1_0802_repeated_dclm1b_bs{batch}_{coordinate['id']}_"
        f"pd_e{target_epoch}_{value['runSuffix']}"
    )
    return name, [
        *base_arguments(coordinate, evaluation=False),
        f"--save-folder={root / 'constant_lr'}",
        f"--trainer.max_duration={{value: {stable_step(target_epoch, batch)}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dense-474m,dclm1b,"
            f"bs{batch},original-tuning,constant-lr,pre-decay,e{target_epoch},lr{lr},wd{wd}]"
        ),
        "--trainer.callbacks.checkpointer.fixed_steps="
        + json.dumps(fixed_steps, separators=(",", ":")),
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=1000",
        "--data_loader.restore_data_order_from_state=true",
        "--data_loader.ignore_fingerprint_mismatch=false",
        (
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.ConstantWithWarmup, "
            f"warmup: {int(coordinate['warmupSteps'])}}}"
        ),
        "--force_exact_trainer_load_path=true",
        f"--trainer.load_path={source}",
        "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=true",
        "--trainer.reset_data_loader_state_on_load_path=false",
        "--train_module.validate_optimizer_hyperparameters_on_load=true",
    ]


def ensure_predecay(
    value: dict[str, Any], mode: str, coordinate: dict[str, Any], state: dict[str, Any]
) -> Path:
    root = claim_output(value, mode, coordinate)
    batch = int(coordinate["globalSequences"])
    target_epoch = int(coordinate["targetEpoch"])
    target = root / "constant_lr" / f"step{stable_step(target_epoch, batch)}"
    if not checkpoint_complete(target, int(coordinate["nprocPerNode"])):
        source = validate_predecay(Path(str(coordinate["sourceCheckpoint"])), coordinate)
        name, arguments = constant_arguments(value, coordinate, source, root)
        state.update({"status": "producer_running", "activeEpoch": target_epoch})
        atomic_json(state_root(coordinate) / "workflow.json", state)
        print(
            f"DENSE474M_ORIGINAL_PD_START mode={mode} coordinate={coordinate['id']} "
            f"epoch={target_epoch} source={source} output={root / 'constant_lr'}",
            flush=True,
        )
        run_torch(
            name,
            arguments,
            state_root(coordinate) / "producer.log",
            int(coordinate["nprocPerNode"]),
        )
    target = validate_predecay(target, coordinate)
    retained = [
        epoch
        for epoch in range(int(coordinate["sourceEpoch"]) + 1, target_epoch + 1)
        if checkpoint_complete(
            root / "constant_lr" / f"step{stable_step(epoch, batch)}",
            int(coordinate["nprocPerNode"]),
        )
    ]
    print(
        f"DENSE474M_ORIGINAL_PD_RETAINED mode={mode} coordinate={coordinate['id']} "
        f"epoch={target_epoch} checkpoint={target} retained={retained}",
        flush=True,
    )
    return target


def post_arguments(
    value: dict[str, Any], coordinate: dict[str, Any], source: Path, output: Path
) -> tuple[str, list[str]]:
    batch = int(coordinate["globalSequences"])
    epoch = int(coordinate["targetEpoch"])
    name = (
        f"dense_474m_step1_0802_repeated_dclm1b_bs{batch}_{coordinate['id']}_"
        f"post_e{epoch}_{value['runSuffix']}"
    )
    scheduler = (
        "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, "
        f"warmup: {int(coordinate['warmupSteps'])}, decay_fraction: {DECAY_FRACTION}}}"
    )
    arguments = common.upsert(
        base_arguments(coordinate, evaluation=True), "--train_module.scheduler=", scheduler
    )
    return name, [
        *arguments,
        f"--save-folder={output}",
        f"--trainer.max_duration={{value: {total_step(epoch, batch)}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dense-474m,dclm1b,"
            f"bs{batch},original-tuning,post-decay,heldout-eval,downstream-eval,e{epoch}]"
        ),
        f"--trainer.callbacks.checkpointer.fixed_steps=[{total_step(epoch, batch)}]",
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
    coordinate: dict[str, Any], checkpoint: Path, output: Path, name: str
) -> list[str]:
    heldout = small.HELDOUT_EVALUATOR.replace(
        "eval_on_finish: true", "eval_on_finish: false, eval_on_startup: true"
    )
    arguments = common.upsert(
        base_arguments(coordinate, evaluation=True),
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
        "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dense-474m,recovered-post-eval]",
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
    print(f"DENSE474M_ORIGINAL_POST_QUARANTINE source={path} destination={destination}", flush=True)


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
    if train_values:
        train = float(train_values[-1])
        result.update({"train": train, "gap": round(validation - train, 6)})
    wandb_values = common.WANDB_RUN.findall(clean)
    if wandb_values:
        result["wandb"] = wandb_values[-1]
    return result


def evaluate(
    value: dict[str, Any], mode: str, coordinate: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    epoch = int(coordinate["targetEpoch"])
    result_path = state_root(coordinate) / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        emit_result(mode, coordinate, result)
        return result
    root = claim_output(value, mode, coordinate)
    batch = int(coordinate["globalSequences"])
    source = validate_predecay(
        root / "constant_lr" / f"step{stable_step(epoch, batch)}", coordinate
    )
    output = root / "post_decay_runs" / f"e{epoch}"
    endpoint = output / f"step{total_step(epoch, batch)}"
    log_path = state_root(coordinate) / "post.log"
    recovery_log = state_root(coordinate) / "post_recovered_eval.log"
    state.update({"status": "post_running", "activeEpoch": epoch})
    atomic_json(state_root(coordinate) / "workflow.json", state)
    if not checkpoint_complete(endpoint, int(coordinate["nprocPerNode"])):
        if output.exists():
            quarantine_partial(output)
        name, arguments = post_arguments(value, coordinate, source, output)
        print(
            f"DENSE474M_ORIGINAL_POST_START mode={mode} coordinate={coordinate['id']} "
            f"epoch={epoch} source={source} output={output}",
            flush=True,
        )
        run_torch(name, arguments, log_path, int(coordinate["nprocPerNode"]))
    if not checkpoint_complete(endpoint, int(coordinate["nprocPerNode"])):
        raise RuntimeError(f"E{epoch} decay exited without complete endpoint {endpoint}")
    try:
        result = parse_result([log_path], epoch)
    except RuntimeError:
        recovered_name = (
            f"dense_474m_{coordinate['id']}_post_e{epoch}_recovered_eval_{value['runSuffix']}"
        )
        run_torch(
            recovered_name,
            recovery_eval_arguments(
                coordinate, endpoint, output / "recovered_eval", recovered_name
            ),
            recovery_log,
            int(coordinate["nprocPerNode"]),
        )
        result = parse_result([log_path, recovery_log], epoch)
    result.update(
        {
            "policy": POLICY,
            "mode": mode,
            "coordinate": coordinate["id"],
            "comparisonGroup": "post_decay",
            "lr": str(coordinate["learningRate"]),
            "wd": str(coordinate["weightDecay"]),
            "batchSequences": batch,
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
    emit_result(mode, coordinate, result)
    return result


def emit_result(mode: str, coordinate: dict[str, Any], result: dict[str, Any]) -> None:
    payload = json.dumps(result, separators=(",", ":"), sort_keys=True)
    print(
        f"DENSE474M_ORIGINAL_POST_RESULT mode={mode} coordinate={coordinate['id']} "
        f"epoch={coordinate['targetEpoch']} json={payload}",
        flush=True,
    )


def choose_bs32(results: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        {
            "learningRate": str(result["lr"]),
            "validationExact": float(result["validationExact"]),
        }
        for result in results
    ]
    candidates.append(
        {
            "learningRate": str(baseline["learningRate"]),
            "validationExact": float(baseline["validationExact"]),
        }
    )
    winner = min(
        candidates, key=lambda item: (item["validationExact"], item["learningRate"] != "1e-3")
    )
    return {
        "policy": POLICY,
        "epoch": 4,
        "weightDecay": "0.1",
        "selectedLearningRate": winner["learningRate"],
        "selectedValidationExact": winner["validationExact"],
        "baselineLearningRate": str(baseline["learningRate"]),
        "baselineValidationExact": float(baseline["validationExact"]),
        "strictlyImprovesBaseline": winner["validationExact"] < float(baseline["validationExact"]),
        "tiePolicy": "retain_1e-3",
        "candidates": candidates,
    }


def run(value: dict[str, Any], mode: str) -> None:
    coordinates = coordinate_values(value, mode)
    results: list[dict[str, Any]] = []
    for coordinate in coordinates:
        claim_output(value, mode, coordinate)
        state_path = state_root(coordinate) / "workflow.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.is_file()
            else {
                "policy": POLICY,
                "workflowId": value["id"],
                "mode": mode,
                "coordinate": coordinate["id"],
                "status": "starting",
                "minRuntimeOmitted": True,
            }
        )
        if state.get("policy") != POLICY or state.get("coordinate") != coordinate["id"]:
            raise RuntimeError("workflow state ownership mismatch")
        atomic_json(state_path, state)
        ensure_predecay(value, mode, coordinate, state)
        result = evaluate(value, mode, coordinate, state)
        results.append(result)
        state.update({"status": "complete", "activeEpoch": coordinate["targetEpoch"]})
        atomic_json(state_path, state)

    if mode == "bs32-probes":
        decision = choose_bs32(results, value["bs32Probes"]["baseline"])
        atomic_json(state_root(coordinates[0]) / "decision.json", decision)
        print(
            "DENSE474M_ORIGINAL_BS32_DECISION json="
            + json.dumps(decision, separators=(",", ":"), sort_keys=True),
            flush=True,
        )
    else:
        result = results[0]
        previous = float(value["bs128Extension"]["previousValidationExact"])
        decision = {
            "policy": POLICY,
            "epoch": 52,
            "validationExact": float(result["validationExact"]),
            "previousEpoch": 48,
            "previousValidationExact": previous,
            "strictlyImprovesE48": float(result["validationExact"]) < previous,
        }
        atomic_json(state_root(coordinates[0]) / "decision.json", decision)
        print(
            "DENSE474M_ORIGINAL_BS128_DECISION json="
            + json.dumps(decision, separators=(",", ":"), sort_keys=True),
            flush=True,
        )
    print(f"DENSE474M_ORIGINAL_WORKFLOW_COMPLETE mode={mode}", flush=True)


def main() -> None:
    args = parse_args()
    value = load_manifest(args.manifest)
    if args.validate_only:
        print(f"validated {args.manifest} for {args.mode}")
        return
    if args.print_plan:
        print(json.dumps(plan(value, args.mode), indent=2))
        return
    run(value, args.mode)


if __name__ == "__main__":
    main()
