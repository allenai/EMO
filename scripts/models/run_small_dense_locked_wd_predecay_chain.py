#!/usr/bin/env python3
"""Run one retry-safe locked-WD pre-decay saturation chain.

This is the successor to the adaptive WD controller.  It never changes WD and
never uses a post-decay score to decide whether constant-LR training should
continue.  Existing exact pre-decay checkpoints are evaluated only from the
configured historical start epoch and only at WSD frontier epochs that the
legacy chain decayed.  New checkpoints are then produced one configured
frontier at a time with a constant learning rate.  At the first pre-decay
non-improvement, the last three
pre-decay checkpoints are independently decayed and the best post-decay result
is selected only among those three post-decay results.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_small_dense_dr_wt_embedwd_chain as adaptive
import run_small_dense_saturation_chain as common

POLICY = "locked_wd_predecay_saturation_v1"
POST_DECAY_SOURCE_COUNT = 3


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_config(config: dict[str, Any]) -> None:
    adaptive.validate_config(config)
    if config.get("policy") != POLICY:
        raise ValueError(f"manifest policy must be {POLICY}")
    locked_wd = str(config.get("lockedWd"))
    if locked_wd not in [str(value) for value in config["wdLadder"]]:
        raise ValueError("lockedWd must be on the historical WD ladder")
    if Decimal(locked_wd) > adaptive.MAX_WEIGHT_DECAY:
        raise ValueError("lockedWd must not exceed 1.0")
    if int(config.get("postDecaySourceCount", 0)) != POST_DECAY_SOURCE_COUNT:
        raise ValueError("postDecaySourceCount must remain exactly three")
    if config.get("comparisonPolicy") != "within_phase_only":
        raise ValueError("pre-decay and post-decay results must remain separate")
    if int(config.get("historicalPreDecayThroughEpoch", 0)) < 3:
        raise ValueError("historical pre-decay boundary must contain at least three epochs")
    start = int(config.get("historicalPreDecayStartEpoch", 0))
    boundary = int(config["historicalPreDecayThroughEpoch"])
    if start > boundary:
        raise ValueError("historical pre-decay start must not exceed its boundary")
    adaptive.targets_through(config, start)
    adaptive.targets_through(config, boundary)


def locked_output(config: dict[str, Any]) -> Path:
    return adaptive.output_for(config, str(config["lockedWd"]))


def policy_state_dir(config: dict[str, Any]) -> Path:
    return Path(str(config["outputRoot"])) / (
        f".bs{config['globalSequences']}_dr_wt_embwd_lr{config['learningRate']}_"
        f"wd{config['lockedWd']}_predecay_policy"
    )


def constant_output(config: dict[str, Any]) -> Path:
    return policy_state_dir(config) / "constant_lr"


def predecay_result_path(config: dict[str, Any], epoch: int) -> Path:
    return policy_state_dir(config) / "pre_decay" / f"e{epoch}.result.json"


def postdecay_result_path(config: dict[str, Any], epoch: int) -> Path:
    return policy_state_dir(config) / "post_decay" / f"e{epoch}.result.json"


def selection_path(config: dict[str, Any]) -> Path:
    return policy_state_dir(config) / "post_decay_selection.json"


def run_name(config: dict[str, Any], phase: str, epoch: int) -> str:
    return (
        f"dense_{config['model']}_step1_0802_repeated_dclm1b_"
        f"bs{config['globalSequences']}_dr_wt_embwd_{phase}_e{epoch}_"
        f"lr{config['learningRate']}_wd{config['lockedWd']}_"
        f"{config['runSuffix']}"
    )


def checkpoint_for_epoch(config: dict[str, Any], epoch: int) -> Path:
    output = (
        locked_output(config)
        if epoch <= int(config["historicalPreDecayThroughEpoch"])
        else constant_output(config)
    )
    return output / (f"step{adaptive.stable_step(epoch, int(config['globalSequences']))}")


def discover_predecay_epochs(config: dict[str, Any]) -> list[int]:
    output = locked_output(config)
    if not output.is_dir():
        raise FileNotFoundError(f"locked WD output does not exist: {output}")
    steps = {
        int(path.name[4:])
        for path in output.glob("step*")
        if path.is_dir() and path.name[4:].isdigit()
    }
    if not steps:
        raise FileNotFoundError(f"locked WD output contains no checkpoints: {output}")
    start = int(config["historicalPreDecayStartEpoch"])
    boundary = int(config["historicalPreDecayThroughEpoch"])
    expected = [epoch for epoch in adaptive.targets_through(config, boundary) if epoch >= start]
    epochs = [
        epoch
        for epoch in expected
        if adaptive.stable_step(epoch, int(config["globalSequences"])) in steps
    ]
    if epochs != expected:
        missing = sorted(set(expected) - set(epochs))
        raise RuntimeError(
            f"locked WD pre-decay lineage has missing scheduled frontiers: {missing}"
        )
    return epochs


def next_predecay_frontier(config: dict[str, Any], previous_epoch: int) -> int:
    index = 0
    while adaptive.target_at(config, index) <= previous_epoch:
        index += 1
    return adaptive.target_at(config, index)


def replace_argument(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    return common.upsert(arguments, prefix, replacement)


def base_arguments(config: dict[str, Any]) -> list[str]:
    batch = int(config["globalSequences"])
    return [
        *adaptive.COMMON_ARGUMENTS,
        *adaptive.MODEL_ARGUMENTS[str(config["model"])],
        f"--data_loader.global_batch_size={batch * adaptive.SEQUENCE_LENGTH}",
        f"--train_module.rank_microbatch_size={int(config['rankMicrobatchSequences']) * adaptive.SEQUENCE_LENGTH}",
        f"--train_module.optim.weight_decay={config['lockedWd']}",
        f"--lr={config['learningRate']}",
        "--model.tie_embeddings=true",
        "--decay-embeddings",
    ]


def evaluation_arguments(
    config: dict[str, Any], checkpoint: Path, output: Path, name: str
) -> list[str]:
    arguments = base_arguments(config)
    arguments = replace_argument(
        arguments,
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=",
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
    )
    arguments.extend(
        (
            "--trainer.callbacks.downstream_evaluator.eval_on_startup=true",
            "--trainer.callbacks.downstream_evaluator.cancel_after_first_eval=true",
        )
    )
    heldout = adaptive.HELDOUT_EVALUATOR.replace(
        "eval_on_finish: true", "eval_on_finish: false, eval_on_startup: true"
    )
    arguments = replace_argument(
        arguments,
        "--trainer.callbacks.heldout_evaluator=",
        f"--trainer.callbacks.heldout_evaluator={heldout}",
    )
    arguments.extend(
        (
            f"--save-folder={output}",
            f"--trainer.callbacks.wandb.name={name}",
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,checkpoint-eval,pre-decay,locked-wd]",
            "--trainer.max_duration={value: 1000000000000, unit: steps}",
            "--trainer.callbacks.checkpointer.enabled=false",
            f"--load_path={checkpoint}",
            "--load_trainer_state=false",
        )
    )
    return arguments


def parse_evaluation(log_path: Path, epoch: int, *, phase: str, checkpoint: Path) -> dict[str, Any]:
    clean = common.ANSI.sub("", log_path.read_text())
    validation_values = common.WANDB_VALIDATION_LOSS.findall(
        clean
    ) or common.VALIDATION_LOSS.findall(clean)
    if not validation_values:
        raise RuntimeError(f"{phase} E{epoch} completed without held-out validation CE")
    accuracy: dict[str, float] = {}
    bpb: dict[str, float] = {}
    for task, value in common.ACCURACY.findall(clean):
        accuracy[common.normalize_task(task)] = 100 * float(value)
    for task, value in common.BPB.findall(clean):
        bpb[common.normalize_task(task)] = float(value)
    missing = [task for task in common.REPORT_TASKS if task not in accuracy or task not in bpb]
    if missing:
        raise RuntimeError(f"{phase} E{epoch} completed without downstream metrics for {missing}")
    validation = float(validation_values[-1])
    result: dict[str, Any] = {
        "epoch": epoch,
        "status": "complete",
        "phase": phase,
        "comparisonGroup": "pre_decay" if phase == "pre_decay" else "post_decay",
        "validation": round(validation, 3),
        "validationExact": validation,
        "downstream": accuracy,
        "downstreamBpb": bpb,
        "acc": accuracy["hellaswag"],
        "bpb": bpb["hellaswag"],
        "avg8Accuracy": sum(accuracy[task] for task in common.AVERAGE_TASKS)
        / len(common.AVERAGE_TASKS),
        "avg8Bpb": sum(bpb[task] for task in common.AVERAGE_TASKS) / len(common.AVERAGE_TASKS),
        "checkpoint": str(checkpoint),
    }
    wandb_values = common.WANDB_RUN.findall(clean)
    if wandb_values:
        result["wandb"] = wandb_values[-1]
    return result


def emit_result(config: dict[str, Any], result: dict[str, Any]) -> None:
    payload = json.dumps(result, separators=(",", ":"), sort_keys=True)
    print(
        f"SMALL_PREDECAY_POLICY_RESULT model={config['model']} "
        f"bs={config['globalSequences']} phase={result['phase']} "
        f"epoch={result['epoch']} wd={config['lockedWd']} json={payload}",
        flush=True,
    )


def evaluate_checkpoint(
    config: dict[str, Any], epoch: int, checkpoint: Path, *, phase: str
) -> dict[str, Any]:
    result_path = (
        predecay_result_path(config, epoch)
        if phase == "pre_decay"
        else postdecay_result_path(config, epoch)
    )
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        emit_result(config, result)
        return result
    eval_output = policy_state_dir(config) / "eval_runs" / phase / f"e{epoch}"
    name = run_name(config, f"{phase}_eval", epoch)
    arguments = evaluation_arguments(config, checkpoint, eval_output, name)
    print(
        f"SMALL_PREDECAY_POLICY_STAGE_START model={config['model']} "
        f"bs={config['globalSequences']} phase={phase}_eval epoch={epoch} "
        f"wd={config['lockedWd']} source={checkpoint} output={eval_output}",
        flush=True,
    )
    subprocess.run(["python", adaptive.TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path = policy_state_dir(config) / "logs" / f"{phase}_eval_e{epoch}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        common.stream_command(
            [
                "torchrun",
                f"--nproc-per-node={config['nprocPerNode']}",
                adaptive.TRAINING_SCRIPT,
                name,
                *arguments,
            ],
            log_file,
        )
    result = parse_evaluation(log_path, epoch, phase=phase, checkpoint=checkpoint)
    result.update(
        {
            "wd": str(config["lockedWd"]),
            "variant": "DR+WT+EmbedWD",
            "policy": POLICY,
            "source": "checkpoint_evaluation",
        }
    )
    atomic_json(result_path, result)
    emit_result(config, result)
    return result


def constant_training_arguments(
    config: dict[str, Any], epoch: int, source: Path, name: str
) -> list[str]:
    batch = int(config["globalSequences"])
    target_step = adaptive.stable_step(epoch, batch)
    output = constant_output(config)
    arguments = base_arguments(config)
    arguments.extend(
        (
            f"--save-folder={output}",
            f"--trainer.max_duration={{value: {target_step}, unit: steps}}",
            f"--trainer.callbacks.wandb.name={name}",
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,constant-lr,pre-decay,locked-wd]",
            f"--trainer.callbacks.checkpointer.fixed_steps=[{target_step}]",
            "--data_loader.restore_data_order_from_state=false",
            "--data_loader.ignore_fingerprint_mismatch=true",
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.ConstantScheduler}",
            "--dynamic-repacking",
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
            "--force_exact_trainer_load_path=true",
            f"--trainer.load_path={source}",
            "--trainer.load_trainer_state=true",
            "--trainer.load_optim_state=true",
            "--trainer.reset_data_loader_state_on_load_path=false",
            "--train_module.validate_optimizer_hyperparameters_on_load=true",
        )
    )
    return arguments


def train_next_predecay(config: dict[str, Any], previous_epoch: int, epoch: int) -> dict[str, Any]:
    target = checkpoint_for_epoch(config, epoch)
    if target.is_dir():
        return evaluate_checkpoint(config, epoch, target, phase="pre_decay")
    source = checkpoint_for_epoch(config, previous_epoch)
    if not source.is_dir():
        raise FileNotFoundError(f"missing exact constant-LR source checkpoint {source}")
    name = run_name(config, "constant_predecay", epoch)
    arguments = constant_training_arguments(config, epoch, source, name)
    print(
        f"SMALL_PREDECAY_POLICY_STAGE_START model={config['model']} "
        f"bs={config['globalSequences']} phase=constant_train epoch={epoch} "
        f"wd={config['lockedWd']} source={source} output={constant_output(config)}",
        flush=True,
    )
    subprocess.run(["python", adaptive.TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path = policy_state_dir(config) / "logs" / f"constant_train_e{epoch}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        common.stream_command(
            [
                "torchrun",
                f"--nproc-per-node={config['nprocPerNode']}",
                adaptive.TRAINING_SCRIPT,
                name,
                *arguments,
            ],
            log_file,
        )
    result = common.parse_stage_result(log_path, epoch)
    result.update(
        {
            "phase": "pre_decay",
            "comparisonGroup": "pre_decay",
            "wd": str(config["lockedWd"]),
            "variant": "DR+WT+EmbedWD",
            "policy": POLICY,
            "scheduler": "constant",
            "sourceCheckpoint": str(source),
            "checkpoint": str(target),
            "source": "constant_lr_training",
        }
    )
    atomic_json(predecay_result_path(config, epoch), result)
    emit_result(config, result)
    return result


def postdecay_training_arguments(
    config: dict[str, Any], epoch: int, source: Path, output: Path, name: str
) -> list[str]:
    batch = int(config["globalSequences"])
    endpoint = adaptive.total_step(epoch, batch)
    arguments = base_arguments(config)
    arguments.extend(
        (
            f"--save-folder={output}",
            f"--trainer.max_duration={{value: {endpoint}, unit: steps}}",
            f"--trainer.callbacks.wandb.name={name}",
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,post-decay,locked-wd]",
            f"--trainer.callbacks.checkpointer.fixed_steps=[{endpoint}]",
            "--data_loader.restore_data_order_from_state=false",
            "--data_loader.ignore_fingerprint_mismatch=true",
            (
                "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, "
                f"units: steps, warmup: {config['warmupSteps']}, "
                f"decay_fraction: {adaptive.DECAY_FRACTION}}}"
            ),
            "--dynamic-repacking",
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
            "--force_exact_trainer_load_path=true",
            f"--trainer.load_path={source}",
            "--trainer.load_trainer_state=true",
            "--trainer.load_optim_state=true",
            "--trainer.reset_data_loader_state_on_load_path=false",
            "--train_module.validate_optimizer_hyperparameters_on_load=true",
        )
    )
    return arguments


def run_postdecay(config: dict[str, Any], epoch: int) -> dict[str, Any]:
    path = postdecay_result_path(config, epoch)
    if path.is_file():
        result = json.loads(path.read_text())
        emit_result(config, result)
        return result
    batch = int(config["globalSequences"])
    source = checkpoint_for_epoch(config, epoch)
    if not source.is_dir():
        raise FileNotFoundError(f"missing post-decay source checkpoint {source}")
    canonical_endpoint = locked_output(config) / f"step{adaptive.total_step(epoch, batch)}"
    historical_result_path = adaptive.result_path(config, str(config["lockedWd"]), epoch)
    historical_result = (
        json.loads(historical_result_path.read_text()) if historical_result_path.is_file() else None
    )
    verified_historical_endpoint = bool(
        epoch <= int(config["historicalPreDecayThroughEpoch"])
        and historical_result
        and historical_result.get("wd") == str(config["lockedWd"])
        and historical_result.get("preDecayCheckpoint") == str(source)
        and historical_result.get("endpointCheckpoint") == str(canonical_endpoint)
        and canonical_endpoint.is_dir()
    )
    if verified_historical_endpoint:
        result = evaluate_checkpoint(config, epoch, canonical_endpoint, phase="post_decay")
        result["sourcePreDecayCheckpoint"] = str(source)
        result["source"] = "existing_canonical_wsd_endpoint"
        atomic_json(path, result)
        return result
    output = policy_state_dir(config) / "post_decay_runs" / f"e{epoch}"
    endpoint = output / f"step{adaptive.total_step(epoch, batch)}"
    if endpoint.is_dir():
        result = evaluate_checkpoint(config, epoch, endpoint, phase="post_decay")
        result["sourcePreDecayCheckpoint"] = str(source)
        result["source"] = "recovered_post_decay_endpoint"
        atomic_json(path, result)
        return result
    name = run_name(config, "postdecay", epoch)
    arguments = postdecay_training_arguments(config, epoch, source, output, name)
    print(
        f"SMALL_PREDECAY_POLICY_STAGE_START model={config['model']} "
        f"bs={config['globalSequences']} phase=post_decay_train epoch={epoch} "
        f"wd={config['lockedWd']} source={source} output={output}",
        flush=True,
    )
    subprocess.run(["python", adaptive.TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path = policy_state_dir(config) / "logs" / f"post_decay_train_e{epoch}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        common.stream_command(
            [
                "torchrun",
                f"--nproc-per-node={config['nprocPerNode']}",
                adaptive.TRAINING_SCRIPT,
                name,
                *arguments,
            ],
            log_file,
        )
    result = common.parse_stage_result(log_path, epoch)
    result.update(
        {
            "phase": "post_decay",
            "comparisonGroup": "post_decay",
            "wd": str(config["lockedWd"]),
            "variant": "DR+WT+EmbedWD",
            "policy": POLICY,
            "sourcePreDecayCheckpoint": str(source),
            "checkpoint": str(endpoint),
            "source": "post_decay_training",
        }
    )
    atomic_json(path, result)
    emit_result(config, result)
    return result


def first_saturation(results: dict[int, dict[str, Any]]) -> int | None:
    epochs = sorted(results)
    for index in range(2, len(epochs)):
        previous = results[epochs[index - 1]]
        current = results[epochs[index]]
        if Decimal(str(current["validationExact"])) >= Decimal(str(previous["validationExact"])):
            return epochs[index]
    return None


def finish_postdecay(
    config: dict[str, Any], saturation_epoch: int, available_epochs: list[int]
) -> None:
    path = selection_path(config)
    if path.is_file():
        selection = json.loads(path.read_text())
    else:
        eligible = [epoch for epoch in available_epochs if epoch <= saturation_epoch]
        sources = eligible[-POST_DECAY_SOURCE_COUNT:]
        if len(sources) != POST_DECAY_SOURCE_COUNT:
            raise RuntimeError("saturation requires exactly three post-decay sources")
        results = {epoch: run_postdecay(config, epoch) for epoch in sources}
        selected_epoch, selected = min(
            results.items(),
            key=lambda item: (Decimal(str(item[1]["validationExact"])), item[0]),
        )
        selection = {
            "status": "complete",
            "policy": POLICY,
            "lockedWd": str(config["lockedWd"]),
            "saturationEpoch": saturation_epoch,
            "preDecayDecisionGroup": "pre_decay",
            "postDecaySelectionGroup": "post_decay",
            "postDecaySourceEpochs": sources,
            "postDecayValidationExact": {
                str(epoch): float(result["validationExact"]) for epoch, result in results.items()
            },
            "selectedPostDecayEpoch": selected_epoch,
            "selectedPostDecayValidationExact": float(selected["validationExact"]),
            "selectedCheckpoint": selected["checkpoint"],
        }
        atomic_json(path, selection)
    print(
        f"SMALL_PREDECAY_POLICY_SELECTED model={config['model']} "
        f"bs={config['globalSequences']} saturation_epoch={selection['saturationEpoch']} "
        f"wd={config['lockedWd']} selected_epoch={selection['selectedPostDecayEpoch']} "
        f"validation={selection['selectedPostDecayValidationExact']} "
        f"json={json.dumps(selection, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )


def run(config: dict[str, Any]) -> None:
    validate_config(config)
    policy_state_dir(config).mkdir(parents=True, exist_ok=True)
    existing_epochs = discover_predecay_epochs(config)
    results: dict[int, dict[str, Any]] = {}
    for epoch in existing_epochs:
        results[epoch] = evaluate_checkpoint(
            config, epoch, checkpoint_for_epoch(config, epoch), phase="pre_decay"
        )
    saturation_epoch = first_saturation(results)
    while saturation_epoch is None:
        previous_epoch = max(results)
        epoch = next_predecay_frontier(config, previous_epoch)
        results[epoch] = train_next_predecay(config, previous_epoch, epoch)
        existing_epochs.append(epoch)
        saturation_epoch = first_saturation(results)
    print(
        f"SMALL_PREDECAY_POLICY_SATURATED model={config['model']} "
        f"bs={config['globalSequences']} epoch={saturation_epoch} "
        f"wd={config['lockedWd']} comparison_group=pre_decay",
        flush=True,
    )
    finish_postdecay(config, saturation_epoch, existing_epochs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    validate_config(config)
    if args.validate_only:
        print(f"validated {args.manifest}")
        return
    run(config)


if __name__ == "__main__":
    main()
