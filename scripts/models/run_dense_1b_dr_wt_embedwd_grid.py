#!/usr/bin/env python3
"""Run one retry-safe Dense-1B PD/POST saturation coordinate.

Each Beaker experiment owns exactly one LR/WD coordinate.  Scheduled
pre-decay (PD) frontiers are trained with constant LR and compared only with
other PD results.  A PD non-improvement provisionally triggers independent WSD
decays from the latest three PD checkpoints.  The coordinate is terminal only
when the newest post-decay (POST) result also fails to improve; otherwise it
continues constant-LR training to the next frontier.

The ``--finalize-only`` mode is used after cross-WD pruning.  It never trains a
new PD frontier, but still produces the requested POST evidence from the latest
three completed PD checkpoints before marking the coordinate pruned.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_small_dense_saturation_chain as common

POLICY = "dense_1b_predecay_postdecay_saturation_v1"
POST_DECAY_SOURCE_COUNT = 3
POST_DECAY_SATURATION_CRITERION = "strict_non_improvement"
TOKENS_PER_EPOCH = 1_000_000_000
SEQUENCE_LENGTH = 4096
DECAY_FRACTION = 0.1
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"
OUTPUT_ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"
ALLOWED_COORDINATES = {
    128: {("1e-3", "0.3"), ("1e-3", "1.0")},
    256: {
        ("1e-3", "0.3"),
        ("1e-3", "1.0"),
        ("2e-3", "0.3"),
        ("2e-3", "1.0"),
    },
    512: {("2e-3", "0.333"), ("2e-3", "1.0")},
}
HELDOUT_EVALUATOR = (
    "{_CLASS_: olmo_core.train.callbacks.evaluator_callback.LMEvaluatorCallbackConfig, "
    "eval_dataset: {_CLASS_: olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig, "
    "tokenizer: {_CLASS_: olmo_core.data.tokenizer.TokenizerConfig, vocab_size: 100278, "
    "eos_token_id: 100257, pad_token_id: 100277, identifier: allenai/dolma2-tokenizer}, "
    "mix: null, mix_base_dir: /weka/oe-training-default/ai2-llm, "
    "subset_manifest: src/olmo_core/data/subsets/0802/dclm_0802_validation.json, "
    "metadata: [{label: dclm-validation-0802}], sequence_length: 4096, "
    "work_dir: /weka/oe-training-default/sewonm/dataset-cache}, eval_interval: null, "
    "eval_on_finish: true, name: heldout}"
)
COMMON_ARGUMENTS = (
    "--data-root=/weka/oe-training-default/ai2-llm",
    "--dataset.mix=null",
    "--dataset.subset_manifest=src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json",
    "--dataset.mix_base_dir=/weka/oe-training-default/ai2-llm",
    "--work-dir=/weka/oe-training-default/sewonm/dataset-cache",
    "--trainer.callbacks.wandb.enabled=true",
    "--trainer.callbacks.wandb.entity=ai2-llm",
    "--trainer.callbacks.wandb.project=sewonm-icsl",
    "--trainer.callbacks.downstream_evaluator.tasks=[]",
    "--trainer.callbacks.downstream_evaluator.eval_interval=null",
    "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
    f"--trainer.callbacks.heldout_evaluator={HELDOUT_EVALUATOR}",
    "--dataset.instance_filter_config={repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}",
    "--model.block.name=default",
    "--model.block.sequence_mixer.qk_norm=null",
    "--init_seed=12536",
    "--data_loader.seed=0",
)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def coordinate_key(lr: str, wd: str) -> str:
    return f"lr{lr}_wd{wd}"


def coordinates(config: dict[str, Any]) -> list[tuple[str, str]]:
    return [(str(item["lr"]), str(item["wd"])) for item in config["coordinates"]]


def coordinate_config(config: dict[str, Any], lr: str, wd: str) -> dict[str, Any]:
    matches = [
        item
        for item in config["coordinates"]
        if str(item["lr"]) == lr and str(item["wd"]) == wd
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one coordinate LR{lr}/WD{wd}")
    return matches[0]


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "globalSequences",
        "nprocPerNode",
        "rankMicrobatchSequences",
        "gradientAccumulation",
        "warmupSteps",
        "coordinates",
        "initialTargets",
        "epochIncrement",
        "maxEpoch",
        "outputRoot",
        "runSuffix",
        "variant",
        "policy",
        "comparisonPolicy",
        "postDecaySourceCount",
        "postDecaySaturationCriterion",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"manifest is missing required keys: {missing}")
    batch = int(config["globalSequences"])
    if batch not in ALLOWED_COORDINATES:
        raise ValueError("only BS128, BS256, and conditional BS512 are authorized")
    if config["policy"] != POLICY:
        raise ValueError(f"manifest policy must be {POLICY}")
    if config["comparisonPolicy"] != "within_phase_only":
        raise ValueError("PD and POST comparisons must stay separate")
    if int(config["postDecaySourceCount"]) != POST_DECAY_SOURCE_COUNT:
        raise ValueError("postDecaySourceCount must remain exactly three")
    if config["postDecaySaturationCriterion"] != POST_DECAY_SATURATION_CRITERION:
        raise ValueError("POST saturation criterion must remain strict_non_improvement")
    if int(config["nprocPerNode"]) != 8:
        raise ValueError("each coordinate must use exactly eight GPUs")
    if int(config["rankMicrobatchSequences"]) != 8:
        raise ValueError("rank microbatch must remain eight sequences")
    if int(config["gradientAccumulation"]) != batch // 64:
        raise ValueError("gradient accumulation does not produce the requested batch")
    if int(config["warmupSteps"]) != 24_576 // batch:
        raise ValueError("warmup must preserve the established token budget")
    requested = coordinates(config)
    if len(requested) != len(set(requested)):
        raise ValueError("manifest coordinates must be unique")
    if not set(requested).issubset(ALLOWED_COORDINATES[batch]):
        raise ValueError(f"unsupported BS{batch} coordinates: {requested}")
    variant = str(config["variant"])
    if batch in {128, 256} and set(requested) != ALLOWED_COORDINATES[batch]:
        raise ValueError(f"BS{batch} must contain the full requested coordinate set")
    if batch == 512 and variant == "Original" and requested != [("2e-3", "0.333")]:
        raise ValueError("conditional original BS512 must be LR2e-3/WD0.333 only")
    if batch == 512 and variant == "DR+WT+EmbedWD" and set(requested) != ALLOWED_COORDINATES[512]:
        raise ValueError("conditional DR+WT+EmbedWD BS512 must contain both requested WDs")
    for lr, wd in requested:
        if Decimal(lr) not in {Decimal("1e-3"), Decimal("2e-3")}:
            raise ValueError(f"unsupported LR {lr}")
        if Decimal(wd) > Decimal("1.0"):
            raise ValueError("WD must not exceed 1.0")
        item = coordinate_config(config, lr, wd)
        if not item.get("output"):
            raise ValueError(f"LR{lr}/WD{wd} is missing an explicit canonical output")
        if variant == "Original" and not item.get("historicalPreDecay"):
            raise ValueError("original BS512 continuation must declare exact historical PD provenance")
    if [int(value) for value in config["initialTargets"]] != [1, 2, 4]:
        raise ValueError("Dense-1B frontier ladder must begin E1 -> E2 -> E4")
    if int(config["epochIncrement"]) != 4 or int(config["maxEpoch"]) < 16:
        raise ValueError("Dense-1B must advance by four epochs with room for three POST sources")
    if str(config["outputRoot"]) != OUTPUT_ROOT:
        raise ValueError(f"outputRoot must remain {OUTPUT_ROOT}")


def total_step(epoch: int, batch: int) -> int:
    return common.total_step(epoch, batch)


def stable_step(epoch: int, batch: int) -> int:
    return common.stable_step(epoch, batch)


def target_at(config: dict[str, Any], index: int) -> int:
    initial = [int(value) for value in config["initialTargets"]]
    if index < len(initial):
        return initial[index]
    return initial[-1] + (index - len(initial) + 1) * int(config["epochIncrement"])


def targets_through(config: dict[str, Any], target: int) -> list[int]:
    values: list[int] = []
    index = 0
    while not values or values[-1] < target:
        values.append(target_at(config, index))
        index += 1
    if values[-1] != target:
        raise ValueError(f"E{target} is not a configured Dense-1B frontier")
    return values


def next_frontier(config: dict[str, Any], previous: int) -> int:
    index = 0
    while target_at(config, index) <= previous:
        index += 1
    return target_at(config, index)


def output_for(config: dict[str, Any], lr: str, wd: str) -> Path:
    return Path(str(coordinate_config(config, lr, wd)["output"]))


def state_dir(config: dict[str, Any], lr: str, wd: str) -> Path:
    return output_for(config, lr, wd) / ".pdpost_policy"


def predecay_result_path(config: dict[str, Any], lr: str, wd: str, epoch: int) -> Path:
    return state_dir(config, lr, wd) / "pre_decay" / f"e{epoch}.result.json"


def postdecay_result_path(config: dict[str, Any], lr: str, wd: str, epoch: int) -> Path:
    return state_dir(config, lr, wd) / "post_decay" / f"e{epoch}.result.json"


def selection_path(config: dict[str, Any], lr: str, wd: str) -> Path:
    return state_dir(config, lr, wd) / "selection.json"


def decision_path(config: dict[str, Any], lr: str, wd: str, epoch: int) -> Path:
    return state_dir(config, lr, wd) / "post_decay_decisions" / f"e{epoch}.json"


def checkpoint_for_epoch(config: dict[str, Any], lr: str, wd: str, epoch: int) -> Path:
    item = coordinate_config(config, lr, wd)
    for historical in item.get("historicalPreDecay", []):
        if int(historical["epoch"]) == epoch:
            return Path(str(historical["checkpoint"]))
    return output_for(config, lr, wd) / f"step{stable_step(epoch, int(config['globalSequences']))}"


def run_name(config: dict[str, Any], lr: str, wd: str, phase: str, epoch: int) -> str:
    variant = "original" if config["variant"] == "Original" else "dr_wt_embwd"
    return (
        f"dense_1b_step1_0802_repeated_dclm1b_bs{config['globalSequences']}_"
        f"{variant}_{phase}_e{epoch}_lr{lr}_wd{wd}_{config['runSuffix']}"
    )


def base_arguments(config: dict[str, Any], lr: str, wd: str) -> list[str]:
    batch = int(config["globalSequences"])
    arguments = [
        *COMMON_ARGUMENTS,
        f"--data_loader.global_batch_size={batch * SEQUENCE_LENGTH}",
        f"--train_module.rank_microbatch_size={int(config['rankMicrobatchSequences']) * SEQUENCE_LENGTH}",
        f"--train_module.optim.weight_decay={wd}",
        f"--lr={lr}",
    ]
    if config["variant"] == "DR+WT+EmbedWD":
        arguments.extend(("--model.tie_embeddings=true", "--decay-embeddings"))
    else:
        arguments.append("--model.tie_embeddings=false")
    return arguments


def phase_metadata(config: dict[str, Any], epoch: int) -> dict[str, Any]:
    adaptive = config["variant"] == "DR+WT+EmbedWD"
    return {
        "variant": str(config["variant"]),
        "dynamicRepacking": bool(adaptive and epoch > 1),
        "weightTying": adaptive,
        "decayEmbeddings": adaptive,
        "dataOrder": "dynamic_repacking" if adaptive and epoch > 1 else "ordinary_shuffled",
    }


def parse_validation(log_path: Path, epoch: int, phase: str, checkpoint: Path) -> dict[str, Any]:
    clean = common.ANSI.sub("", log_path.read_text())
    validation_values = common.WANDB_VALIDATION_LOSS.findall(clean) or common.VALIDATION_LOSS.findall(clean)
    if not validation_values:
        raise RuntimeError(f"{phase} E{epoch} completed without held-out validation CE")
    train_values = common.TRAIN_LOSS.findall(clean)
    validation = float(validation_values[-1])
    result: dict[str, Any] = {
        "epoch": epoch,
        "status": "complete",
        "phase": phase,
        "comparisonGroup": "pre_decay" if phase == "pre_decay" else "post_decay",
        "validation": round(validation, 3),
        "validationExact": validation,
        "checkpoint": str(checkpoint),
    }
    if train_values:
        train = float(train_values[-1])
        result.update({"train": train, "gap": round(validation - train, 6)})
    wandb_values = common.WANDB_RUN.findall(clean)
    if wandb_values:
        result["wandb"] = wandb_values[-1]
    return result


def emit_result(config: dict[str, Any], lr: str, wd: str, result: dict[str, Any]) -> None:
    payload = json.dumps(result, separators=(",", ":"), sort_keys=True)
    print(
        f"DENSE1B_PDPOST_RESULT bs={config['globalSequences']} lr={lr} wd={wd} "
        f"phase={result['phase']} epoch={result['epoch']} json={payload}",
        flush=True,
    )


def run_torch(config: dict[str, Any], name: str, arguments: list[str], log_path: Path) -> None:
    subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        common.stream_command(
            [
                "torchrun",
                f"--nproc-per-node={config['nprocPerNode']}",
                TRAINING_SCRIPT,
                name,
                *arguments,
            ],
            log_file,
        )


def evaluation_arguments(
    config: dict[str, Any], lr: str, wd: str, checkpoint: Path, output: Path, name: str
) -> list[str]:
    heldout = HELDOUT_EVALUATOR.replace(
        "eval_on_finish: true", "eval_on_finish: false, eval_on_startup: true"
    )
    arguments = base_arguments(config, lr, wd)
    arguments = common.upsert(
        arguments,
        "--trainer.callbacks.heldout_evaluator=",
        f"--trainer.callbacks.heldout_evaluator={heldout}",
    )
    arguments.extend(
        (
            f"--save-folder={output}",
            f"--trainer.callbacks.wandb.name={name}",
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dense-1b,pdpost,checkpoint-eval]",
            "--trainer.max_duration={value: 1000000000000, unit: steps}",
            "--trainer.callbacks.checkpointer.enabled=false",
            f"--load_path={checkpoint}",
            "--load_trainer_state=false",
        )
    )
    return arguments


def evaluate_checkpoint(
    config: dict[str, Any], lr: str, wd: str, epoch: int, checkpoint: Path, *, phase: str
) -> dict[str, Any]:
    path = (
        predecay_result_path(config, lr, wd, epoch)
        if phase == "pre_decay"
        else postdecay_result_path(config, lr, wd, epoch)
    )
    if path.is_file():
        result = json.loads(path.read_text())
        emit_result(config, lr, wd, result)
        return result
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"missing exact {phase} checkpoint {checkpoint}")
    output = state_dir(config, lr, wd) / "eval_runs" / phase / f"e{epoch}"
    name = run_name(config, lr, wd, f"{phase}_eval", epoch)
    arguments = evaluation_arguments(config, lr, wd, checkpoint, output, name)
    print(
        f"DENSE1B_PDPOST_STAGE_START bs={config['globalSequences']} lr={lr} wd={wd} "
        f"phase={phase}_eval epoch={epoch} source={checkpoint} output={output}",
        flush=True,
    )
    log_path = state_dir(config, lr, wd) / "logs" / f"{phase}_eval_e{epoch}.log"
    run_torch(config, name, arguments, log_path)
    result = parse_validation(log_path, epoch, phase, checkpoint)
    result.update(
        {
            "policy": POLICY,
            "lr": lr,
            "wd": wd,
            "source": "checkpoint_evaluation",
            **phase_metadata(config, epoch),
        }
    )
    atomic_json(path, result)
    emit_result(config, lr, wd, result)
    return result


def predecay_training_arguments(
    config: dict[str, Any], lr: str, wd: str, epoch: int, source: Path | None, name: str
) -> list[str]:
    batch = int(config["globalSequences"])
    target = stable_step(epoch, batch)
    output = output_for(config, lr, wd)
    arguments = base_arguments(config, lr, wd)
    arguments.extend(
        (
            f"--save-folder={output}",
            f"--trainer.max_duration={{value: {target}, unit: steps}}",
            f"--trainer.callbacks.wandb.name={name}",
            (
                "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dense-1b,"
                f"pdpost,pre-decay,constant-lr,bs{batch},lr{lr},wd{wd}]"
            ),
            f"--trainer.callbacks.checkpointer.fixed_steps=[{target}]",
            f"--data_loader.ignore_fingerprint_mismatch={'true' if source else 'false'}",
            (
                "--train_module.scheduler={_CLASS_: "
                "olmo_core.optim.scheduler.ConstantWithWarmup, "
                f"warmup: {config['warmupSteps']}}}"
            ),
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        )
    )
    adaptive = config["variant"] == "DR+WT+EmbedWD"
    if adaptive and epoch > 1:
        arguments.append("--dynamic-repacking")
    if source is not None:
        arguments.extend(
            (
                f"--data_loader.restore_data_order_from_state={'false' if adaptive else 'true'}",
                "--force_exact_trainer_load_path=true",
                f"--trainer.load_path={source}",
                "--trainer.load_trainer_state=true",
                "--trainer.load_optim_state=true",
                "--trainer.reset_data_loader_state_on_load_path=false",
                "--train_module.validate_optimizer_hyperparameters_on_load=true",
            )
        )
    return arguments


def train_predecay(
    config: dict[str, Any], lr: str, wd: str, previous_epoch: int | None, epoch: int
) -> dict[str, Any]:
    target = checkpoint_for_epoch(config, lr, wd, epoch)
    if target.is_dir():
        return evaluate_checkpoint(config, lr, wd, epoch, target, phase="pre_decay")
    source = (
        checkpoint_for_epoch(config, lr, wd, previous_epoch)
        if previous_epoch is not None
        else None
    )
    if source is not None and not source.is_dir():
        raise FileNotFoundError(f"missing exact PD source checkpoint {source}")
    name = run_name(config, lr, wd, "constant_predecay", epoch)
    arguments = predecay_training_arguments(config, lr, wd, epoch, source, name)
    print(
        f"DENSE1B_PDPOST_STAGE_START bs={config['globalSequences']} lr={lr} wd={wd} "
        f"phase=pre_decay_train epoch={epoch} previous_epoch={previous_epoch or 0} "
        f"source={source or 'fresh'} output={output_for(config, lr, wd)}",
        flush=True,
    )
    log_path = state_dir(config, lr, wd) / "logs" / f"pre_decay_train_e{epoch}.log"
    run_torch(config, name, arguments, log_path)
    result = parse_validation(log_path, epoch, "pre_decay", target)
    result.update(
        {
            "policy": POLICY,
            "lr": lr,
            "wd": wd,
            "scheduler": "constant_with_warmup",
            "sourceCheckpoint": str(source) if source else None,
            "source": "constant_lr_training",
            **phase_metadata(config, epoch),
        }
    )
    atomic_json(predecay_result_path(config, lr, wd, epoch), result)
    emit_result(config, lr, wd, result)
    return result


def postdecay_training_arguments(
    config: dict[str, Any], lr: str, wd: str, epoch: int, source: Path, output: Path, name: str
) -> list[str]:
    batch = int(config["globalSequences"])
    endpoint = total_step(epoch, batch)
    arguments = base_arguments(config, lr, wd)
    arguments.extend(
        (
            f"--save-folder={output}",
            f"--trainer.max_duration={{value: {endpoint}, unit: steps}}",
            f"--trainer.callbacks.wandb.name={name}",
            (
                "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dense-1b,"
                f"pdpost,post-decay,bs{batch},lr{lr},wd{wd}]"
            ),
            f"--trainer.callbacks.checkpointer.fixed_steps=[{endpoint}]",
            "--data_loader.ignore_fingerprint_mismatch=true",
            (
                "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, "
                f"units: steps, warmup: {config['warmupSteps']}, decay_fraction: {DECAY_FRACTION}}}"
            ),
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
            f"--data_loader.restore_data_order_from_state={'false' if config['variant'] == 'DR+WT+EmbedWD' else 'true'}",
            "--force_exact_trainer_load_path=true",
            f"--trainer.load_path={source}",
            "--trainer.load_trainer_state=true",
            "--trainer.load_optim_state=true",
            "--trainer.reset_data_loader_state_on_load_path=false",
            "--train_module.validate_optimizer_hyperparameters_on_load=true",
        )
    )
    if config["variant"] == "DR+WT+EmbedWD" and epoch > 1:
        arguments.append("--dynamic-repacking")
    return arguments


def run_postdecay(config: dict[str, Any], lr: str, wd: str, epoch: int) -> dict[str, Any]:
    path = postdecay_result_path(config, lr, wd, epoch)
    if path.is_file():
        result = json.loads(path.read_text())
        emit_result(config, lr, wd, result)
        return result
    source = checkpoint_for_epoch(config, lr, wd, epoch)
    if not source.is_dir():
        raise FileNotFoundError(f"missing exact POST source checkpoint {source}")
    output = state_dir(config, lr, wd) / "post_decay_runs" / f"e{epoch}"
    endpoint = output / f"step{total_step(epoch, int(config['globalSequences']))}"
    if endpoint.is_dir():
        return evaluate_checkpoint(config, lr, wd, epoch, endpoint, phase="post_decay")
    name = run_name(config, lr, wd, "postdecay", epoch)
    arguments = postdecay_training_arguments(config, lr, wd, epoch, source, output, name)
    print(
        f"DENSE1B_PDPOST_STAGE_START bs={config['globalSequences']} lr={lr} wd={wd} "
        f"phase=post_decay_train epoch={epoch} source={source} output={output}",
        flush=True,
    )
    log_path = state_dir(config, lr, wd) / "logs" / f"post_decay_train_e{epoch}.log"
    run_torch(config, name, arguments, log_path)
    result = parse_validation(log_path, epoch, "post_decay", endpoint)
    result.update(
        {
            "policy": POLICY,
            "lr": lr,
            "wd": wd,
            "sourcePreDecayCheckpoint": str(source),
            "source": "post_decay_training",
            **phase_metadata(config, epoch),
        }
    )
    atomic_json(path, result)
    emit_result(config, lr, wd, result)
    return result


def latest_predecay_saturation(results: dict[int, dict[str, Any]]) -> int | None:
    epochs = sorted(results)
    if len(epochs) < POST_DECAY_SOURCE_COUNT:
        return None
    previous, current = results[epochs[-2]], results[epochs[-1]]
    if Decimal(str(current["validationExact"])) >= Decimal(str(previous["validationExact"])):
        return epochs[-1]
    return None


def postdecay_saturated(results: dict[int, dict[str, Any]]) -> bool:
    epochs = sorted(results)
    if len(epochs) != POST_DECAY_SOURCE_COUNT:
        raise ValueError("POST saturation requires exactly three results")
    previous, current = results[epochs[-2]], results[epochs[-1]]
    return Decimal(str(current["validationExact"])) >= Decimal(
        str(previous["validationExact"])
    )


def finish_postdecay(
    config: dict[str, Any], lr: str, wd: str, trigger_epoch: int, epochs: list[int], *, pruned: bool
) -> bool:
    sources = [epoch for epoch in sorted(epochs) if epoch <= trigger_epoch][-POST_DECAY_SOURCE_COUNT:]
    if len(sources) != POST_DECAY_SOURCE_COUNT:
        raise RuntimeError("POST finalization requires exactly three completed PD sources")
    results = {epoch: run_postdecay(config, lr, wd, epoch) for epoch in sources}
    saturated = postdecay_saturated(results)
    selected_epoch, selected = min(
        results.items(), key=lambda item: (Decimal(str(item[1]["validationExact"])), item[0])
    )
    decision: dict[str, Any] = {
        "status": "pruned" if pruned else "complete" if saturated else "continue",
        "policy": POLICY,
        "lr": lr,
        "wd": wd,
        "variant": str(config["variant"]),
        "trigger": "cross_wd_prune" if pruned else "pre_decay_non_improvement",
        "triggerEpoch": trigger_epoch,
        "preDecayDecisionGroup": "pre_decay",
        "postDecayDecisionGroup": "post_decay",
        "postDecaySaturationCriterion": POST_DECAY_SATURATION_CRITERION,
        "postDecaySaturated": saturated,
        "postDecaySourceEpochs": sources,
        "postDecayValidationExact": {
            str(epoch): float(result["validationExact"]) for epoch, result in results.items()
        },
        "postDecaySelectionGroup": "post_decay",
        "selectedPostDecayEpoch": selected_epoch,
        "selectedPostDecayValidationExact": float(selected["validationExact"]),
        "selectedCheckpoint": selected["checkpoint"],
    }
    if not pruned and not saturated:
        decision["nextPreDecayEpoch"] = next_frontier(config, trigger_epoch)
    atomic_json(decision_path(config, lr, wd, trigger_epoch), decision)
    atomic_json(selection_path(config, lr, wd), decision)
    marker = "DENSE1B_PDPOST_PRUNED" if pruned else (
        "DENSE1B_PDPOST_SELECTED" if saturated else "DENSE1B_PDPOST_CONTINUE"
    )
    print(
        f"{marker} bs={config['globalSequences']} lr={lr} wd={wd} "
        f"trigger_epoch={trigger_epoch} selected_epoch={selected_epoch} "
        f"validation={selected['validationExact']} json="
        f"{json.dumps(decision, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )
    return pruned or saturated


def recover_predecay_results(config: dict[str, Any], lr: str, wd: str) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    item = coordinate_config(config, lr, wd)
    for historical in sorted(item.get("historicalPreDecay", []), key=lambda value: int(value["epoch"])):
        epoch = int(historical["epoch"])
        results[epoch] = evaluate_checkpoint(
            config, lr, wd, epoch, Path(str(historical["checkpoint"])), phase="pre_decay"
        )
    for path in sorted(
        (state_dir(config, lr, wd) / "pre_decay").glob("e*.result.json")
        if (state_dir(config, lr, wd) / "pre_decay").is_dir()
        else []
    ):
        result = json.loads(path.read_text())
        results[int(result["epoch"])] = result
    return results


def run(config: dict[str, Any], lr: str, wd: str, *, finalize_only: bool) -> None:
    validate_config(config)
    coordinate_config(config, lr, wd)
    state_dir(config, lr, wd).mkdir(parents=True, exist_ok=True)
    existing_selection = selection_path(config, lr, wd)
    if existing_selection.is_file():
        selection = json.loads(existing_selection.read_text())
        if selection.get("status") in {"complete", "pruned"}:
            print(
                f"DENSE1B_PDPOST_ALREADY_TERMINAL bs={config['globalSequences']} "
                f"lr={lr} wd={wd} status={selection['status']}",
                flush=True,
            )
            return
    results = recover_predecay_results(config, lr, wd)
    if finalize_only:
        if len(results) < POST_DECAY_SOURCE_COUNT:
            raise RuntimeError("cannot prune before three scheduled PD checkpoints are complete")
        finish_postdecay(config, lr, wd, max(results), sorted(results), pruned=True)
        return
    item = coordinate_config(config, lr, wd)
    if not results:
        first = int(item.get("firstEpoch", target_at(config, 0)))
        results[first] = train_predecay(config, lr, wd, None, first)
    while True:
        trigger = latest_predecay_saturation(results)
        if trigger is not None:
            print(
                f"DENSE1B_PDPOST_PROVISIONAL bs={config['globalSequences']} lr={lr} "
                f"wd={wd} epoch={trigger} comparison_group=pre_decay",
                flush=True,
            )
            if finish_postdecay(config, lr, wd, trigger, sorted(results), pruned=False):
                return
        previous = max(results)
        epoch = next_frontier(config, previous)
        if epoch > int(config["maxEpoch"]):
            raise RuntimeError(
                f"LR{lr}/WD{wd} reached E{previous} without confirmed POST saturation"
            )
        results[epoch] = train_predecay(config, lr, wd, previous, epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lr", required=True)
    parser.add_argument("--wd", required=True)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    validate_config(config)
    coordinate_config(config, args.lr, args.wd)
    if args.validate_only:
        print(f"validated {args.manifest} LR{args.lr}/WD{args.wd}")
        return
    run(config, args.lr, args.wd, finalize_only=args.finalize_only)


if __name__ == "__main__":
    main()
