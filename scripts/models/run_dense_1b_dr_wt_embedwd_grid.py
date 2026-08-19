#!/usr/bin/env python3
"""Run one retry-safe Dense-1B DR+WT+EmbedWD LR/WD grid chain.

One Beaker task owns one batch-size study. Every LR/WD coordinate has its own
canonical output directory and optimizer trajectory. E1 starts with ordinary
packing, dynamic repacking begins at E2, and every later stage resumes that
coordinate's exact preceding pre-decay checkpoint. All requested coordinates
are evaluated at every frontier; the chain stops when the best held-out CE no
longer strictly improves.
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

TOKENS_PER_EPOCH = 1_000_000_000
SEQUENCE_LENGTH = 4096
DECAY_FRACTION = 0.1
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"
OUTPUT_ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"
ALLOWED_COORDINATES = {
    128: (("1e-3", "0.3"), ("1e-3", "1.0")),
    256: (
        ("1e-3", "0.333"),
        ("1e-3", "1.0"),
        ("2e-3", "0.333"),
        ("2e-3", "1.0"),
    ),
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def coordinates(config: dict[str, Any]) -> list[tuple[str, str]]:
    return [(str(item["lr"]), str(item["wd"])) for item in config["coordinates"]]


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
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"manifest is missing required keys: {missing}")
    batch = int(config["globalSequences"])
    if batch not in ALLOWED_COORDINATES:
        raise ValueError("only BS128 and BS256 are authorized")
    if int(config["nprocPerNode"]) != 8:
        raise ValueError("the 1B grid must use eight GPUs")
    if int(config["rankMicrobatchSequences"]) != 8:
        raise ValueError("the 1B grid must use rank microbatch eight")
    if int(config["gradientAccumulation"]) != batch // 64:
        raise ValueError("gradient accumulation does not produce the requested global batch")
    if int(config["warmupSteps"]) != 24_576 // batch:
        raise ValueError("warmup must preserve the established token budget")
    requested = tuple(coordinates(config))
    if requested != ALLOWED_COORDINATES[batch]:
        raise ValueError(f"BS{batch} coordinate grid must be {ALLOWED_COORDINATES[batch]}")
    for lr, wd in requested:
        if Decimal(lr) not in {Decimal("1e-3"), Decimal("2e-3")}:
            raise ValueError(f"unsupported LR {lr}")
        if Decimal(wd) > Decimal("1.0"):
            raise ValueError("WD must not exceed 1.0")
    if [int(value) for value in config["initialTargets"]] != [1, 2, 4]:
        raise ValueError("the 1B grid must begin E1 -> E2 -> E4")
    if int(config["epochIncrement"]) != 4 or int(config["maxEpoch"]) != 64:
        raise ValueError("the 1B grid must advance by four epochs through at most E64")
    if str(config["outputRoot"]) != OUTPUT_ROOT:
        raise ValueError(f"outputRoot must remain {OUTPUT_ROOT}")
    outputs = [output_for(config, lr, wd) for lr, wd in requested]
    if len(outputs) != len(set(outputs)):
        raise ValueError("every LR/WD coordinate must have a distinct output directory")


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
        raise ValueError(f"E{target} is not on the configured frontier ladder")
    return values


def output_for(config: dict[str, Any], lr: str, wd: str) -> Path:
    return Path(str(config["outputRoot"])) / (
        f"bs{config['globalSequences']}_dr_wt_embwd_lr{lr}_wd{wd}"
    )


def coordinate_key(lr: str, wd: str) -> str:
    return f"lr{lr}_wd{wd}"


def state_dir(config: dict[str, Any]) -> Path:
    return Path(str(config["outputRoot"])) / (
        f".bs{config['globalSequences']}_dr_wt_embwd_lr_wd_grid"
    )


def result_path(config: dict[str, Any], lr: str, wd: str, epoch: int) -> Path:
    return output_for(config, lr, wd) / f".dr_wt_embwd_grid_e{epoch}.result.json"


def frontier_path(config: dict[str, Any], epoch: int) -> Path:
    return state_dir(config) / f".frontier_e{epoch}.json"


def run_name(config: dict[str, Any], lr: str, wd: str, epoch: int) -> str:
    return (
        "dense_1b_step1_0802_repeated_dclm1b_wsd_"
        f"bs{config['globalSequences']}_dr_wt_embwd_e{epoch}_lr{lr}_wd{wd}_"
        f"warmup{config['warmupSteps']}_{config['runSuffix']}"
    )


def recovery_checkpoint(
    config: dict[str, Any], lr: str, wd: str, previous_epoch: int, epoch: int
) -> Path | None:
    batch = int(config["globalSequences"])
    output = output_for(config, lr, wd)
    endpoint = output / f"step{total_step(epoch, batch)}"
    if endpoint.is_dir():
        print(
            f"DENSE1B_DRWTEMBWD_GRID_EVAL_RECOVERY bs={batch} epoch={epoch} "
            f"lr={lr} wd={wd} checkpoint={endpoint}",
            flush=True,
        )
        return endpoint
    candidates: list[Path] = []
    if previous_epoch:
        candidates.append(output / f"step{stable_step(previous_epoch, batch)}")
    candidates.extend(
        output / f"step{stable_step(candidate, batch)}"
        for candidate in range(previous_epoch + 1, epoch + 1)
    )
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    if existing:
        selected = existing[-1]
        print(
            f"DENSE1B_DRWTEMBWD_GRID_RESUME bs={batch} epoch={epoch} "
            f"lr={lr} wd={wd} checkpoint={selected}",
            flush=True,
        )
        return selected
    if previous_epoch:
        raise FileNotFoundError(
            f"no exact LR{lr}/WD{wd} checkpoint exists for E{epoch}: {candidates}"
        )
    return None


def stage_arguments(
    config: dict[str, Any],
    lr: str,
    wd: str,
    previous_epoch: int,
    epoch: int,
    checkpoint: Path | None,
) -> list[str]:
    batch = int(config["globalSequences"])
    output = output_for(config, lr, wd)
    retained = [stable_step(value, batch) for value in range(previous_epoch + 1, epoch + 1)]
    pending = [step for step in retained if not (output / f"step{step}").is_dir()]
    arguments = [
        *COMMON_ARGUMENTS,
        f"--save-folder={output}",
        f"--trainer.max_duration={{value: {epoch * TOKENS_PER_EPOCH}, unit: tokens}}",
        f"--trainer.callbacks.wandb.name={run_name(config, lr, wd, epoch)}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dense-1b,"
            "repeated-data,data-loader-study,dr_wt_embwd,"
            f"bs{batch},e{epoch},lr{lr},wd{wd},lr-wd-grid,wsd]"
        ),
        "--trainer.callbacks.checkpointer.fixed_steps="
        + json.dumps(pending, separators=(",", ":")),
        f"--data_loader.global_batch_size={batch * SEQUENCE_LENGTH}",
        "--data_loader.restore_data_order_from_state=false",
        f"--data_loader.ignore_fingerprint_mismatch={'true' if epoch > 1 else 'false'}",
        f"--train_module.rank_microbatch_size={int(config['rankMicrobatchSequences']) * SEQUENCE_LENGTH}",
        (
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, "
            f"units: steps, warmup: {config['warmupSteps']}, decay_fraction: {DECAY_FRACTION}}}"
        ),
        f"--train_module.optim.weight_decay={wd}",
        f"--lr={lr}",
        "--model.tie_embeddings=true",
        "--decay-embeddings",
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
    ]
    if epoch > 1:
        arguments.append("--dynamic-repacking")
    if checkpoint is not None:
        arguments.extend(
            (
                "--force_exact_trainer_load_path=true",
                f"--trainer.load_path={checkpoint}",
                "--trainer.load_trainer_state=true",
                "--trainer.load_optim_state=true",
                "--trainer.reset_data_loader_state_on_load_path=false",
                "--train_module.validate_optimizer_hyperparameters_on_load=true",
            )
        )
    return arguments


def emit_stage(
    config: dict[str, Any], lr: str, wd: str, epoch: int, result: dict[str, Any]
) -> None:
    payload = json.dumps(result, separators=(",", ":"), sort_keys=True)
    print(
        f"DENSE1B_DRWTEMBWD_GRID_STAGE_RESULT bs={config['globalSequences']} "
        f"epoch={epoch} lr={lr} wd={wd} json={payload}",
        flush=True,
    )


def run_stage(
    config: dict[str, Any], lr: str, wd: str, previous_epoch: int, epoch: int
) -> dict[str, Any]:
    path = result_path(config, lr, wd, epoch)
    if path.is_file():
        result = json.loads(path.read_text())
        emit_stage(config, lr, wd, epoch, result)
        return result
    output = output_for(config, lr, wd)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = recovery_checkpoint(config, lr, wd, previous_epoch, epoch)
    name = run_name(config, lr, wd, epoch)
    arguments = stage_arguments(config, lr, wd, previous_epoch, epoch, checkpoint)
    print(
        f"DENSE1B_DRWTEMBWD_GRID_STAGE_START bs={config['globalSequences']} "
        f"epoch={epoch} lr={lr} wd={wd} previous_epoch={previous_epoch} output={output}",
        flush=True,
    )
    subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path = output / f".dr_wt_embwd_grid_e{epoch}.log"
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
    result = common.parse_stage_result(log_path, epoch)
    result.update(
        {
            "lr": lr,
            "wd": wd,
            "variant": "DR+WT+EmbedWD",
            "dataOrder": "original_e1" if epoch == 1 else "dynamic_repacking",
            "dynamicRepacking": epoch > 1,
            "weightTying": True,
            "decayEmbeddings": True,
            "embeddingWeightDecay": wd,
            "output": str(output),
            "sourceCheckpoint": str(checkpoint) if checkpoint is not None else None,
            "preDecayCheckpoint": str(
                output / f"step{stable_step(epoch, int(config['globalSequences']))}"
            ),
            "endpointCheckpoint": str(
                output / f"step{total_step(epoch, int(config['globalSequences']))}"
            ),
            "sequential": True,
        }
    )
    atomic_json(path, result)
    emit_stage(config, lr, wd, epoch, result)
    return result


def ensure_trajectory(config: dict[str, Any], lr: str, wd: str, target: int) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    previous = 0
    for epoch in targets_through(config, target):
        result = run_stage(config, lr, wd, previous, epoch)
        previous = epoch
    assert result is not None
    return result


def choose_frontier(
    config: dict[str, Any], index: int, epoch: int, results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    selected_key, selected = min(
        results.items(),
        key=lambda item: (
            Decimal(str(item[1]["validationExact"])),
            -Decimal(str(item[1]["wd"])),
            Decimal(str(item[1]["lr"])),
        ),
    )
    previous_epoch = target_at(config, index - 1) if index else None
    previous_validation = None
    if previous_epoch is not None:
        previous = json.loads(frontier_path(config, previous_epoch).read_text())
        previous_validation = float(previous["selectedValidationExact"])
    selected_validation = float(selected["validationExact"])
    if epoch >= int(config["maxEpoch"]):
        action = "stop_max_epoch"
    elif previous_validation is None or selected_validation < previous_validation:
        action = "continue"
    else:
        action = "stop"
    return {
        "epoch": epoch,
        "status": "complete",
        "variant": "DR+WT+EmbedWD",
        "candidates": list(results),
        "candidateValidationExact": {
            key: float(result["validationExact"]) for key, result in results.items()
        },
        "selectedCoordinate": selected_key,
        "selectedLr": selected["lr"],
        "selectedWd": selected["wd"],
        "selectedValidation": round(selected_validation, 3),
        "selectedValidationExact": selected_validation,
        "previousEpoch": previous_epoch,
        "previousSelectedValidationExact": previous_validation,
        "decision": action,
    }


def emit_frontier(config: dict[str, Any], frontier: dict[str, Any]) -> None:
    payload = json.dumps(frontier, separators=(",", ":"), sort_keys=True)
    print(
        f"DENSE1B_DRWTEMBWD_GRID_FRONTIER_RESULT bs={config['globalSequences']} "
        f"epoch={frontier['epoch']} json={payload}",
        flush=True,
    )


def run(config: dict[str, Any]) -> None:
    validate_config(config)
    state_dir(config).mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        epoch = target_at(config, index)
        path = frontier_path(config, epoch)
        if path.is_file():
            frontier = json.loads(path.read_text())
            emit_frontier(config, frontier)
        else:
            coordinate_values = coordinates(config)
            print(
                f"DENSE1B_DRWTEMBWD_GRID_FRONTIER_START bs={config['globalSequences']} "
                f"epoch={epoch} candidates="
                + ",".join(coordinate_key(lr, wd) for lr, wd in coordinate_values),
                flush=True,
            )
            results = {
                coordinate_key(lr, wd): ensure_trajectory(config, lr, wd, epoch)
                for lr, wd in coordinate_values
            }
            frontier = choose_frontier(config, index, epoch, results)
            atomic_json(path, frontier)
            emit_frontier(config, frontier)
        if frontier["decision"] != "continue":
            print(
                f"DENSE1B_DRWTEMBWD_GRID_SATURATED bs={config['globalSequences']} "
                f"epoch={epoch} lr={frontier['selectedLr']} wd={frontier['selectedWd']} "
                f"reason={frontier['decision']}",
                flush=True,
            )
            return
        index += 1


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
