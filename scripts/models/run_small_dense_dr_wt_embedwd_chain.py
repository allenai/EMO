#!/usr/bin/env python3
"""Run one retry-safe adaptive small-dense DR+WT+EmbedWD chain.

One Beaker task owns one model-size/batch-size study.  Every fixed-WD
trajectory keeps a single canonical save folder.  E1 is the common packed-data
bootstrap used by the existing Dense-1B DR study; dynamic repacking begins at
E2.  Later stages always restore the same WD trajectory's exact pre-decay
checkpoint.

The initial frontier evaluates the baseline-optimal WD and its immediate
neighbors.  Each later frontier evaluates the preceding winner and one WD
level higher.  A newly introduced higher WD is backfilled from E1 so WD is
never changed inside an optimizer trajectory.  Atomic stage/frontier records
make Beaker task retries idempotent.
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
MAX_WEIGHT_DECAY = Decimal("1.0")
ALLOWED_OUTPUT_PREFIX = "/weka/oe-training-default/sewonm/icsl/models/"
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"
DOWNSTREAM_TASKS = (
    "[arc_easy, arc_challenge, boolq, csqa_val_rc_5shot, hellaswag, "
    "openbookqa_test_rc_5shot, piqa, socialiqa_val_rc_5shot, winogrande]"
)
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
    f"--trainer.callbacks.downstream_evaluator.tasks={DOWNSTREAM_TASKS}",
    "--trainer.callbacks.downstream_evaluator.eval_interval=null",
    "--trainer.callbacks.downstream_evaluator.eval_on_finish=true",
    f"--trainer.callbacks.heldout_evaluator={HELDOUT_EVALUATOR}",
    "--dataset.instance_filter_config={repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}",
    "--model.block.name=default",
    "--model.block.sequence_mixer.qk_norm=null",
    "--init_seed=12536",
    "--data_loader.seed=0",
)
MODEL_ARGUMENTS = {
    "474m": (
        "--model-size=1B",
        "--model.d_model=1024",
        "--model.n_layers=16",
        "--model.block.sequence_mixer.n_heads=16",
        "--model.block.feed_forward.hidden_size=4096",
    ),
    "153m": ("--model-size=153M",),
}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "model",
        "globalSequences",
        "nprocPerNode",
        "rankMicrobatchSequences",
        "warmupSteps",
        "learningRate",
        "wdLadder",
        "baselineOptimalWd",
        "initialWds",
        "initialTargets",
        "epochIncrement",
        "outputRoot",
        "runSuffix",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"manifest is missing required keys: {missing}")
    model = str(config["model"])
    if model not in MODEL_ARGUMENTS:
        raise ValueError(f"unsupported model {model}")
    batch = int(config["globalSequences"])
    expected_gpus = 4 if batch == 64 else 8
    if batch not in {64, 128, 256, 512}:
        raise ValueError("globalSequences must be one of 64, 128, 256, 512")
    if int(config["nprocPerNode"]) != expected_gpus:
        raise ValueError(f"BS{batch} must use {expected_gpus} GPUs")
    if int(config["rankMicrobatchSequences"]) != 16:
        raise ValueError("rank microbatch must remain 16 sequences")
    if batch % (expected_gpus * 16):
        raise ValueError("global batch is not divisible by GPUs x rank microbatch")
    if Decimal(str(config["learningRate"])) != Decimal("0.002"):
        raise ValueError("the adaptive study is locked to LR2e-3")

    ladder = [str(value) for value in config["wdLadder"]]
    if not ladder or len(ladder) != len(set(ladder)):
        raise ValueError("WD ladder must be nonempty and contain unique values")
    decimal_ladder = [Decimal(value) for value in ladder]
    if decimal_ladder != sorted(decimal_ladder):
        raise ValueError("WD ladder must be strictly increasing")
    if max(decimal_ladder) > MAX_WEIGHT_DECAY:
        raise ValueError("WD ladder must not exceed 1.0")
    center = str(config["baselineOptimalWd"])
    if center not in ladder:
        raise ValueError("baselineOptimalWd is not on the WD ladder")
    center_index = ladder.index(center)
    if center_index == 0 or center_index == len(ladder) - 1:
        raise ValueError("baselineOptimalWd must have lower and higher neighbors")
    expected_initial = ladder[center_index - 1 : center_index + 2]
    if [str(value) for value in config["initialWds"]] != expected_initial:
        raise ValueError(f"initialWds must be immediate neighbors {expected_initial}")

    targets = [int(value) for value in config["initialTargets"]]
    expected = [1, 2, 4, 8, 16, 24] if model == "474m" else [1, 2, 4, 8, 16, 32, 48]
    if targets != expected:
        raise ValueError(f"{model} must start with target ladder {expected}")
    expected_increment = 8 if model == "474m" else 16
    if int(config["epochIncrement"]) != expected_increment:
        raise ValueError(f"{model} must use increment {expected_increment}")

    overrides = config.get("outputOverrides", {})
    if not isinstance(overrides, dict):
        raise TypeError("outputOverrides must be a WD-to-path mapping")
    unknown_wds = sorted(set(overrides) - set(ladder))
    if unknown_wds:
        raise ValueError(f"outputOverrides contains WDs outside the ladder: {unknown_wds}")
    outputs = [str(output_for(config, wd)) for wd in ladder]
    if len(outputs) != len(set(outputs)):
        raise ValueError("every fixed-WD trajectory must have a distinct output directory")
    model_output_root = str(config["outputRoot"]).rstrip("/") + "/"
    for output in outputs:
        if not output.startswith(ALLOWED_OUTPUT_PREFIX):
            raise ValueError(f"output directory is outside the approved model root: {output}")
        if not output.startswith(model_output_root):
            raise ValueError(f"output directory is outside this model's output root: {output}")


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
    targets: list[int] = []
    index = 0
    while not targets or targets[-1] < target:
        targets.append(target_at(config, index))
        index += 1
    if targets[-1] != target:
        raise ValueError(f"E{target} is not on the configured epoch ladder")
    return targets


def output_for(config: dict[str, Any], wd: str) -> Path:
    override = config.get("outputOverrides", {}).get(wd)
    if override is not None:
        return Path(str(override))
    return Path(str(config["outputRoot"])) / (
        f"bs{config['globalSequences']}_dr_wt_embwd_lr{config['learningRate']}_wd{wd}"
    )


def parse_output_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        wd, separator, raw_path = value.partition("=")
        if not separator or not wd or not raw_path:
            raise ValueError(f"output override must have the form WD=/absolute/path: {value}")
        path = Path(raw_path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"output override must be a normalized absolute path: {raw_path}")
        normalized = str(path)
        if not normalized.startswith(ALLOWED_OUTPUT_PREFIX):
            raise ValueError(f"output override is outside the approved model root: {normalized}")
        if wd in overrides:
            raise ValueError(f"duplicate output override for WD{wd}")
        overrides[wd] = normalized
    return overrides


def state_dir(config: dict[str, Any]) -> Path:
    return Path(str(config["outputRoot"])) / (
        f".bs{config['globalSequences']}_dr_wt_embwd_lr{config['learningRate']}_adaptive"
    )


def result_path(config: dict[str, Any], wd: str, epoch: int) -> Path:
    return output_for(config, wd) / f".dr_wt_embwd_e{epoch}.result.json"


def frontier_path(config: dict[str, Any], epoch: int) -> Path:
    return state_dir(config) / f".frontier_e{epoch}.json"


def run_name(config: dict[str, Any], wd: str, epoch: int) -> str:
    return (
        f"dense_{config['model']}_step1_0802_repeated_dclm1b_wsd_"
        f"bs{config['globalSequences']}_dr_wt_embwd_e{epoch}_"
        f"lr{config['learningRate']}_wd{wd}_warmup{config['warmupSteps']}_"
        f"{config['runSuffix']}"
    )


def recovery_checkpoint(
    config: dict[str, Any], wd: str, previous_epoch: int, epoch: int
) -> Path | None:
    output = output_for(config, wd)
    endpoint = output / f"step{total_step(epoch, int(config['globalSequences']))}"
    if endpoint.is_dir():
        print(
            f"SMALL_DRWTEMBWD_EVAL_RECOVERY model={config['model']} "
            f"bs={config['globalSequences']} epoch={epoch} wd={wd} checkpoint={endpoint}",
            flush=True,
        )
        return endpoint

    candidates: list[Path] = []
    if previous_epoch:
        candidates.append(
            output / f"step{stable_step(previous_epoch, int(config['globalSequences']))}"
        )
    candidates.extend(
        output / f"step{stable_step(candidate, int(config['globalSequences']))}"
        for candidate in range(previous_epoch + 1, epoch + 1)
    )
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    if existing:
        selected = existing[-1]
        print(
            f"SMALL_DRWTEMBWD_RESUME model={config['model']} bs={config['globalSequences']} "
            f"epoch={epoch} wd={wd} checkpoint={selected}",
            flush=True,
        )
        return selected
    if previous_epoch:
        raise FileNotFoundError(
            f"no exact WD{wd} pre-decay checkpoint exists for E{epoch}: {candidates}"
        )
    return None


def stage_arguments(
    config: dict[str, Any], wd: str, previous_epoch: int, epoch: int, checkpoint: Path | None
) -> list[str]:
    batch = int(config["globalSequences"])
    output = output_for(config, wd)
    name = run_name(config, wd, epoch)
    retained = [stable_step(value, batch) for value in range(previous_epoch + 1, epoch + 1)]
    arguments = [
        *COMMON_ARGUMENTS,
        *MODEL_ARGUMENTS[str(config["model"])],
        f"--save-folder={output}",
        f"--trainer.max_duration={{value: {epoch * TOKENS_PER_EPOCH}, unit: tokens}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,"
            f"dense-{config['model']},repeated-data,dclm-train-only,data-loader-study,"
            f"dr_wt_embwd,bs{batch},e{epoch},lr2e-3,wd{wd},adaptive-wd-chain,wsd]"
        ),
        "--trainer.callbacks.checkpointer.fixed_steps="
        + json.dumps(retained, separators=(",", ":")),
        f"--data_loader.global_batch_size={batch * SEQUENCE_LENGTH}",
        "--data_loader.restore_data_order_from_state=false",
        f"--data_loader.ignore_fingerprint_mismatch={'true' if epoch > 1 else 'false'}",
        f"--train_module.rank_microbatch_size={int(config['rankMicrobatchSequences']) * SEQUENCE_LENGTH}",
        (
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, "
            f"units: steps, warmup: {config['warmupSteps']}, decay_fraction: {DECAY_FRACTION}}}"
        ),
        f"--train_module.optim.weight_decay={wd}",
        f"--lr={config['learningRate']}",
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


def emit_stage(config: dict[str, Any], wd: str, epoch: int, result: dict[str, Any]) -> None:
    payload = json.dumps(result, separators=(",", ":"), sort_keys=True)
    print(
        f"SMALL_DRWTEMBWD_STAGE_RESULT model={config['model']} "
        f"bs={config['globalSequences']} epoch={epoch} wd={wd} json={payload}",
        flush=True,
    )


def run_stage(
    config: dict[str, Any], wd: str, previous_epoch: int, epoch: int
) -> dict[str, Any]:
    path = result_path(config, wd, epoch)
    if path.is_file():
        result = json.loads(path.read_text())
        emit_stage(config, wd, epoch, result)
        print(
            f"SMALL_DRWTEMBWD_STAGE_REUSED model={config['model']} "
            f"bs={config['globalSequences']} epoch={epoch} wd={wd}",
            flush=True,
        )
        return result

    output = output_for(config, wd)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = recovery_checkpoint(config, wd, previous_epoch, epoch)
    name = run_name(config, wd, epoch)
    arguments = stage_arguments(config, wd, previous_epoch, epoch, checkpoint)
    print(
        f"SMALL_DRWTEMBWD_STAGE_START model={config['model']} "
        f"bs={config['globalSequences']} epoch={epoch} wd={wd} "
        f"previous_epoch={previous_epoch} output={output}",
        flush=True,
    )
    subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)

    log_path = output / f".dr_wt_embwd_e{epoch}.log"
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
            "lr": str(config["learningRate"]),
            "wd": wd,
            "variant": "DR+WT+EmbedWD",
            "dataOrder": "original_e1" if epoch == 1 else "dynamic_repacking",
            "dynamicRepacking": epoch > 1,
            "weightTying": True,
            "decayEmbeddings": True,
            "embeddingWeightDecay": wd,
            "output": str(output),
            "sourceCheckpoint": str(checkpoint) if checkpoint is not None else None,
            "preDecayCheckpoint": str(output / f"step{stable_step(epoch, int(config['globalSequences']))}"),
            "endpointCheckpoint": str(output / f"step{total_step(epoch, int(config['globalSequences']))}"),
            "sequential": True,
        }
    )
    atomic_json(path, result)
    emit_stage(config, wd, epoch, result)
    print(
        f"SMALL_DRWTEMBWD_STAGE_COMPLETE model={config['model']} "
        f"bs={config['globalSequences']} epoch={epoch} wd={wd} "
        f"validation={result['validationExact']}",
        flush=True,
    )
    return result


def ensure_trajectory(config: dict[str, Any], wd: str, target: int) -> dict[str, Any]:
    targets = targets_through(config, target)
    result: dict[str, Any] | None = None
    previous = 0
    for epoch in targets:
        result = run_stage(config, wd, previous, epoch)
        previous = epoch
    assert result is not None
    return result


def next_higher(config: dict[str, Any], wd: str) -> str | None:
    ladder = [str(value) for value in config["wdLadder"]]
    index = ladder.index(wd)
    if index + 1 >= len(ladder):
        return None
    candidate = ladder[index + 1]
    return candidate if Decimal(candidate) <= MAX_WEIGHT_DECAY else None


def candidate_wds(config: dict[str, Any], index: int) -> list[str]:
    if index == 0:
        return [str(value) for value in config["initialWds"]]
    previous_epoch = target_at(config, index - 1)
    previous = json.loads(frontier_path(config, previous_epoch).read_text())
    selected = str(previous["selectedWd"])
    candidates = [selected]
    higher = next_higher(config, selected)
    if higher is not None:
        candidates.append(higher)
    return candidates


def choose_frontier(
    config: dict[str, Any], index: int, epoch: int, results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    selected_wd, selected_result = min(
        results.items(),
        key=lambda item: (
            Decimal(str(item[1]["validationExact"])),
            -Decimal(item[0]),
        ),
    )
    previous_epoch = target_at(config, index - 1) if index else None
    previous_validation = None
    if previous_epoch is not None:
        previous = json.loads(frontier_path(config, previous_epoch).read_text())
        previous_validation = float(previous["selectedValidationExact"])
    selected_validation = float(selected_result["validationExact"])
    action = (
        "continue"
        if previous_validation is None or selected_validation < previous_validation
        else "stop"
    )
    return {
        "epoch": epoch,
        "status": "complete",
        "variant": "DR+WT+EmbedWD",
        "candidates": list(results),
        "candidateValidationExact": {
            wd: float(result["validationExact"]) for wd, result in results.items()
        },
        "selectedWd": selected_wd,
        "selectedValidation": round(selected_validation, 3),
        "selectedValidationExact": selected_validation,
        "previousEpoch": previous_epoch,
        "previousSelectedValidationExact": previous_validation,
        "decision": action,
    }


def emit_frontier(config: dict[str, Any], frontier: dict[str, Any]) -> None:
    payload = json.dumps(frontier, separators=(",", ":"), sort_keys=True)
    print(
        f"SMALL_DRWTEMBWD_FRONTIER_RESULT model={config['model']} "
        f"bs={config['globalSequences']} epoch={frontier['epoch']} json={payload}",
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
            print(
                f"SMALL_DRWTEMBWD_FRONTIER_REUSED model={config['model']} "
                f"bs={config['globalSequences']} epoch={epoch} "
                f"selected_wd={frontier['selectedWd']} action={frontier['decision']}",
                flush=True,
            )
        else:
            candidates = candidate_wds(config, index)
            print(
                f"SMALL_DRWTEMBWD_FRONTIER_START model={config['model']} "
                f"bs={config['globalSequences']} epoch={epoch} candidates={','.join(candidates)}",
                flush=True,
            )
            results = {wd: ensure_trajectory(config, wd, epoch) for wd in candidates}
            frontier = choose_frontier(config, index, epoch, results)
            atomic_json(path, frontier)
            emit_frontier(config, frontier)
            print(
                f"SMALL_DRWTEMBWD_FRONTIER_COMPLETE model={config['model']} "
                f"bs={config['globalSequences']} epoch={epoch} "
                f"selected_wd={frontier['selectedWd']} "
                f"validation={frontier['selectedValidationExact']} "
                f"action={frontier['decision']}",
                flush=True,
            )
        if frontier["decision"] == "stop":
            print(
                f"SMALL_DRWTEMBWD_SATURATED model={config['model']} "
                f"bs={config['globalSequences']} epoch={epoch} "
                f"selected_wd={frontier['selectedWd']}",
                flush=True,
            )
            return
        index += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-override",
        action="append",
        default=[],
        metavar="WD=/ABSOLUTE/PATH",
        help="Reroute one fixed-WD trajectory during a guarded recovery submission",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    overrides = dict(config.get("outputOverrides", {}))
    overrides.update(parse_output_overrides(args.output_override))
    config["outputOverrides"] = overrides
    validate_config(config)
    if args.validate_only:
        print(f"validated {args.manifest}")
        return
    run(config)


if __name__ == "__main__":
    main()
