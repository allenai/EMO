#!/usr/bin/env python3
"""Run one isolated Pool-3B small-model WSD decay and heldout evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dense_1b_dr_wt_embedwd_grid as dense1b
import run_dense_small_pool3b_checkpoint_producer as producer
import run_small_dense_dr_wt_embedwd_chain as small
import run_small_dense_saturation_chain as common

POLICY = "dense_small_pool3b_checkpoint_evaluator_v1"
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load(
    manifest_path: Path, coordinate_id: str, epoch: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = producer.load_manifest(manifest_path)
    item = producer.coordinate(config, coordinate_id)
    producer.validate_coordinate(config, item, check_filesystem=False)
    if epoch not in producer.target_epochs(item, int(config["maxEpoch"])):
        raise ValueError(f"E{epoch} is not a retained checkpoint for {coordinate_id}")
    return config, item


def source_checkpoint(item: dict[str, Any], epoch: int) -> Path:
    step = producer.stable_step(epoch, producer.TARGET_POOL_TOKENS, int(item["batchSequences"]))
    return Path(str(item["output"])) / f"step{step}"


def validate_source(item: dict[str, Any], epoch: int) -> Path:
    source = source_checkpoint(item, epoch)
    config_path = source / "config.json"
    if (
        not source.is_dir()
        or not config_path.is_file()
        or not (source / "model_and_optim").is_dir()
    ):
        raise FileNotFoundError(f"incomplete exact pre-decay checkpoint {source}")
    value = json.loads(config_path.read_text())
    model = str(item["model"])
    batch = int(item["batchSequences"])
    d_model, layers = producer.MODEL_SHAPES[model]
    checks = {
        "d_model": (int(value["model"]["d_model"]), d_model),
        "n_layers": (int(value["model"]["n_layers"]), layers),
        "global_batch_size": (
            int(value["data_loader"]["global_batch_size"]),
            batch * producer.SEQUENCE_LENGTH,
        ),
        "tie_embeddings": (bool(value["model"]["tie_embeddings"]), True),
        "dynamic_repacking": (bool(value["dataset"]["dynamic_repacking"]), True),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    optim = value["train_module"]["optim"]
    if Decimal(str(optim["lr"])) != Decimal(str(item["learningRate"])):
        mismatches.append("learning_rate")
    if Decimal(str(optim["weight_decay"])) != Decimal(str(item["weightDecay"])):
        mismatches.append("weight_decay")
    if optim.get("group_overrides") != []:
        mismatches.append("embedding_weight_decay")
    if mismatches:
        raise RuntimeError(f"pre-decay checkpoint recipe mismatch {mismatches}: {source}")
    return source


def state_dir(item: dict[str, Any]) -> Path:
    return Path(str(item["output"])) / ".constant_checkpoint_evaluator_pool3b_v1"


def base_arguments(config: dict[str, Any], item: dict[str, Any], *, heldout: str) -> list[str]:
    arguments = producer.base_arguments(item, str(config["repeatedManifest"]))
    return common.upsert(
        arguments,
        "--trainer.callbacks.heldout_evaluator=",
        f"--trainer.callbacks.heldout_evaluator={heldout}",
    )


def run_torch(name: str, arguments: list[str], log_path: Path) -> None:
    subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        common.stream_command(
            ["torchrun", "--nproc-per-node=8", TRAINING_SCRIPT, name, *arguments],
            log_file,
        )


def postdecay_arguments(
    config: dict[str, Any],
    item: dict[str, Any],
    epoch: int,
    source: Path,
    output: Path,
    name: str,
) -> list[str]:
    batch = int(item["batchSequences"])
    endpoint = producer.total_step(epoch, producer.TARGET_POOL_TOKENS, batch)
    arguments = base_arguments(config, item, heldout=small.HELDOUT_EVALUATOR)
    arguments = common.upsert(
        arguments,
        "--train_module.scheduler=",
        (
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, "
            f"units: steps, warmup: {24576 // batch}, decay_fraction: {producer.DECAY_FRACTION}}}"
        ),
    )
    return [
        *arguments,
        "--dynamic-repacking",
        f"--save-folder={output}",
        f"--trainer.max_duration={{value: {endpoint}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,"
            f"dense-{item['model']},checkpoint-evaluator,post-decay,bs{batch},"
            f"e{epoch},lr{item['learningRate']},wd{item['weightDecay']}]"
        ),
        f"--trainer.callbacks.checkpointer.fixed_steps=[{endpoint}]",
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


def evaluation_arguments(
    config: dict[str, Any],
    item: dict[str, Any],
    checkpoint: Path,
    output: Path,
    name: str,
) -> list[str]:
    heldout = small.HELDOUT_EVALUATOR.replace(
        "eval_on_finish: true", "eval_on_finish: false, eval_on_startup: true"
    )
    return [
        *base_arguments(config, item, heldout=heldout),
        "--dynamic-repacking",
        f"--save-folder={output}",
        f"--trainer.callbacks.wandb.name={name}",
        "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,checkpoint-evaluator,recovered-eval]",
        "--trainer.max_duration={value: 1000000000000, unit: steps}",
        "--trainer.callbacks.checkpointer.enabled=false",
        f"--load_path={checkpoint}",
        "--load_trainer_state=false",
    ]


def run(config: dict[str, Any], item: dict[str, Any], epoch: int) -> dict[str, Any]:
    result_path = state_dir(item) / "results" / f"e{epoch}.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        print(
            f"DENSE_SMALL_CHECKPOINT_EVALUATOR_RESULT id={item['id']} epoch={epoch} "
            f"json={json.dumps(result, separators=(',', ':'), sort_keys=True)}",
            flush=True,
        )
        print(
            f"DENSE_SMALL_CHECKPOINT_EVALUATOR_COMPLETE id={item['id']} epoch={epoch}",
            flush=True,
        )
        return result

    source = validate_source(item, epoch)
    output = state_dir(item) / "post_decay_runs" / f"e{epoch}"
    endpoint_step = producer.total_step(
        epoch, producer.TARGET_POOL_TOKENS, int(item["batchSequences"])
    )
    endpoint = output / f"step{endpoint_step}"
    name = f"{item['id']}-post-e{epoch}-v1"
    if endpoint.is_dir():
        eval_name = f"{name}-recovered-eval"
        log_path = state_dir(item) / "logs" / f"e{epoch}_recovered_eval.log"
        run_torch(
            eval_name,
            evaluation_arguments(config, item, endpoint, output / "eval", eval_name),
            log_path,
        )
    else:
        log_path = state_dir(item) / "logs" / f"e{epoch}.log"
        print(
            f"DENSE_SMALL_CHECKPOINT_EVALUATOR_START id={item['id']} epoch={epoch} "
            f"source={source} output={output}",
            flush=True,
        )
        run_torch(
            name,
            postdecay_arguments(config, item, epoch, source, output, name),
            log_path,
        )
    if not endpoint.is_dir():
        raise RuntimeError(f"E{epoch} decay exited without endpoint {endpoint}")
    result = dense1b.parse_validation(log_path, epoch, "post_decay", endpoint)
    result.update(
        {
            "policy": POLICY,
            "producerPolicy": producer.POLICY,
            "model": str(item["model"]),
            "batchSequences": int(item["batchSequences"]),
            "lr": str(item["learningRate"]),
            "wd": str(item["weightDecay"]),
            "variant": "DR+WT+EmbedWD",
            "dynamicRepacking": True,
            "weightTying": True,
            "decayEmbeddings": True,
            "sourcePreDecayCheckpoint": str(source),
            "source": "independent_wsd_decay_and_heldout_eval",
        }
    )
    atomic_json(result_path, result)
    print(
        f"DENSE_SMALL_CHECKPOINT_EVALUATOR_RESULT id={item['id']} epoch={epoch} "
        f"json={json.dumps(result, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )
    print(
        f"DENSE_SMALL_CHECKPOINT_EVALUATOR_COMPLETE id={item['id']} epoch={epoch}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coordinate", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config, item = load(args.manifest, args.coordinate, args.epoch)
    if args.validate_only:
        print(f"validated small evaluator {args.coordinate} E{args.epoch}")
        return
    run(config, item, args.epoch)


if __name__ == "__main__":
    main()
