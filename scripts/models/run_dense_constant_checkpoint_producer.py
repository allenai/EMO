#!/usr/bin/env python3
"""Resume one exact DR+WT+EmbedWD checkpoint and only produce PD checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dense_1b_dr_wt_embedwd_grid as dense1b
import run_small_dense_dr_wt_embedwd_chain as small
import run_small_dense_saturation_chain as common

POLICY = "dense_constant_checkpoint_producers_v1"
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"
SEQUENCE_LENGTH = 4096
DECAY_FRACTION = 0.1
ALLOWED_ROOT = Path("/weka/oe-training-default/sewonm/icsl/models")
POOL3B_MANIFEST = (
    "/weka/oe-training-default/sewonm/icsl/data/dclm_0802_nested_1b_3b_9b/"
    "manifests/dclm_0802_nested_train_3b_flat_dynamic_repacking.json"
)
SMALL_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json"
MODEL_SHAPES = {
    "1b": (2048, 16),
    "474m": (1024, 16),
    "153m": (512, 12),
}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def total_step(epoch: int, pool_tokens: int, batch: int) -> int:
    return math.ceil(epoch * pool_tokens / (batch * SEQUENCE_LENGTH))


def stable_step(epoch: int, pool_tokens: int, batch: int) -> int:
    endpoint = total_step(epoch, pool_tokens, batch)
    return endpoint - round(DECAY_FRACTION * endpoint) - 1


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("policy") != POLICY:
        raise ValueError(f"manifest policy must be {POLICY}")
    if int(manifest.get("maxEpoch", 0)) != 256:
        raise ValueError("producer ceiling must remain E256")
    return manifest


def coordinate(manifest: dict[str, Any], coordinate_id: str) -> dict[str, Any]:
    matches = [
        item for item in manifest.get("producerCoordinates", []) if item.get("id") == coordinate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one producer coordinate {coordinate_id}")
    return matches[0]


def target_epochs(item: dict[str, Any], max_epoch: int) -> list[int]:
    first = int(item["firstCheckpointEpoch"])
    increment = int(item["epochIncrement"])
    if first <= 0 or increment <= 0 or first % increment:
        raise ValueError("checkpoint schedule must be a positive aligned arithmetic progression")
    return list(range(first, max_epoch + 1, increment))


def expected_output(item: dict[str, Any]) -> Path:
    model = str(item["model"])
    pool = str(item["pool"])
    batch = int(item["batchSequences"])
    lr = str(item["learningRate"])
    wd = str(item["weightDecay"])
    return ALLOWED_ROOT / f"dense_{model}_{pool}" / f"bs{batch}_dr_wt_embwd_lr{lr}_wd{wd}"


def validate_coordinate(item: dict[str, Any], max_epoch: int, *, check_source: bool) -> None:
    model = str(item.get("model"))
    if model not in MODEL_SHAPES:
        raise ValueError(f"unsupported model {model}")
    batch = int(item["batchSequences"])
    if model == "1b" and batch not in {64, 128}:
        raise ValueError("1B producers are restricted to BS64/BS128")
    if model != "1b" and batch not in {128, 256}:
        raise ValueError("small producers are restricted to BS128/BS256")
    expected_lr = Decimal("1e-3") if model == "1b" else Decimal("2e-3")
    if Decimal(str(item["learningRate"])) != expected_lr:
        raise ValueError(f"{model} uses the wrong learning rate")
    if model == "1b" and Decimal(str(item["weightDecay"])) != Decimal("0.3"):
        raise ValueError("1B weight decay must remain 0.3")
    if Path(str(item["output"])) != expected_output(item):
        raise ValueError(f"coordinate output must remain {expected_output(item)}")
    source = Path(str(item["sourceCheckpoint"]))
    if source.parent != expected_output(item):
        raise ValueError("source checkpoint must remain in the coordinate's canonical output")
    pool_tokens = int(item["poolTokens"])
    expected_source = stable_step(int(item["sourceEpoch"]), pool_tokens, batch)
    if source.name != f"step{expected_source}":
        raise ValueError(f"source must be exact stable checkpoint step{expected_source}")
    targets = target_epochs(item, max_epoch)
    if not targets or targets[-1] != max_epoch:
        raise ValueError("checkpoint schedule must terminate exactly at E256")
    existing = [int(value) for value in item.get("existingScheduledEpochs", [])]
    if any(epoch not in targets and not (model == "1b" and epoch in {8, 16}) for epoch in existing):
        raise ValueError("existing scheduled epoch is outside the authorized ladder")
    if existing != sorted(set(existing)) or any(epoch > int(item["sourceEpoch"]) for epoch in existing):
        raise ValueError("existing scheduled epochs must be sorted and no later than the source")
    if check_source:
        validate_source_checkpoint(item)


def validate_source_checkpoint(item: dict[str, Any]) -> None:
    source = Path(str(item["sourceCheckpoint"]))
    config_path = source / "config.json"
    if not source.is_dir() or not config_path.is_file() or not (source / "model_and_optim").is_dir():
        raise FileNotFoundError(f"incomplete exact source checkpoint {source}")
    config = json.loads(config_path.read_text())
    model = str(item["model"])
    batch = int(item["batchSequences"])
    expected_d, expected_layers = MODEL_SHAPES[model]
    checks = {
        "d_model": (int(config["model"]["d_model"]), expected_d),
        "n_layers": (int(config["model"]["n_layers"]), expected_layers),
        "global_batch_size": (int(config["data_loader"]["global_batch_size"]), batch * SEQUENCE_LENGTH),
        "tie_embeddings": (bool(config["model"]["tie_embeddings"]), True),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    optim = config["train_module"]["optim"]
    if Decimal(str(optim["lr"])) != Decimal(str(item["learningRate"])):
        mismatches.append("learning_rate")
    if Decimal(str(optim["weight_decay"])) != Decimal(str(item["weightDecay"])):
        mismatches.append("weight_decay")
    if optim.get("group_overrides") != []:
        mismatches.append("embedding_weight_decay")
    if mismatches:
        raise RuntimeError(f"source checkpoint recipe mismatch {mismatches}: {source}")
    expected_manifest = POOL3B_MANIFEST if model == "1b" else SMALL_MANIFEST
    if config["dataset"].get("subset_manifest") != expected_manifest:
        raise RuntimeError(f"source checkpoint uses the wrong data pool: {source}")
    if int(item["sourceEpoch"]) > 1 and config["dataset"].get("dynamic_repacking") is not True:
        raise RuntimeError(f"source checkpoint is missing dynamic repacking: {source}")


def base_arguments(item: dict[str, Any]) -> list[str]:
    model = str(item["model"])
    batch = int(item["batchSequences"])
    rank_microbatch = 8 if model == "1b" else 16
    common_arguments = list(dense1b.COMMON_ARGUMENTS if model == "1b" else small.COMMON_ARGUMENTS)
    manifest = POOL3B_MANIFEST if model == "1b" else SMALL_MANIFEST
    common_arguments = common.upsert(
        common_arguments, "--dataset.subset_manifest=", f"--dataset.subset_manifest={manifest}"
    )
    common_arguments = common.upsert(
        common_arguments,
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=",
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
    )
    heldout = (dense1b.HELDOUT_EVALUATOR if model == "1b" else small.HELDOUT_EVALUATOR).replace(
        "eval_on_finish: true", "eval_on_finish: false"
    )
    common_arguments = common.upsert(
        common_arguments,
        "--trainer.callbacks.heldout_evaluator=",
        f"--trainer.callbacks.heldout_evaluator={heldout}",
    )
    model_arguments = [] if model == "1b" else list(small.MODEL_ARGUMENTS[model])
    return [
        *common_arguments,
        *model_arguments,
        f"--data_loader.global_batch_size={batch * SEQUENCE_LENGTH}",
        f"--train_module.rank_microbatch_size={rank_microbatch * SEQUENCE_LENGTH}",
        f"--train_module.optim.weight_decay={item['weightDecay']}",
        f"--lr={item['learningRate']}",
        "--model.tie_embeddings=true",
        "--decay-embeddings",
    ]


def state_dir(item: dict[str, Any]) -> Path:
    return Path(str(item["output"])) / ".constant_checkpoint_producer_v1"


def run(item: dict[str, Any], max_epoch: int) -> None:
    validate_coordinate(item, max_epoch, check_source=True)
    source = Path(str(item["sourceCheckpoint"]))
    output = Path(str(item["output"]))
    pool_tokens = int(item["poolTokens"])
    batch = int(item["batchSequences"])
    epochs = target_epochs(item, max_epoch)
    pending_epochs = [
        epoch
        for epoch in epochs
        if epoch > int(item["sourceEpoch"])
        and not (output / f"step{stable_step(epoch, pool_tokens, batch)}").is_dir()
    ]
    if not pending_epochs:
        print(f"DENSE_CHECKPOINT_PRODUCER_COMPLETE id={item['id']} source={source}", flush=True)
        return
    fixed_steps = [stable_step(epoch, pool_tokens, batch) for epoch in pending_epochs]
    target_step = fixed_steps[-1]
    name = f"{item['id']}-constant-pd-e{item['sourceEpoch']}-e{max_epoch}-v1"
    tags = (
        "[pretraining,step1,0802,checkpoint-producer,constant-lr,pre-decay,"
        f"dense-{item['model']},bs{batch},lr{item['learningRate']},wd{item['weightDecay']}]"
    )
    arguments = [
        *base_arguments(item),
        f"--save-folder={output}",
        f"--trainer.max_duration={{value: {target_step}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        f"--trainer.callbacks.wandb.tags={tags}",
        "--trainer.callbacks.checkpointer.fixed_steps=" + json.dumps(fixed_steps, separators=(",", ":")),
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        "--data_loader.restore_data_order_from_state=false",
        "--data_loader.ignore_fingerprint_mismatch=true",
        "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.ConstantScheduler}",
        "--dynamic-repacking",
        "--force_exact_trainer_load_path=true",
        f"--trainer.load_path={source}",
        "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=true",
        "--trainer.reset_data_loader_state_on_load_path=false",
        "--train_module.validate_optimizer_hyperparameters_on_load=true",
    ]
    print(
        f"DENSE_CHECKPOINT_PRODUCER_START id={item['id']} source={source} "
        f"epochs={pending_epochs} steps={fixed_steps} output={output}",
        flush=True,
    )
    subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path = state_dir(item) / "producer.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "policy": POLICY,
        "id": item["id"],
        "status": "running",
        "sourceCheckpoint": str(source),
        "targetEpochs": pending_epochs,
        "targetSteps": fixed_steps,
        "scheduler": "constant",
        "evaluationEnabled": False,
    }
    atomic_json(state_dir(item) / "producer.json", state)
    with log_path.open("a") as log_file:
        common.stream_command(
            ["torchrun", "--nproc-per-node=8", TRAINING_SCRIPT, name, *arguments], log_file
        )
    missing = [epoch for epoch, step in zip(pending_epochs, fixed_steps) if not (output / f"step{step}").is_dir()]
    if missing:
        raise RuntimeError(f"producer exited without scheduled checkpoints {missing}")
    state["status"] = "complete"
    atomic_json(state_dir(item) / "producer.json", state)
    print(f"DENSE_CHECKPOINT_PRODUCER_COMPLETE id={item['id']} epochs={pending_epochs}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coordinate", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    item = coordinate(manifest, args.coordinate)
    validate_coordinate(item, int(manifest["maxEpoch"]), check_source=not args.validate_only)
    if args.validate_only:
        print(f"validated producer {args.coordinate}")
        return
    run(item, int(manifest["maxEpoch"]))


if __name__ == "__main__":
    main()
