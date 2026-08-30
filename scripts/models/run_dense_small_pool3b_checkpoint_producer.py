#!/usr/bin/env python3
"""Bridge one small Dense model from Pool-1B E1 to Pool-3B and retain PD checkpoints.

Each coordinate loads the exact Pool-1B pre-decay E1 model, trainer, and optimizer
state.  It resets only the data-loader position and consumes the fresh, disjoint
1B-to-3B extension until the Pool-3B pre-decay E1 optimizer step.  It then resets
the data loader again, switches to the sealed flattened Pool-3B manifest with
dynamic repacking, and trains at constant LR through the requested checkpoint
ladder.  Neither stage performs decay or evaluation.
"""

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
import run_small_dense_dr_wt_embedwd_chain as small
import run_small_dense_saturation_chain as common

POLICY = "dense_small_pool3b_checkpoint_producers_v2"
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"
SEQUENCE_LENGTH = 4096
DECAY_FRACTION = 0.1
SOURCE_POOL_TOKENS = 1_000_000_000
TARGET_POOL_TOKENS = 3_000_000_000
SOURCE_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json"
ALLOWED_ROOT = Path("/weka/oe-training-default/sewonm/icsl/models")
MODEL_SHAPES = {"474m": (1024, 16), "153m": (512, 12)}
EXPECTED_COORDINATES = {
    ("474m", 128, "0.1"),
    ("474m", 128, "0.3"),
    ("474m", 256, "0.1"),
    ("474m", 256, "0.333"),
    ("153m", 128, "0.033"),
    ("153m", 128, "0.1"),
    ("153m", 256, "0.033"),
    ("153m", 256, "0.1"),
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


def expected_output(item: dict[str, Any]) -> Path:
    return (
        ALLOWED_ROOT
        / f"dense_{item['model']}_dclm3b"
        / (
            f"bs{item['batchSequences']}_dr_wt_embwd_"
            f"lr{item['learningRate']}_wd{item['weightDecay']}"
        )
    )


def expected_source(item: dict[str, Any]) -> Path:
    batch = int(item["batchSequences"])
    return (
        ALLOWED_ROOT
        / f"dense_{item['model']}_dclm1b"
        / (
            f"bs{batch}_dr_wt_embwd_"
            f"lr{item['learningRate']}_wd{item['weightDecay']}"
        )
        / f"step{stable_step(1, SOURCE_POOL_TOKENS, batch)}"
    )


def target_epochs(item: dict[str, Any], max_epoch: int) -> list[int]:
    first = int(item["firstCheckpointEpoch"])
    increment = int(item["epochIncrement"])
    if first <= 1 or increment <= 0 or first % increment:
        raise ValueError("checkpoint schedule must be an aligned arithmetic progression after E1")
    epochs = list(range(first, max_epoch + 1, increment))
    if not epochs or epochs[-1] != max_epoch:
        raise ValueError("checkpoint schedule must terminate exactly at maxEpoch")
    return epochs


def load_manifest(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("policy") != POLICY:
        raise ValueError(f"manifest policy must be {POLICY}")
    if int(config.get("sourcePoolTokens", 0)) != SOURCE_POOL_TOKENS:
        raise ValueError("source pool must remain the nominal 1B pool")
    if int(config.get("targetPoolTokens", 0)) != TARGET_POOL_TOKENS:
        raise ValueError("target pool must remain the nominal 3B pool")
    if config.get("sourcePool") != "dclm1b" or config.get("targetPool") != "dclm3b":
        raise ValueError("manifest must encode the DCLM-1B to DCLM-3B transition")
    if int(config.get("maxEpoch", 0)) != 256:
        raise ValueError("producer ceiling must remain E256")
    coordinates = config.get("producerCoordinates") or []
    if len(coordinates) != 8:
        raise ValueError("manifest must contain exactly eight replacement producers")
    actual = {
        (str(item["model"]), int(item["batchSequences"]), str(item["weightDecay"]))
        for item in coordinates
    }
    if actual != EXPECTED_COORDINATES:
        raise ValueError(f"replacement coordinate set mismatch: {sorted(actual)}")
    ids = [str(item["id"]) for item in coordinates]
    if len(ids) != len(set(ids)):
        raise ValueError("producer IDs must be unique")
    return config


def coordinate(config: dict[str, Any], coordinate_id: str) -> dict[str, Any]:
    matches = [
        item for item in config["producerCoordinates"] if item.get("id") == coordinate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one producer coordinate {coordinate_id}")
    return matches[0]


def validate_coordinate(
    config: dict[str, Any], item: dict[str, Any], *, check_filesystem: bool
) -> None:
    model = str(item["model"])
    batch = int(item["batchSequences"])
    if model not in MODEL_SHAPES or batch not in {128, 256}:
        raise ValueError("replacement producers are restricted to 474M/153M and BS128/BS256")
    if Decimal(str(item["learningRate"])) != Decimal("2e-3"):
        raise ValueError("small-model learning rate must remain 2e-3")
    if Path(str(item["sourceCheckpoint"])) != expected_source(item):
        raise ValueError(f"source must be exact Pool-1B E1 checkpoint {expected_source(item)}")
    if Path(str(item["output"])) != expected_output(item):
        raise ValueError(f"output must be isolated at {expected_output(item)}")
    if Path(str(item["sourceCheckpoint"])).parent == Path(str(item["output"])):
        raise ValueError("source and Pool-3B output directories must never be shared")
    expected_increment = 16 if model == "474m" else 32
    if (
        int(item["firstCheckpointEpoch"]) != expected_increment
        or int(item["epochIncrement"]) != expected_increment
    ):
        raise ValueError(f"{model} checkpoint schedule must advance every {expected_increment} epochs")
    target_epochs(item, int(config["maxEpoch"]))
    if check_filesystem:
        validate_source_checkpoint(item)
        validate_pool_lineage(config)


def validate_source_checkpoint(item: dict[str, Any]) -> None:
    source = Path(str(item["sourceCheckpoint"]))
    config_path = source / "config.json"
    if not source.is_dir() or not config_path.is_file() or not (source / "model_and_optim").is_dir():
        raise FileNotFoundError(f"incomplete exact source checkpoint {source}")
    value = json.loads(config_path.read_text())
    model = str(item["model"])
    batch = int(item["batchSequences"])
    d_model, layers = MODEL_SHAPES[model]
    checks = {
        "d_model": (int(value["model"]["d_model"]), d_model),
        "n_layers": (int(value["model"]["n_layers"]), layers),
        "global_batch_size": (
            int(value["data_loader"]["global_batch_size"]),
            batch * SEQUENCE_LENGTH,
        ),
        "tie_embeddings": (bool(value["model"]["tie_embeddings"]), True),
        "dynamic_repacking": (bool(value["dataset"]["dynamic_repacking"]), False),
        "source_manifest": (value["dataset"].get("subset_manifest"), SOURCE_MANIFEST),
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
        raise RuntimeError(f"source checkpoint recipe mismatch {mismatches}: {source}")


def validate_pool_lineage(config: dict[str, Any]) -> None:
    bridge = Path(str(config["bridgeManifest"]))
    repeated = Path(str(config["repeatedManifest"]))
    composite = Path(str(config["compositeManifest"]))
    audit_path = Path(str(config["poolAudit"]))
    for path in (bridge, repeated, composite, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"required sealed Pool-3B artifact is unavailable: {path}")
    bridge_value = json.loads(bridge.read_text())
    composite_value = json.loads(composite.read_text())
    repeated_value = json.loads(repeated.read_text())
    audit_value = json.loads(audit_path.read_text())
    if int(bridge_value["selection"]["requested_tokens"]) != 2_000_000_000:
        raise RuntimeError("bridge manifest must contain the fresh nominal 2B extension")
    if int(composite_value["selection"]["requested_tokens"]) != TARGET_POOL_TOKENS:
        raise RuntimeError("composite manifest must contain the nominal 3B pool")
    if composite_value.get("nestedness_audit", {}).get("passed") is not True:
        raise RuntimeError("composite Pool-3B nestedness audit failed")
    if audit_value.get("audit", {}).get("passed") is not True:
        raise RuntimeError("Pool-3B disjointness audit failed")
    flattened = repeated_value.get("flattened_from", {})
    if (
        len(repeated_value.get("entries") or []) != 1
        or int(repeated_value["entries"][0].get("start_instance", -1)) != 0
        or flattened.get("document_order_preserved") is not True
        or int(repeated_value["selection"]["requested_tokens"]) != TARGET_POOL_TOKENS
    ):
        raise RuntimeError("repeated Pool-3B manifest is not the sealed flattened lineage")


def base_arguments(item: dict[str, Any], manifest: str) -> list[str]:
    model = str(item["model"])
    batch = int(item["batchSequences"])
    arguments = list(small.COMMON_ARGUMENTS)
    arguments = common.upsert(
        arguments, "--dataset.subset_manifest=", f"--dataset.subset_manifest={manifest}"
    )
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
    heldout = small.HELDOUT_EVALUATOR.replace(
        "eval_on_finish: true", "eval_on_finish: false"
    )
    arguments = common.upsert(
        arguments,
        "--trainer.callbacks.heldout_evaluator=",
        f"--trainer.callbacks.heldout_evaluator={heldout}",
    )
    return [
        *arguments,
        *small.MODEL_ARGUMENTS[model],
        f"--data_loader.global_batch_size={batch * SEQUENCE_LENGTH}",
        f"--train_module.rank_microbatch_size={16 * SEQUENCE_LENGTH}",
        f"--train_module.optim.weight_decay={item['weightDecay']}",
        f"--lr={item['learningRate']}",
        "--model.tie_embeddings=true",
        "--decay-embeddings",
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        "--data_loader.restore_data_order_from_state=false",
        "--data_loader.ignore_fingerprint_mismatch=true",
        "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.ConstantScheduler}",
        "--force_exact_trainer_load_path=true",
        "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=true",
        "--trainer.reset_data_loader_state_on_load_path=true",
        "--train_module.validate_optimizer_hyperparameters_on_load=true",
    ]


def bridge_arguments(config: dict[str, Any], item: dict[str, Any]) -> list[str]:
    batch = int(item["batchSequences"])
    target = stable_step(1, TARGET_POOL_TOKENS, batch)
    name = f"{item['id']}-fresh-2b-bridge-to-pd-e1-v2"
    return [
        *base_arguments(item, str(config["bridgeManifest"])),
        f"--save-folder={item['output']}",
        f"--trainer.max_duration={{value: {target}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-bridge,"
            f"dense-{item['model']},dr_wt_embwd,bs{batch},lr{item['learningRate']},"
            f"wd{item['weightDecay']},fresh-2b,pre-decay-e1]"
        ),
        f"--trainer.callbacks.checkpointer.fixed_steps=[{target}]",
        f"--trainer.load_path={item['sourceCheckpoint']}",
    ]


def continuation_arguments(config: dict[str, Any], item: dict[str, Any]) -> list[str]:
    batch = int(item["batchSequences"])
    bridge_step = stable_step(1, TARGET_POOL_TOKENS, batch)
    epochs = target_epochs(item, int(config["maxEpoch"]))
    fixed_steps = [stable_step(epoch, TARGET_POOL_TOKENS, batch) for epoch in epochs]
    name = f"{item['id']}-constant-pd-e1-e{config['maxEpoch']}-v2"
    return [
        *base_arguments(item, str(config["repeatedManifest"])),
        "--dynamic-repacking",
        f"--save-folder={item['output']}",
        f"--trainer.max_duration={{value: {fixed_steps[-1]}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,"
            f"dense-{item['model']},dr_wt_embwd,bs{batch},lr{item['learningRate']},"
            f"wd{item['weightDecay']},constant-lr,pre-decay]"
        ),
        "--trainer.callbacks.checkpointer.fixed_steps="
        + json.dumps(fixed_steps, separators=(",", ":")),
        f"--trainer.load_path={item['output']}/step{bridge_step}",
    ]


def state_dir(item: dict[str, Any]) -> Path:
    return Path(str(item["output"])) / ".constant_checkpoint_producer_pool3b_v2"


def run_torch(name: str, arguments: list[str], log_path: Path) -> None:
    subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        common.stream_command(
            ["torchrun", "--nproc-per-node=8", TRAINING_SCRIPT, name, *arguments],
            log_file,
        )


def run(config: dict[str, Any], item: dict[str, Any]) -> None:
    validate_coordinate(config, item, check_filesystem=True)
    output = Path(str(item["output"]))
    batch = int(item["batchSequences"])
    bridge_step = stable_step(1, TARGET_POOL_TOKENS, batch)
    bridge_checkpoint = output / f"step{bridge_step}"
    epochs = target_epochs(item, int(config["maxEpoch"]))
    fixed_steps = [stable_step(epoch, TARGET_POOL_TOKENS, batch) for epoch in epochs]
    state = {
        "policy": POLICY,
        "id": item["id"],
        "status": "running",
        "sourceCheckpoint": item["sourceCheckpoint"],
        "sourcePool": "dclm1b",
        "targetPool": "dclm3b",
        "bridgeCheckpoint": str(bridge_checkpoint),
        "targetEpochs": epochs,
        "targetSteps": fixed_steps,
        "scheduler": "constant",
        "evaluationEnabled": False,
    }
    atomic_json(state_dir(item) / "producer.json", state)

    if not bridge_checkpoint.is_dir():
        existing = list(output.glob("step*")) if output.is_dir() else []
        if existing:
            raise RuntimeError(
                f"refusing Pool-3B bridge into output with unexpected checkpoints: {existing}"
            )
        state["phase"] = "fresh_2b_bridge_to_predecay_e1"
        atomic_json(state_dir(item) / "producer.json", state)
        arguments = bridge_arguments(config, item)
        name = f"{item['id']}-fresh-2b-bridge-to-pd-e1-v2"
        print(
            f"DENSE_SMALL_POOL3B_BRIDGE_START id={item['id']} "
            f"source={item['sourceCheckpoint']} target={bridge_checkpoint} "
            f"manifest={config['bridgeManifest']} reset_data_loader=true",
            flush=True,
        )
        run_torch(name, arguments, state_dir(item) / "bridge.log")
    if not bridge_checkpoint.is_dir():
        raise RuntimeError(f"bridge exited without exact Pool-3B E1 checkpoint {bridge_checkpoint}")
    print(
        f"DENSE_SMALL_POOL3B_BRIDGE_COMPLETE id={item['id']} checkpoint={bridge_checkpoint}",
        flush=True,
    )

    missing_epochs = [
        epoch
        for epoch, step in zip(epochs, fixed_steps)
        if not (output / f"step{step}").is_dir()
    ]
    if missing_epochs:
        state["phase"] = "repacked_shuffled_pool3b_constant_lr"
        state["pendingEpochs"] = missing_epochs
        atomic_json(state_dir(item) / "producer.json", state)
        arguments = continuation_arguments(config, item)
        name = f"{item['id']}-constant-pd-e1-e{config['maxEpoch']}-v2"
        print(
            f"DENSE_SMALL_POOL3B_PRODUCER_START id={item['id']} source={bridge_checkpoint} "
            f"epochs={missing_epochs} manifest={config['repeatedManifest']} "
            "dynamic_repacking=true reset_data_loader=true evaluation=false",
            flush=True,
        )
        run_torch(name, arguments, state_dir(item) / "producer.log")
    missing_epochs = [
        epoch
        for epoch, step in zip(epochs, fixed_steps)
        if not (output / f"step{step}").is_dir()
    ]
    if missing_epochs:
        raise RuntimeError(f"producer exited without scheduled checkpoints {missing_epochs}")
    state.update({"status": "complete", "phase": "complete", "pendingEpochs": []})
    atomic_json(state_dir(item) / "producer.json", state)
    print(
        f"DENSE_SMALL_POOL3B_PRODUCER_COMPLETE id={item['id']} epochs={epochs}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coordinate", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_manifest(args.manifest)
    item = coordinate(config, args.coordinate)
    validate_coordinate(config, item, check_filesystem=not args.validate_only)
    if args.validate_only:
        print(f"validated {args.coordinate}")
        return
    run(config, item)


if __name__ == "__main__":
    main()
