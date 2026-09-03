#!/usr/bin/env python3
"""Continue the selected 153M Pool-3B BS256 producer from E256 to E384.

The producer loads the exact clean E256 pre-decay checkpoint, preserves the
model/trainer/optimizer state, repeats the sealed Pool-3B data with dynamic
repacking, and checkpoints every four epochs for preemption recovery.
Evaluations run as isolated WSD branches only at E320 and E384 and never
advance the producer.  Once E384 is complete, recovery-only checkpoints are
deleted and the evaluation checkpoints are preserved.
"""

from __future__ import annotations

import argparse
import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_small_pool3b_checkpoint_evaluator as evaluator
import run_dense_small_pool3b_checkpoint_producer as producer

POLICY = "dense_small_pool3b_bs256_e384_continuation_v2"
DEFAULT_MANIFEST = Path(
    "scripts/models/manifests/dense-small-pool3b-bs256-e384-continuation-v1.json"
)
SOURCE_EPOCH = 256
TARGET_EPOCHS = (320, 384)
EVALUATION_EPOCHS = (320, 384)
CHECKPOINT_INTERVAL_EPOCHS = 4
CHECKPOINT_EPOCHS = tuple(
    range(
        SOURCE_EPOCH + CHECKPOINT_INTERVAL_EPOCHS, TARGET_EPOCHS[-1] + 1, CHECKPOINT_INTERVAL_EPOCHS
    )
)
CLEANUP_EPOCHS = tuple(epoch for epoch in CHECKPOINT_EPOCHS if epoch not in EVALUATION_EPOCHS)
EXPECTED_ID = "dense-153m-dclm3b-bs256-lr2e-3-wd0.1"
EXPECTED_OUTPUT = Path(
    "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm3b/"
    "bs256_dr_wt_embwd_lr2e-3_wd0.1_throughput_recovery_r2"
)
EXPECTED_CANONICAL_OUTPUT = Path(
    "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm3b/bs256_dr_wt_embwd_lr2e-3_wd0.1"
)
EXPECTED_REPEATED_MANIFEST = Path(
    "/weka/oe-training-default/sewonm/icsl/data/dclm_0802_nested_1b_3b_9b/"
    "manifests/dclm_0802_nested_train_3b_flat_dynamic_repacking.json"
)


def load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text())
    item = config.get("producerCoordinate")
    if not isinstance(item, dict):
        raise ValueError("continuation manifest must contain one producerCoordinate")
    validate(config, item, check_filesystem=False)
    return config, item


def checkpoint_step(epoch: int) -> int:
    return producer.stable_step(epoch, producer.TARGET_POOL_TOKENS, 256)


def checkpoint_complete(path: Path) -> bool:
    model_and_optim = path / "model_and_optim"
    train_state = path / "train"
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and model_and_optim.is_dir()
        and (model_and_optim / ".metadata").is_file()
        and any(model_and_optim.glob("*.distcp"))
        and train_state.is_dir()
        and all((train_state / f"rank{rank}.pt").is_file() for rank in range(8))
    )


def validate_pool_artifacts(config: dict[str, Any]) -> None:
    repeated = Path(str(config["repeatedManifest"]))
    composite = Path(str(config["compositeManifest"]))
    audit = Path(str(config["poolAudit"]))
    for path in (repeated, composite, audit):
        if not path.is_file():
            raise FileNotFoundError(f"missing sealed Pool-3B artifact {path}")
    repeated_value = json.loads(repeated.read_text())
    composite_value = json.loads(composite.read_text())
    audit_value = json.loads(audit.read_text())
    flattened = repeated_value.get("flattened_from", {})
    if (
        repeated != EXPECTED_REPEATED_MANIFEST
        or len(repeated_value.get("entries") or []) != 1
        or int(repeated_value["entries"][0].get("start_instance", -1)) != 0
        or flattened.get("document_order_preserved") is not True
        or int(repeated_value["selection"]["requested_tokens"]) != producer.TARGET_POOL_TOKENS
    ):
        raise RuntimeError("continuation repeated manifest is not the sealed Pool-3B lineage")
    if (
        int(composite_value["selection"]["requested_tokens"]) != producer.TARGET_POOL_TOKENS
        or composite_value.get("nestedness_audit", {}).get("passed") is not True
        or audit_value.get("audit", {}).get("passed") is not True
    ):
        raise RuntimeError("Pool-3B lineage audit failed")


def validate_source(item: dict[str, Any]) -> Path:
    source = Path(str(item["sourceCheckpoint"]))
    expected = EXPECTED_OUTPUT / f"step{checkpoint_step(SOURCE_EPOCH)}"
    if source != expected or not checkpoint_complete(source):
        raise FileNotFoundError(f"incomplete exact E256 pre-decay checkpoint {expected}")
    value = json.loads((source / "config.json").read_text())
    checks = {
        "d_model": (int(value["model"]["d_model"]), 512),
        "n_layers": (int(value["model"]["n_layers"]), 12),
        "global_batch_size": (
            int(value["data_loader"]["global_batch_size"]),
            256 * producer.SEQUENCE_LENGTH,
        ),
        "tie_embeddings": (bool(value["model"]["tie_embeddings"]), True),
        "dynamic_repacking": (bool(value["dataset"]["dynamic_repacking"]), True),
        "subset_manifest": (
            Path(str(value["dataset"]["subset_manifest"])),
            EXPECTED_REPEATED_MANIFEST,
        ),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    optim = value["train_module"]["optim"]
    if Decimal(str(optim["lr"])) != Decimal("2e-3"):
        mismatches.append("learning_rate")
    if Decimal(str(optim["weight_decay"])) != Decimal("0.1"):
        mismatches.append("weight_decay")
    if optim.get("group_overrides") != []:
        mismatches.append("embedding_weight_decay")
    if mismatches:
        raise RuntimeError(f"E256 continuation source mismatch {mismatches}: {source}")
    return source


def validate(config: dict[str, Any], item: dict[str, Any], *, check_filesystem: bool) -> None:
    expected = {
        "policy": POLICY,
        "sourcePool": "dclm3b",
        "sourcePoolTokens": producer.TARGET_POOL_TOKENS,
        "targetPool": "dclm3b",
        "targetPoolTokens": producer.TARGET_POOL_TOKENS,
        "sourceEpoch": SOURCE_EPOCH,
        "targetEpochs": list(TARGET_EPOCHS),
        "evaluationEpochs": list(EVALUATION_EPOCHS),
        "checkpointIntervalEpochs": CHECKPOINT_INTERVAL_EPOCHS,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"continuation manifest mismatch for {key}")
    item_expected = {
        "id": EXPECTED_ID,
        "model": "153m",
        "batchSequences": 256,
        "gpuCount": 8,
        "rankMicrobatchSequences": 16,
        "gradientAccumulationSteps": 2,
        "learningRate": "2e-3",
        "weightDecay": "0.1",
        "canonicalOutput": str(EXPECTED_CANONICAL_OUTPUT),
        "output": str(EXPECTED_OUTPUT),
        "sourceCheckpoint": str(EXPECTED_OUTPUT / f"step{checkpoint_step(SOURCE_EPOCH)}"),
        "baseExperiment": "01KZ6Q4DJ8J994A6SQ39MEGTZ2",
    }
    for key, value in item_expected.items():
        if item.get(key) != value:
            raise ValueError(f"continuation coordinate mismatch for {key}")
    if Path(str(config["repeatedManifest"])) != EXPECTED_REPEATED_MANIFEST:
        raise ValueError("continuation must use the exact sealed repeated Pool-3B manifest")
    if check_filesystem:
        validate_pool_artifacts(config)
        validate_source(item)


def pending_state(item: dict[str, Any]) -> tuple[int, Path, list[int]]:
    source = Path(str(item["sourceCheckpoint"]))
    complete_epochs = [
        epoch
        for epoch in CHECKPOINT_EPOCHS
        if checkpoint_complete(EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}")
    ]
    for epoch in CHECKPOINT_EPOCHS:
        path = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
        if path.exists() and epoch not in complete_epochs:
            raise RuntimeError(f"refusing to overwrite incomplete checkpoint {path}")
    resume_epoch = max([SOURCE_EPOCH, *complete_epochs])
    resume = (
        source
        if resume_epoch == SOURCE_EPOCH
        else EXPECTED_OUTPUT / f"step{checkpoint_step(resume_epoch)}"
    )
    pending = [epoch for epoch in CHECKPOINT_EPOCHS if epoch > resume_epoch]
    return resume_epoch, resume, pending


def cleanup_nonessential_checkpoints(output: Path, state_dir: Path) -> list[int]:
    missing_essential = [
        epoch
        for epoch in EVALUATION_EPOCHS
        if not checkpoint_complete(output / f"step{checkpoint_step(epoch)}")
    ]
    if missing_essential:
        raise RuntimeError(
            f"refusing checkpoint cleanup before essential checkpoints are complete: {missing_essential}"
        )
    removed: list[int] = []
    for epoch in CLEANUP_EPOCHS:
        path = output / f"step{checkpoint_step(epoch)}"
        if not path.exists():
            continue
        if not checkpoint_complete(path):
            raise RuntimeError(f"refusing to delete incomplete checkpoint {path}")
        shutil.rmtree(path)
        removed.append(epoch)
    producer.atomic_json(
        state_dir / "checkpoint_cleanup.json",
        {
            "policy": POLICY,
            "status": "complete",
            "removedEpochs": removed,
            "preservedEvaluationEpochs": list(EVALUATION_EPOCHS),
        },
    )
    return removed


def producer_arguments(
    config: dict[str, Any], item: dict[str, Any], resume: Path, pending: list[int]
) -> list[str]:
    output = Path(str(item["output"]))
    target_steps = [checkpoint_step(epoch) for epoch in pending]
    name = f"{EXPECTED_ID}-constant-pd-e256-e384-v2"
    return [
        *producer.base_arguments(item, str(config["repeatedManifest"])),
        "--dynamic-repacking",
        f"--save-folder={output}",
        f"--trainer.max_duration={{value: {target_steps[-1]}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,"
            "dense-153m,dr_wt_embwd,bs256,lr2e-3,wd0.1,constant-lr,"
            "pre-decay,e256-e384-continuation,dense-checkpoint-recovery]"
        ),
        "--trainer.callbacks.checkpointer.fixed_steps="
        + json.dumps(target_steps, separators=(",", ":")),
        f"--trainer.load_path={resume}",
    ]


def run_producer(config: dict[str, Any], item: dict[str, Any]) -> None:
    validate(config, item, check_filesystem=True)
    output = Path(str(item["output"]))
    source = Path(str(item["sourceCheckpoint"]))
    state_dir = output / ".constant_checkpoint_producer_pool3b_e256_e384_v2"
    resume_epoch, resume, pending = pending_state(item)
    if not pending:
        cleanup_nonessential_checkpoints(output, state_dir)
        print(
            f"DENSE_SMALL_POOL3B_PRODUCER_COMPLETE id={EXPECTED_ID} epochs={list(TARGET_EPOCHS)}",
            flush=True,
        )
        return
    producer.atomic_json(
        state_dir / "producer.json",
        {
            "policy": POLICY,
            "id": EXPECTED_ID,
            "status": "running",
            "phase": "repacked_shuffled_pool3b_constant_lr",
            "sourceEpoch": SOURCE_EPOCH,
            "sourceCheckpoint": str(source),
            "resumeEpoch": resume_epoch,
            "resumeCheckpoint": str(resume),
            "output": str(output),
            "targetEpochs": list(TARGET_EPOCHS),
            "checkpointIntervalEpochs": CHECKPOINT_INTERVAL_EPOCHS,
            "checkpointEpochs": list(CHECKPOINT_EPOCHS),
            "checkpointSteps": [checkpoint_step(epoch) for epoch in CHECKPOINT_EPOCHS],
            "pendingEpochs": pending,
            "evaluationEpochs": list(EVALUATION_EPOCHS),
            "evaluationEnabled": False,
            "gpuCount": 8,
            "rankMicrobatchSequences": 16,
            "gradientAccumulationSteps": 2,
        },
    )
    print(
        f"DENSE_SMALL_POOL3B_PRODUCER_START id={EXPECTED_ID} source={resume} "
        f"source_epoch={resume_epoch} checkpoint_epochs={pending} "
        f"evaluation_epochs={list(EVALUATION_EPOCHS)} manifest={config['repeatedManifest']} "
        "dynamic_repacking=true reset_data_loader=true evaluation=false",
        flush=True,
    )
    name = f"{EXPECTED_ID}-constant-pd-e256-e384-v2"
    producer.run_torch(
        name,
        producer_arguments(config, item, resume, pending),
        state_dir / "producer.log",
    )
    _, _, remaining = pending_state(item)
    if remaining:
        raise RuntimeError(f"continuation exited without complete checkpoints {remaining}")
    removed = cleanup_nonessential_checkpoints(output, state_dir)
    producer.atomic_json(
        state_dir / "producer.json",
        {
            "policy": POLICY,
            "id": EXPECTED_ID,
            "status": "complete",
            "phase": "complete",
            "sourceEpoch": SOURCE_EPOCH,
            "sourceCheckpoint": str(source),
            "resumeEpoch": resume_epoch,
            "resumeCheckpoint": str(resume),
            "output": str(output),
            "targetEpochs": list(TARGET_EPOCHS),
            "checkpointIntervalEpochs": CHECKPOINT_INTERVAL_EPOCHS,
            "checkpointEpochs": list(CHECKPOINT_EPOCHS),
            "checkpointSteps": [checkpoint_step(epoch) for epoch in CHECKPOINT_EPOCHS],
            "pendingEpochs": [],
            "evaluationEpochs": list(EVALUATION_EPOCHS),
            "removedRecoveryEpochs": removed,
            "evaluationEnabled": False,
            "gpuCount": 8,
            "rankMicrobatchSequences": 16,
            "gradientAccumulationSteps": 2,
        },
    )
    print(
        f"DENSE_SMALL_POOL3B_PRODUCER_COMPLETE id={EXPECTED_ID} epochs={list(TARGET_EPOCHS)}",
        flush=True,
    )


def run_evaluator(config: dict[str, Any], item: dict[str, Any], epoch: int) -> None:
    validate(config, item, check_filesystem=True)
    if epoch not in EVALUATION_EPOCHS:
        raise ValueError(f"E{epoch} is not an authorized continuation evaluation")
    evaluator.run(config, item, epoch, str(EXPECTED_OUTPUT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--evaluate-epoch", type=int, choices=EVALUATION_EPOCHS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config, item = load(args.manifest)
    if args.validate_only:
        validate(config, item, check_filesystem=False)
        mode = f"POST E{args.evaluate_epoch}" if args.evaluate_epoch else "producer"
        print(f"validated {EXPECTED_ID} E256-E384 {mode}")
        return
    if args.evaluate_epoch is not None:
        run_evaluator(config, item, args.evaluate_epoch)
    else:
        run_producer(config, item)


if __name__ == "__main__":
    main()
