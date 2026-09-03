#!/usr/bin/env python3
"""Continue 474M Pool-3B BS256 from the exact E96 pre-decay checkpoint."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_small_pool3b_checkpoint_producer as producer


POLICY = "dense_474m_pool3b_bs256_e96_continuation_v1"
DEFAULT_MANIFEST = Path(
    "scripts/models/manifests/dense-474m-pool3b-bs256-e96-continuation-v1.json"
)
SOURCE_EPOCH = 96
TARGET_EPOCHS = (112, 128, 144, 160, 176, 192, 208, 224, 240, 256)
EVALUATION_EPOCHS = TARGET_EPOCHS
EXPECTED_ID = "dense-474m-dclm3b-bs256-lr2e-3-wd0.1"
EXPECTED_OUTPUT = Path(
    "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm3b/"
    "bs256_dr_wt_embwd_lr2e-3_wd0.1"
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
        or int(repeated_value["selection"]["requested_tokens"])
        != producer.TARGET_POOL_TOKENS
    ):
        raise RuntimeError("continuation repeated manifest is not sealed Pool-3B")
    if (
        int(composite_value["selection"]["requested_tokens"])
        != producer.TARGET_POOL_TOKENS
        or composite_value.get("nestedness_audit", {}).get("passed") is not True
        or audit_value.get("audit", {}).get("passed") is not True
    ):
        raise RuntimeError("Pool-3B lineage audit failed")


def validate_source(item: dict[str, Any]) -> Path:
    source = Path(str(item["sourceCheckpoint"]))
    expected = EXPECTED_OUTPUT / f"step{checkpoint_step(SOURCE_EPOCH)}"
    if source != expected or not checkpoint_complete(source):
        raise FileNotFoundError(f"incomplete exact E96 pre-decay checkpoint {expected}")
    value = json.loads((source / "config.json").read_text())
    checks = {
        "d_model": (int(value["model"]["d_model"]), 1024),
        "n_layers": (int(value["model"]["n_layers"]), 16),
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
    mismatches = [name for name, values in checks.items() if values[0] != values[1]]
    optim = value["train_module"]["optim"]
    if Decimal(str(optim["lr"])) != Decimal("2e-3"):
        mismatches.append("learning_rate")
    if Decimal(str(optim["weight_decay"])) != Decimal("0.1"):
        mismatches.append("weight_decay")
    if optim.get("group_overrides") != []:
        mismatches.append("embedding_weight_decay")
    if mismatches:
        raise RuntimeError(f"E96 continuation source mismatch {mismatches}: {source}")
    return source


def validate(
    config: dict[str, Any], item: dict[str, Any], *, check_filesystem: bool
) -> None:
    expected = {
        "policy": POLICY,
        "sourcePool": "dclm3b",
        "sourcePoolTokens": producer.TARGET_POOL_TOKENS,
        "targetPool": "dclm3b",
        "targetPoolTokens": producer.TARGET_POOL_TOKENS,
        "sourceEpoch": SOURCE_EPOCH,
        "targetEpochs": list(TARGET_EPOCHS),
        "evaluationEpochs": list(EVALUATION_EPOCHS),
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(f"continuation manifest mismatch for {key}")
    item_expected = {
        "id": EXPECTED_ID,
        "model": "474m",
        "batchSequences": 256,
        "gpuCount": 8,
        "rankMicrobatchSequences": 16,
        "gradientAccumulationSteps": 2,
        "learningRate": "2e-3",
        "weightDecay": "0.1",
        "output": str(EXPECTED_OUTPUT),
        "sourceCheckpoint": str(
            EXPECTED_OUTPUT / f"step{checkpoint_step(SOURCE_EPOCH)}"
        ),
        "baseExperiment": "01M1AAHXQ1XNX4RZ2R44RGAV4R",
    }
    for key, expected_value in item_expected.items():
        if item.get(key) != expected_value:
            raise ValueError(f"continuation coordinate mismatch for {key}")
    if Path(str(config["repeatedManifest"])) != EXPECTED_REPEATED_MANIFEST:
        raise ValueError("continuation must use the sealed repeated Pool-3B manifest")
    if check_filesystem:
        validate_pool_artifacts(config)
        validate_source(item)


def pending_state(item: dict[str, Any]) -> tuple[int, Path, list[int]]:
    source = Path(str(item["sourceCheckpoint"]))
    complete_epochs = [
        epoch
        for epoch in TARGET_EPOCHS
        if checkpoint_complete(EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}")
    ]
    for epoch in TARGET_EPOCHS:
        path = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
        if path.exists() and epoch not in complete_epochs:
            raise RuntimeError(f"refusing to overwrite incomplete checkpoint {path}")
    resume_epoch = max([SOURCE_EPOCH, *complete_epochs])
    resume = (
        source
        if resume_epoch == SOURCE_EPOCH
        else EXPECTED_OUTPUT / f"step{checkpoint_step(resume_epoch)}"
    )
    pending = [epoch for epoch in TARGET_EPOCHS if epoch > resume_epoch]
    return resume_epoch, resume, pending


def producer_arguments(
    config: dict[str, Any], item: dict[str, Any], resume: Path, pending: list[int]
) -> list[str]:
    target_steps = [checkpoint_step(epoch) for epoch in pending]
    return [
        *producer.base_arguments(item, str(config["repeatedManifest"])),
        "--dynamic-repacking",
        f"--save-folder={EXPECTED_OUTPUT}",
        f"--trainer.max_duration={{value: {target_steps[-1]}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={EXPECTED_ID}-constant-pd-e96-e256-v1",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,"
            "dense-474m,dr_wt_embwd,bs256,lr2e-3,wd0.1,constant-lr,pre-decay,"
            "e96-e256-continuation]"
        ),
        "--trainer.callbacks.checkpointer.fixed_steps="
        + json.dumps(target_steps, separators=(",", ":")),
        f"--trainer.load_path={resume}",
    ]


def run(config: dict[str, Any], item: dict[str, Any]) -> None:
    validate(config, item, check_filesystem=True)
    resume_epoch, resume, pending = pending_state(item)
    if not pending:
        print(f"DENSE_SMALL_POOL3B_PRODUCER_COMPLETE id={EXPECTED_ID}", flush=True)
        return
    state_dir = EXPECTED_OUTPUT / ".constant_checkpoint_producer_pool3b_e96_e256_v1"
    state = {
        "policy": POLICY,
        "id": EXPECTED_ID,
        "status": "running",
        "phase": "repacked_shuffled_pool3b_constant_lr",
        "sourceEpoch": SOURCE_EPOCH,
        "sourceCheckpoint": str(item["sourceCheckpoint"]),
        "resumeEpoch": resume_epoch,
        "resumeCheckpoint": str(resume),
        "output": str(EXPECTED_OUTPUT),
        "targetEpochs": list(TARGET_EPOCHS),
        "pendingEpochs": pending,
        "evaluationEpochs": list(EVALUATION_EPOCHS),
        "evaluationEnabled": False,
        "gpuCount": 8,
        "rankMicrobatchSequences": 16,
        "gradientAccumulationSteps": 2,
    }
    producer.atomic_json(state_dir / "producer.json", state)
    print(
        f"DENSE_SMALL_POOL3B_PRODUCER_START id={EXPECTED_ID} source={resume} "
        f"source_epoch={resume_epoch} epochs={pending} "
        f"manifest={config['repeatedManifest']} dynamic_repacking=true "
        "reset_data_loader=true evaluation=false",
        flush=True,
    )
    producer.run_torch(
        f"{EXPECTED_ID}-constant-pd-e96-e256-v1",
        producer_arguments(config, item, resume, pending),
        state_dir / "producer.log",
    )
    _, _, remaining = pending_state(item)
    if remaining:
        raise RuntimeError(f"continuation exited without complete checkpoints {remaining}")
    state.update({"status": "complete", "phase": "complete", "pendingEpochs": []})
    producer.atomic_json(state_dir / "producer.json", state)
    print(f"DENSE_SMALL_POOL3B_PRODUCER_COMPLETE id={EXPECTED_ID}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config, item = load(args.manifest)
    if args.validate_only:
        validate(config, item, check_filesystem=False)
        print(f"validated {EXPECTED_ID} E96-E256 producer")
        return
    run(config, item)


if __name__ == "__main__":
    main()
