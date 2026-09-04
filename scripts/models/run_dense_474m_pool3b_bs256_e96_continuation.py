#!/usr/bin/env python3
"""Continue 474M Pool-3B BS256 from the exact E96 pre-decay checkpoint.

Checkpoint every two epochs for preemption recovery. At every 16-epoch target,
the same job immediately performs isolated WSD decay plus heldout evaluation
and stops before the next training block on strict non-improvement. Remove
recovery-only checkpoints after saturation or the E256 hard ceiling.
"""

from __future__ import annotations

import argparse
import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_small_pool3b_checkpoint_producer as producer
import run_dense_small_pool3b_checkpoint_evaluator as evaluator

POLICY = "dense_474m_pool3b_bs256_e96_integrated_v4"
RESTART_POLICY = "dense_474m_pool3b_bs256_e96_continuation_v3"
DEFAULT_MANIFEST = Path("scripts/models/manifests/dense-474m-pool3b-bs256-e96-continuation-v1.json")
SOURCE_EPOCH = 96
TARGET_EPOCHS = (112, 128, 144, 160, 176, 192, 208, 224, 240, 256)
EVALUATION_EPOCHS = TARGET_EPOCHS
CHECKPOINT_INTERVAL_EPOCHS = 2
CHECKPOINT_EPOCHS = tuple(
    range(
        SOURCE_EPOCH + CHECKPOINT_INTERVAL_EPOCHS, TARGET_EPOCHS[-1] + 1, CHECKPOINT_INTERVAL_EPOCHS
    )
)
CLEANUP_EPOCHS = tuple(epoch for epoch in CHECKPOINT_EPOCHS if epoch not in EVALUATION_EPOCHS)
EXPECTED_ID = "dense-474m-dclm3b-bs256-lr2e-3-wd0.1"
EXPECTED_OUTPUT = Path(
    "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm3b/bs256_dr_wt_embwd_lr2e-3_wd0.1"
)
STATE_NAME = ".constant_checkpoint_producer_pool3b_e96_e256_v3"
QUARANTINE_NAME = ".pre_exact_e96_restart_v3_quarantine"
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
        raise RuntimeError("continuation repeated manifest is not sealed Pool-3B")
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
        "sourceCheckpoint": str(EXPECTED_OUTPUT / f"step{checkpoint_step(SOURCE_EPOCH)}"),
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


def pending_state(item: dict[str, Any], target_epoch: int) -> tuple[int, Path, list[int]]:
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
    pending = [
        epoch for epoch in CHECKPOINT_EPOCHS if resume_epoch < epoch <= target_epoch
    ]
    return resume_epoch, resume, pending


def initialize_exact_e96_restart(state_dir: Path) -> list[int]:
    """Quarantine later checkpoints once so this lineage starts at exact E96.

    Earlier attempts could leave a complete later checkpoint in the canonical
    output.  It must not silently become the source of this explicitly-E96
    restart.  The quarantine move is reversible and the marker makes retries
    resume only checkpoints produced by this v3 lineage.
    """
    marker = state_dir / "exact_e96_restart.json"
    quarantine = EXPECTED_OUTPUT / QUARANTINE_NAME
    if marker.is_file():
        value = json.loads(marker.read_text())
        if (
            value.get("policy") != RESTART_POLICY
            or value.get("status") != "complete"
            or int(value.get("sourceEpoch", -1)) != SOURCE_EPOCH
        ):
            raise RuntimeError(f"invalid exact-E96 restart marker {marker}")
        return [int(epoch) for epoch in value.get("quarantinedEpochs", [])]

    quarantine.mkdir(parents=True, exist_ok=True)
    quarantined: list[int] = []
    for epoch in CHECKPOINT_EPOCHS:
        path = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
        destination = quarantine / path.name
        if path.exists() and destination.exists():
            raise RuntimeError(f"checkpoint exists both live and quarantined: {path}")
        if path.exists():
            path.rename(destination)
        if destination.exists():
            quarantined.append(epoch)
    producer.atomic_json(
        marker,
        {
            "policy": RESTART_POLICY,
            "status": "complete",
            "sourceEpoch": SOURCE_EPOCH,
            "sourceCheckpoint": str(
                EXPECTED_OUTPUT / f"step{checkpoint_step(SOURCE_EPOCH)}"
            ),
            "quarantinedEpochs": quarantined,
            "quarantine": str(quarantine),
        },
    )
    return quarantined


def result_path(item: dict[str, Any], epoch: int) -> Path:
    return evaluator.state_dir(item, str(EXPECTED_OUTPUT)) / "results" / f"e{epoch}.json"


def load_post_result(item: dict[str, Any], epoch: int) -> dict[str, Any]:
    path = result_path(item, epoch)
    if not path.is_file():
        raise FileNotFoundError(f"missing POST E{epoch} result {path}")
    result = json.loads(path.read_text())
    if (
        result.get("status") != "complete"
        or result.get("comparisonGroup") != "post_decay"
        or int(result.get("epoch", -1)) != epoch
        or Path(str(result.get("sourcePreDecayCheckpoint")))
        != EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
    ):
        raise RuntimeError(f"unhealthy or unmatched POST E{epoch} result {path}")
    Decimal(str(result["validationExact"]))
    return result


def cleanup_nonessential_checkpoints(
    item: dict[str, Any], state_dir: Path, terminal_epoch: int
) -> list[int]:
    essential = [epoch for epoch in EVALUATION_EPOCHS if epoch <= terminal_epoch]
    missing_essential = [
        epoch
        for epoch in essential
        if not checkpoint_complete(EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}")
    ]
    if missing_essential:
        raise RuntimeError(
            f"refusing checkpoint cleanup before essential checkpoints are complete: {missing_essential}"
        )
    removed: list[int] = []
    load_post_result(item, SOURCE_EPOCH)
    for epoch in essential:
        load_post_result(item, epoch)
    keep = {SOURCE_EPOCH, *essential}
    for epoch in CHECKPOINT_EPOCHS:
        if epoch in keep:
            continue
        path = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
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
            "terminalEpoch": terminal_epoch,
            "removedEpochs": removed,
            "preservedEvaluationEpochs": [SOURCE_EPOCH, *essential],
        },
    )
    return removed


def producer_arguments(
    config: dict[str, Any], item: dict[str, Any], resume: Path, pending: list[int]
) -> list[str]:
    target_steps = [checkpoint_step(epoch) for epoch in pending]
    return [
        *producer.base_arguments(item, str(config["repeatedManifest"])),
        "--dynamic-repacking",
        f"--save-folder={EXPECTED_OUTPUT}",
        f"--trainer.max_duration={{value: {target_steps[-1]}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={EXPECTED_ID}-integrated-e96-e256-v4",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,"
            "dense-474m,dr_wt_embwd,bs256,lr2e-3,wd0.1,constant-lr,pre-decay,"
            "e96-e256-continuation,dense-checkpoint-recovery]"
        ),
        "--trainer.callbacks.checkpointer.fixed_steps="
        + json.dumps(target_steps, separators=(",", ":")),
        f"--trainer.load_path={resume}",
    ]


def run_producer_stage(
    config: dict[str, Any],
    item: dict[str, Any],
    target_epoch: int,
    state_dir: Path,
    quarantined: list[int],
) -> None:
    resume_epoch, resume, pending = pending_state(item, target_epoch)
    if not pending:
        return
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
        "checkpointIntervalEpochs": CHECKPOINT_INTERVAL_EPOCHS,
        "checkpointEpochs": list(CHECKPOINT_EPOCHS),
        "pendingEpochs": pending,
        "evaluationEpochs": list(EVALUATION_EPOCHS),
        "evaluationEnabled": True,
        "integratedEvaluation": True,
        "stageTargetEpoch": target_epoch,
        "quarantinedPreRestartEpochs": quarantined,
        "gpuCount": 8,
        "rankMicrobatchSequences": 16,
        "gradientAccumulationSteps": 2,
    }
    producer.atomic_json(state_dir / "producer.json", state)
    print(
        f"DENSE_SMALL_POOL3B_PRODUCER_START id={EXPECTED_ID} source={resume} "
        f"source_epoch={resume_epoch} epochs={pending} "
        f"manifest={config['repeatedManifest']} dynamic_repacking=true "
        f"reset_data_loader=true integrated_evaluation_at={target_epoch}",
        flush=True,
    )
    producer.run_torch(
        f"{EXPECTED_ID}-integrated-e96-e256-v4",
        producer_arguments(config, item, resume, pending),
        state_dir / "producer.log",
    )
    _, _, remaining = pending_state(item, target_epoch)
    if remaining:
        raise RuntimeError(f"continuation exited without complete checkpoints {remaining}")
    state.update({"status": "stage_complete", "phase": "post", "pendingEpochs": []})
    producer.atomic_json(state_dir / "producer.json", state)


def decision_for(
    item: dict[str, Any], epoch: int, result: dict[str, Any]
) -> dict[str, Any]:
    previous_epoch = epoch - 16
    previous = load_post_result(item, previous_epoch)
    previous_value = Decimal(str(previous["validationExact"]))
    current_value = Decimal(str(result["validationExact"]))
    saturated = current_value >= previous_value
    terminal = saturated or epoch == EVALUATION_EPOCHS[-1]
    return {
        "policy": POLICY,
        "status": "saturated" if saturated else "terminal_e256" if terminal else "improving",
        "comparisonGroup": "post_decay_only",
        "criterion": "strict_non_improvement",
        "epochs": [previous_epoch, epoch],
        "validationExact": {
            str(previous_epoch): float(previous_value),
            str(epoch): float(current_value),
        },
        "producerStoppedAfterEpoch": epoch if terminal else None,
        "nextProducerEpoch": None if terminal else epoch + 16,
        "hardTerminalEpoch": EVALUATION_EPOCHS[-1],
    }


def run(config: dict[str, Any], item: dict[str, Any]) -> None:
    validate(config, item, check_filesystem=True)
    state_dir = EXPECTED_OUTPUT / STATE_NAME
    quarantined = initialize_exact_e96_restart(state_dir)
    for epoch in EVALUATION_EPOCHS:
        decision_path = state_dir / "decisions" / f"e{epoch}.json"
        if decision_path.is_file():
            decision = json.loads(decision_path.read_text())
            if decision.get("nextProducerEpoch") is None:
                print(
                    f"DENSE_POOL3B_INTEGRATED_JOB_COMPLETE id={EXPECTED_ID} "
                    f"terminal_epoch={epoch} status={decision['status']}",
                    flush=True,
                )
                return
            continue
        run_producer_stage(config, item, epoch, state_dir, quarantined)
        checkpoint = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
        if not checkpoint_complete(checkpoint):
            raise RuntimeError(f"integrated job is missing exact PD E{epoch}")
        print(
            f"DENSE_POOL3B_INTEGRATED_PD_RETAINED id={EXPECTED_ID} "
            f"epoch={epoch} checkpoint={checkpoint}",
            flush=True,
        )
        result = evaluator.run(config, item, epoch, str(EXPECTED_OUTPUT))
        print(
            f"DENSE_POOL3B_INTEGRATED_POST_RESULT id={EXPECTED_ID} epoch={epoch} "
            f"json={json.dumps(result, separators=(',', ':'), sort_keys=True)}",
            flush=True,
        )
        decision = decision_for(item, epoch, result)
        if decision.get("nextProducerEpoch") is None:
            decision["removedRecoveryEpochs"] = cleanup_nonessential_checkpoints(
                item, state_dir, epoch
            )
        producer.atomic_json(decision_path, decision)
        print(
            f"DENSE_POOL3B_INTEGRATED_DECISION id={EXPECTED_ID} epoch={epoch} "
            f"json={json.dumps(decision, separators=(',', ':'), sort_keys=True)}",
            flush=True,
        )
        if decision.get("nextProducerEpoch") is None:
            print(
                f"DENSE_POOL3B_INTEGRATED_JOB_COMPLETE id={EXPECTED_ID} "
                f"terminal_epoch={epoch} status={decision['status']}",
                flush=True,
            )
            return
    raise RuntimeError("integrated continuation reached no terminal decision")


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
