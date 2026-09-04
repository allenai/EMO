#!/usr/bin/env python3
"""Run a gated Dense-1B Pool-3B BS128 continuation from exact PD E32.

The integrated job checkpoints every epoch and immediately performs the
isolated WSD decay plus heldout evaluation at E48, E64, and E80 before it may
continue training. E48 stops on a strict regression from E32; E64 stops on
non-improvement from E48; E80 is the hard terminal ceiling. Once a terminal
decision is validated, recovery-only checkpoints are deleted.
"""

from __future__ import annotations

import argparse
import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_1b_checkpoint_evaluator as evaluator
import run_dense_constant_checkpoint_producer as producer

POLICY = "dense_1b_pool3b_bs128_e32_e80_integrated_v2"
DEFAULT_MANIFEST = Path("scripts/models/manifests/dense-1b-pool3b-bs128-e32-continuation-v1.json")
SOURCE_EPOCH = 32
EVALUATION_EPOCHS = (48, 64, 80)
HARD_TERMINAL_EPOCH = 80
CHECKPOINT_INTERVAL_EPOCHS = 1
EXPECTED_ID = "dense-1b-dclm3b-bs128-lr1e-3-wd0.3"
EXPECTED_OUTPUT = Path(
    "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm3b/bs128_dr_wt_embwd_lr1e-3_wd0.3"
)
EXPECTED_POOL_MANIFEST = Path(producer.POOL3B_MANIFEST)
STATE_NAME = ".dense_1b_pool3b_bs128_e32_e80_integrated_v2"
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"


def load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text())
    item = config.get("producerCoordinate")
    if not isinstance(item, dict):
        raise ValueError("continuation manifest must contain one producerCoordinate")
    validate(config, item, check_filesystem=False)
    return config, item


def checkpoint_step(epoch: int) -> int:
    return producer.stable_step(epoch, 3_000_000_000, 128)


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


def validate(config: dict[str, Any], item: dict[str, Any], *, check_filesystem: bool) -> None:
    expected = {
        "policy": POLICY,
        "sourcePool": "dclm3b",
        "sourcePoolTokens": 3_000_000_000,
        "sourceEpoch": SOURCE_EPOCH,
        "checkpointIntervalEpochs": CHECKPOINT_INTERVAL_EPOCHS,
        "evaluationEpochs": list(EVALUATION_EPOCHS),
        "hardTerminalEpoch": HARD_TERMINAL_EPOCH,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"continuation manifest mismatch for {key}")
    baseline = config.get("baselinePostDecay") or {}
    if (
        int(baseline.get("epoch", -1)) != SOURCE_EPOCH
        or Decimal(str(baseline.get("validationExact"))) != Decimal("2.78441")
        or Path(str(baseline.get("sourcePreDecayCheckpoint")))
        != EXPECTED_OUTPUT / f"step{checkpoint_step(SOURCE_EPOCH)}"
        or baseline.get("experiment") != "01M1CZVK1DAB18SBPCXBHSAT75"
    ):
        raise ValueError("E32 POST baseline provenance mismatch")
    item_expected = {
        "id": EXPECTED_ID,
        "model": "1b",
        "pool": "dclm3b",
        "poolTokens": 3_000_000_000,
        "batchSequences": 128,
        "gpuCount": 8,
        "rankMicrobatchSequences": 8,
        "gradientAccumulationSteps": 2,
        "learningRate": "1e-3",
        "weightDecay": "0.3",
        "output": str(EXPECTED_OUTPUT),
        "sourceCheckpoint": str(EXPECTED_OUTPUT / f"step{checkpoint_step(SOURCE_EPOCH)}"),
        "sourceExperiment": "01M19ZSWGEB7R6K3Y0CDXZ6G5A",
        "baseExperiment": "01M19ZSWGEB7R6K3Y0CDXZ6G5A",
    }
    for key, value in item_expected.items():
        if item.get(key) != value:
            raise ValueError(f"continuation coordinate mismatch for {key}")
    if check_filesystem:
        validate_source(item)


def validate_source(item: dict[str, Any]) -> Path:
    source = Path(str(item["sourceCheckpoint"]))
    if source != EXPECTED_OUTPUT / f"step{checkpoint_step(SOURCE_EPOCH)}":
        raise RuntimeError("continuation source is not exact PD E32")
    if not checkpoint_complete(source):
        raise FileNotFoundError(f"incomplete exact E32 pre-decay checkpoint {source}")
    producer.validate_source_checkpoint(
        {**item, "sourceEpoch": SOURCE_EPOCH, "sourceCheckpoint": str(source)}
    )
    return source


def state_dir() -> Path:
    return EXPECTED_OUTPUT / STATE_NAME


def post_result_path(item: dict[str, Any], epoch: int) -> Path:
    return evaluator.state_dir(item) / "post_decay" / f"e{epoch}.result.json"


def load_post_result(item: dict[str, Any], epoch: int) -> dict[str, Any]:
    path = post_result_path(item, epoch)
    if not path.is_file():
        raise FileNotFoundError(f"missing validated POST E{epoch} result {path}")
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


def producer_gate(config: dict[str, Any], item: dict[str, Any], target_epoch: int) -> None:
    if target_epoch == 48:
        return
    e48 = load_post_result(item, 48)
    e48_value = Decimal(str(e48["validationExact"]))
    baseline = Decimal(str(config["baselinePostDecay"]["validationExact"]))
    if e48_value > baseline:
        raise RuntimeError("E48 is worse than E32; E64 is not authorized")
    if target_epoch == 64:
        return
    e64 = load_post_result(item, 64)
    if Decimal(str(e64["validationExact"])) >= e48_value:
        raise RuntimeError("E64 is saturated relative to E48; E80 is not authorized")


def checkpoint_epochs(target_epoch: int) -> tuple[int, ...]:
    if target_epoch not in EVALUATION_EPOCHS:
        raise ValueError(f"E{target_epoch} is not an authorized producer target")
    return tuple(range(SOURCE_EPOCH + 1, target_epoch + 1))


def pending_state(item: dict[str, Any], target_epoch: int) -> tuple[int, Path, list[int]]:
    epochs = checkpoint_epochs(target_epoch)
    complete_epochs = [
        epoch
        for epoch in epochs
        if checkpoint_complete(EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}")
    ]
    for epoch in epochs:
        path = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
        if path.exists() and epoch not in complete_epochs:
            raise RuntimeError(f"refusing to overwrite incomplete checkpoint {path}")
    resume_epoch = max([SOURCE_EPOCH, *complete_epochs])
    resume = (
        Path(str(item["sourceCheckpoint"]))
        if resume_epoch == SOURCE_EPOCH
        else EXPECTED_OUTPUT / f"step{checkpoint_step(resume_epoch)}"
    )
    pending = [epoch for epoch in epochs if epoch > resume_epoch]
    return resume_epoch, resume, pending


def producer_arguments(
    item: dict[str, Any], target_epoch: int, resume: Path, pending: list[int]
) -> list[str]:
    target_steps = [checkpoint_step(epoch) for epoch in pending]
    name = f"{EXPECTED_ID}-constant-pd-e32-e{target_epoch}-dense-v1"
    return [
        *producer.base_arguments(item),
        f"--save-folder={EXPECTED_OUTPUT}",
        f"--trainer.max_duration={{value: {target_steps[-1]}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,"
            "dense-1b,dr_wt_embwd,bs128,lr1e-3,wd0.3,constant-lr,pre-decay,"
            "e32-e80-continuation,dense-checkpoint-recovery]"
        ),
        "--trainer.callbacks.checkpointer.fixed_steps="
        + json.dumps(target_steps, separators=(",", ":")),
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        "--data_loader.restore_data_order_from_state=false",
        "--data_loader.ignore_fingerprint_mismatch=true",
        "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.ConstantScheduler}",
        "--dynamic-repacking",
        "--force_exact_trainer_load_path=true",
        f"--trainer.load_path={resume}",
        "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=true",
        "--trainer.reset_data_loader_state_on_load_path=false",
        "--train_module.validate_optimizer_hyperparameters_on_load=true",
    ]


def run_producer(config: dict[str, Any], item: dict[str, Any], target_epoch: int) -> None:
    validate(config, item, check_filesystem=True)
    producer_gate(config, item, target_epoch)
    resume_epoch, resume, pending = pending_state(item, target_epoch)
    if not pending:
        print(
            f"DENSE_CHECKPOINT_PRODUCER_COMPLETE id={EXPECTED_ID} epoch={target_epoch}",
            flush=True,
        )
        return
    producer.atomic_json(
        state_dir() / "producer.json",
        {
            "policy": POLICY,
            "id": EXPECTED_ID,
            "status": "running",
            "sourceEpoch": SOURCE_EPOCH,
            "sourceCheckpoint": str(item["sourceCheckpoint"]),
            "resumeEpoch": resume_epoch,
            "resumeCheckpoint": str(resume),
            "targetEpoch": target_epoch,
            "checkpointIntervalEpochs": CHECKPOINT_INTERVAL_EPOCHS,
            "pendingEpochs": pending,
            "evaluationEpochs": list(EVALUATION_EPOCHS),
            "output": str(EXPECTED_OUTPUT),
        },
    )
    arguments = producer_arguments(item, target_epoch, resume, pending)
    name = f"{EXPECTED_ID}-constant-pd-e32-e{target_epoch}-dense-v1"
    print(
        f"DENSE_CHECKPOINT_PRODUCER_START id={EXPECTED_ID} source={resume} "
        f"source_epoch={resume_epoch} epochs={pending} output={EXPECTED_OUTPUT}",
        flush=True,
    )
    evaluator.run_torch(name, arguments, state_dir() / f"producer_e{target_epoch}.log")
    _, _, remaining = pending_state(item, target_epoch)
    if remaining:
        raise RuntimeError(f"producer exited without complete checkpoints {remaining}")
    producer.atomic_json(
        state_dir() / "producer.json",
        {
            "policy": POLICY,
            "id": EXPECTED_ID,
            "status": "complete",
            "sourceEpoch": SOURCE_EPOCH,
            "sourceCheckpoint": str(item["sourceCheckpoint"]),
            "resumeEpoch": resume_epoch,
            "resumeCheckpoint": str(resume),
            "targetEpoch": target_epoch,
            "checkpointIntervalEpochs": CHECKPOINT_INTERVAL_EPOCHS,
            "pendingEpochs": [],
            "evaluationEpochs": list(EVALUATION_EPOCHS),
            "output": str(EXPECTED_OUTPUT),
        },
    )
    print(
        f"DENSE_CHECKPOINT_PRODUCER_COMPLETE id={EXPECTED_ID} epoch={target_epoch}",
        flush=True,
    )


def cleanup_nonessential_checkpoints(item: dict[str, Any], terminal_epoch: int) -> list[int]:
    required = [epoch for epoch in EVALUATION_EPOCHS if epoch <= terminal_epoch]
    for epoch in required:
        checkpoint = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
        if not checkpoint_complete(checkpoint):
            raise RuntimeError(f"refusing cleanup before exact PD E{epoch} is complete")
        load_post_result(item, epoch)
    removed: list[int] = []
    for epoch in range(SOURCE_EPOCH + 1, terminal_epoch + 1):
        if epoch in EVALUATION_EPOCHS:
            continue
        checkpoint = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
        if not checkpoint.exists():
            continue
        if not checkpoint_complete(checkpoint):
            raise RuntimeError(f"refusing to delete incomplete checkpoint {checkpoint}")
        shutil.rmtree(checkpoint)
        removed.append(epoch)
    producer.atomic_json(
        state_dir() / "checkpoint_cleanup.json",
        {
            "policy": POLICY,
            "status": "complete",
            "terminalEpoch": terminal_epoch,
            "removedEpochs": removed,
            "preservedEvaluationEpochs": [SOURCE_EPOCH, *required],
        },
    )
    return removed


def evaluator_decision(
    config: dict[str, Any], item: dict[str, Any], epoch: int, result: dict[str, Any]
) -> dict[str, Any]:
    previous_epoch = SOURCE_EPOCH if epoch == 48 else epoch - 16
    previous_value = (
        Decimal(str(config["baselinePostDecay"]["validationExact"]))
        if previous_epoch == SOURCE_EPOCH
        else Decimal(str(load_post_result(item, previous_epoch)["validationExact"]))
    )
    current_value = Decimal(str(result["validationExact"]))
    if epoch == 48:
        terminal = current_value > previous_value
        status = "worse" if terminal else "improving"
    elif epoch == 64:
        terminal = current_value >= previous_value
        status = "saturated" if terminal else "improving"
    else:
        terminal = True
        status = "terminal_e80"
    return {
        "policy": POLICY,
        "status": status,
        "comparisonGroup": "post_decay_only",
        "criterion": "e48_strict_regression_then_strict_non_improvement",
        "epochs": [previous_epoch, epoch],
        "validationExact": {
            str(previous_epoch): float(previous_value),
            str(epoch): float(current_value),
        },
        "producerCancellationRequested": False,
        "producerStoppedAfterEpoch": epoch if terminal else None,
        "nextProducerEpoch": None if terminal else epoch + 16,
        "hardTerminalEpoch": HARD_TERMINAL_EPOCH,
    }


def run_evaluator(
    config: dict[str, Any], item: dict[str, Any], epoch: int
) -> dict[str, Any]:
    validate(config, item, check_filesystem=True)
    if epoch not in EVALUATION_EPOCHS:
        raise ValueError(f"E{epoch} is not an authorized evaluator")
    producer_gate(config, item, epoch)
    result = evaluator.run_epoch(item, epoch)
    decision = evaluator_decision(config, item, epoch, result)
    terminal_epoch = decision.get("producerStoppedAfterEpoch")
    if terminal_epoch is not None:
        decision["removedRecoveryEpochs"] = cleanup_nonessential_checkpoints(
            item, int(terminal_epoch)
        )
    producer.atomic_json(state_dir() / "decisions" / f"e{epoch}.json", decision)
    print(
        f"DENSE1B_CHECKPOINT_EVALUATOR_COMPLETE id={EXPECTED_ID} "
        f"status={decision['status']} "
        f"json={json.dumps(decision, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )
    return decision


def run_integrated(config: dict[str, Any], item: dict[str, Any]) -> None:
    """Run each constant-LR stage and its gate in one Beaker task."""
    validate(config, item, check_filesystem=True)
    for epoch in EVALUATION_EPOCHS:
        decision_path = state_dir() / "decisions" / f"e{epoch}.json"
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
        run_producer(config, item, epoch)
        checkpoint = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
        if not checkpoint_complete(checkpoint):
            raise RuntimeError(f"integrated job is missing exact PD E{epoch}")
        print(
            f"DENSE_POOL3B_INTEGRATED_PD_RETAINED id={EXPECTED_ID} "
            f"epoch={epoch} checkpoint={checkpoint}",
            flush=True,
        )
        decision = run_evaluator(config, item, epoch)
        result = load_post_result(item, epoch)
        print(
            f"DENSE_POOL3B_INTEGRATED_POST_RESULT id={EXPECTED_ID} epoch={epoch} "
            f"json={json.dumps(result, separators=(',', ':'), sort_keys=True)}",
            flush=True,
        )
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
    parser.add_argument("--mode", choices=("integrated", "producer", "evaluator"), required=True)
    parser.add_argument("--epoch", type=int, choices=EVALUATION_EPOCHS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config, item = load(args.manifest)
    if args.validate_only:
        suffix = "" if args.epoch is None else f" E{args.epoch}"
        print(f"validated {EXPECTED_ID} {args.mode}{suffix}")
        return
    if args.mode == "integrated":
        if args.epoch is not None:
            raise SystemExit("integrated mode does not accept --epoch")
        run_integrated(config, item)
    elif args.epoch is None:
        raise SystemExit("producer and evaluator modes require --epoch")
    elif args.mode == "producer":
        run_producer(config, item, args.epoch)
    else:
        run_evaluator(config, item, args.epoch)


if __name__ == "__main__":
    main()
