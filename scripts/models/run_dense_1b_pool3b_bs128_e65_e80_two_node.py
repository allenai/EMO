#!/usr/bin/env python3
"""Resume exact 1B Pool-3B BS128 E65 PD to E80 on two 8-GPU nodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_1b_checkpoint_evaluator as evaluator
import run_dense_1b_pool3b_bs128_e32_continuation as original
import run_dense_constant_checkpoint_producer as producer

POLICY = "dense_1b_pool3b_bs128_e65_e80_two_node_integrated_v1"
DEFAULT_MANIFEST = Path(
    "scripts/models/manifests/dense-1b-pool3b-bs128-e65-e80-two-node-v1.json"
)
EXPECTED_ID = original.EXPECTED_ID
EXPECTED_OUTPUT = original.EXPECTED_OUTPUT
SOURCE_EPOCH = 65
TARGET_EPOCH = 80
CHECKPOINT_EPOCHS = tuple(range(SOURCE_EPOCH + 1, TARGET_EPOCH + 1))
STATE_DIR = EXPECTED_OUTPUT / ".dense_1b_pool3b_bs128_e65_e80_two_node_v1"
TRAINING_SCRIPT = original.TRAINING_SCRIPT


def checkpoint_step(epoch: int) -> int:
    return original.checkpoint_step(epoch)


def checkpoint_world_size(path: Path) -> int | None:
    if not (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "model_and_optim" / ".metadata").is_file()
        and any((path / "model_and_optim").glob("*.distcp"))
    ):
        return None
    ranks = sorted(
        int(value.stem.removeprefix("rank"))
        for value in (path / "train").glob("rank*.pt")
        if value.stem.removeprefix("rank").isdigit()
    )
    if ranks == list(range(8)):
        return 8
    if ranks == list(range(16)):
        return 16
    return None


def is_leader() -> bool:
    return int(os.environ.get("BEAKER_REPLICA_RANK", "0")) == 0


def load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text())
    item = config.get("producerCoordinate")
    if not isinstance(item, dict):
        raise ValueError("manifest must contain producerCoordinate")
    validate(config, item, filesystem=False)
    return config, item


def validate(config: dict[str, Any], item: dict[str, Any], *, filesystem: bool) -> None:
    expected = {
        "policy": POLICY,
        "sourcePool": "dclm3b",
        "sourcePoolTokens": 3_000_000_000,
        "sourceEpoch": SOURCE_EPOCH,
        "targetEpoch": TARGET_EPOCH,
        "checkpointIntervalEpochs": 1,
        "evaluationEpochs": [TARGET_EPOCH],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"manifest mismatch for {key}: {config.get(key)!r}")
    item_expected = {
        "id": EXPECTED_ID,
        "model": "1b",
        "pool": "dclm3b",
        "poolTokens": 3_000_000_000,
        "batchSequences": 128,
        "gpuCount": 16,
        "gpusPerNode": 8,
        "nodeCount": 2,
        "rankMicrobatchSequences": 8,
        "gradientAccumulationSteps": 1,
        "learningRate": "1e-3",
        "weightDecay": "0.3",
        "output": str(EXPECTED_OUTPUT),
        "sourceCheckpoint": str(EXPECTED_OUTPUT / f"step{checkpoint_step(SOURCE_EPOCH)}"),
    }
    for key, value in item_expected.items():
        if item.get(key) != value:
            raise ValueError(f"coordinate mismatch for {key}: {item.get(key)!r}")
    if not filesystem:
        return
    source = Path(str(item["sourceCheckpoint"]))
    if checkpoint_world_size(source) != 8:
        raise FileNotFoundError(f"incomplete exact eight-rank E65 checkpoint {source}")
    value = json.loads((source / "config.json").read_text())
    optim = value["train_module"]["optim"]
    checks = [
        int(value["data_loader"]["global_batch_size"]) == 128 * producer.SEQUENCE_LENGTH,
        bool(value["model"]["tie_embeddings"]),
        bool(value["dataset"]["dynamic_repacking"]),
        Decimal(str(optim["lr"])) == Decimal("1e-3"),
        Decimal(str(optim["weight_decay"])) == Decimal("0.3"),
        optim.get("group_overrides") == [],
    ]
    if not all(checks):
        raise RuntimeError("E65 source does not preserve the selected DR+WT+EmbedWD recipe")


def torchrun(name: str, arguments: list[str], log_path: Path, stage: str) -> None:
    rank = int(os.environ.get("BEAKER_REPLICA_RANK", "0"))
    if int(os.environ.get("NUM_NODES", "2")) != 2:
        raise RuntimeError("two-node continuation requires exactly two nodes")
    if rank == 0:
        subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    node_log = (
        log_path
        if rank == 0
        else log_path.with_name(f"{log_path.stem}.node{rank}{log_path.suffix}")
    )
    command = [
        "torchrun",
        "--nnodes=2",
        "--nproc-per-node=8",
        f"--node-rank={rank}",
        "--rdzv-backend=static",
        f"--rdzv-endpoint={os.environ['BEAKER_LEADER_REPLICA_HOSTNAME']}:{os.environ['GANTRY_RDZV_PORT']}",
        f"--rdzv-id={os.environ['GANTRY_RDZV_ID']}-{stage}",
        "--rdzv-conf=read_timeout=420",
        TRAINING_SCRIPT,
        name,
        *arguments,
    ]
    with node_log.open("a") as handle:
        producer.common.stream_command(command, handle)


def producer_arguments(item: dict[str, Any], resume: Path, pending: list[int]) -> list[str]:
    target_steps = [checkpoint_step(epoch) for epoch in pending]
    name = f"{EXPECTED_ID}-constant-pd-e65-e80-two-node-v1"
    return [
        *producer.base_arguments(item),
        f"--save-folder={EXPECTED_OUTPUT}",
        f"--trainer.max_duration={{value: {target_steps[-1]}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,dense-1b,dr_wt_embwd,bs128,lr1e-3,wd0.3,constant-lr,pre-decay,e65-e80-continuation,two-node]",
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


def wait_for(path: Path, timeout: int = 7200) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.is_file():
            return json.loads(path.read_text())
        time.sleep(5)
    raise TimeoutError(f"timed out waiting for leader result {path}")


def run(config: dict[str, Any], item: dict[str, Any]) -> None:
    validate(config, item, filesystem=True)
    complete: list[int] = []
    for epoch in range(SOURCE_EPOCH, TARGET_EPOCH + 1):
        path = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
        world_size = checkpoint_world_size(path)
        if world_size is not None:
            complete.append(epoch)
        elif path.exists():
            raise RuntimeError(f"refusing to overwrite incomplete checkpoint {path}")
    resume_epoch = max(complete)
    pending = [epoch for epoch in CHECKPOINT_EPOCHS if epoch > resume_epoch]
    if pending:
        resume = EXPECTED_OUTPUT / f"step{checkpoint_step(resume_epoch)}"
        if is_leader():
            producer.atomic_json(
                STATE_DIR / "producer.json",
                {
                    "policy": POLICY,
                    "status": "running",
                    "sourceEpoch": SOURCE_EPOCH,
                    "resumeEpoch": resume_epoch,
                    "resumeCheckpoint": str(resume),
                    "targetEpoch": TARGET_EPOCH,
                    "pendingEpochs": pending,
                    "nodeCount": 2,
                    "gpuCount": 16,
                    "rankMicrobatchSequences": 8,
                    "gradientAccumulationSteps": 1,
                },
            )
            print(
                f"DENSE_POOL3B_E80_TWO_NODE_PD_START id={EXPECTED_ID} "
                f"source_epoch={resume_epoch} source={resume} target_epoch={TARGET_EPOCH}",
                flush=True,
            )
        torchrun(
            f"{EXPECTED_ID}-constant-pd-e65-e80-two-node-v1",
            producer_arguments(item, resume, pending),
            STATE_DIR / "producer.log",
            "pd",
        )
    target = EXPECTED_OUTPUT / f"step{checkpoint_step(TARGET_EPOCH)}"
    if checkpoint_world_size(target) != 16:
        raise RuntimeError(f"producer exited without exact complete 16-rank E80 checkpoint {target}")
    if is_leader():
        print(
            f"DENSE_POOL3B_INTEGRATED_PD_RETAINED id={EXPECTED_ID} epoch=80 checkpoint={target}",
            flush=True,
        )

    result_path = original.post_result_path(item, TARGET_EPOCH)
    if not result_path.is_file():
        output = evaluator.state_dir(item) / "post_decay_runs" / f"e{TARGET_EPOCH}"
        endpoint_step = producer.total_step(
            TARGET_EPOCH, int(item["poolTokens"]), int(item["batchSequences"])
        )
        endpoint = output / f"step{endpoint_step}"
        name = f"{EXPECTED_ID}-postdecay-e80-two-node-v1"
        if endpoint.is_dir():
            log_path = evaluator.state_dir(item) / "logs" / "postdecay_e80_two_node_recovered_eval.log"
            arguments = evaluator.evaluation_arguments(
                item, endpoint, output / "eval-two-node", f"{name}-recovered-eval"
            )
        else:
            log_path = evaluator.state_dir(item) / "logs" / "postdecay_e80_two_node.log"
            arguments = evaluator.postdecay_arguments(
                item, TARGET_EPOCH, target, output, name
            )
        if is_leader():
            print(
                f"DENSE1B_CHECKPOINT_EVALUATOR_START id={EXPECTED_ID} epoch=80 "
                f"source={target} output={output}",
                flush=True,
            )
        torchrun(name, arguments, log_path, "post")
        if is_leader():
            if not endpoint.is_dir():
                raise RuntimeError(f"E80 decay exited without endpoint {endpoint}")
            result = evaluator.parse_result(log_path, item, TARGET_EPOCH, endpoint, target)
            result.update(
                {
                    "policy": POLICY,
                    "producerPolicy": POLICY,
                    "source": "integrated_two_node_wsd_decay_and_heldout_eval",
                }
            )
            producer.atomic_json(result_path, result)
    result = json.loads(result_path.read_text()) if is_leader() else wait_for(result_path)
    if is_leader():
        decision = original.evaluator_decision(config, item, TARGET_EPOCH, result)
        decision["policy"] = POLICY
        decision["removedRecoveryEpochs"] = original.cleanup_nonessential_checkpoints(
            item, TARGET_EPOCH
        )
        producer.atomic_json(original.state_dir() / "decisions" / "e80.json", decision)
        print(
            f"DENSE_POOL3B_INTEGRATED_POST_RESULT id={EXPECTED_ID} epoch=80 "
            f"json={json.dumps(result, separators=(',', ':'), sort_keys=True)}",
            flush=True,
        )
        print(
            f"DENSE_POOL3B_INTEGRATED_DECISION id={EXPECTED_ID} epoch=80 "
            f"json={json.dumps(decision, separators=(',', ':'), sort_keys=True)}",
            flush=True,
        )
        print(
            f"DENSE_POOL3B_INTEGRATED_JOB_COMPLETE id={EXPECTED_ID} "
            "terminal_epoch=80 status=terminal_e80",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config, item = load(args.manifest)
    if args.validate_only:
        print(f"validated {EXPECTED_ID} E65-E80 two-node continuation")
        return
    run(config, item)


if __name__ == "__main__":
    main()
