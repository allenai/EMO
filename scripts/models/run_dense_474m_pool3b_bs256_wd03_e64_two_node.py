#!/usr/bin/env python3
"""Continue exact 474M Pool-3B BS256 WD0.3 E32 PD to E64 on two nodes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_1b_dr_wt_embedwd_grid as dense1b
import run_dense_small_pool3b_checkpoint_evaluator as evaluator
import run_dense_small_pool3b_checkpoint_producer as producer

POLICY = "dense_474m_pool3b_bs256_wd03_e64_two_node_v1"
DEFAULT_MANIFEST = Path(
    "scripts/models/manifests/dense-474m-pool3b-bs256-wd0.3-e32-e64-two-node-v1.json"
)
EXPECTED_ID = "dense-474m-dclm3b-bs256-lr2e-3-wd0.3-probe-v1"
EXPECTED_OUTPUT = Path(
    "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm3b/"
    "bs256_dr_wt_embwd_lr2e-3_wd0.3"
)
SOURCE_EPOCH = 32
TARGET_EPOCH = 64
CHECKPOINT_EPOCHS = tuple(range(36, 65, 4))
STATE_DIR = EXPECTED_OUTPUT / ".dense_474m_pool3b_bs256_wd03_e64_two_node_v1"
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"


def checkpoint_step(epoch: int) -> int:
    return producer.stable_step(epoch, producer.TARGET_POOL_TOKENS, 256)


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
        "sourcePoolTokens": producer.TARGET_POOL_TOKENS,
        "targetPoolTokens": producer.TARGET_POOL_TOKENS,
        "sourceEpoch": SOURCE_EPOCH,
        "targetEpochs": [TARGET_EPOCH],
        "evaluationEpochs": [TARGET_EPOCH],
        "checkpointIntervalEpochs": 4,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"manifest mismatch for {key}: {config.get(key)!r}")
    item_expected = {
        "id": EXPECTED_ID,
        "model": "474m",
        "pool": "dclm3b",
        "batchSequences": 256,
        "gpuCount": 16,
        "gpusPerNode": 8,
        "nodeCount": 2,
        "rankMicrobatchSequences": 16,
        "gradientAccumulationSteps": 1,
        "learningRate": "2e-3",
        "weightDecay": "0.3",
        "output": str(EXPECTED_OUTPUT),
        "sourceCheckpoint": str(EXPECTED_OUTPUT / f"step{checkpoint_step(SOURCE_EPOCH)}"),
    }
    for key, value in item_expected.items():
        if item.get(key) != value:
            raise ValueError(f"coordinate mismatch for {key}: {item.get(key)!r}")
    if not filesystem:
        return
    source = Path(item["sourceCheckpoint"])
    if checkpoint_world_size(source) != 8:
        raise FileNotFoundError(f"incomplete exact eight-rank E32 source {source}")
    value = json.loads((source / "config.json").read_text())
    optim = value["train_module"]["optim"]
    checks = [
        int(value["model"]["d_model"]) == 1024,
        int(value["model"]["n_layers"]) == 16,
        int(value["data_loader"]["global_batch_size"]) == 256 * producer.SEQUENCE_LENGTH,
        bool(value["model"]["tie_embeddings"]),
        bool(value["dataset"]["dynamic_repacking"]),
        Decimal(str(optim["lr"])) == Decimal("2e-3"),
        Decimal(str(optim["weight_decay"])) == Decimal("0.3"),
        optim.get("group_overrides") == [],
    ]
    if not all(checks):
        raise RuntimeError("E32 source does not preserve the selected DR+WT+EmbedWD recipe")


def torchrun(name: str, arguments: list[str], log_path: Path, stage: str) -> None:
    rank = int(os.environ.get("BEAKER_REPLICA_RANK", "0"))
    if int(os.environ.get("NUM_NODES", "2")) != 2:
        raise RuntimeError("E64 continuation requires exactly two nodes")
    if rank == 0:
        subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    node_log = log_path if rank == 0 else log_path.with_name(
        f"{log_path.stem}.node{rank}{log_path.suffix}"
    )
    command = [
        "torchrun", "--nnodes=2", "--nproc-per-node=8", f"--node-rank={rank}",
        "--rdzv-backend=static",
        f"--rdzv-endpoint={os.environ['BEAKER_LEADER_REPLICA_HOSTNAME']}:{os.environ['GANTRY_RDZV_PORT']}",
        f"--rdzv-id={os.environ['GANTRY_RDZV_ID']}-{stage}",
        "--rdzv-conf=read_timeout=420", TRAINING_SCRIPT, name, *arguments,
    ]
    with node_log.open("a") as handle:
        producer.common.stream_command(command, handle)


def producer_arguments(config: dict[str, Any], item: dict[str, Any], resume: Path, pending: list[int]) -> list[str]:
    target_steps = [checkpoint_step(epoch) for epoch in pending]
    arguments = producer.base_arguments(item, str(config["repeatedManifest"]))
    arguments = producer.common.upsert(
        arguments,
        "--data_loader.restore_data_order_from_state=",
        "--data_loader.restore_data_order_from_state=true",
    )
    arguments = producer.common.upsert(
        arguments,
        "--trainer.reset_data_loader_state_on_load_path=",
        "--trainer.reset_data_loader_state_on_load_path=false",
    )
    return [
        *arguments,
        "--dynamic-repacking",
        f"--save-folder={EXPECTED_OUTPUT}",
        f"--trainer.max_duration={{value: {target_steps[-1]}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={EXPECTED_ID}-constant-pd-e32-e64-two-node-v1",
        "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,dense-474m,dr_wt_embwd,bs256,lr2e-3,wd0.3,constant-lr,pre-decay,e32-e64-continuation,two-node]",
        "--trainer.callbacks.checkpointer.fixed_steps=" + json.dumps(target_steps, separators=(",", ":")),
        f"--trainer.load_path={resume}",
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
    for epoch in range(SOURCE_EPOCH, TARGET_EPOCH + 1, 4):
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
            producer.atomic_json(STATE_DIR / "producer.json", {
                "policy": POLICY,
                "status": "running",
                "sourceEpoch": SOURCE_EPOCH,
                "resumeEpoch": resume_epoch,
                "resumeCheckpoint": str(resume),
                "targetEpoch": TARGET_EPOCH,
                "pendingEpochs": pending,
                "nodeCount": 2,
                "gpuCount": 16,
            })
            print(
                f"DENSE474M_POOL3B_WD03_E64_PD_START id={EXPECTED_ID} "
                f"source_epoch={resume_epoch} source={resume} target_epoch={TARGET_EPOCH}",
                flush=True,
            )
        torchrun(
            f"{EXPECTED_ID}-constant-pd-e32-e64-two-node-v1",
            producer_arguments(config, item, resume, pending),
            STATE_DIR / "producer.log",
            "pd",
        )
    target = EXPECTED_OUTPUT / f"step{checkpoint_step(TARGET_EPOCH)}"
    if checkpoint_world_size(target) != 16:
        raise RuntimeError(f"producer exited without exact complete 16-rank E64 checkpoint {target}")
    if is_leader():
        print(f"DENSE474M_POOL3B_WD03_E64_PD_RETAINED id={EXPECTED_ID} epoch=64 checkpoint={target}", flush=True)

    result_path = evaluator.state_dir(item) / "results" / "e64.json"
    if not result_path.is_file():
        post_output = evaluator.state_dir(item) / "post_decay_runs" / "e64"
        endpoint = post_output / f"step{producer.total_step(TARGET_EPOCH, producer.TARGET_POOL_TOKENS, 256)}"
        name = f"{EXPECTED_ID}-post-e64-two-node-v1"
        log_path = evaluator.state_dir(item) / "logs" / "e64_two_node.log"
        arguments = (
            evaluator.evaluation_arguments(config, item, endpoint, post_output / "eval-two-node", f"{name}-recovered-eval")
            if endpoint.is_dir()
            else evaluator.postdecay_arguments(config, item, TARGET_EPOCH, target, post_output, name)
        )
        if is_leader():
            print(f"DENSE_SMALL_CHECKPOINT_EVALUATOR_START id={EXPECTED_ID} epoch=64 source={target} output={post_output}", flush=True)
        torchrun(name, arguments, log_path, "post")
        if is_leader():
            if not endpoint.is_dir():
                raise RuntimeError(f"E64 decay exited without endpoint {endpoint}")
            result = dense1b.parse_validation(log_path, TARGET_EPOCH, "post_decay", endpoint)
            result.update({
                "policy": POLICY,
                "producerPolicy": POLICY,
                "model": "474m",
                "batchSequences": 256,
                "lr": "2e-3",
                "wd": "0.3",
                "variant": "DR+WT+EmbedWD",
                "dynamicRepacking": True,
                "weightTying": True,
                "decayEmbeddings": True,
                "sourcePreDecayCheckpoint": str(target),
                "source": "integrated_two_node_wsd_decay_and_heldout_eval",
            })
            evaluator.atomic_json(result_path, result)
    result = json.loads(result_path.read_text()) if is_leader() else wait_for(result_path)
    if is_leader():
        previous = json.loads((evaluator.state_dir(item) / "results" / "e32.json").read_text())
        decision = {
            "policy": POLICY,
            "status": "terminal_e64",
            "comparisonGroup": "post_decay_only",
            "criterion": "user_authorized_e64_endpoint",
            "epochs": [32, 64],
            "validationExact": {"32": previous["validationExact"], "64": result["validationExact"]},
            "producerStoppedAfterEpoch": 64,
            "nextProducerEpoch": None,
            "hardTerminalEpoch": 64,
        }
        removed = []
        for epoch in CHECKPOINT_EPOCHS:
            if epoch == TARGET_EPOCH:
                continue
            path = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
            if checkpoint_world_size(path) == 16:
                shutil.rmtree(path)
                removed.append(epoch)
        decision["removedRecoveryEpochs"] = removed
        producer.atomic_json(STATE_DIR / "decisions" / "e64.json", decision)
        print(f"DENSE474M_POOL3B_WD03_E64_POST_RESULT id={EXPECTED_ID} epoch=64 json={json.dumps(result,separators=(',',':'),sort_keys=True)}", flush=True)
        print(f"DENSE474M_POOL3B_WD03_E64_DECISION id={EXPECTED_ID} epoch=64 json={json.dumps(decision,separators=(',',':'),sort_keys=True)}", flush=True)
        print(f"DENSE474M_POOL3B_WD03_E64_JOB_COMPLETE id={EXPECTED_ID} terminal_epoch=64 status=terminal_e64", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config, item = load(args.manifest)
    if args.validate_only:
        print(f"validated {EXPECTED_ID} E32-E64 two-node continuation")
        return
    run(config, item)


if __name__ == "__main__":
    main()
