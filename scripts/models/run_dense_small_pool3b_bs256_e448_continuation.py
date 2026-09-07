#!/usr/bin/env python3
"""Continue exact 153M Pool-3B E384 PD to E448 on two 8-GPU nodes."""

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
import run_small_dense_dr_wt_embedwd_chain as small

POLICY = "dense_small_pool3b_bs256_e448_integrated_v1"
DEFAULT_MANIFEST = Path("scripts/models/manifests/dense-small-pool3b-bs256-e448-continuation-v1.json")
EXPECTED_ID = "dense-153m-dclm3b-bs256-lr2e-3-wd0.1"
SOURCE_EPOCH = 384
TARGET_EPOCH = 448
CHECKPOINT_EPOCHS = tuple(range(388, 449, 4))
EXPECTED_OUTPUT = Path("/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm3b/bs256_dr_wt_embwd_lr2e-3_wd0.1_throughput_recovery_r2")
STATE_DIR = EXPECTED_OUTPUT / ".constant_checkpoint_producer_pool3b_e384_e448_integrated_v1"
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"


def checkpoint_step(epoch: int) -> int:
    return producer.stable_step(epoch, producer.TARGET_POOL_TOKENS, 256)


def checkpoint_complete(path: Path, ranks: int) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "model_and_optim" / ".metadata").is_file()
        and any((path / "model_and_optim").glob("*.distcp"))
        and all((path / "train" / f"rank{rank}.pt").is_file() for rank in range(ranks))
    )


def is_leader() -> bool:
    return int(os.environ.get("BEAKER_REPLICA_RANK", "0")) == 0


def load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text())
    item = config["producerCoordinate"]
    validate(config, item, filesystem=False)
    return config, item


def validate(config: dict[str, Any], item: dict[str, Any], *, filesystem: bool) -> None:
    expected = {
        "policy": POLICY, "sourceEpoch": SOURCE_EPOCH, "targetEpochs": [TARGET_EPOCH],
        "evaluationEpochs": [TARGET_EPOCH], "checkpointIntervalEpochs": 4,
        "sourcePoolTokens": producer.TARGET_POOL_TOKENS,
        "targetPoolTokens": producer.TARGET_POOL_TOKENS,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"manifest mismatch for {key}: {config.get(key)!r}")
    item_expected = {
        "id": EXPECTED_ID, "model": "153m", "batchSequences": 256,
        "gpuCount": 16, "gpusPerNode": 8, "nodeCount": 2,
        "rankMicrobatchSequences": 16, "gradientAccumulationSteps": 1,
        "learningRate": "2e-3", "weightDecay": "0.1",
        "output": str(EXPECTED_OUTPUT),
        "sourceCheckpoint": str(EXPECTED_OUTPUT / f"step{checkpoint_step(SOURCE_EPOCH)}"),
    }
    for key, value in item_expected.items():
        if item.get(key) != value:
            raise ValueError(f"coordinate mismatch for {key}: {item.get(key)!r}")
    if filesystem:
        source = Path(item["sourceCheckpoint"])
        if not checkpoint_complete(source, 8):
            raise FileNotFoundError(f"incomplete exact E384 source {source}")
        value = json.loads((source / "config.json").read_text())
        optim = value["train_module"]["optim"]
        checks = [
            int(value["data_loader"]["global_batch_size"]) == 256 * producer.SEQUENCE_LENGTH,
            bool(value["model"]["tie_embeddings"]),
            bool(value["dataset"]["dynamic_repacking"]),
            Decimal(str(optim["lr"])) == Decimal("2e-3"),
            Decimal(str(optim["weight_decay"])) == Decimal("0.1"),
            optim.get("group_overrides") == [],
        ]
        if not all(checks):
            raise RuntimeError("E384 source does not preserve the selected DR+WT+EmbedWD recipe")


def torchrun(name: str, arguments: list[str], log_path: Path, stage: str) -> None:
    rank = int(os.environ.get("BEAKER_REPLICA_RANK", "0"))
    nodes = int(os.environ.get("NUM_NODES", "2"))
    if nodes != 2:
        raise RuntimeError("E448 continuation requires exactly two nodes")
    if rank == 0:
        subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    node_log = log_path if rank == 0 else log_path.with_name(f"{log_path.stem}.node{rank}{log_path.suffix}")
    command = [
        "torchrun", "--nnodes=2", "--nproc-per-node=8", f"--node-rank={rank}",
        "--rdzv-backend=static",
        f"--rdzv-endpoint={os.environ['BEAKER_LEADER_REPLICA_HOSTNAME']}:{os.environ['GANTRY_RDZV_PORT']}",
        f"--rdzv-id={os.environ['GANTRY_RDZV_ID']}-{stage}", "--rdzv-conf=read_timeout=420",
        TRAINING_SCRIPT, name, *arguments,
    ]
    with node_log.open("a") as handle:
        producer.common.stream_command(command, handle)


def producer_arguments(config: dict[str, Any], item: dict[str, Any], resume: Path, pending: list[int]) -> list[str]:
    target_steps = [checkpoint_step(epoch) for epoch in pending]
    name = f"{EXPECTED_ID}-constant-pd-e384-e448-v1"
    return [
        *producer.base_arguments(item, str(config["repeatedManifest"])), "--dynamic-repacking",
        f"--save-folder={EXPECTED_OUTPUT}",
        f"--trainer.max_duration={{value: {target_steps[-1]}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,dense-153m,dr_wt_embwd,bs256,lr2e-3,wd0.1,constant-lr,pre-decay,e384-e448-continuation,two-node]",
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
    complete = [e for e in CHECKPOINT_EPOCHS if checkpoint_complete(EXPECTED_OUTPUT / f"step{checkpoint_step(e)}", 16)]
    for epoch in CHECKPOINT_EPOCHS:
        path = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
        if path.exists() and epoch not in complete:
            raise RuntimeError(f"refusing incomplete checkpoint {path}")
    resume_epoch = max([SOURCE_EPOCH, *complete])
    pending = [e for e in CHECKPOINT_EPOCHS if e > resume_epoch]
    if pending:
        resume = EXPECTED_OUTPUT / f"step{checkpoint_step(resume_epoch)}"
        if is_leader():
            producer.atomic_json(STATE_DIR / "producer.json", {
                "policy": POLICY, "status": "running", "sourceEpoch": SOURCE_EPOCH,
                "resumeEpoch": resume_epoch, "targetEpoch": TARGET_EPOCH,
                "pendingEpochs": pending, "gpuCount": 16, "nodeCount": 2,
                "rankMicrobatchSequences": 16, "gradientAccumulationSteps": 1,
            })
            print(f"DENSE_POOL3B_E448_PD_START id={EXPECTED_ID} source_epoch={resume_epoch} target_epoch=448", flush=True)
        torchrun(f"{EXPECTED_ID}-constant-pd-e384-e448-v1", producer_arguments(config, item, resume, pending), STATE_DIR / "producer.log", "pd")
    target = EXPECTED_OUTPUT / f"step{checkpoint_step(TARGET_EPOCH)}"
    if not checkpoint_complete(target, 16):
        raise RuntimeError(f"producer exited without exact E448 checkpoint {target}")
    if is_leader():
        print(f"DENSE_POOL3B_INTEGRATED_PD_RETAINED id={EXPECTED_ID} epoch=448 checkpoint={target}", flush=True)

    result_path = evaluator.state_dir(item, str(EXPECTED_OUTPUT)) / "results" / "e448.json"
    if not result_path.is_file():
        post_output = evaluator.state_dir(item, str(EXPECTED_OUTPUT)) / "post_decay_runs" / "e448"
        endpoint_step = producer.total_step(TARGET_EPOCH, producer.TARGET_POOL_TOKENS, 256)
        endpoint = post_output / f"step{endpoint_step}"
        name = f"{EXPECTED_ID}-post-e448-v1"
        log_path = evaluator.state_dir(item, str(EXPECTED_OUTPUT)) / "logs" / "e448.log"
        if endpoint.is_dir():
            eval_name = f"{name}-recovered-eval"
            args = evaluator.evaluation_arguments(config, item, endpoint, post_output / "eval", eval_name)
        else:
            args = evaluator.postdecay_arguments(config, item, TARGET_EPOCH, target, post_output, name)
        if is_leader():
            print(f"DENSE_SMALL_CHECKPOINT_EVALUATOR_START id={EXPECTED_ID} epoch=448 source={target} output={post_output}", flush=True)
        torchrun(name, args, log_path, "post")
        if is_leader():
            if not endpoint.is_dir():
                raise RuntimeError(f"E448 decay exited without endpoint {endpoint}")
            result = dense1b.parse_validation(log_path, TARGET_EPOCH, "post_decay", endpoint)
            result.update({
                "policy": POLICY, "producerPolicy": POLICY, "model": "153m",
                "batchSequences": 256, "lr": "2e-3", "wd": "0.1",
                "variant": "DR+WT+EmbedWD", "dynamicRepacking": True,
                "weightTying": True, "decayEmbeddings": True,
                "sourcePreDecayCheckpoint": str(target),
                "source": "integrated_two_node_wsd_decay_and_heldout_eval",
            })
            evaluator.atomic_json(result_path, result)
    result = json.loads(result_path.read_text()) if is_leader() else wait_for(result_path)
    if is_leader():
        previous = json.loads((evaluator.state_dir(item, str(EXPECTED_OUTPUT)) / "results" / "e384.json").read_text())
        saturated = Decimal(str(result["validationExact"])) >= Decimal(str(previous["validationExact"]))
        decision = {
            "policy": POLICY, "status": "saturated" if saturated else "terminal_e448",
            "comparisonGroup": "post_decay_only", "criterion": "strict_non_improvement",
            "epochs": [384, 448], "validationExact": {"384": previous["validationExact"], "448": result["validationExact"]},
            "producerStoppedAfterEpoch": 448, "nextProducerEpoch": None, "hardTerminalEpoch": 448,
        }
        for epoch in CHECKPOINT_EPOCHS:
            if epoch == TARGET_EPOCH:
                continue
            path = EXPECTED_OUTPUT / f"step{checkpoint_step(epoch)}"
            if checkpoint_complete(path, 16):
                shutil.rmtree(path)
        producer.atomic_json(STATE_DIR / "decisions" / "e448.json", decision)
        print(f"DENSE_POOL3B_INTEGRATED_POST_RESULT id={EXPECTED_ID} epoch=448 json={json.dumps(result,separators=(',',':'),sort_keys=True)}", flush=True)
        print(f"DENSE_POOL3B_INTEGRATED_DECISION id={EXPECTED_ID} epoch=448 json={json.dumps(decision,separators=(',',':'),sort_keys=True)}", flush=True)
        print(f"DENSE_POOL3B_INTEGRATED_JOB_COMPLETE id={EXPECTED_ID} terminal_epoch=448 status={decision['status']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config, item = load(args.manifest)
    if args.validate_only:
        print(f"validated {EXPECTED_ID} E384-E448 two-node continuation")
        return
    run(config, item)


if __name__ == "__main__":
    main()
