#!/usr/bin/env python3
"""Run the matched LR4e-3 153M Pool-3B BS256 saturation trajectory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_1b_dr_wt_embedwd_grid as dense1b
import run_dense_small_pool3b_bs256_e512_continuation as distributed
import run_dense_small_pool3b_checkpoint_evaluator as evaluator
import run_dense_small_pool3b_checkpoint_producer as producer
import run_small_dense_dr_wt_embedwd_chain as small

POLICY = "dense_153m_pool3b_bs256_lr4e3_wd01_saturation_v1"
DEFAULT_MANIFEST = Path(
    "scripts/models/manifests/dense-153m-pool3b-bs256-lr4e3-wd01-saturation-v1.json"
)
EXPECTED_ID = "dense-153m-dclm3b-bs256-lr4e-3-wd0.1"
OUTPUT = Path(
    "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm3b/"
    "bs256_dr_wt_embwd_lr4e-3_wd0.1"
)
BOOTSTRAP_OUTPUT = Path(
    "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm1b/"
    "bs256_dr_wt_embwd_lr4e-3_wd0.1"
)
STATE_DIR = OUTPUT / ".lr4e3_wd01_saturation_v1"
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"
CHECKPOINT_EPOCHS = tuple(range(32, 385, 32))
EVALUATION_EPOCHS = (32, 64, 96, 128, 160, 192, 224, 256, 320, 384)


def checkpoint_step(epoch: int, pool_tokens: int = producer.TARGET_POOL_TOKENS) -> int:
    return producer.stable_step(epoch, pool_tokens, 256)


def is_leader() -> bool:
    return int(os.environ.get("BEAKER_REPLICA_RANK", "0")) == 0


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if is_leader():
        producer.atomic_json(path, value)


def wait_for(path: Path, timeout: int = 7200) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.is_file():
            return json.loads(path.read_text())
        time.sleep(5)
    raise TimeoutError(f"timed out waiting for leader result {path}")


def load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text())
    item = config["producerCoordinate"]
    validate(config, item, filesystem=False)
    return config, item


def validate(config: dict[str, Any], item: dict[str, Any], *, filesystem: bool) -> None:
    expected = {
        "policy": POLICY,
        "sourcePoolTokens": 1_000_000_000,
        "targetPoolTokens": producer.TARGET_POOL_TOKENS,
        "checkpointEpochs": list(CHECKPOINT_EPOCHS),
        "evaluationEpochs": list(EVALUATION_EPOCHS),
        "maxEpoch": 384,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"manifest mismatch for {key}: {config.get(key)!r}")
    item_expected = {
        "id": EXPECTED_ID,
        "model": "153m",
        "batchSequences": 256,
        "gpuCount": 16,
        "gpusPerNode": 8,
        "nodeCount": 2,
        "rankMicrobatchSequences": 16,
        "gradientAccumulationSteps": 1,
        "learningRate": "4e-3",
        "weightDecay": "0.1",
        "bootstrapOutput": str(BOOTSTRAP_OUTPUT),
        "sourceCheckpoint": str(BOOTSTRAP_OUTPUT / f"step{checkpoint_step(1, 1_000_000_000)}"),
        "output": str(OUTPUT),
        "stopOnAdjacentPostNonImprovement": True,
    }
    for key, value in item_expected.items():
        if item.get(key) != value:
            raise ValueError(f"coordinate mismatch for {key}: {item.get(key)!r}")
    if filesystem:
        producer.validate_pool_lineage(config)


def torchrun(name: str, arguments: list[str], log_path: Path, stage: str) -> None:
    distributed.torchrun(name, arguments, log_path, stage)


def common_arguments(
    config: dict[str, Any], item: dict[str, Any], manifest: str, *, heldout: bool = False
) -> list[str]:
    arguments = producer.base_arguments(item, manifest)
    heldout_config = small.HELDOUT_EVALUATOR
    if not heldout:
        heldout_config = heldout_config.replace("eval_on_finish: true", "eval_on_finish: false")
    arguments = producer.common.upsert(
        arguments,
        "--trainer.callbacks.heldout_evaluator=",
        f"--trainer.callbacks.heldout_evaluator={heldout_config}",
    )
    return arguments


def bootstrap_arguments(config: dict[str, Any], item: dict[str, Any]) -> list[str]:
    step = checkpoint_step(1, 1_000_000_000)
    name = f"{EXPECTED_ID}-bootstrap-pool1b-e1"
    arguments = producer.common.upsert(
        common_arguments(config, item, str(config["sourceManifest"])),
        "--train_module.scheduler=",
        "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.ConstantWithWarmup, warmup: 96}",
    )
    return [
        *arguments,
        f"--save-folder={BOOTSTRAP_OUTPUT}",
        f"--trainer.max_duration={{value: {step}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool1b-bootstrap,dense-153m,dr_wt_embwd,bs256,lr4e-3,wd0.1]",
        f"--trainer.callbacks.checkpointer.fixed_steps=[{step}]",
    ]


def train_arguments(
    config: dict[str, Any], item: dict[str, Any], source: Path, pending_epochs: list[int]
) -> list[str]:
    steps = [checkpoint_step(epoch) for epoch in pending_epochs]
    target = pending_epochs[-1]
    name = f"{EXPECTED_ID}-constant-pd-to-e{target}"
    return [
        *common_arguments(config, item, str(config["repeatedManifest"])),
        "--dynamic-repacking",
        f"--save-folder={OUTPUT}",
        f"--trainer.max_duration={{value: {steps[-1]}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,dense-153m,dr_wt_embwd,bs256,lr4e-3,wd0.1,constant-lr,two-node]",
        "--trainer.callbacks.checkpointer.fixed_steps=" + json.dumps(steps, separators=(",", ":")),
        f"--trainer.load_path={source}",
    ]


def ensure_bootstrap(config: dict[str, Any], item: dict[str, Any]) -> Path:
    source = Path(str(item["sourceCheckpoint"]))
    if distributed.checkpoint_complete(source, 16):
        return source
    if BOOTSTRAP_OUTPUT.exists() and any(BOOTSTRAP_OUTPUT.iterdir()):
        unexpected = [path for path in BOOTSTRAP_OUTPUT.iterdir() if path != STATE_DIR]
        if unexpected and not source.exists():
            raise RuntimeError(f"refusing nonempty bootstrap output without E1: {unexpected}")
    if is_leader():
        print(f"DENSE153M_POOL3B_LR4_BOOTSTRAP_START target={source}", flush=True)
    torchrun(
        f"{EXPECTED_ID}-bootstrap-pool1b-e1",
        bootstrap_arguments(config, item),
        STATE_DIR / "bootstrap.log",
        "bootstrap",
    )
    if not distributed.checkpoint_complete(source, 16):
        raise RuntimeError(f"bootstrap exited without complete source {source}")
    return source


def ensure_bridge(config: dict[str, Any], item: dict[str, Any]) -> Path:
    bridge = OUTPUT / f"step{checkpoint_step(1)}"
    if distributed.checkpoint_complete(bridge, 16):
        return bridge
    source = ensure_bootstrap(config, item)
    producer.validate_source_checkpoint(item)
    if OUTPUT.exists():
        unexpected = [path for path in OUTPUT.iterdir() if path != STATE_DIR]
        if unexpected and not bridge.exists():
            raise RuntimeError(f"refusing nonempty Pool-3B output before bridge: {unexpected}")
    if is_leader():
        print(f"DENSE153M_POOL3B_LR4_BRIDGE_START source={source} target={bridge}", flush=True)
    torchrun(
        f"{EXPECTED_ID}-fresh-2b-bridge-to-pd-e1",
        producer.bridge_arguments(config, item),
        STATE_DIR / "bridge.log",
        "bridge",
    )
    if not distributed.checkpoint_complete(bridge, 16):
        raise RuntimeError(f"bridge exited without complete Pool-3B E1 {bridge}")
    return bridge


def evaluate(config: dict[str, Any], item: dict[str, Any], epoch: int) -> dict[str, Any]:
    result_path = STATE_DIR / "results" / f"e{epoch}.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    source = OUTPUT / f"step{checkpoint_step(epoch)}"
    if not distributed.checkpoint_complete(source, 16):
        raise FileNotFoundError(f"incomplete E{epoch} PD source {source}")
    post_output = STATE_DIR / "post_decay_runs" / f"e{epoch}"
    endpoint = post_output / f"step{producer.total_step(epoch, producer.TARGET_POOL_TOKENS, 256)}"
    name = f"{EXPECTED_ID}-post-e{epoch}"
    log_path = STATE_DIR / "logs" / f"post_e{epoch}.log"
    if distributed.checkpoint_complete(endpoint, 16):
        arguments = evaluator.evaluation_arguments(
            config, item, endpoint, post_output / "recovered_eval", f"{name}-recovered-eval"
        )
    else:
        arguments = evaluator.postdecay_arguments(config, item, epoch, source, post_output, name)
    if is_leader():
        print(f"DENSE153M_POOL3B_LR4_POST_START epoch={epoch} source={source}", flush=True)
    torchrun(name, arguments, log_path, f"post-e{epoch}")
    if is_leader():
        if not distributed.checkpoint_complete(endpoint, 16):
            raise RuntimeError(f"E{epoch} POST exited without complete endpoint {endpoint}")
        result = dense1b.parse_validation(log_path, epoch, "post_decay", endpoint)
        result.update(
            {
                "policy": POLICY,
                "model": "153m",
                "batchSequences": 256,
                "lr": "4e-3",
                "wd": "0.1",
                "variant": "DR+WT+EmbedWD",
                "sourcePreDecayCheckpoint": str(source),
                "source": "integrated_two_node_wsd_decay_and_heldout_eval",
            }
        )
        producer.atomic_json(result_path, result)
        print(
            f"DENSE153M_POOL3B_LR4_POST_RESULT epoch={epoch} "
            f"json={json.dumps(result, separators=(',', ':'), sort_keys=True)}",
            flush=True,
        )
    return json.loads(result_path.read_text()) if is_leader() else wait_for(result_path)


def run(config: dict[str, Any], item: dict[str, Any]) -> None:
    validate(config, item, filesystem=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_OUTPUT.mkdir(parents=True, exist_ok=True)
    state = {
        "policy": POLICY,
        "id": EXPECTED_ID,
        "status": "running",
        "checkpointEpochs": list(CHECKPOINT_EPOCHS),
        "evaluationEpochs": list(EVALUATION_EPOCHS),
        "gpuCount": 16,
        "nodeCount": 2,
        "rankMicrobatchSequences": 16,
        "gradientAccumulationSteps": 1,
    }
    atomic_json(STATE_DIR / "producer.json", state)
    source = ensure_bridge(config, item)
    completed_evaluations: list[int] = []
    for target in EVALUATION_EPOCHS:
        pending = [
            epoch
            for epoch in CHECKPOINT_EPOCHS
            if epoch <= target
            and not distributed.checkpoint_complete(OUTPUT / f"step{checkpoint_step(epoch)}", 16)
        ]
        if pending:
            completed = [
                epoch
                for epoch in CHECKPOINT_EPOCHS
                if epoch < pending[0]
                and distributed.checkpoint_complete(OUTPUT / f"step{checkpoint_step(epoch)}", 16)
            ]
            active_source = OUTPUT / f"step{checkpoint_step(max(completed))}" if completed else source
            state.update({"status": "producer_running", "currentEpoch": target})
            atomic_json(STATE_DIR / "producer.json", state)
            if is_leader():
                print(
                    f"DENSE153M_POOL3B_LR4_PD_START target={target} pending={pending} source={active_source}",
                    flush=True,
                )
            torchrun(
                f"{EXPECTED_ID}-constant-pd-to-e{target}",
                train_arguments(config, item, active_source, pending),
                STATE_DIR / "logs" / f"pd_to_e{target}.log",
                f"pd-e{target}",
            )
        target_path = OUTPUT / f"step{checkpoint_step(target)}"
        if not distributed.checkpoint_complete(target_path, 16):
            raise RuntimeError(f"missing complete E{target} PD checkpoint")
        if is_leader():
            print(f"DENSE153M_POOL3B_LR4_PD_RETAINED epoch={target} checkpoint={target_path}", flush=True)
        state.update({"status": "post_running", "currentEpoch": target})
        atomic_json(STATE_DIR / "producer.json", state)
        result = evaluate(config, item, target)
        completed_evaluations.append(target)
        if len(completed_evaluations) >= 2:
            previous_epoch = completed_evaluations[-2]
            previous = json.loads((STATE_DIR / "results" / f"e{previous_epoch}.json").read_text())
            if Decimal(str(result["validationExact"])) >= Decimal(str(previous["validationExact"])):
                state.update(
                    {
                        "status": "saturated",
                        "currentEpoch": target,
                        "saturationDecision": {
                            "previousEpoch": previous_epoch,
                            "currentEpoch": target,
                            "criterion": "adjacent_post_validationExact_non_improvement",
                            "previousValidationExact": previous["validationExact"],
                            "currentValidationExact": result["validationExact"],
                        },
                    }
                )
                atomic_json(STATE_DIR / "producer.json", state)
                if is_leader():
                    print(f"DENSE153M_POOL3B_LR4_SATURATED epoch={target} previousEpoch={previous_epoch}", flush=True)
                return
    state.update({"status": "complete", "currentEpoch": EVALUATION_EPOCHS[-1]})
    atomic_json(STATE_DIR / "producer.json", state)
    if is_leader():
        print("DENSE153M_POOL3B_LR4_COMPLETE hard_ceiling=384", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config, item = load(args.manifest)
    if args.validate_only:
        print(f"validated {EXPECTED_ID}")
        return
    run(config, item)


if __name__ == "__main__":
    main()
