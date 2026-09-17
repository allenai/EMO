#!/usr/bin/env python3
"""Run the matched LR4e-3 153M Pool-3B BS256 saturation trajectory."""

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
EVALUATION_EPOCHS = (64, 128, 192, 256, 320, 384)
RUNTIME_LOG_DIR = Path("/tmp/icsl-dense-153m-pool3b-lr4e3-wd01")


def checkpoint_step(epoch: int, pool_tokens: int = producer.TARGET_POOL_TOKENS) -> int:
    return producer.stable_step(epoch, pool_tokens, 256)


# Recovery checkpoints do not need to land on an exact epoch boundary: they are
# only used to bound lost work after preemption. Exact 32-epoch PD checkpoints
# remain permanent via ``fixed_steps`` below. The checkpointer keeps only the
# newest ephemeral checkpoint and removes its predecessor after the new save is
# complete, so this adds at most one recovery-only checkpoint to the trajectory.
RECOVERY_SAVE_INTERVAL_STEPS = checkpoint_step(1)


def recovery_checkpoint_steps_through(target_epoch: int) -> list[int]:
    """Return recovery-only steps that may exist through an exact PD target."""
    target_step = checkpoint_step(target_epoch)
    permanent_steps = {checkpoint_step(1), *map(checkpoint_step, CHECKPOINT_EPOCHS)}
    return [
        step
        for step in range(
            RECOVERY_SAVE_INTERVAL_STEPS,
            target_step + 1,
            RECOVERY_SAVE_INTERVAL_STEPS,
        )
        if step not in permanent_steps
    ]


def resume_checkpoint_steps_before(target_epoch: int) -> list[int]:
    """Return every authorized permanent or recovery step before a target."""
    target_step = checkpoint_step(target_epoch)
    steps = {
        checkpoint_step(1),
        *(
            checkpoint_step(epoch)
            for epoch in CHECKPOINT_EPOCHS
            if checkpoint_step(epoch) < target_step
        ),
        *recovery_checkpoint_steps_through(target_epoch),
    }
    return sorted(step for step in steps if step < target_step)


def latest_resume_checkpoint(source: Path, target_epoch: int) -> Path:
    latest = source
    latest_step = checkpoint_step(1)
    for step in resume_checkpoint_steps_before(target_epoch):
        path = OUTPUT / f"step{step}"
        if step > latest_step and distributed.checkpoint_complete(path, 16):
            latest = path
            latest_step = step
    return latest


def cleanup_recovery_checkpoints(target_epoch: int) -> list[int]:
    """Remove recovery-only checkpoints after the exact target is durable."""
    target_path = OUTPUT / f"step{checkpoint_step(target_epoch)}"
    if not distributed.checkpoint_complete(target_path, 16):
        raise RuntimeError(f"refusing recovery cleanup before complete target {target_path}")
    removed: list[int] = []
    for step in recovery_checkpoint_steps_through(target_epoch):
        path = OUTPUT / f"step{step}"
        if not path.exists():
            continue
        if not distributed.checkpoint_complete(path, 16):
            raise RuntimeError(f"refusing to delete incomplete recovery checkpoint {path}")
        shutil.rmtree(path)
        removed.append(step)
    return removed


def post_recovery_checkpoint_steps(epoch: int) -> list[int]:
    """Return recovery-only WSD steps between the PD source and POST endpoint."""
    source_step = checkpoint_step(epoch)
    endpoint_step = producer.total_step(epoch, producer.TARGET_POOL_TOKENS, 256)
    first_step = (
        (source_step // RECOVERY_SAVE_INTERVAL_STEPS) + 1
    ) * RECOVERY_SAVE_INTERVAL_STEPS
    return list(range(first_step, endpoint_step, RECOVERY_SAVE_INTERVAL_STEPS))


def latest_post_resume_checkpoint(post_output: Path, source: Path, epoch: int) -> Path:
    latest = source
    latest_step = checkpoint_step(epoch)
    for step in post_recovery_checkpoint_steps(epoch):
        path = post_output / f"step{step}"
        if step > latest_step and distributed.checkpoint_complete(path, 16):
            latest = path
            latest_step = step
    return latest


def cleanup_post_recovery_checkpoints(post_output: Path, epoch: int) -> list[int]:
    endpoint_step = producer.total_step(epoch, producer.TARGET_POOL_TOKENS, 256)
    endpoint = post_output / f"step{endpoint_step}"
    if not distributed.checkpoint_complete(endpoint, 16):
        raise RuntimeError(f"refusing POST recovery cleanup before complete endpoint {endpoint}")
    removed: list[int] = []
    for step in post_recovery_checkpoint_steps(epoch):
        path = post_output / f"step{step}"
        if not path.exists():
            continue
        if not distributed.checkpoint_complete(path, 16):
            raise RuntimeError(f"refusing to delete incomplete POST recovery checkpoint {path}")
        shutil.rmtree(path)
        removed.append(step)
    return removed


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


def runtime_log(name: str) -> Path:
    """Keep verbose per-step logs off the checkpoint filesystem."""
    return RUNTIME_LOG_DIR / name


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
    # The Pool-1B bootstrap is a new trajectory.  The shared Pool-3B producer
    # arguments require exact loading for continuations, but there is no source
    # checkpoint at this stage.
    arguments = producer.common.upsert(
        arguments,
        "--force_exact_trainer_load_path=",
        "--force_exact_trainer_load_path=false",
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
    arguments = producer.common.upsert(
        common_arguments(config, item, str(config["repeatedManifest"])),
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
        (
            "--trainer.callbacks.checkpointer.ephemeral_save_interval="
            f"{RECOVERY_SAVE_INTERVAL_STEPS}"
        ),
    )
    return [
        *arguments,
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
        runtime_log("bootstrap.log"),
        "bootstrap",
    )
    if not distributed.checkpoint_complete(source, 16):
        raise RuntimeError(f"bootstrap exited without complete source {source}")
    return source


def ensure_bridge(config: dict[str, Any], item: dict[str, Any]) -> Path:
    # A retry may enter here after the canonical Pool-3B trajectory already
    # contains durable checkpoints.  Prefer the newest verified checkpoint
    # before applying the empty-output guard, which is only for a fresh bridge.
    authorized_steps = {
        checkpoint_step(1),
        *map(checkpoint_step, CHECKPOINT_EPOCHS),
        *recovery_checkpoint_steps_through(CHECKPOINT_EPOCHS[-1]),
    }
    for step in sorted(authorized_steps, reverse=True):
        existing = OUTPUT / f"step{step}"
        if distributed.checkpoint_complete(existing, 16):
            return existing
    bridge = OUTPUT / f"step{checkpoint_step(1)}"
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
        runtime_log("bridge.log"),
        "bridge",
    )
    if not distributed.checkpoint_complete(bridge, 16):
        raise RuntimeError(f"bridge exited without complete Pool-3B E1 {bridge}")
    return bridge


def postdecay_resume_arguments(
    config: dict[str, Any],
    item: dict[str, Any],
    epoch: int,
    source: Path,
    resume_source: Path,
    post_output: Path,
    name: str,
) -> list[str]:
    arguments = evaluator.postdecay_arguments(
        config, item, epoch, resume_source, post_output, name
    )
    arguments = producer.common.upsert(
        arguments,
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
        (
            "--trainer.callbacks.checkpointer.ephemeral_save_interval="
            f"{RECOVERY_SAVE_INTERVAL_STEPS}"
        ),
    )
    if resume_source != source:
        # A WSD recovery checkpoint contains the optimizer's decayed current LR,
        # which intentionally differs from the trajectory's original command LR.
        # Load that optimizer/scheduler state exactly instead of rejecting it.
        arguments = producer.common.upsert(
            arguments,
            "--train_module.validate_optimizer_hyperparameters_on_load=",
            "--train_module.validate_optimizer_hyperparameters_on_load=false",
        )
    return arguments


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
    log_path = runtime_log(f"post_e{epoch}.log")
    endpoint_complete = distributed.checkpoint_complete(endpoint, 16)
    if endpoint_complete:
        resume_source = endpoint
        arguments = evaluator.evaluation_arguments(
            config, item, endpoint, post_output / "recovered_eval", f"{name}-recovered-eval"
        )
    else:
        resume_source = latest_post_resume_checkpoint(post_output, source, epoch)
        arguments = postdecay_resume_arguments(
            config,
            item,
            epoch,
            source,
            resume_source,
            post_output,
            name,
        )
    if is_leader():
        print(
            f"DENSE153M_POOL3B_LR4_POST_START epoch={epoch} source={source} "
            f"resume={resume_source}",
            flush=True,
        )
    torchrun(name, arguments, log_path, f"post-e{epoch}")
    if is_leader():
        if not distributed.checkpoint_complete(endpoint, 16):
            raise RuntimeError(f"E{epoch} POST exited without complete endpoint {endpoint}")
        removed = cleanup_post_recovery_checkpoints(post_output, epoch)
        if removed:
            print(
                f"DENSE153M_POOL3B_LR4_POST_RECOVERY_CLEANUP epoch={epoch} "
                f"removedSteps={removed}",
                flush=True,
            )
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
            active_source = latest_resume_checkpoint(source, target)
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
                runtime_log(f"pd_to_e{target}.log"),
                f"pd-e{target}",
            )
        target_path = OUTPUT / f"step{checkpoint_step(target)}"
        if not distributed.checkpoint_complete(target_path, 16):
            raise RuntimeError(f"missing complete E{target} PD checkpoint")
        if is_leader():
            removed = cleanup_recovery_checkpoints(target)
            if removed:
                print(
                    f"DENSE153M_POOL3B_LR4_RECOVERY_CLEANUP target={target} "
                    f"removedSteps={removed}",
                    flush=True,
                )
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
