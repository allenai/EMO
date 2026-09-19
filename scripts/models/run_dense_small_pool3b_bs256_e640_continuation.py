#!/usr/bin/env python3
"""Continue exact 153M Pool-3B E576 PD to E640 on two 8-GPU nodes."""

from __future__ import annotations

import argparse
import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_1b_dr_wt_embedwd_grid as dense1b
import run_dense_small_pool3b_bs256_e512_continuation as base
import run_dense_small_pool3b_checkpoint_evaluator as evaluator
import run_dense_small_pool3b_checkpoint_producer as producer

POLICY = "dense_small_pool3b_bs256_e640_integrated_protected_v1"
DEFAULT_MANIFEST = Path("scripts/models/manifests/dense-small-pool3b-bs256-e640-continuation-v1.json")
EXPECTED_ID = "dense-153m-dclm3b-bs256-lr2e-3-wd0.1"
SOURCE_EPOCH = 576
TARGET_EPOCH = 640
CHECKPOINT_EPOCHS = tuple(range(580, 641, 4))
EXPECTED_OUTPUT = base.EXPECTED_OUTPUT
STATE_DIR = EXPECTED_OUTPUT / ".constant_checkpoint_producer_pool3b_e576_e640_integrated_v1"
POST_START_STEP = base.checkpoint_step(TARGET_EPOCH)
POST_ENDPOINT_STEP = producer.total_step(TARGET_EPOCH, producer.TARGET_POOL_TOKENS, 256)
POST_RECOVERY_STEPS = tuple(
    POST_START_STEP + (POST_ENDPOINT_STEP - POST_START_STEP) * quarter // 4
    for quarter in range(1, 5)
)


def configure_base() -> None:
    base.POLICY = POLICY
    base.DEFAULT_MANIFEST = DEFAULT_MANIFEST
    base.EXPECTED_ID = EXPECTED_ID
    base.SOURCE_EPOCH = SOURCE_EPOCH
    base.TARGET_EPOCH = TARGET_EPOCH
    base.CHECKPOINT_EPOCHS = CHECKPOINT_EPOCHS
    base.EXPECTED_OUTPUT = EXPECTED_OUTPUT
    base.STATE_DIR = STATE_DIR


def producer_arguments(config: dict[str, Any], item: dict[str, Any], resume: Path, pending: list[int]) -> list[str]:
    target_steps = [base.checkpoint_step(epoch) for epoch in pending]
    name = f"{EXPECTED_ID}-constant-pd-e576-e640-v1"
    return [
        *producer.base_arguments(item, str(config["repeatedManifest"])), "--dynamic-repacking",
        f"--save-folder={EXPECTED_OUTPUT}",
        f"--trainer.max_duration={{value: {target_steps[-1]}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,dense-153m,dr_wt_embwd,bs256,lr2e-3,wd0.1,constant-lr,pre-decay,e576-e640-continuation,two-node]",
        "--trainer.callbacks.checkpointer.fixed_steps=" + json.dumps(target_steps, separators=(",", ":")),
        f"--trainer.load_path={resume}",
        "--force_exact_trainer_load_path=true",
    ]


def postdecay_arguments(
    config: dict[str, Any],
    item: dict[str, Any],
    source: Path,
    output: Path,
    name: str,
    resume_step: int,
) -> list[str]:
    pending_steps = [step for step in POST_RECOVERY_STEPS if step > resume_step]
    if not pending_steps:
        raise RuntimeError("post-decay endpoint is already complete")
    arguments = evaluator.postdecay_arguments(config, item, TARGET_EPOCH, source, output, name)
    fixed_steps = (
        "--trainer.callbacks.checkpointer.fixed_steps="
        + json.dumps(pending_steps, separators=(",", ":"))
    )
    arguments = [
        fixed_steps if value.startswith("--trainer.callbacks.checkpointer.fixed_steps=") else value
        for value in arguments
    ]
    if resume_step > POST_START_STEP:
        # A WSD recovery checkpoint contains the optimizer's decayed current LR,
        # which intentionally differs from the trajectory's original command LR.
        # Restore that optimizer/scheduler state exactly instead of rejecting it.
        arguments = producer.common.upsert(
            arguments,
            "--train_module.validate_optimizer_hyperparameters_on_load=",
            "--train_module.validate_optimizer_hyperparameters_on_load=false",
        )
    return arguments


def run(config: dict[str, Any], item: dict[str, Any]) -> None:
    base.validate(config, item, filesystem=True)
    complete = [
        epoch for epoch in CHECKPOINT_EPOCHS
        if base.checkpoint_complete(EXPECTED_OUTPUT / f"step{base.checkpoint_step(epoch)}", 16)
    ]
    for epoch in CHECKPOINT_EPOCHS:
        path = EXPECTED_OUTPUT / f"step{base.checkpoint_step(epoch)}"
        if path.exists() and epoch not in complete:
            raise RuntimeError(f"refusing incomplete checkpoint {path}")
    resume_epoch = max([SOURCE_EPOCH, *complete])
    pending = [epoch for epoch in CHECKPOINT_EPOCHS if epoch > resume_epoch]
    if pending:
        resume = EXPECTED_OUTPUT / f"step{base.checkpoint_step(resume_epoch)}"
        if base.is_leader():
            producer.atomic_json(STATE_DIR / "producer.json", {
                "policy": POLICY,
                "status": "running",
                "sourceEpoch": SOURCE_EPOCH,
                "resumeEpoch": resume_epoch,
                "targetEpoch": TARGET_EPOCH,
                "pendingEpochs": pending,
                "gpuCount": 16,
                "nodeCount": 2,
                "rankMicrobatchSequences": 16,
                "gradientAccumulationSteps": 1,
            })
            print(
                f"DENSE_POOL3B_E640_PD_START id={EXPECTED_ID} "
                f"source_epoch={resume_epoch} target_epoch=640",
                flush=True,
            )
        base.torchrun(
            f"{EXPECTED_ID}-constant-pd-e576-e640-v1",
            producer_arguments(config, item, resume, pending),
            STATE_DIR / "producer.log",
            "pd",
        )

    target = EXPECTED_OUTPUT / f"step{base.checkpoint_step(TARGET_EPOCH)}"
    if not base.checkpoint_complete(target, 16):
        raise RuntimeError(f"producer exited without exact E640 checkpoint {target}")
    if base.is_leader():
        print(
            f"DENSE_POOL3B_INTEGRATED_PD_RETAINED id={EXPECTED_ID} "
            f"epoch=640 checkpoint={target}",
            flush=True,
        )

    result_path = evaluator.state_dir(item, str(EXPECTED_OUTPUT)) / "results" / "e640.json"
    if not result_path.is_file():
        post_output = evaluator.state_dir(item, str(EXPECTED_OUTPUT)) / "post_decay_runs" / "e640"
        endpoint = post_output / f"step{POST_ENDPOINT_STEP}"
        name = f"{EXPECTED_ID}-post-e640-v1"
        log_path = evaluator.state_dir(item, str(EXPECTED_OUTPUT)) / "logs" / "e640.log"
        complete_post_steps = []
        for step in POST_RECOVERY_STEPS:
            checkpoint = post_output / f"step{step}"
            if base.checkpoint_complete(checkpoint, 16):
                complete_post_steps.append(step)
            elif checkpoint.exists():
                raise RuntimeError(f"refusing incomplete post-decay checkpoint {checkpoint}")
        if POST_ENDPOINT_STEP in complete_post_steps:
            log_path = evaluator.state_dir(item, str(EXPECTED_OUTPUT)) / "logs" / "e640_recovered_eval.log"
            active_source = endpoint
            args = evaluator.evaluation_arguments(
                config, item, endpoint, post_output / "eval", f"{name}-recovered-eval"
            )
        else:
            resume_step = max([POST_START_STEP, *complete_post_steps])
            resume = target if resume_step == POST_START_STEP else post_output / f"step{resume_step}"
            active_source = resume
            args = postdecay_arguments(config, item, resume, post_output, name, resume_step)
        if base.is_leader():
            print(
                f"DENSE_SMALL_CHECKPOINT_EVALUATOR_START id={EXPECTED_ID} epoch=640 "
                f"source={active_source} output={post_output}",
                flush=True,
            )
        base.torchrun(name, args, log_path, "post")
        if base.is_leader():
            if not endpoint.is_dir():
                raise RuntimeError(f"E640 decay exited without endpoint {endpoint}")
            result = dense1b.parse_validation(log_path, TARGET_EPOCH, "post_decay", endpoint)
            result.update({
                "policy": POLICY,
                "producerPolicy": POLICY,
                "model": "153m",
                "batchSequences": 256,
                "lr": "2e-3",
                "wd": "0.1",
                "variant": "DR+WT+EmbedWD",
                "dynamicRepacking": True,
                "weightTying": True,
                "decayEmbeddings": True,
                "sourcePreDecayCheckpoint": str(target),
                "source": "integrated_two_node_wsd_decay_and_heldout_eval",
            })
            evaluator.atomic_json(result_path, result)

    result = json.loads(result_path.read_text()) if base.is_leader() else base.wait_for(result_path)
    if base.is_leader():
        previous = json.loads(
            (evaluator.state_dir(item, str(EXPECTED_OUTPUT)) / "results" / "e576.json").read_text()
        )
        saturated = Decimal(str(result["validationExact"])) >= Decimal(str(previous["validationExact"]))
        decision = {
            "policy": POLICY,
            "status": "saturated" if saturated else "terminal_e640",
            "comparisonGroup": "post_decay_only",
            "criterion": "strict_non_improvement",
            "epochs": [576, 640],
            "validationExact": {
                "576": previous["validationExact"],
                "640": result["validationExact"],
            },
            "producerStoppedAfterEpoch": 640,
            "nextProducerEpoch": None,
            "hardTerminalEpoch": 640,
        }
        for epoch in CHECKPOINT_EPOCHS:
            if epoch != TARGET_EPOCH:
                path = EXPECTED_OUTPUT / f"step{base.checkpoint_step(epoch)}"
                if base.checkpoint_complete(path, 16):
                    shutil.rmtree(path)
        for step in POST_RECOVERY_STEPS[:-1]:
            path = evaluator.state_dir(item, str(EXPECTED_OUTPUT)) / "post_decay_runs" / "e640" / f"step{step}"
            if base.checkpoint_complete(path, 16):
                shutil.rmtree(path)
        producer.atomic_json(STATE_DIR / "decisions" / "e640.json", decision)
        print(
            f"DENSE_POOL3B_INTEGRATED_POST_RESULT id={EXPECTED_ID} epoch=640 "
            f"json={json.dumps(result,separators=(',',':'),sort_keys=True)}",
            flush=True,
        )
        print(
            f"DENSE_POOL3B_INTEGRATED_DECISION id={EXPECTED_ID} epoch=640 "
            f"json={json.dumps(decision,separators=(',',':'),sort_keys=True)}",
            flush=True,
        )
        print(
            f"DENSE_POOL3B_INTEGRATED_JOB_COMPLETE id={EXPECTED_ID} "
            f"terminal_epoch=640 status={decision['status']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    configure_base()
    config, item = base.load(args.manifest)
    if args.validate_only:
        print(f"validated {EXPECTED_ID} E576-E640 two-node continuation")
        return
    run(config, item)


if __name__ == "__main__":
    main()
