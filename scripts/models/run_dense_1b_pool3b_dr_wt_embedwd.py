#!/usr/bin/env python3
"""Run one Dense-1B DR+WT+EmbedWD coordinate on the sealed 3B pool.

The constant-LR branch retains an internal E1 checkpoint and the requested
E4/E8/.../E32 pre-decay checkpoints.  No checkpoint before E32 is evaluated.
POST evaluations begin at E32 and then advance by sixteen epochs until the
first strict non-improvement relative to the immediately preceding POST result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dense_1b_dr_wt_embedwd_grid as chain

POLICY = "dense_1b_pool3b_dr_wt_embedwd_postdecay_saturation_v1"
POOL_TOKENS = 3_000_000_000
SEQUENCE_LENGTH = 4096
DECAY_FRACTION = 0.1
POST_DECAY_START_EPOCH = 32
CHECKPOINT_ONLY_EPOCHS = [1, 4, 8, 12, 16, 20, 24, 28]
INITIAL_TARGETS = [1, 4, 8, 12, 16, 20, 24, 28, 32]
POST_EPOCH_INCREMENT = 16
POST_DECAY_SOURCE_COUNT = 2
OUTPUT_ROOT = (
    "/weka/oe-training-default/sewonm/icsl/models/"
    "dense_1b_pool3b_dr_wt_embwd"
)
POOL_ROOT = (
    "/weka/oe-training-default/sewonm/icsl/data/"
    "dclm_0802_nested_1b_3b_9b"
)
FULL_POOL = f"{POOL_ROOT}/manifests/dclm_0802_nested_train_3b.json"
POOL_AUDIT = f"{POOL_ROOT}/manifests/dclm_0802_nested_1b_3b_9b.pool.json"


def total_step(epoch: int, batch: int) -> int:
    return math.ceil(epoch * POOL_TOKENS / (batch * SEQUENCE_LENGTH))


def stable_step(epoch: int, batch: int) -> int:
    endpoint = total_step(epoch, batch)
    return endpoint - round(DECAY_FRACTION * endpoint) - 1


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "globalSequences",
        "nprocPerNode",
        "rankMicrobatchSequences",
        "gradientAccumulation",
        "warmupSteps",
        "coordinates",
        "initialTargets",
        "epochIncrement",
        "maxEpoch",
        "outputRoot",
        "runSuffix",
        "variant",
        "policy",
        "comparisonPolicy",
        "postDecaySourceCount",
        "postDecaySaturationCriterion",
        "postDecayStartEpoch",
        "checkpointOnlyEpochs",
        "postDecayEvaluation",
        "poolTokens",
        "poolManifest",
        "poolAudit",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"manifest is missing required keys: {missing}")
    batch = int(config["globalSequences"])
    if batch not in {64, 128}:
        raise ValueError("the Pool-3B study is restricted to BS64 and BS128")
    if config["policy"] != POLICY:
        raise ValueError(f"manifest policy must be {POLICY}")
    if config["variant"] != "DR+WT+EmbedWD":
        raise ValueError("variant must remain DR+WT+EmbedWD")
    if config["comparisonPolicy"] != "post_decay_only":
        raise ValueError("saturation must use POST only")
    if int(config["postDecaySourceCount"]) != POST_DECAY_SOURCE_COUNT:
        raise ValueError("POST saturation must compare adjacent POST results")
    if config["postDecaySaturationCriterion"] != "strict_non_improvement":
        raise ValueError("POST saturation criterion must remain strict_non_improvement")
    if int(config["postDecayStartEpoch"]) != POST_DECAY_START_EPOCH:
        raise ValueError("POST evaluation must begin at E32")
    if [int(x) for x in config["checkpointOnlyEpochs"]] != CHECKPOINT_ONLY_EPOCHS:
        raise ValueError("checkpoint-only epochs must be E1 and every E4 through E28")
    if [int(x) for x in config["initialTargets"]] != INITIAL_TARGETS:
        raise ValueError("constant-LR checkpoint ladder must be E1,E4,...,E32")
    if int(config["epochIncrement"]) != POST_EPOCH_INCREMENT:
        raise ValueError("frontiers after E32 must advance by sixteen epochs")
    if config["postDecayEvaluation"] != "e32_then_every_16_epochs":
        raise ValueError("POST evaluation schedule must be E32,E48,E64,...")
    if int(config["poolTokens"]) != POOL_TOKENS:
        raise ValueError("one Pool-3B epoch must mean exactly 3B tokens")
    if config["poolManifest"] != FULL_POOL or config["poolAudit"] != POOL_AUDIT:
        raise ValueError("manifest must use the sealed nested Pool-3B lineage")
    if int(config["nprocPerNode"]) != 8 or int(config["rankMicrobatchSequences"]) != 8:
        raise ValueError("each coordinate must use eight GPUs and rank microbatch 8")
    if int(config["gradientAccumulation"]) != batch // 64:
        raise ValueError("gradient accumulation does not produce the requested batch")
    if int(config["warmupSteps"]) != 24_576 // batch:
        raise ValueError("warmup must preserve the established token budget")
    if str(config["outputRoot"]) != OUTPUT_ROOT:
        raise ValueError(f"outputRoot must remain {OUTPUT_ROOT}")
    coordinates = config["coordinates"]
    if len(coordinates) != 1:
        raise ValueError("each persistent experiment must own exactly one coordinate")
    coordinate = coordinates[0]
    if Decimal(str(coordinate["lr"])) != Decimal("1e-3"):
        raise ValueError("learning rate must remain 1e-3")
    if Decimal(str(coordinate["wd"])) != Decimal("0.3"):
        raise ValueError("weight decay must remain 0.3")
    expected_output = f"{OUTPUT_ROOT}/bs{batch}_lr1e-3_wd0.3"
    if coordinate.get("output") != expected_output:
        raise ValueError(f"coordinate output must remain {expected_output}")
    if int(config["maxEpoch"]) < 64:
        raise ValueError("maxEpoch must leave room for adjacent POST comparisons")


def validate_pool_lineage(config: dict[str, Any]) -> None:
    manifest = Path(str(config["poolManifest"]))
    audit_path = Path(str(config["poolAudit"]))
    if not manifest.is_file() or not audit_path.is_file():
        raise FileNotFoundError("sealed Pool-3B manifest or audit is unavailable")
    audit = json.loads(audit_path.read_text()).get("audit", {})
    if not (
        audit.get("passed") is True
        and audit.get("base_document_overlap") == 0
        and audit.get("chunk_document_overlap") == 0
    ):
        raise RuntimeError(f"Pool-3B disjointness audit failed: {audit}")
    print(
        "DENSE1B_POOL3B_AUDIT_OK "
        f"manifest={manifest} audit={audit_path} unique_tokens={POOL_TOKENS}",
        flush=True,
    )


def configure_shared_runner() -> None:
    chain.POLICY = POLICY
    chain.POST_DECAY_SOURCE_COUNT = POST_DECAY_SOURCE_COUNT
    chain.POST_DECAY_START_EPOCH = POST_DECAY_START_EPOCH
    chain.CHECKPOINT_ONLY_EPOCHS = CHECKPOINT_ONLY_EPOCHS
    chain.TOKENS_PER_EPOCH = POOL_TOKENS
    chain.DECAY_FRACTION = DECAY_FRACTION
    chain.OUTPUT_ROOT = OUTPUT_ROOT
    chain.total_step = total_step
    chain.stable_step = stable_step
    chain.validate_config = validate_config
    chain.COMMON_ARGUMENTS = (
        "--data-root=/weka/oe-training-default/ai2-llm",
        "--dataset.mix=null",
        f"--dataset.subset_manifest={FULL_POOL}",
        "--dataset.mix_base_dir=/weka/oe-training-default/ai2-llm",
        "--work-dir=/weka/oe-training-default/sewonm/dataset-cache",
        "--trainer.callbacks.wandb.enabled=true",
        "--trainer.callbacks.wandb.entity=ai2-llm",
        "--trainer.callbacks.wandb.project=sewonm-icsl",
        "--trainer.callbacks.downstream_evaluator.tasks=[]",
        "--trainer.callbacks.downstream_evaluator.eval_interval=null",
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
        f"--trainer.callbacks.heldout_evaluator={chain.HELDOUT_EVALUATOR}",
        "--dataset.instance_filter_config={repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}",
        "--model.block.name=default",
        "--model.block.sequence_mixer.qk_norm=null",
        "--init_seed=12536",
        "--data_loader.seed=0",
    )

    def run_name(config: dict[str, Any], lr: str, wd: str, phase: str, epoch: int) -> str:
        return (
            f"dense_1b_pool3b_bs{config['globalSequences']}_dr_wt_embwd_"
            f"{phase}_e{epoch}_lr{lr}_wd{wd}_{config['runSuffix']}"
        )

    chain.run_name = run_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    validate_config(config)
    configure_shared_runner()
    coordinate = config["coordinates"][0]
    if args.validate_only:
        print(f"validated {args.manifest}")
        return
    validate_pool_lineage(config)
    chain.run(config, str(coordinate["lr"]), str(coordinate["wd"]), finalize_only=False)


if __name__ == "__main__":
    main()
