#!/usr/bin/env bash
# Window-3 baseline of the random-pool-selection control (user request 2026-09-24): the random-pool EMO 512e model
# (pool size uniform in [64, 512], random experts in the pool) continued jointly from its 30B checkpoint (olmoe3_275m_randsel64_30b/step57221)
# to 130B tokens (steps 57,221 -> 247,956) with the same recipe, constant LR, 4 jupiter nodes x 8 H100, allocated
# (baseline_keeper.sh relaunches it after the 8-h multi-node cut), saving at the matched checkpoints of the other 130B controls.
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_randsel64_130b_baseline.sh [dry_run]
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_EMO_POOL_SELECT=random
export OLMOE3_EMO_MIN_POOL=64
export OLMOE3_TOKENS=130000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=66481,116479,166478,216477,247956
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_randsel64_130b}"
export OLMOE3_WANDB_TAGS=130b,olmoe3_squares_randsel64k4,pool_select_random,pool_min64
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_randsel64_30b/step57221 \
    --trainer.load_strategy=always
