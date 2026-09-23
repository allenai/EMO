#!/usr/bin/env bash
# Baseline for the random-pool-selection control (user request 2026-09-23): the EMO 512e model trained with RANDOM per-document
# expert pools (olmoe3_275m_emo_randsel_10b, pool size uniform in [16, 512]) continued jointly from its 10B checkpoint to 30B on the
# same stream with the same recipe (OLMOE3_EMO_POOL_SELECT=random), saving at every matched point of the 512e controls
# (windows 1 and 2). 2 jupiter nodes x 8 H100, allocated (baseline_keeper.sh relaunches it after the 8-h multi-node cut).
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_randsel_30b_baseline.sh [dry_run]
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_EMO_POOL_SELECT=random
export OLMOE3_TOKENS=30000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-2}"
export OLMOE3_FIXED_STEPS=20000,25000,30000,35000,38148,39073,44073,49073,54073,57221
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_randsel_30b}"
export OLMOE3_WANDB_TAGS=30b,olmoe3_squares_randsel4,pool_select_random
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/debug_validation/olmoe3_275m_emo_randsel_10b/step19074 \
    --trainer.load_strategy=always
