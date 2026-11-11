#!/usr/bin/env bash
##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_randsel_30b_baseline.sh"
# DESCRIPTION:
#     Joint baseline of the MIXED-POOL EMO 512e model (olmoe3_275m_emo_mix20_10b: 20% of the documents get a random pool, 80% EMO's
#     relevance pool; user request 2026-11-11): continued from its 10B checkpoint to 30B on the same stream with the same recipe
#     (OLMOE3_EMO_POOL_SELECT=mix:0.2), saving at every matched point of the 512e controls (windows 1 and 2), as olmoe3_275m_randsel_30b did
#     for the random-pool model. Reference line for the mix20 4- and 8-square controls at sub-model LR 4e-4 (squares_big.sh mix20 k4|k8).
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_mix20_30b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_EMO_POOL_SELECT=mix:0.2
export OLMOE3_TOKENS=30000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-2}"
export OLMOE3_FIXED_STEPS=20000,25000,30000,35000,38148,39073,44073,49073,54073,57221
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_mix20_30b}"
export OLMOE3_WANDB_TAGS=30b,olmoe3_squares_mix20rand8,pool_select_mix0.2
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/debug_validation/olmoe3_275m_emo_mix20_10b/step19074 \
    --trainer.load_strategy=always
