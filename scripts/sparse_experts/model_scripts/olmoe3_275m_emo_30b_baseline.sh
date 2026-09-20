##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_20b_baseline.sh"
# DESCRIPTION:
#     EMO 512e baseline continued from its 20B checkpoint (olmoe3_275m_emo_20b_1node/step38147) to 30B tokens (steps
#     38,147 -> 57,221) on the same stream, constant LR (WSD trunk): the reference for window 2 of the EMO random-partition
#     control (report Q3 block A). Permanent checkpoints at the five matched points of the 20B -> 30B window (39073, 44073,
#     49073, 54073, 57221), one node like the 20B continuation.
#
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_emo_30b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_TOKENS=30000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-1}"
export OLMOE3_FIXED_STEPS=39073,44073,49073,54073,57221
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_emo_30b_1node}"
export OLMOE3_WANDB_TAGS=30b,olmoe3_squares_emorand
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_emo_20b_1node/step38147 \
    --trainer.load_strategy=always
