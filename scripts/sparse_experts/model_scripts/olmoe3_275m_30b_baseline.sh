##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_20b_baseline.sh"
# DESCRIPTION:
#     olmoe3_squares_std window 2: continue the standard-MoE baseline (olmoe3_275m_20b_1node, step 38147 = 20B tokens)
#     to 30B tokens on one node, trainer state included so the data order continues (stream steps 38147 -> 57221).
#     Fixed checkpoints at the same five progress fractions of the new 10B window as window 1
#     (926/5926/10926/15926/19074 of 19074 -> steps 39073/44073/49073/54073/57221).
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_30b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=0
export OLMOE3_TOKENS=30000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-1}"
export OLMOE3_FIXED_STEPS=39073,44073,49073,54073,57221
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_30b_1node}"
export OLMOE3_WANDB_TAGS=30b,olmoe3_squares_std
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_20b_1node/step38147 \
    --trainer.load_strategy=always
