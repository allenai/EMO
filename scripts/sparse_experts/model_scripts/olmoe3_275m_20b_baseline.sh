##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_20b_baseline.sh"
# DESCRIPTION:
#     olmoe3_squares_std baseline: continue the STANDARD-routing 512e model (olmoe3_275m_10b, WSD
#     trunk, constant LR 8e-4) from its step-19074 checkpoint for 10B more tokens (to 20B) in a new
#     save folder, full trainer + optimizer state loaded (same next-10B data order). One node with
#     4-way gradient accumulation (the 4-node copies of the EMO baseline never scheduled).
#
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_20b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=0
export OLMOE3_TOKENS=20000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-1}"
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_20b_1node}"
export OLMOE3_WANDB_TAGS=20b,olmoe3_squares_std
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_10b/step19074 \
    --trainer.load_strategy=always \
    --trainer.callbacks.checkpointer.max_checkpoints=null
