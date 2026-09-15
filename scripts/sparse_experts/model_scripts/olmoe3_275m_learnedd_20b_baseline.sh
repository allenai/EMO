##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_learnedd_10b.sh"
# DESCRIPTION:
#     olmoe3_squares baseline for the learned-pool EMO arm: continue olmoe3_275m_emo_learnedd_10b from its
#     step-19074 checkpoint (trainer state included) to 20B tokens on one node, every 5000-step checkpoint kept;
#     same learned-d settings as the 10B run (coverage signal, threshold 0.002).
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_learnedd_20b_baseline.sh [dry_run]
##############################################################
export OLMOE3_TOKENS=20000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-1}"
export OLMOE3_PPL_EVAL_INTERVAL=0
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_learnedd_20b_1node}"
export OLMOE3_WANDB_TAGS=20b,olmoe3_squares_learnedd
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_emo_learnedd_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_emo_learnedd_10b/step19074 \
    --trainer.load_strategy=always \
    --trainer.callbacks.checkpointer.max_checkpoints=null
