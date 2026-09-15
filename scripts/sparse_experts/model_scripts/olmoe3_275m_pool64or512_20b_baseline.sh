##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_pool64or512_10b.sh"
# DESCRIPTION:
#     olmoe3_squares baseline for the pool-{64,512} EMO arm: continue olmoe3_275m_emo_pool64or512_10b from its
#     step-19074 checkpoint (trainer state included, so the data order continues) to 20B tokens on one node,
#     every 5000-step checkpoint kept.  bash scripts/sparse_experts/model_scripts/olmoe3_275m_pool64or512_20b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_EMO_POOL_DIST=choice:64,512
export OLMOE3_TOKENS=20000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-1}"
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_pool64or512_20b_1node}"
export OLMOE3_WANDB_TAGS=20b,olmoe3_squares_pool64or512
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_emo_pool64or512_10b/step19074 \
    --trainer.load_strategy=always \
    --trainer.callbacks.checkpointer.max_checkpoints=null
