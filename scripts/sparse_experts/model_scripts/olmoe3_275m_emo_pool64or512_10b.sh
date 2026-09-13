##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_10b.sh"
# DESCRIPTION:
#     EMO 512-expert arm trained from scratch for 10B tokens exactly like olmoe3_275m_emo_10b (WSD trunk,
#     LR 8e-4, no decay, 5000-step checkpoints), except that each document's expert-pool size is a
#     random choice between 64 and 512 (OLMOE3_EMO_POOL_DIST=choice:64,512) instead of uniform on
#     [16, 512]; eval pool 512. One node (8 GPUs x 2 seq x 4 accumulation = the same 64-seq batch).
#
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_emo_pool64or512_10b.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_EMO_POOL_DIST=choice:64,512
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-1}"
export OLMOE3_RUNNAME=olmoe3_275m_emo_pool64or512_10b
export OLMOE3_WANDB_TAGS=10b
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
