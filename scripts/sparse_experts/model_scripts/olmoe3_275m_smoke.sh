# DESCRIPTION:
#     Smoke test for olmoe3_275m_10b.sh: same model / data / topology, 40 optimizer steps
#     (~21M tokens) so the whole stack runs once (image, pinned OLMo-core install, Dolma 3.5 index
#     build from S3, FLA KDA + flash-attn 2 kernels, fused MoE-v2 blocks, 4-node DDP, checkpoint
#     to weka, W&B).
#     Distinct run name + save root so the real run can never resume from it.
#
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_smoke.sh [dry_run]
##############################################################
export OLMOE3_TOKENS=$((40 * 524288))
export OLMOE3_RUNNAME=olmoe3_275m_smoke
export OLMOE3_WANDB_TAGS=smoke
export OLMOE3_SAVE_ROOT=/weka/oe-training-default/ryanwang/EMO/sparse_experts/smoke
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.callbacks.checkpointer.save_interval=20 \
    --trainer.callbacks.checkpointer.ephemeral_save_interval=null
