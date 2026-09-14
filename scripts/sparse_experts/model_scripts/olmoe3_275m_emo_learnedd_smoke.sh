##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_learnedd_10b.sh"
# DESCRIPTION:
#     40-step smoke test of the learned-d EMO arm (mechanics + d statistics in W&B): 1 node, warm-up
#     over 20 steps so both the unrestricted and the restricted phase run, in-loop ppl eval at step 40.
#
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_emo_learnedd_smoke.sh [dry_run]
##############################################################
export OLMOE3_TOKENS=$((40 * 524288))
export OLMOE3_LD_WARMUP=20
export OLMOE3_PPL_EVAL_INTERVAL=40
export OLMOE3_RUNNAME=olmoe3_275m_emo_learnedd_smoke
export OLMOE3_WANDB_TAGS=smoke
export OLMOE3_SAVE_ROOT=/weka/oe-training-default/ryanwang/EMO/sparse_experts/smoke
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_emo_learnedd_10b.sh" "${1:-launch}" \
    --trainer.callbacks.checkpointer.save_interval=20 \
    --trainer.callbacks.checkpointer.ephemeral_save_interval=null
