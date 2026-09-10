##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_10b.sh"
# DESCRIPTION:
#     Smoke test for the 2000-expert / EP=2 (sync_1d) arm: one jupiter node (8 GPUs -> EP mesh
#     (4, 2)), 40 optimizer steps with grad accumulation (8 ranks x 2 seq x 4 = the fixed
#     64-sequence global batch), checkpoint at step 20 and 40 so the EP-sharded expert state
#     round-trips. Distinct run name + save root so the real run can never resume from it.
#     Pass OLMOE3_EMO=1 to smoke the EMO variant.
#
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_smoke.sh [dry_run]
##############################################################
export OLMOE3_TOKENS=$((40 * 524288))
export OLMOE3_NUM_NODES=1
export OLMOE3_RUNNAME="olmoe3_275m_2000e${OLMOE3_EMO:+_emo}_smoke"
export OLMOE3_WANDB_TAGS=smoke
export OLMOE3_SAVE_ROOT=/weka/oe-training-default/ryanwang/EMO/sparse_experts/smoke
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_2000e_10b.sh" "${1:-launch}" \
    --trainer.callbacks.checkpointer.save_interval=20 \
    --trainer.callbacks.checkpointer.ephemeral_save_interval=null
