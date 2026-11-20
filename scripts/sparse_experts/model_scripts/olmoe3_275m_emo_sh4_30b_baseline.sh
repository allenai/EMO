##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_sh4_10b.sh"
# DESCRIPTION:
#     Joint baseline of the 4-shared-expert EMO 512e model continued from its 10B checkpoint to 30B (constant LR 8e-4, 4 nodes):
#     the reference for its 4- / 8-sub-model controls (squares_shared4.sh). Permanent checkpoints at the matched points
#     30,000 (15.7B), 38,148 (20B) and 57,221 (30B); baseline_keeper.sh resumes the run across the 8 h multi-node cut.
#
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_emo_sh4_30b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_NUM_SHARED=4
export OLMOE3_TOKENS=30000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=30000,38148,57221
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_emo_sh4_30b}"
export OLMOE3_WANDB_TAGS=30b,shared4
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_emo_sh4_10b/step19074 \
    --trainer.load_strategy=always
