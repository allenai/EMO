##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_1024e_h272_k16_emo_10b.sh"
# DESCRIPTION:
#     Half-size-expert EMO (1024 experts x hidden 272, top-16) joint baseline continued from its 10B checkpoint
#     (olmoe3_275m_1024e_h272_k16_emo_10b/step19074) to 30B tokens (steps 19,074 -> 57,221), constant LR 8e-4, 4 nodes, EP=2
#     sync_1d like the 10B run: the reference for the eight-square control of this model (report Q5 box 5F). Permanent
#     checkpoints at the control's three matched points (30000 = ~16B, 38148 = 20B, 57221 = 30B); baseline_keeper.sh resumes it.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_1024e_h272_k16_emo_30b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=1024
export OLMOE3_EXPERT_HIDDEN=272
export OLMOE3_TOP_K=16
export OLMOE3_EP_SIZE=2
export OLMOE3_EP_PATH=sync_1d
export OLMOE3_EMO=1
export OLMOE3_TOKENS=30000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=30000,38148,57221
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_1024e_h272_k16_emo_30b}"
export OLMOE3_WANDB_TAGS=30b,halfexperts,olmoe3_squares_h272k16rand8
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_1024e_h272_k16_emo_10b/step19074 \
    --trainer.load_strategy=always
