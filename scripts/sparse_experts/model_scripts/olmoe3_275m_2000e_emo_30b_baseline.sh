##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_emo_10b.sh"
# DESCRIPTION:
#     EMO 2000e joint baseline continued from its 10B checkpoint (olmoe3_275m_2000e_emo_10b/step19074, the ladder's 512e recipe
#     with 2000 routed experts of the same size, EP=2 sync_1d) to 30B tokens (steps 19,074 -> 57,221), constant LR 8e-4, 4 nodes:
#     the reference for the four- and eight-square controls of this model (report Q5 box 5G). Permanent checkpoints at the controls'
#     three matched points (30000 = ~16B, 38148 = 20B, 57221 = 30B); baseline_keeper.sh resumes it across the 8 h multi-node cut.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_emo_30b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=2000
export OLMOE3_EP_SIZE=2
export OLMOE3_EP_PATH=sync_1d
export OLMOE3_EMO=1
export OLMOE3_TOKENS=30000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=30000,38148,57221
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_2000e_emo_30b}"
export OLMOE3_WANDB_TAGS=30b,olmoe3_squares_emo2000rand8
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_2000e_emo_10b/step19074 \
    --trainer.load_strategy=always
