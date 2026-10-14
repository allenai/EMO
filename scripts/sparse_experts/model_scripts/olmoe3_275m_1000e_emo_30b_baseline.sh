##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_1000e_emo_10b.sh"
# DESCRIPTION:
#     EMO 1000e joint baseline continued from its 10B checkpoint (olmoe3_275m_1000e_emo_10b/step19074, the same 10B of
#     data as the 512e EMO model) to 30B tokens (steps 19,074 -> 57,221) on the same stream, constant LR 8e-4 (WSD trunk),
#     4 nodes like the 10B run (EP1): the reference for the 8-square EMO 1000e control (report Q5 box 5E). Permanent
#     checkpoints at the three matched points of that control (30000 = ~16B, 38148 = 20B, 57221 = 30B); the rolling
#     500-step ephemeral checkpoint lets baseline_keeper.sh resume it across the 8-hour multi-node runtime cut.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_1000e_emo_30b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=1000
export OLMOE3_EMO=1
export OLMOE3_TOKENS=30000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=30000,38148,57221
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_1000e_emo_30b}"
export OLMOE3_WANDB_TAGS=30b,olmoe3_squares_emo1000rand8
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_1000e_emo_10b/step19074 \
    --trainer.load_strategy=always
