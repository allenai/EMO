##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_1000e_emo_61b_baseline.sh"
# DESCRIPTION:
#     EMO 1000e joint baseline continued from its 61B checkpoint (olmoe3_275m_1000e_emo_61b/step116479) to the 87B, 113B and 130B
#     matched points (steps 166,478 / 216,477 / 247,956), constant LR 8e-4, 4 nodes: the reference for the
#     four-square LR-4e-4 control's window 3 beyond 61B (user request 2026-11-10). baseline_keeper.sh resumes the run across the
#     8 h multi-node cut (the trainer resumes from the latest checkpoint in the run folder).
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_1000e_emo_130b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=1000
export OLMOE3_EMO=1
export OLMOE3_TOKENS=130000355328   # 247,956 steps x 524,288 tokens
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=166478,216477,247956
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_1000e_emo_130b}"
export OLMOE3_WANDB_TAGS=130b,olmoe3_squares_emo1000rand4
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_1000e_emo_61b/step116479 \
    --trainer.load_strategy=always
