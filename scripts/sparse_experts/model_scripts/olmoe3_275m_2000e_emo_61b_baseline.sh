##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_emo_30b_baseline.sh"
# DESCRIPTION:
#     EMO 2000e joint baseline continued from its 30B checkpoint (olmoe3_275m_2000e_emo_30b/step57221) to the 61B matched
#     point (step 116,479 = 61.07B tokens), constant LR 8e-4, 4 nodes, EP=2 sync_1d: the reference for the
#     four-square control's third window (report Q5 box 5G; user request 2026-10-18). One permanent checkpoint, at
#     116,479; baseline_keeper.sh resumes the run across the 8 h multi-node cut.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_emo_61b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=2000
export OLMOE3_EP_SIZE=2
export OLMOE3_EP_PATH=sync_1d
export OLMOE3_EMO=1
export OLMOE3_TOKENS=61068017792   # 116,479 steps x 524,288 tokens
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=116479
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_2000e_emo_61b}"
export OLMOE3_WANDB_TAGS=61b,olmoe3_squares_emo2000rand4
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_2000e_emo_30b/step57221 \
    --trainer.load_strategy=always
