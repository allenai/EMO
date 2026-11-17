##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_mix20_30b_baseline.sh"
# DESCRIPTION:
#     Mixed-pool EMO 512e (20% random pools) joint baseline continued from its 30B checkpoint
#     (olmoe3_275m_mix20_30b/step57221) to the 61B matched point (step 116,479 = 61.07B tokens), constant LR 8e-4,
#     4 nodes: the reference for the re-partitioned sub-model arms' extension to 61B (user request 2026-11-17;
#     squares_repartition_ext.sh). Permanent checkpoints at the matched points 66,481 (34.9B) and 116,479;
#     baseline_keeper.sh resumes the run across the 8 h multi-node cut.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_mix20_61b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_EMO_POOL_SELECT=mix:0.2
export OLMOE3_TOKENS=61068017792   # 116,479 steps x 524,288 tokens
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=66481,116479
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_mix20_61b}"
export OLMOE3_WANDB_TAGS=61b,olmoe3_squares_mix20rand8,pool_select_mix0.2
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_mix20_30b/step57221 \
    --trainer.load_strategy=always
