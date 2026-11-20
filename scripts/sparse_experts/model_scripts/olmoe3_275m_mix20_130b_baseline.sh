##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_mix20_61b_baseline.sh"
# DESCRIPTION:
#     Mixed-pool EMO 512e (20% random pools) joint baseline continued from its 61B checkpoint
#     (olmoe3_275m_mix20_61b/step116479) to 130B (step 247,956), constant LR 8e-4, 4 nodes: the reference for the
#     4-sub-model re-partition-every-5B arm's extension to 130B (user request 2026-11-20; squares_repartition_ext.sh
#     joint130 / rp5b130). Checkpoints at the standard matched points 166,478 (87B), 216,477 (113B), 247,956 (130B) and at
#     the arm's cycle ends at 70B / 80B / ... / 120B (133,513 ... 228,881; evaluated, then deleted by the driver).
#     baseline_keeper.sh resumes the run across the 8 h multi-node cut.
#
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_mix20_130b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_EMO_POOL_SELECT=mix:0.2
export OLMOE3_TOKENS=130000355328   # 247,956 steps x 524,288 tokens
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=133513,152589,166478,171661,190737,209809,216477,228881,247956
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_mix20_130b}"
export OLMOE3_WANDB_TAGS=130b,olmoe3_squares_mix20rand4_rp5b,pool_select_mix0.2
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_mix20_61b/step116479 \
    --trainer.load_strategy=always
