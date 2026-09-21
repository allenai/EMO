##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_130b_baseline.sh"
# DESCRIPTION:
#     EMO 512e baseline continued from its 30B checkpoint (olmoe3_275m_emo_30b_1node/step57221) to 130B tokens (steps
#     57,221 -> 247,956), constant LR, 4 nodes: the reference for window 3 of the EMO random-partition control (report Q3
#     block A), with the same matched checkpoints as the standard one (66481, 116479, 166478, 216477, 247956).
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_TOKENS=130000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=66481,116479,166478,216477,247956
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_emo_130b}"
export OLMOE3_WANDB_TAGS=130b,olmoe3_squares_emorand
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_emo_30b_1node/step57221 \
    --trainer.load_strategy=always
