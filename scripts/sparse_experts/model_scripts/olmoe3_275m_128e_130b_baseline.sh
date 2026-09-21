##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_128e_10b.sh"
# DESCRIPTION:
#     Standard-routing 128-expert model (same expert size as the 512e model, top-16 of 128) continued jointly from its 10B
#     checkpoint (olmoe3_275m_128e_10b/step19074) to 130B tokens (steps 19,074 -> 247,956), constant LR, 4 nodes: the baseline
#     of the 128e random-partition controls (report Q3 block F), with checkpoints at every matched point of the 512e controls.
##############################################################
export OLMOE3_NUM_EXPERTS=128
export OLMOE3_EMO=0
export OLMOE3_TOKENS=130000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=20000,25000,30000,35000,38148,39073,44073,49073,54073,57221,66481,116479,166478,216477,247956
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_128e_130b}"
export OLMOE3_WANDB_TAGS=130b,olmoe3_squares_s128rand
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_128e_10b/step19074 \
    --trainer.load_strategy=always
