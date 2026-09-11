##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh"
# DESCRIPTION:
#     olmoe3_squares_std sub-model g (0..3): the STANDARD-routing 512e model (olmoe3_275m_10b) with
#     layers 2-9 cut down to block-group g's experts (layer 1 whole), initialised from the sliced
#     step-19074 checkpoint, trained on group g's re-packed documents of the next 10B tokens with
#     plain top-16 routing (no EMO); otherwise identical to the EMO squares.
#
#   SQUARE_GROUP=g bash scripts/sparse_experts/model_scripts/olmoe3_275m_std_square.sh [dry_run]
##############################################################
export OLMOE3_EMO=0
export SQUARES_NAME=olmoe3_squares_std
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_square${SQUARE_GROUP:?set SQUARE_GROUP}}"
export OLMOE3_WANDB_TAGS=olmoe3_squares_std,square
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_emo_square.sh" "$@"
