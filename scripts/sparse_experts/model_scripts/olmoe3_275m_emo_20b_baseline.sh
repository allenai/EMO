##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_10b.sh"
# DESCRIPTION:
#     olmoe3_squares baseline: continue the EMO 512e model (olmoe3_275m_emo_10b, WSD trunk, constant
#     LR 8e-4) from its step-19074 checkpoint for 10B more tokens (to 20B) in a NEW save folder.
#     Trainer state (step, data-loader position, RNG) and the full optimizer state are loaded from
#     the checkpoint, so the run consumes exactly the next 10B tokens of the same Dolma order
#     (steps 19074..38147 = the stream the sub-models are partitioned from). Permanent checkpoints
#     every 5000 steps -> 20000 / 25000 / 30000 / 35000 / 38148; the sub-models checkpoint at the
#     same fractions of progress so intermediate merges can be compared.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_emo_20b_baseline.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_TOKENS=20000000000
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_emo_20b}"
export OLMOE3_WANDB_TAGS=20b,olmoe3_squares
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_emo_10b/step19074 \
    --trainer.load_strategy=always
