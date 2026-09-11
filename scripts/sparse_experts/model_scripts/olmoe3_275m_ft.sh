##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_10b.sh"
# DESCRIPTION:
#     olmoe3_squares post-merge finetuning: resume training from a hybrid checkpoint built by
#     olmoe3_squares/make_finetune_start.py (model + carried Adam state from the merged squares or a
#     baseline, trainer state pinned to FT_START_STEP) for FT_STEPS more steps (default 954 =
#     0.5B tokens) on the training stream past the held-out window, constant LR 8e-4, one node,
#     one permanent checkpoint at the end.
#
#   FT_START=<hybrid dir> FT_START_STEP=38548 OLMOE3_EMO=1 OLMOE3_RUNNAME=<name> \
#     bash scripts/sparse_experts/model_scripts/olmoe3_275m_ft.sh [dry_run]
##############################################################
: "${FT_START:?set FT_START (dir containing step<FT_START_STEP>)}"
FT_START_STEP="${FT_START_STEP:-38548}"; FT_STEPS="${FT_STEPS:-954}"
end=$((FT_START_STEP + FT_STEPS))
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO="${OLMOE3_EMO:-1}"
export OLMOE3_TOKENS=$((end * 524288))
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-1}"
export OLMOE3_FIXED_STEPS="$end"
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:?set OLMOE3_RUNNAME}"
export OLMOE3_WANDB_TAGS="${OLMOE3_WANDB_TAGS:-olmoe3_squares,finetune}"
echo "finetune $OLMOE3_RUNNAME: steps $FT_START_STEP -> $end ($FT_STEPS steps = $((FT_STEPS * 524288 / 1000000))M tokens)"
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path="$FT_START/step$FT_START_STEP" \
    --trainer.load_strategy=always \
    --trainer.callbacks.checkpointer.max_checkpoints=null
