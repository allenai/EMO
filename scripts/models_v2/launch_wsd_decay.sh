#!/bin/bash
# Reusable WSD decay-branch (anneal) launcher for models_v2.
#
# WSD workflow: train a STABLE TRUNK once (flat peak LR), then fork cheap DECAY BRANCHES from a
# stable checkpoint — each branch loads that checkpoint (weights + optimizer state + step counter)
# and linearly decays the LR to 0 over a chosen token budget. This avoids retraining the stable
# phase for every "what if we stopped & decayed here" experiment.
#
# This wraps the anneal entry point src/scripts/train/olmoe-1B-7B_fsl_anneal.py, which:
#   * reads the branch checkpoint's global_step and auto-extracts its (peak) LR,
#   * builds a WSD schedule (warmup=checkpoint_step -> skip; decay=anneal_steps -> linear to 0),
#   * loads full state incl. OPTIMIZER (continues the step counter from the branch point),
#   * sets max_duration = checkpoint_step + anneal_steps.
# It is otherwise identical to the plain olmoe-1B-7B_fsl.py for the stdMoE/EMO model-types.
#
# Parameterize a branch via env vars (defaults below launch the headline run):
#   TRUNK_RUN   trunk run dir under MODELS_DIR (default stdmoe_64exp_50b_wsd)
#   CKPT_STEP   branch-point checkpoint step in the trunk (default 8941 = 37.5B, stable)
#   DECAY_B     decay length in BILLIONS of tokens, may be fractional (default 12.5)
#   num_experts / num_shared_experts / lb   stdMoE knobs (match the trunk)
#
# Default run: anneal from stdmoe_64exp_50b_wsd/step8941 (37.5B) and decay for 12.5B tokens, so
# total training reaches exactly 50B (end at step 11921) -- directly comparable to the 50B
# trunk/baseline finals.
#
# NAMING + LAYOUT (decay-family convention):
#   * branches save HIERARCHICALLY inside the trunk dir:
#       <MODELS_DIR>/<TRUNK_RUN>/anneals/s<step>_<len>b/   (e.g. .../anneals/s8941_12p5b/)
#     -- separate from the trunk's own step* dirs. "<len>" uses 'p' for the decimal (12.5 -> 12p5).
#   * flat WandB/Beaker run name: <TRUNK_RUN>_anneal_s<step>_<len>b  (unique, searchable)
#   * WandB tag wsd_decay on every branch so the decay family filters together.
#
# FUTURE TRUNKS: to branch at arbitrary points, a trunk should keep permanent stable checkpoints
# every 5B via fixed_steps = [1192, 2384, 3576, 4768, 5960, 7153, 8345, 9537] (round(k*5e9 /
# 4,194,304), k=1..8 -> 5,10,...,40B) plus the final. The existing trunk only kept 12.5/25/37.5B,
# which is why the only branch point available now is step8941 (37.5B).
#
# Usage:
#   git add . && git commit && git push origin <branch>        # gantry clones from origin
#   MODE=beaker bash scripts/models_v2/launch_wsd_decay.sh      # headline run
#   CKPT_STEP=4768 DECAY_B=20 MODE=beaker bash scripts/models_v2/launch_wsd_decay.sh   # another branch
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_v2"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8   # match the trunk: at lb!=0 the reduce-dp batch-LB stats depend on node count
BEAKER_GPUS=8

# stdMoE config (match the trunk being branched).
num_experts="${num_experts:-64}"
num_shared_experts="${num_shared_experts:-1}"
lb="${lb:-1e-1}"

# Branch spec.
TRUNK_RUN="${TRUNK_RUN:-stdmoe_64exp_50b_wsd}"
CKPT_STEP="${CKPT_STEP:-8941}"
DECAY_B="${DECAY_B:-12.5}"

# Derived (awk for fractional B; bash $(()) is integer-only, and awk %d is 32-bit so use %.0f).
anneal_tokens=$(awk "BEGIN{printf \"%.0f\", ${DECAY_B}*1000000000}")
label="$(echo "${DECAY_B}" | sed 's/\./p/')b"      # 12.5 -> 12p5b
branch_leaf="s${CKPT_STEP}_${label}"                # s8941_12p5b
runname="${TRUNK_RUN}_anneal_${branch_leaf}"        # stdmoe_64exp_50b_wsd_anneal_s8941_12p5b
anneal_checkpoint="${MODELS_DIR}/${TRUNK_RUN}/step${CKPT_STEP}"
save_folder="${MODELS_DIR}/${TRUNK_RUN}/anneals/${branch_leaf}"

echo "WSD decay branch:"
echo "  trunk:            ${TRUNK_RUN}"
echo "  branch checkpoint:${anneal_checkpoint}"
echo "  decay tokens:     ${anneal_tokens} (${DECAY_B}B)"
echo "  run name:         ${runname}"
echo "  save folder:      ${save_folder}"

# No --lr / --scheduler / --*_steps: the anneal entry point sets WSD internally and extracts the
# peak LR from the checkpoint. Model flags mirror stdmoe_64exp_50b_wsd.sh exactly.
launch src/scripts/train/olmoe-1B-7B_fsl_anneal.py "$runname" \
		--save-folder="${save_folder}" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--model-type="moe_lbreducedp_sharedexp" \
		--num_shared_experts=$num_shared_experts \
		--model.block.feed_forward_moe.num_experts=${num_experts} \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, wsd_decay]" \
		--anneal-tokens=${anneal_tokens} \
		--anneal-checkpoint=${anneal_checkpoint}
