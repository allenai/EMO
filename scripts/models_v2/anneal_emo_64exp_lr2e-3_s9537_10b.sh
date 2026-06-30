#!/bin/bash
# Dedicated WSD decay branch for the EMO 64e lr2e-3 stable trunk (emo_64exp_50b_wsd_lr2e-3):
# anneal from its 40B checkpoint (step9537) for 10B tokens -> ends at step11921 = 50B total. The
# anneal entry point (src/scripts/train/olmoe-1B-7B_fsl_anneal.py) auto-extracts the trunk's flat
# peak LR (2e-3) + global_step from the checkpoint and decays the LR linearly to 0 over 10B tokens.
#
# This is the EMO counterpart to anneal_lr2e-3_s9537_10b.sh / anneal_128exp_lr2e-3_s9537_10b.sh (the
# stdMoE 64e/128e 2e-3 decay@40B/10B branches). Those wrap the reusable launch_wsd_decay.sh, but that
# launcher hardcodes the stdMoE model flags (model-type=moe_lbreducedp_sharedexp, no document-expert
# pool / doc-length args). EMO needs its own architecture flags, so this script is SELF-CONTAINED and
# calls the anneal entry point directly -- but it reuses the same decay-family naming + layout:
#   * saves hierarchically to <trunk>/anneals/s9537_10b/
#   * flat run name emo_64exp_50b_wsd_lr2e-3_anneal_s9537_10b
#   * WandB tag wsd_decay so the decay family filters together.
#
# Model flags MUST match the trunk emo_64exp_50b_wsd_lr2e-3.sh exactly (the model is rebuilt from CLI
# and then the checkpoint state is loaded into it -- a mismatch would fail the load): EMO model-type,
# 64 experts / 1 shared, lb 1e-1, generate_doc_lengths, flash_2, document-expert pool min=8/max=64/
# eval=64. No --lr / --scheduler / --*_steps here: the anneal entry sets WSD (warmup=ckpt_step skip,
# decay=anneal_steps linear to 0) and max_duration = ckpt_step + anneal_steps internally.
#
#   git add . && git commit && git push origin <branch>      # gantry clones from origin
#   MODE=beaker bash scripts/models_v2/anneal_emo_64exp_lr2e-3_s9537_10b.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_v2"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8   # match the trunk: at lb!=0 the reduce-dp batch-LB stats depend on node count
BEAKER_GPUS=8

# EMO config (match emo_64exp_50b_wsd_lr2e-3.sh exactly).
lb=1e-1
num_shared_experts=1
num_experts=64
min_document_expert_pool=8
max_document_expert_pool=64
eval_document_expert_pool=64

# Branch spec.
TRUNK_RUN="emo_64exp_50b_wsd_lr2e-3"
CKPT_STEP=9537    # 40B permanent fixed_step checkpoint in the trunk
DECAY_B=10        # decay 10B tokens -> 40B + 10B = 50B (end step 11921)

anneal_tokens=$(awk "BEGIN{printf \"%.0f\", ${DECAY_B}*1000000000}")
branch_leaf="s${CKPT_STEP}_${DECAY_B}b"                       # s9537_10b
runname="${TRUNK_RUN}_anneal_${branch_leaf}"                  # emo_64exp_50b_wsd_lr2e-3_anneal_s9537_10b
anneal_checkpoint="${MODELS_DIR}/${TRUNK_RUN}/step${CKPT_STEP}"
save_folder="${MODELS_DIR}/${TRUNK_RUN}/anneals/${branch_leaf}"

echo "EMO WSD decay branch:"
echo "  trunk:             ${TRUNK_RUN}"
echo "  branch checkpoint: ${anneal_checkpoint}"
echo "  decay tokens:      ${anneal_tokens} (${DECAY_B}B)"
echo "  run name:          ${runname}"
echo "  save folder:       ${save_folder}"

launch src/scripts/train/olmoe-1B-7B_fsl_anneal.py "$runname" \
		--save-folder="${save_folder}" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--model.block.feed_forward_moe.num_experts=${num_experts} \
		--dataset.generate_doc_lengths=true \
		--model.block.sequence_mixer.backend=flash_2 \
		--model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool" \
		--min_document_expert_pool=${min_document_expert_pool} \
		--max_document_expert_pool=${max_document_expert_pool} \
		--eval_document_expert_pool=${eval_document_expert_pool} \
		--num_shared_experts=$num_shared_experts \
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
