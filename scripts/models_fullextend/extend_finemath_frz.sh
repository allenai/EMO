#!/usr/bin/env bash
# models_fullextend extendability eval -- continual-pretrain step.
#
# Takes a ghost-trained checkpoint that has been grown by one instantiated expert
# (128 -> 129; see add_expert_all.sh / add_expert_to_checkpoint.py) and continually
# pretrains it on FineMath with EVERYTHING frozen except the new expert's MLP:
#   - backbone (embeddings, attention, feed_forward_norm, router, lm_head) hard-frozen
#     via --model.freeze_params;
#   - the expert MLP tensor stays trainable, but --freeze-new-expert masks its gradient
#     to update only the last non-shared expert (index 127) and restores the frozen rows
#     each step to undo AdamW weight decay (WD kept at the base 0.1).
# The new expert is forced into every document pool (num_forced_experts=1) and its
# token/document activation rate is logged (num_new_experts=1).
#
# The hypothesis: a model pretrained with ghost experts absorbs the new expert better
# (more activation, larger downstream gains, less forgetting) than one that wasn't.
#
# Usage:
#   bash scripts/models_fullextend/extend_finemath_frz.sh <uniform|usage|random>
#   MODE=beaker bash scripts/models_fullextend/extend_finemath_frz.sh uniform
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

VARIANT="${1:?usage: extend_finemath_frz.sh <uniform|usage|random>}"
case "$VARIANT" in
    uniform|usage|random) ;;
    *) echo "VARIANT must be one of uniform|usage|random (got '$VARIANT')"; exit 1 ;;
esac

EXPERIMENT_NAME="models_fullextend"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

# 10B-token continual-pretrain. Node count is scientifically FREE here: the only cross-DP
# reduction in the randpool router (the reduce-dp batch-level LB stat, router lines ~518-519)
# is gated by the LB-loss block, and we run with lb=0, so it contributes zero gradient; the
# expert-pool routing decision itself is purely per-rank/local. With the global batch fixed in
# config, 8 and 16 nodes give identical training dynamics -- pick by available compute.
# NB: assign UNCONDITIONALLY (not `${BEAKER_NODES:-N}`). launch_common.sh, sourced above,
# already sets BEAKER_NODES=16 at source-time, so a `:-` default here would silently no-op.
BEAKER_NODES=16
BEAKER_GPUS=8

# Router / model geometry (matches the ghost recipe, now with one added expert).
min_document_expert_pool=8
max_document_expert_pool=128     # randomly sampled per doc in [8, 128], matching the ghost pretraining recipe
eval_document_expert_pool=32
num_shared_experts=1
num_experts=129                  # 128 original + 1 new
num_new_experts=1

lr=4e-4
wd=0.1                           # base-recipe weight decay (frozen experts protected by the restorer)
lb=0                             # no load-balancing loss during the frozen continual-pretrain

num_billion_tokens=10
num_tokens=$((num_billion_tokens * 1000000000))

base_model_path="${MODELS_DIR}/emo_1b14b_130b_ghost_${VARIANT}_always_detachF/step11921-plus${num_new_experts}"
runname="emo_1b14b_130b_ghost_${VARIANT}_extend${num_new_experts}_finemath_frz"

launch src/scripts/train/olmoe-1B-7B_fsl_extension.py "$runname" \
    --save-folder="${MODELS_DIR}/${runname}" \
    --work-dir="${DATASET_CACHE}" \
    --data-root="${DATA_ROOT}" \
    --load-path="${base_model_path}/model_and_optim" \
    --no-load-optim-state \
    --num-tokens=${num_tokens} \
    --lr=${lr} \
    --weight-decay=${wd} \
    --freeze-new-expert \
    --num-new-experts=${num_new_experts} \
    --model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool" \
    --min_document_expert_pool=${min_document_expert_pool} \
    --max_document_expert_pool=${max_document_expert_pool} \
    --eval_document_expert_pool=${eval_document_expert_pool} \
    --num_shared_experts=${num_shared_experts} \
    --dataset.mix=mj_finemath4plus \
    --dataset.generate_doc_lengths=true \
    --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
    --model.block.name="moe" \
    --model.block.feed_forward_moe.num_experts=${num_experts} \
    --model.block.feed_forward_moe.lb_loss_weight=${lb} \
    --model.block.feed_forward_moe.router.num_forced_experts=${num_new_experts} \
    --model.block.feed_forward_moe.router.num_new_experts=${num_new_experts} \
    --model.block.sequence_mixer.backend=flash_2 \
    --model.block.sequence_mixer.qk_norm=null \
    --model.freeze_params='[embeddings.*, blocks.*.attention*, blocks.*.feed_forward_norm.*, blocks.*.feed_forward_moe.router.*, lm_head.*]' \
    --trainer.callbacks.checkpointer.save_interval=600 \
    --trainer.callbacks.checkpointer.ephemeral_save_interval=200 \
    --trainer.callbacks.checkpointer.pre_train_checkpoint=false \
    --trainer.callbacks.downstream_evaluator.eval_interval=250 \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.entity=ryanyxw \
    --trainer.callbacks.wandb.project=emo-extension \
    --trainer.callbacks.wandb.name="${runname}" \
    --trainer.callbacks.wandb.tags="[extension, contpretrain, finemath, frz, ${EXPERIMENT_NAME}, ghost_${VARIANT}]"
