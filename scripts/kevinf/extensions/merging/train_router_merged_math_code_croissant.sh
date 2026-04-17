#!/bin/bash
# Train only the router on the merged 140-expert model (128 base + 4 math + 4 code + 4 croissant).
# All expert weights and non-MoE params are frozen — only the router learns to route
# tokens to the right domain experts.

NUM_NEW_EXPERTS=12  # 4 math + 4 code + 4 croissant
TOTAL_EXPERTS=$((128+${NUM_NEW_EXPERTS}))

MERGED_MODEL_PATH="/weka/oe-training-default/kevinf/extension-experiments/merged-math-code-croissant-train-act"

NUM_BILLION_TOKENS=1
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))

LR=4e-4

RUN_NAME="rt-merged_math_code_croissant_train_act_${NUM_BILLION_TOKENS}B_lr_${LR}"

python -m olmo_core.launch.beaker \
  --name ${RUN_NAME} \
  --gpus 8 \
  --nodes 4 \
  --weka=oe-training-default \
  --shared-filesystem \
  --workspace ai2/flex2 \
  --cluster ai2/jupiter \
  --preemptible \
  --allow-dirty \
  --priority urgent \
  --env-secret "GITHUB_TOKEN=KEVINF_GITHUB_TOKEN" "WANDB_API_KEY=KEVINF_WANDB_API_KEY" "BEAKER_TOKEN=KEVINF_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=KEVINF_HF_TOKEN" \
  -- src/scripts/akshitab/add_finegrained_expert/train_router.py \
    ${RUN_NAME} \
    --trainer.load_path="${MERGED_MODEL_PATH}/model_and_optim" \
    --save-folder="/weka/oe-training-default/kevinf/extension-experiments/${RUN_NAME}" \
    --dataset.mix=base_math_code \
    --work-dir="/weka/oe-training-default/kevinf/dataset-cache" \
    --trainer.max_duration="{value: ${NUM_TOKENS}, unit: tokens}" \
    --trainer.callbacks.wandb="{enabled: true, entity: allennlp, project: flex2-extensions-kevinf, name: ${RUN_NAME}, tags: [extension, router-training, merged]}" \
    --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
    --model.block.name="moe" \
    --model.block.attention.qk_norm=null \
    --model.block.feed_forward_moe.lb_loss_weight=1e-2 \
    --train_module.scheduler.warmup_fraction=0.1 \
    --lr=${LR} \
    --num-new-experts=${NUM_NEW_EXPERTS}
