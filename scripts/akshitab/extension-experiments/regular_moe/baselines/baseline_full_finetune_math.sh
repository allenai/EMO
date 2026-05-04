source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

TOTAL_EXPERTS=128

# Full finetuning - nothing frozen
EXPERTS_TO_TRAIN=$(seq -s, 0 $((TOTAL_EXPERTS - 1)) | sed 's/,$//')

# BASE_MODEL_PATH="${NONSHARED_BASE}"
BASE_MODEL_PATH="${REGULAR_BASE}"

NUM_BILLION_TOKENS=10
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))

LR=4e-4 #4e-4  # 4e-3, #4e-5

RUN_NAME="moereducedp512sharedexp1_1b14b_${TOTAL_EXPERTS}experts_full_finetune_math_${NUM_BILLION_TOKENS}B_lr_${LR}"

echo $RUN_NAME

launch src/scripts/akshitab/add_finegrained_expert/train_selected_experts.py ${RUN_NAME} \
		--trainer.load_path="${BASE_MODEL_PATH}/model_and_optim" \
		--save-folder="${MODELS}/${RUN_NAME}" \
		--dataset.mix=mj_finemath4plus \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration="{value: ${NUM_TOKENS}, unit: tokens}" \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=akshitab \
		--trainer.callbacks.wandb.project=olmoe-modular \
		--trainer.callbacks.wandb.name="${RUN_NAME}" \
		--trainer.callbacks.wandb.tags='[extension]' \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.feed_forward_moe.lb_loss_weight=1e-2 \
		--model.freeze_params='[]' \
        --train_module.scheduler.warmup_fraction=0.1 \
        --lr=${LR} \
        --base-model-config="${BASE_MODEL_PATH}" \
        --experts-to-train=${EXPERTS_TO_TRAIN}
