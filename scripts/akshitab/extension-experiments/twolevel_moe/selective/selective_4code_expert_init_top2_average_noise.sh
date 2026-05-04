source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

# Top 4 code experts (excluding shared expert 127) from router analysis
# python src/scripts/eval/router_analysis.py --router-files router_evals/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301_step30995-hf/task-mbpp-router.jsonl
EXPERTS_TO_TRAIN=63,26,6,19

TOTAL_EXPERTS=128

BASE_MODEL_PATH="${TWOLEVEL_BASE}"

NUM_BILLION_TOKENS=10
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))

LR=4e-4 #4e-4  # 4e-3, #4e-5

RUN_NAME="twolevel_1b14b_${TOTAL_EXPERTS}experts_${EXPERTS_TO_TRAIN//,/_}_trained_code_${NUM_BILLION_TOKENS}B_lr_${LR}"

echo $RUN_NAME

launch src/scripts/akshitab/add_finegrained_expert/train_selected_experts.py ${RUN_NAME} \
		--trainer.load_path="${BASE_MODEL_PATH}/model_and_optim" \
		--save-folder="${MODELS}/${RUN_NAME}" \
		--dataset.mix=code_mix \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration="{value: ${NUM_TOKENS}, unit: tokens}" \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=akshitab \
		--trainer.callbacks.wandb.project=olmoe-modular \
		--trainer.callbacks.wandb.name="${RUN_NAME}" \
		--trainer.callbacks.wandb.tags='[extension]' \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.feed_forward_moe.lb_loss_weight=1e-2 \
		--model.block.sequence_mixer.backend=torch \
        --train_module.scheduler.warmup_fraction=0.1 \
        --lr=${LR} \
        --base-model-config="${BASE_MODEL_PATH}" \
        --experts-to-train=${EXPERTS_TO_TRAIN} \
