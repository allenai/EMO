source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

EXPERTS_TO_TRAIN=76,41,120,3  # top 4 experts as per python src/scripts/eval/router_analysis.py --router-files router_evals/moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123_step30995-hf/task-mbpp-router.jsonl

TOTAL_EXPERTS=128

BASE_MODEL_PATH="${NONSHARED_BASE}"

NUM_BILLION_TOKENS=10
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))

LR=4e-4 #4e-4  # 4e-3, #4e-5

RUN_NAME="moe1b14b_${TOTAL_EXPERTS}experts_${EXPERTS_TO_TRAIN//,/_}_trained_code_no_router_${NUM_BILLION_TOKENS}B_lr_${LR}"

echo $RUN_NAME

launch src/scripts/akshitab/add_finegrained_expert/train_selected_experts_no_router.py ${RUN_NAME} \
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
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--model.block.feed_forward_moe.lb_loss_weight=1e-2 \
        --train_module.scheduler.warmup_fraction=0.1 \
        --lr=${LR} \
        --experts-to-train=${EXPERTS_TO_TRAIN}
