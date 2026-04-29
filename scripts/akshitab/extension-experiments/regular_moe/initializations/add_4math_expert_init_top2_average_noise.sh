# PARENT: NONE
# DESCRIPTION:
#     - Extension experiment (math): Add new expert initialized from AVERAGE, then train only the new expert on math dataset
# STATUS: USED
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"



NUM_NEW_EXPERTS=4
TOTAL_EXPERTS=$((128+${NUM_NEW_EXPERTS}))

# Part 1: Add new expert
# BASE_MODEL_PATH="${NONSHARED_BASE}"
# NEW_BASE_MODEL_PATH="${EXTENSIONS}/moe_1b14b_${TOTAL_EXPERTS}experts_olmoe-mix_130B_1103_step30995_init_top2_average"

BASE_MODEL_PATH="${REGULAR_BASE}"
NEW_BASE_MODEL_PATH="${EXTENSIONS}/moereducedp512sharedexp1_1b14b_${TOTAL_EXPERTS}experts_0308_step30995_init_top2_average_noise"

NUM_BILLION_TOKENS=10
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))

EVAL_DIR="s3://ai2-sewonm/akshitab/mose/evals/extensions/moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308_step30995-hf"

# Run this once; on weka
# top 2: [69, 30]
# python src/scripts/akshitab/add_finegrained_expert/add_new_expert.py \
# 	-c ${BASE_MODEL_PATH}\
# 	-o ${NEW_BASE_MODEL_PATH} \
# 	--num_new_experts ${NUM_NEW_EXPERTS} \
# 	--init_method similar \
#     --activation_file ${EVAL_DIR}/task-gsm8k_generation_0shot_test-router.jsonl \
#     -k 2 \
# 	--num_shared_experts 1 --exclude_experts 127


LR=4e-4 #4e-4  # 4e-3, #4e-5

# # Part 2: Train with new expert
NUM_SHARED_EXPERTS=1
INSERT_POS=$((128 - NUM_SHARED_EXPERTS))
EXPERTS_TO_TRAIN=$(seq -s, $INSERT_POS $((INSERT_POS + NUM_NEW_EXPERTS - 1)) | sed 's/,$//')

echo $EXPERTS_TO_TRAIN

RUN_NAME="moereducedp512sharedexp1_${TOTAL_EXPERTS}experts_${NUM_NEW_EXPERTS}trained_math_init_top2_average_${NUM_BILLION_TOKENS}B_lr_${LR}"

launch src/scripts/akshitab/add_finegrained_expert/train_selected_experts.py ${RUN_NAME} \
		--trainer.load_path="${NEW_BASE_MODEL_PATH}/model_and_optim" \
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
        --train_module.scheduler.warmup_fraction=0.1 \
        --lr=${LR} \
        --base-model-config="${NEW_BASE_MODEL_PATH}" \
        --experts-to-train=${EXPERTS_TO_TRAIN}