# PARENT: NONE
# DESCRIPTION:
#     - Extension experiment (math): Add new expert initialized from AVERAGE, then train only the new expert on math dataset
# STATUS: USED
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"



NUM_NEW_EXPERTS=4
TOTAL_EXPERTS=$((132+${NUM_NEW_EXPERTS}))

# Part 1: Add new expert
BASE_MODEL_PATH="${MODELS}/freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_10B_lr_4e-4/step2385"
NEW_BASE_MODEL_PATH="${EXTENSIONS}/moe_1b14b_128base_4math_10B_4code_init_top2_starcoder_average_noise"

# EVAL_DIR="s3://ai2-sewonm/akshitab/mose/evals/extensions/moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123_step30995-hf"

# Run this once; on weka
# python src/scripts/akshitab/add_finegrained_expert/add_new_expert.py \
# 	-c ${BASE_MODEL_PATH}\
# 	-o ${NEW_BASE_MODEL_PATH} \
# 	--num_new_experts ${NUM_NEW_EXPERTS} \
# 	--init_method similar \
#     --activation_file task-mbpp-router.jsonl \
#     -k 2 \
# 	--noise_std_fraction 0.1

NUM_BILLION_TOKENS=10
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))

LR=4e-4 #4e-4  # 4e-3, #4e-5

# # Part 2: Train with new expert
# RUN_NAME="ff-moe1b14b_${TOTAL_EXPERTS}experts_${NUM_NEW_EXPERTS}trained_starcoder_init_top2_average_noise_${NUM_BILLION_TOKENS}B_lr_${LR}"
RUN_NAME="ff-moe_1b14b_128base_4math_10B_${NUM_NEW_EXPERTS}code_init_top2_code_mix_average_noise_${NUM_BILLION_TOKENS}B_lr_${LR}"

launch src/scripts/akshitab/add_finegrained_expert/train_new_expert.py ${RUN_NAME} \
		--trainer.load_path="${NEW_BASE_MODEL_PATH}/model_and_optim" \
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
        --train_module.scheduler.warmup_fraction=0.1 \
        --lr=${LR} \
        --base-model-config="${NEW_BASE_MODEL_PATH}" \
        --num-experts-to-train=${NUM_NEW_EXPERTS}