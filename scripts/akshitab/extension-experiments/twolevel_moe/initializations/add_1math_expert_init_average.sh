# PARENT: NONE
# DESCRIPTION:
#     - Extension experiment (math): Add new expert initialized from AVERAGE, then train only the new expert on math dataset
# STATUS: USED
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"



NUM_NEW_EXPERTS=1
TOTAL_EXPERTS=$((128+${NUM_NEW_EXPERTS}))

# Part 1: Add new expert
BASE_MODEL_PATH="${BASE_MODELS}/twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995"
NEW_BASE_MODEL_PATH="${EXTENSIONS}/twolevelbatchlb-32_1b14b_${TOTAL_EXPERTS}experts_stability_prenorm_noqknorm_1121_step30995_init_average"

NUM_BILLION_TOKENS=5
NUM_TOKENS=25 #$((NUM_BILLION_TOKENS * 1000000000))

# Run this once; on weka
# python src/scripts/akshitab/add_finegrained_expert/add_new_expert.py \
# 	-c ${BASE_MODEL_PATH}\
# 	-o ${NEW_BASE_MODEL_PATH} \
# 	--num_new_experts 1 \
# 	--init_method average


LR=4e-4 #4e-4  # 4e-3, #4e-5

# # Part 2: Train with new expert
#RUN_NAME="freeze-fix-twolevel_${TOTAL_EXPERTS}experts_${NUM_NEW_EXPERTS}trained_math_init_average_${NUM_BILLION_TOKENS}B_lr_${LR}"
RUN_NAME="test-hf-converter"

launch src/scripts/akshitab/add_finegrained_expert/train_new_expert.py ${RUN_NAME} \
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
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--model.block.feed_forward_moe.lb_loss_weight=1e-2 \
        --train_module.scheduler.warmup_fraction=0.1 \
        --lr=${LR} \
        --num-experts-to-train=${NUM_NEW_EXPERTS}