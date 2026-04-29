# PARENT: NONE
# DESCRIPTION:
#     - Extension experiment (math): Add new expert initialized from AVERAGE, then train only the new expert on math dataset
# STATUS: USED
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"



NUM_NEW_EXPERTS=8 # 4 math + 4 code experts
TOTAL_EXPERTS=$((128+${NUM_NEW_EXPERTS}))

MERGED_MODEL_PATH="${MODELS}/merged_twolevel_1b14b_128base_4math_10B_4code_mix_10B_forced_init_top2_average_noise"


NUM_BILLION_TOKENS=1
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))

LR=4e-4 #4e-4  # 4e-3, #4e-5

# # Part 2: Train with new expert
RUN_NAME="rt-merged_twolevel_1b14b_128base_4math_10B_4code_mix_10B_forced_init_top2_average_noise_${NUM_BILLION_TOKENS}B_lr_${LR}"

launch src/scripts/akshitab/add_finegrained_expert/train_router.py ${RUN_NAME} \
		--trainer.load_path="${MERGED_MODEL_PATH}/model_and_optim" \
		--save-folder="${MODELS}/${RUN_NAME}" \
		--dataset.mix=base_math_code \
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
        --num-new-experts=${NUM_NEW_EXPERTS}