# PARENT: NONE
# DESCRIPTION:
#     - Extension experiment (math): Add new expert initialized from AVERAGE, then train only the new expert on math dataset
# STATUS: USED
##############################################################


NUM_NEW_EXPERTS=8 # 4 math + 4 code experts
TOTAL_EXPERTS=$((128+${NUM_NEW_EXPERTS}))

MERGED_MODEL_PATH="/weka/oe-training-default/akshitab/FlexMoE/models/merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise"


NUM_BILLION_TOKENS=1
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))

LR=4e-4 #4e-4  # 4e-3, #4e-5

# # Part 2: Train with new expert
RUN_NAME="rt-merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise_${NUM_BILLION_TOKENS}B_lr_${LR}"

python -m olmo_core.launch.beaker \
  --name ${RUN_NAME} \
	--gpus 8 \
  --nodes 4 \
	--weka=oe-training-default \
  --shared-filesystem \
	--workspace ai2/flex2 \
	--cluster ai2/jupiter \
  --is_private_repo \
	--preemptible \
	--allow-dirty \
	--priority urgent \
	--env-secret "GITHUB_TOKEN=AKSHITAB_GITHUB_TOKEN" "WANDB_API_KEY=AKSHITAB_WANDB_API_KEY" "BEAKER_TOKEN=AKSHITAB_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
	-- src/scripts/akshitab/add_finegrained_expert/train_router.py \
    ${RUN_NAME} \
		--trainer.load_path="${MERGED_MODEL_PATH}/model_and_optim" \
		--save-folder="/weka/oe-training-default/akshitab/FlexMoE/models/${RUN_NAME}" \
		--dataset.mix=proxy_mix_base_math_code \
		--work-dir="/weka/oe-training-default/akshitab/dataset-cache" \
		--trainer.max_duration="{value: ${NUM_TOKENS}, unit: tokens}" \
		--trainer.callbacks.wandb="{enabled: true, entity: akshitab, project: olmoe-modular, name: ${RUN_NAME}, tags: [extension]}" \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.attention.qk_norm=null \
		--model.block.feed_forward_moe.lb_loss_weight=1e-2 \
        --train_module.scheduler.warmup_fraction=0.1 \
        --lr=${LR} \
        --num-new-experts=${NUM_NEW_EXPERTS}