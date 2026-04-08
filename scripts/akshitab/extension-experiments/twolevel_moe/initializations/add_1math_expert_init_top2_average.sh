# PARENT: NONE
# DESCRIPTION:
#     - Extension experiment (math): Add new expert initialized from AVERAGE, then train only the new expert on math dataset
# STATUS: USED
##############################################################


NUM_NEW_EXPERTS=1
TOTAL_EXPERTS=$((128+${NUM_NEW_EXPERTS}))

# Part 1: Add new expert
BASE_MODEL_PATH="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995"
NEW_BASE_MODEL_PATH="/weka/oe-training-default/akshitab/FlexMoE/models/extensions/twolevelbatchlb-32_1b14b_${TOTAL_EXPERTS}experts_stability_prenorm_noqknorm_1121_step30995_init_top2_average"

NUM_BILLION_TOKENS=5
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))

# EVAL_DIR="s3://ai2-sewonm/akshitab/mose/evals/extensions/moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995-hf"

# # Run this once; on weka
# python src/scripts/akshitab/add_finegrained_expert/add_new_expert.py \
# 	-c ${BASE_MODEL_PATH}\
# 	-o ${NEW_BASE_MODEL_PATH} \
# 	--num_new_experts 1 \
# 	--init_method similar \
#     --activation_file ${EVAL_DIR}/task-gsm8k_generation_test_0shot-router.jsonl \
#     -k 2


LR=4e-4 #4e-4  # 4e-3, #4e-5

# # Part 2: Train with new expert
RUN_NAME="freeze-fix-twolevel_${TOTAL_EXPERTS}experts_${NUM_NEW_EXPERTS}trained_math_init_top2_average_${NUM_BILLION_TOKENS}B_lr_${LR}"

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
	--env-secret "GITHUB_TOKEN=AKSHITAB_GITHUB_TOKEN" "WANDB_API_KEY=AKSHITAB_WANDB_API_KEY" "BEAKER_TOKEN=AKSHITAB_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
	-- src/scripts/akshitab/add_finegrained_expert/train_new_expert.py \
    ${RUN_NAME} \
		--trainer.load_path="${NEW_BASE_MODEL_PATH}/model_and_optim" \
		--save-folder="/weka/oe-training-default/akshitab/FlexMoE/models/${RUN_NAME}" \
		--dataset.mix=mj_finemath4plus \
		--work-dir="/weka/oe-training-default/akshitab/dataset-cache" \
		--trainer.max_duration="{value: ${NUM_TOKENS}, unit: tokens}" \
		--trainer.callbacks.wandb="{enabled: true, entity: akshitab, project: olmoe-modular, name: ${RUN_NAME}, tags: [extension]}" \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.attention.qk_norm=null \
		--model.block.feed_forward_moe.lb_loss_weight=1e-2 \
        --train_module.scheduler.warmup_fraction=0.1 \
        --lr=${LR} \
        --num-experts-to-train=${NUM_NEW_EXPERTS}