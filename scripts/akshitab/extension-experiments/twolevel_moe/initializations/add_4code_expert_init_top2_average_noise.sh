# PARENT: NONE
# DESCRIPTION:
#     - Extension experiment (code): Add new expert initialized from top2 average + noise, then train only the new expert on code dataset
# STATUS: NEW
##############################################################


NUM_NEW_EXPERTS=4
TOTAL_EXPERTS=$((128+${NUM_NEW_EXPERTS}))

# Part 1: Add new expert
BASE_MODEL_PATH="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995"
NEW_BASE_MODEL_PATH="/weka/oe-training-default/akshitab/FlexMoE/models/extensions/twolevelbatchlbreducedp512sharedexp1randpool_1b14b_${TOTAL_EXPERTS}experts_0301_step30995_init_top2_code_average_noise"

# Run this once; on weka
python src/scripts/akshitab/add_finegrained_expert/add_new_expert.py \
	-c ${BASE_MODEL_PATH}\
	-o ${NEW_BASE_MODEL_PATH} \
	--num_new_experts ${NUM_NEW_EXPERTS} \
	--init_method similar \
    --activation_file s3://ai2-sewonm/akshitab/mose/evals/extensions/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301_step30995-hf/task-mbpp-router.jsonl \
    -k 2 \
	--noise_std_fraction 0.1


# # Part 2: Train with new expert
NUM_BILLION_TOKENS=10
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))


LR=4e-4 #4e-4  # 4e-3, #4e-5

# # Part 2: Train with new expert
RUN_NAME="twolevel_${TOTAL_EXPERTS}experts_${NUM_NEW_EXPERTS}trained_code_mix_init_top2_average_noise_${NUM_BILLION_TOKENS}B_lr_${LR}"

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
		--dataset.mix=code_mix \
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