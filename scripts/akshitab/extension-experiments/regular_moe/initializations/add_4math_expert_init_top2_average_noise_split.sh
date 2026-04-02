# PARENT: add_4math_expert_init_top2_average_noise.sh
# DESCRIPTION:
#     - Same as parent but uses split-expert-params for memory-efficient training.
#     - Pipeline: add_new_expert → convert to split → train with --split-expert-params → convert back
# STATUS: NEW
##############################################################


NUM_NEW_EXPERTS=4
TOTAL_EXPERTS=$((128+${NUM_NEW_EXPERTS}))

# Part 1: Add new expert (same as parent — outputs regular checkpoint)
# BASE_MODEL_PATH="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995"

BASE_MODEL_PATH="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995"
NEW_BASE_MODEL_PATH="/weka/oe-training-default/akshitab/FlexMoE/models/extensions/moereducedp512sharedexp1_1b14b_${TOTAL_EXPERTS}experts_0308_step30995_init_top2_average_noise"

EVAL_DIR="s3://ai2-sewonm/akshitab/mose/evals/extensions/moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308_step30995-hf"

NUM_SHARED_EXPERTS=1
INSERT_POS=$((128 - NUM_SHARED_EXPERTS))
EXPERTS_TO_TRAIN=$(seq -s, $INSERT_POS $((INSERT_POS + NUM_NEW_EXPERTS - 1)) | sed 's/,$//')

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


# Part 2: Convert regular checkpoint to split format (run once; on weka)
SPLIT_MODEL_PATH="${NEW_BASE_MODEL_PATH}_split"

# python src/scripts/akshitab/add_finegrained_expert/convert_split_checkpoint.py \
# 	--checkpoint-path ${NEW_BASE_MODEL_PATH} \
# 	--save-path ${SPLIT_MODEL_PATH} \
# 	--experts-to-train ${EXPERTS_TO_TRAIN} \
# 	--to-split


# Part 3: Train with split-expert-params
NUM_BILLION_TOKENS=10
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))

LR=4e-4 #4e-4  # 4e-3, #4e-5

RUN_NAME="moereducedp512sharedexp1_${TOTAL_EXPERTS}experts_${NUM_NEW_EXPERTS}trained_math_init_top2_average_split_${NUM_BILLION_TOKENS}B_lr_${LR}"

echo "EXPERTS_TO_TRAIN=${EXPERTS_TO_TRAIN}"
echo "RUN_NAME=${RUN_NAME}"

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
	-- src/scripts/akshitab/add_finegrained_expert/train_selected_experts.py \
    ${RUN_NAME} \
		--trainer.load_path="${SPLIT_MODEL_PATH}/model_and_optim" \
		--save-folder="/weka/oe-training-default/akshitab/FlexMoE/models/${RUN_NAME}" \
		--dataset.mix=mj_finemath4plus \
		--work-dir="/weka/oe-training-default/akshitab/dataset-cache" \
		--trainer.max_duration="{value: ${NUM_TOKENS}, unit: tokens}" \
		--trainer.callbacks.wandb="{enabled: true, entity: akshitab, project: olmoe-modular, name: ${RUN_NAME}, tags: [extension]}" \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.feed_forward_moe.lb_loss_weight=1e-2 \
        --train_module.scheduler.warmup_fraction=0.1 \
        --lr=${LR} \
        --base-model-config="${NEW_BASE_MODEL_PATH}" \
        --experts-to-train=${EXPERTS_TO_TRAIN} \
        --split-expert-params


# Part 4: Convert trained split checkpoint back to regular (run after training; on weka)
# TRAINED_CHECKPOINT="/weka/oe-training-default/akshitab/FlexMoE/models/${RUN_NAME}/stepXXXX"  # fill in actual step
# python src/scripts/akshitab/add_finegrained_expert/convert_split_checkpoint.py \
# 	--checkpoint-path ${TRAINED_CHECKPOINT} \
# 	--save-path ${TRAINED_CHECKPOINT}_regular \
# 	--experts-to-train ${EXPERTS_TO_TRAIN} \
# 	--to-regular
