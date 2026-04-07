#!/bin/bash
# PARENT: add_math_experts_training_activations.sh (must run first to create extended checkpoint)
# DESCRIPTION:
#     - Stage 3: Train the 4 new experts + router on mj_finemath4plus
#     - Extended checkpoint created by add_math_experts_training_activations.sh (Stage 2)
# STATUS: NEW
##############################################################

NUM_NEW_EXPERTS=4
TOTAL_EXPERTS=$((128+${NUM_NEW_EXPERTS}))

NEW_BASE_MODEL_PATH="/weka/oe-training-default/kevinf/FlexMoE/models/extensions/moereducedp512sharedexp1_1b14b_${TOTAL_EXPERTS}experts_0308_step30995_init_top2_average_train_act"

NUM_BILLION_TOKENS=10
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))

LR=4e-4

# Stage 3: Train the new experts
NUM_SHARED_EXPERTS=1
INSERT_POS=$((128 - NUM_SHARED_EXPERTS))
EXPERTS_TO_TRAIN=$(seq -s, $INSERT_POS $((INSERT_POS + NUM_NEW_EXPERTS - 1)) | sed 's/,$//')

echo $EXPERTS_TO_TRAIN

RUN_NAME="moereducedp512sharedexp1_${TOTAL_EXPERTS}experts_${NUM_NEW_EXPERTS}trained_math_init_top2_average_train_act_${NUM_BILLION_TOKENS}B_lr_${LR}"

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
	--env-secret "GITHUB_TOKEN=KEVINF_GITHUB_TOKEN" "WANDB_API_KEY=KEVINF_WANDB_API_KEY" "BEAKER_TOKEN=KEVINF_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=KEVINF_HF_TOKEN" \
	-- src/scripts/akshitab/add_finegrained_expert/train_selected_experts.py \
    ${RUN_NAME} \
		--trainer.load_path="${NEW_BASE_MODEL_PATH}/model_and_optim" \
		--save-folder="/weka/oe-training-default/kevinf/FlexMoE/models/${RUN_NAME}" \
		--dataset.mix=mj_finemath4plus \
		--work-dir="/weka/oe-training-default/kevinf/dataset-cache" \
		--trainer.max_duration="{value: ${NUM_TOKENS}, unit: tokens}" \
		--trainer.callbacks.wandb="{enabled: true, entity: kevinfarhat, project: olmoe-modular, name: ${RUN_NAME}, tags: [extension, training-activations]}" \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.feed_forward_moe.lb_loss_weight=1e-2 \
        --train_module.scheduler.warmup_fraction=0.1 \
        --lr=${LR} \
        --base-model-config="${NEW_BASE_MODEL_PATH}" \
        --experts-to-train=${EXPERTS_TO_TRAIN}
