#!/bin/bash
# PARENT: math_training_data_activations.sh (must run first to produce activation file)
# DESCRIPTION:
#     - Stage 2: Add 4 new experts to moereducedp512sharedexp1 using training-data activations
#     - Reads activation file from S3, reads base checkpoint from WEKA, writes extended checkpoint to WEKA
#     - CPU-only job (no GPU needed), just checkpoint manipulation
# STATUS: NEW
##############################################################

NUM_NEW_EXPERTS=4
TOTAL_EXPERTS=$((128+${NUM_NEW_EXPERTS}))

BASE_MODEL_PATH="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995"
NEW_BASE_MODEL_PATH="/weka/oe-training-default/kevinf/FlexMoE/models/extensions/moereducedp512sharedexp1_1b14b_${TOTAL_EXPERTS}experts_0308_step30995_init_top2_average_train_act"

# Activation file from training data (produced by math_training_data_activations.sh)
ACTIVATION_FILE="s3://ai2-kevinf/FlexMoE/training-activations/moereducedp512sharedexp1_1b14b_0308_step30995/mj_finemath4plus-mj_finemath4plus-router.jsonl"

CLUSTER="ai2/jupiter-cirrascale-2"
JOB_NAME="add-experts-math-training-act"

gantry run \
    --name $JOB_NAME \
    --weka oe-training-default:/weka/oe-training-default \
    --install "pip install setuptools uv && UV_CACHE_DIR=/tmp/uv-cache uv pip install -e '.[eval,transformers]'" \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --cluster $CLUSTER \
    --priority urgent \
    --allow-dirty \
    --gpus 1 \
    --preemptible \
    --env-secret AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY \
    -- \
    bash -c "PYTHONPATH=. python -u src/scripts/akshitab/add_finegrained_expert/add_new_expert.py -c ${BASE_MODEL_PATH} -o ${NEW_BASE_MODEL_PATH} --num_new_experts ${NUM_NEW_EXPERTS} --init_method similar --activation_file ${ACTIVATION_FILE} -k 2 --num_shared_experts 1 --exclude_experts 127"
