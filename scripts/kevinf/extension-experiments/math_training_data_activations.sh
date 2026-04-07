#!/bin/bash
# PARENT: NONE
# DESCRIPTION:
#     - Compute average router activations on mj_finemath4plus training data (Issue #26)
#     - Produces activation file for expert selection via training data instead of eval tasks
# STATUS: NEW
##############################################################

MODEL_PATH="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995-hf"
MIX="mj_finemath4plus"
MIX_BASE_DIR="/weka/oe-training-default/ai2-llm"
OUTPUT_DIR="s3://ai2-kevinf/FlexMoE/training-activations/moereducedp512sharedexp1_1b14b_0308_step30995"
BATCH_SIZE=16
SEQ_LENGTH=4096
MAX_TOKENS=25000000  # 25M tokens — ~10 min on 4 GPUs with batch_size=16
CLUSTER="ai2/jupiter-cirrascale-2"

JOB_NAME="logits-training-math-moereducedp512sharedexp1"

# Local run (uncomment for debugging):
# PYTHONPATH=. python -u src/scripts/eval/launch_logits_training.py --model ${MODEL_PATH} --mix ${MIX} --mix-base-dir ${MIX_BASE_DIR} --output-dir ./claude_outputs/training-activations --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} --max-tokens 1000000

gantry run \
    --name $JOB_NAME \
    --weka oe-training-default:/weka/oe-training-default \
    --install "pip install setuptools uv && UV_CACHE_DIR=/tmp/uv-cache uv pip install -e '.[eval,transformers]'" \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --cluster $CLUSTER \
    --priority urgent \
    --allow-dirty \
    --gpus 4 \
    --preemptible \
    --env-secret HF_TOKEN=KEVINF_HF_TOKEN \
    --env-secret AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY \
    -- \
    bash -c "PYTHONPATH=. python -u src/scripts/eval/launch_logits_training.py --model ${MODEL_PATH} --mix ${MIX} --mix-base-dir ${MIX_BASE_DIR} --output-dir ${OUTPUT_DIR} --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} --max-tokens ${MAX_TOKENS}"
