#!/bin/bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PIPELINE_DIR}/common.sh"

[[ $# -ge 1 ]] || die "Usage: stage2_add_experts.sh <experiment-name>"
load_experiment "$1"

assert_stage1_activation_exists
assert_stage2_checkpoint_absent

log_note "Launching Stage 2: add experts"
log_note "  experiment:  ${EXPERIMENT_NAME}"
log_note "  base ckpt:   ${BASE_MODEL_PATH}"
log_note "  activations: ${ACTIVATION_FILE}"
log_note "  output:      ${NEW_BASE_MODEL_PATH}"
log_note "  cluster:     ${STAGE2_CLUSTER}"

gantry run \
    --name "${EXPERIMENT_NAME}-stage2" \
    --weka "${WEKA_BUCKET}:/weka/${WEKA_BUCKET}" \
    --install "${GANTRY_INSTALL_CMD}" \
    --budget "${BUDGET}" \
    --workspace "${WORKSPACE}" \
    --cluster "${STAGE2_CLUSTER}" \
    --priority urgent \
    --allow-dirty \
    --gpus "${STAGE2_GPUS}" \
    --preemptible \
    --env-secret AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY \
    -- \
    bash -c "PYTHONPATH=. python -u src/scripts/akshitab/add_finegrained_expert/add_new_expert.py -c ${BASE_MODEL_PATH} -o ${NEW_BASE_MODEL_PATH} --num_new_experts ${NUM_NEW_EXPERTS} --init_method ${INIT_METHOD} --activation_file ${ACTIVATION_FILE} -k ${INIT_K} --num_shared_experts ${NUM_SHARED_EXPERTS} --exclude_experts ${EXCLUDE_EXPERTS}"
