
#!/bin/bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PIPELINE_DIR}/common.sh"

[[ $# -ge 1 ]] || die "Usage: stage1_compute_activations.sh <experiment-name>"
load_experiment "$1"

if stage1_activation_exists; then
    warn "Stage 1 activation file already exists at '${ACTIVATION_FILE}'"
fi

log_note "Launching Stage 1: compute activations"
log_note "  experiment:  ${EXPERIMENT_NAME}"
log_note "  model:       ${BASE_MODEL_HF_PATH}"
log_note "  mix:         ${MIX}"
log_note "  output:      ${ACTIVATION_FILE}"
log_note "  cluster:     ${STAGE1_CLUSTER}"

gantry run \
    --name "${EXPERIMENT_NAME}-stage1" \
    --weka "${WEKA_BUCKET}:/weka/${WEKA_BUCKET}" \
    --install "${GANTRY_INSTALL_CMD}" \
    --budget "${BUDGET}" \
    --workspace "${WORKSPACE}" \
    --cluster "${STAGE1_CLUSTER}" \
    --priority urgent \
    --allow-dirty \
    --gpus "${STAGE1_GPUS}" \
    --preemptible \
    --env-secret HF_TOKEN=KEVINF_HF_TOKEN \
    --env-secret AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY \
    -- \
    bash -c "PYTHONPATH=. python -u src/scripts/eval/launch_logits_training.py --model ${BASE_MODEL_HF_PATH} --mix ${MIX} --mix-base-dir ${MIX_BASE_DIR} --output-dir ${ACTIVATION_OUTPUT_DIR} --batch-size ${STAGE1_BATCH_SIZE} --seq-length ${STAGE1_SEQ_LENGTH} --max-tokens ${STAGE1_MAX_TOKENS}"
