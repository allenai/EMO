#!/bin/bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PIPELINE_DIR}/common.sh"

[[ $# -ge 1 ]] || die "Usage: stage3_train_experts.sh <experiment-name> [run-name]"
load_experiment "$1"

DEFAULT_RUN_NAME="$(default_stage3_run_name)"
RUN_NAME="${2:-${DEFAULT_RUN_NAME}}"

assert_stage2_checkpoint_exists
assert_stage3_save_folder_safe "${RUN_NAME}"
assert_beaker_launcher_ready

if [[ -z "${2:-}" ]]; then
    log_note "No run name provided, auto-generated '${RUN_NAME}'"
fi

log_note "Launching Stage 3: train experts"
log_note "  experiment:  ${EXPERIMENT_NAME}"
log_note "  input ckpt:  ${NEW_BASE_MODEL_PATH}"
log_note "  run name:    ${RUN_NAME}"
log_note "  save folder: $(stage3_save_folder "${RUN_NAME}")"
log_note "  experts:     ${EXPERTS_TO_TRAIN}"
log_note "  lr:          ${STAGE3_LR}"
log_note "  tokens:      ${STAGE3_NUM_BILLION_TOKENS}B"
log_note "  cluster:     ${STAGE3_CLUSTER}"

python -m olmo_core.launch.beaker \
    --name "${RUN_NAME}" \
    --gpus "${STAGE3_GPUS}" \
    --nodes "${STAGE3_NODES}" \
    --weka="${WEKA_BUCKET}" \
    --shared-filesystem \
    --workspace "${WORKSPACE}" \
    --cluster "${STAGE3_CLUSTER}" \
    --preemptible \
    --allow-dirty \
    --priority urgent \
    --env-secret "GITHUB_TOKEN=KEVINF_GITHUB_TOKEN" "WANDB_API_KEY=KEVINF_WANDB_API_KEY" "BEAKER_TOKEN=KEVINF_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=KEVINF_HF_TOKEN" \
    -- src/scripts/akshitab/add_finegrained_expert/train_selected_experts.py \
    "${RUN_NAME}" \
    --trainer.load_path="${NEW_BASE_MODEL_PATH}/model_and_optim" \
    --save-folder="$(stage3_save_folder "${RUN_NAME}")" \
    --dataset.mix="${MIX}" \
    --work-dir="${DATASET_CACHE_DIR}" \
    --trainer.max_duration="{value: ${NUM_TOKENS}, unit: tokens}" \
    --trainer.callbacks.wandb="{enabled: true, entity: ${WANDB_ENTITY}, project: ${WANDB_PROJECT}, name: ${RUN_NAME}, tags: [extension, ${EXPERIMENT_NAME}]}" \
    --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
    --model.block.feed_forward_moe.lb_loss_weight="${STAGE3_LB_LOSS_WEIGHT}" \
    --train_module.scheduler.warmup_fraction="${STAGE3_WARMUP_FRACTION}" \
    --lr="${STAGE3_LR}" \
    --base-model-config="${NEW_BASE_MODEL_PATH}" \
    --experts-to-train="${EXPERTS_TO_TRAIN}"
