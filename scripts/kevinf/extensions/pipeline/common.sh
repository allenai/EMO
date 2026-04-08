#!/bin/bash
# Shared config and helpers for the extension pipeline.
#
# Usage from a stage script:
#   source "${PIPELINE_DIR}/common.sh"
#   load_experiment "$1"

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="${PIPELINE_DIR}/../experiments"

# ── Logging / control flow ──────────────────────────────────
log_note() { echo "[extension-pipeline] $*"; }
die()      { echo "[extension-pipeline] ERROR: $*" >&2; exit 1; }
warn()     { echo "[extension-pipeline] WARNING: $*" >&2; }

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Missing required command '$1'"
}

# ── Config loading ──────────────────────────────────────────
load_experiment() {
    local name="$1"
    local config_file="${EXPERIMENTS_DIR}/${name}.yaml"
    [[ -f "${config_file}" ]] || die "Experiment config not found: ${config_file}"
    require_command python3
    eval "$(python3 "${PIPELINE_DIR}/parse_config.py" "${config_file}")"
    _derive_paths
}

_derive_paths() {
    TOTAL_EXPERTS=$((128 + NUM_NEW_EXPERTS))
    INSERT_POS=$((128 - NUM_SHARED_EXPERTS))
    EXPERTS_TO_TRAIN="$(seq -s, "${INSERT_POS}" "$((INSERT_POS + NUM_NEW_EXPERTS - 1))" | sed 's/,$//')"
    NUM_TOKENS=$((STAGE3_NUM_BILLION_TOKENS * 1000000000))

    BASE_MODEL_STEP="$(basename "${BASE_MODEL_PATH}")"

    # All artifacts under one WEKA root: {experiment}/{step}/
    EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/weka/oe-training-default/kevinf/extension-experiments}"
    STEP_ROOT="${EXPERIMENTS_ROOT}/${EXPERIMENT_NAME}/${BASE_MODEL_STEP}"
    DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/weka/oe-training-default/kevinf/dataset-cache}"

    ACTIVATION_OUTPUT_DIR="${ACTIVATION_OUTPUT_DIR:-${STEP_ROOT}/activations}"
    ACTIVATION_FILE="${ACTIVATION_FILE:-${ACTIVATION_OUTPUT_DIR}/${MIX}-router.jsonl}"
    NEW_BASE_MODEL_PATH="${NEW_BASE_MODEL_PATH:-${STEP_ROOT}/extended-checkpoint}"

    WEKA_BUCKET="${WEKA_BUCKET:-oe-training-default}"
    WEKA_PROFILE="${WEKA_PROFILE:-WEKA}"
    GANTRY_INSTALL_CMD="${GANTRY_INSTALL_CMD:-pip install setuptools uv && UV_CACHE_DIR=/tmp/uv-cache uv pip install -e '.[eval,transformers]'}"
    WORKSPACE="${WORKSPACE:-ai2/flex2}"
    BUDGET="${BUDGET:-ai2/oceo}"

    RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
}

# ── Naming helpers ──────────────────────────────────────────
default_stage3_run_name() {
    echo "${EXPERIMENT_NAME}_lr${STAGE3_LR}_${STAGE3_NUM_BILLION_TOKENS}B_${RUN_TIMESTAMP}"
}

stage3_save_folder() {
    echo "${STEP_ROOT}/runs/$1"
}

# ── Remote artifact checks ──────────────────────────────────
have_aws() { command -v aws >/dev/null 2>&1; }

maybe_warn_missing_aws() {
    have_aws || warn "AWS CLI not found, skipping remote artifact checks"
}

weka_path_to_s3() {
    local path="$1"
    [[ "${path}" == /weka/* ]] || die "Expected a /weka/... path, got '${path}'"
    local stripped="${path#/weka/}"
    local bucket="${stripped%%/*}"
    local key="${stripped#*/}"
    echo "s3://${bucket}/${key}"
}

weka_object_exists() { aws --profile "${WEKA_PROFILE}" s3 ls "$(weka_path_to_s3 "$1")" >/dev/null 2>&1; }
weka_prefix_exists() { aws --profile "${WEKA_PROFILE}" s3 ls "$(weka_path_to_s3 "$1")/" >/dev/null 2>&1; }

stage1_activation_exists() {
    have_aws && weka_object_exists "${ACTIVATION_FILE}"
}

stage2_checkpoint_exists() {
    have_aws \
        && weka_object_exists "${NEW_BASE_MODEL_PATH}/config.json" \
        && weka_object_exists "${NEW_BASE_MODEL_PATH}/model_and_optim/.metadata"
}

stage3_save_folder_exists() {
    have_aws && weka_prefix_exists "$(stage3_save_folder "$1")"
}

# ── Assertions ──────────────────────────────────────────────
assert_stage1_activation_exists() {
    maybe_warn_missing_aws
    if have_aws && ! stage1_activation_exists; then
        die "Missing Stage 1 activation at '${ACTIVATION_FILE}'"
    fi
}

assert_stage2_checkpoint_exists() {
    maybe_warn_missing_aws
    if have_aws && ! stage2_checkpoint_exists; then
        die "Missing Stage 2 checkpoint at '${NEW_BASE_MODEL_PATH}'"
    fi
}

assert_stage2_checkpoint_absent() {
    maybe_warn_missing_aws
    if have_aws && stage2_checkpoint_exists; then
        if [[ "${ALLOW_STAGE2_REUSE:-0}" == "1" ]]; then
            warn "Stage 2 checkpoint exists at '${NEW_BASE_MODEL_PATH}', continuing (ALLOW_STAGE2_REUSE=1)"
        else
            die "Stage 2 checkpoint already exists at '${NEW_BASE_MODEL_PATH}'. Set ALLOW_STAGE2_REUSE=1 to reuse."
        fi
    fi
}

assert_stage3_save_folder_safe() {
    local run_name="$1"
    maybe_warn_missing_aws
    if have_aws && stage3_save_folder_exists "${run_name}"; then
        if [[ "${ALLOW_STAGE3_RESUME:-0}" == "1" ]]; then
            warn "Stage 3 folder exists at '$(stage3_save_folder "${run_name}")', continuing (ALLOW_STAGE3_RESUME=1)"
        else
            die "Stage 3 folder already exists at '$(stage3_save_folder "${run_name}")'. Pick a new name or set ALLOW_STAGE3_RESUME=1."
        fi
    fi
}

assert_beaker_launcher_ready() {
    require_command python
    python -c "import olmo_core.launch.beaker" >/dev/null 2>&1 \
        || die "Beaker launcher import failed. Activate the repo environment first."
}

# ── Status ──────────────────────────────────────────────────
print_pipeline_status() {
    local run_name="${1:-$(default_stage3_run_name)}"

    echo "Experiment: ${EXPERIMENT_NAME}"
    echo "  base checkpoint:    ${BASE_MODEL_PATH}"
    echo "  base HF checkpoint: ${BASE_MODEL_HF_PATH}"
    echo "  mix:                ${MIX}"
    echo "  new experts:        ${NUM_NEW_EXPERTS} (total: ${TOTAL_EXPERTS})"
    echo "  experts to train:   ${EXPERTS_TO_TRAIN}"
    echo ""
    echo "Artifact paths:"
    echo "  experiment root:      ${STEP_ROOT}"
    echo "  activation file:      ${ACTIVATION_FILE}"
    echo "  extension checkpoint: ${NEW_BASE_MODEL_PATH}"
    echo "  stage3 save folder:   $(stage3_save_folder "${run_name}")"
    echo "  stage3 run name:      ${run_name}"
    echo "  W&B: ${WANDB_ENTITY}/${WANDB_PROJECT}/${run_name}"

    if ! have_aws; then
        echo ""
        echo "Artifact status: unavailable (aws CLI not found)"
        return 0
    fi

    echo ""
    echo "Artifact status:"
    if stage1_activation_exists; then echo "  stage1 activation: PRESENT"; else echo "  stage1 activation: missing"; fi
    if stage2_checkpoint_exists; then echo "  stage2 checkpoint: PRESENT"; else echo "  stage2 checkpoint: missing"; fi
    if stage3_save_folder_exists "${run_name}"; then echo "  stage3 save folder: PRESENT"; else echo "  stage3 save folder: missing"; fi
}
