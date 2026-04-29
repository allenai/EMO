#!/usr/bin/env bash
# Common preamble for extension-experiment run scripts.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"
#
# Override defaults via environment variables before sourcing:
#   PREFIX=/path/to/FlexMoE     # root for outputs (default: weka path)
#   BASE_MODELS=...             # root for upstream base checkpoints
#   MODE=beaker                 # default: local (uses torchrun)
#   NPROC=8                     # GPUs per node when MODE=local
#   BEAKER_GPUS / BEAKER_NODES  # cluster sizing when MODE=beaker

# Output root: trained checkpoints, dataset cache, etc.
PREFIX="${PREFIX:-/weka/oe-training-default/akshitab/FlexMoE}"

# Base models root (read-only, owned by upstream).
BASE_MODELS="${BASE_MODELS:-/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models}"

# Derived paths.
EXTENSIONS="${EXTENSIONS:-${PREFIX}/models/extensions}"
MODELS="${MODELS:-${PREFIX}/models}"
DATASET_CACHE="${DATASET_CACHE:-${PREFIX}/dataset-cache}"

# Root for the OLMo data mix (passed to training scripts as --data-root).
DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/ai2-llm}"

# S3 base for eval / logits outputs (used by launch_beaker_evals*.sh and launch_beaker_logits*.sh).
EVALS_S3_BASE="${EVALS_S3_BASE:-s3://ai2-sewonm/akshitab/mose/evals/extensions}"

# Specific base checkpoints used across templates.
REGULAR_BASE="${BASE_MODELS}/moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995"
TWOLEVEL_BASE="${BASE_MODELS}/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995"
NONSHARED_BASE="${BASE_MODELS}/moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995"

# Launcher mode: "local" (torchrun) or "beaker" (python -m olmo_core.launch.beaker).
MODE="${MODE:-local}"

# Local launcher knobs.
NPROC="${NPROC:-8}"

# Beaker launcher knobs (used only when MODE=beaker).
BEAKER_GPUS="${BEAKER_GPUS:-8}"
BEAKER_NODES="${BEAKER_NODES:-4}"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/flex2}"
BEAKER_CLUSTER="${BEAKER_CLUSTER:-ai2/jupiter}"
BEAKER_PRIORITY="${BEAKER_PRIORITY:-urgent}"
BEAKER_WEKA="${BEAKER_WEKA:-oe-training-default}"
BEAKER_ENV_SECRETS=(
    "GITHUB_TOKEN=AKSHITAB_GITHUB_TOKEN"
    "WANDB_API_KEY=AKSHITAB_WANDB_API_KEY"
    "BEAKER_TOKEN=AKSHITAB_BEAKER_TOKEN"
    "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID"
    "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY"
    "HF_TOKEN=RYAN_HF_TOKEN"
)

# launch <script_path> <run_name> [args...]
#   Submits to beaker (when MODE=beaker) or runs locally with torchrun otherwise.
#   The first positional arg is the python script path; the second is the run name
#   (used both as the beaker job name and as the first positional arg of the script);
#   the rest are forwarded as script args.
launch() {
    local script="$1"
    local run_name="$2"
    shift 2

    if [[ "${MODE}" == "beaker" ]]; then
        python -m olmo_core.launch.beaker \
            --name "${run_name}" \
            --gpus "${BEAKER_GPUS}" \
            --nodes "${BEAKER_NODES}" \
            --weka="${BEAKER_WEKA}" \
            --shared-filesystem \
            --workspace "${BEAKER_WORKSPACE}" \
            --cluster "${BEAKER_CLUSTER}" \
            --preemptible \
            --allow-dirty \
            --priority "${BEAKER_PRIORITY}" \
            --env-secret "${BEAKER_ENV_SECRETS[@]}" \
            -- "${script}" "${run_name}" --data-root="${DATA_ROOT}" "$@"
    else
        torchrun --nproc-per-node="${NPROC}" "${script}" "${run_name}" --data-root="${DATA_ROOT}" "$@"
    fi
}
