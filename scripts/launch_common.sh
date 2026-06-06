#!/usr/bin/env bash
# Common preamble for ryanwang training scripts (models_0116/, extensions/).
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"
#
# Override defaults via environment variables before sourcing:
#   PREFIX=/path/to/Emo     # root for model outputs (default: weka path)
#   DATASET_CACHE=...           # dataset cache root (default: weka path)
#   MODE=beaker                 # default: local (uses torchrun)
#   NPROC=8                     # GPUs per node when MODE=local
#   BEAKER_GPUS / BEAKER_NODES  # cluster sizing when MODE=beaker

# Output root for trained checkpoints.
PREFIX="${PREFIX:-/weka/oe-training-default/ryanwang/phdbrainstorm/Emo}"
MODELS_DIR="${MODELS_DIR:-${PREFIX}/models}"

# Dataset cache (lives outside the model output root on weka).
DATASET_CACHE="${DATASET_CACHE:-/weka/oe-training-default/ryanwang/dataset-cache}"

# Root for the OLMo data mix (passed to training scripts as --data-root).
DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/ai2-llm}"

# Launcher mode: "local" (torchrun) or "beaker" (python -m olmo_core.launch.beaker).
MODE="${MODE:-local}"

# Local launcher knobs.
NPROC="${NPROC:-8}"

# Beaker launcher knobs (used only when MODE=beaker).
BEAKER_GPUS="${BEAKER_GPUS:-8}"
BEAKER_NODES="${BEAKER_NODES:-16}"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/flex2}"
BEAKER_CLUSTER="${BEAKER_CLUSTER:-ai2/jupiter}"
BEAKER_PRIORITY="${BEAKER_PRIORITY:-urgent}"
BEAKER_WEKA="${BEAKER_WEKA:-oe-training-default}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tylerr/olmo-core-tch280cu128-2025-11-25}"
BEAKER_ENV_SECRETS=(
    "GITHUB_TOKEN=RYAN_GITHUB_TOKEN"
    "WANDB_API_KEY=RYAN_WANDB_API_KEY"
    "BEAKER_TOKEN=RYAN_BEAKER_TOKEN"
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
            --beaker-image "${BEAKER_IMAGE}" \
            --preemptible \
            --allow-dirty \
            --priority "${BEAKER_PRIORITY}" \
            --env-secret "${BEAKER_ENV_SECRETS[@]}" \
            --env "S3_PROFILE=" \
            -- "${script}" "${run_name}" --data-root="${DATA_ROOT}" "$@"
    else
        torchrun --nproc-per-node="${NPROC}" "${script}" "${run_name}" --data-root="${DATA_ROOT}" "$@"
    fi
}
