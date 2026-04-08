#!/bin/bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PIPELINE_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: run.sh <experiment-name> <command> [args...]

Commands:
  status           Show resolved paths and artifact status.
  stage1           Launch Stage 1 (compute activations).
  stage2           Launch Stage 2 (add experts).
  stage3 [name]    Launch Stage 3 (train experts). Auto-generates a
                   timestamped name if omitted.
  next   [name]    Launch the next missing stage.

Environment overrides (most exported config fields can be overridden):
  STAGE3_LR=1e-4 bash run.sh math-ta-01 stage3
  STAGE3_NUM_BILLION_TOKENS=5 bash run.sh math-ta-01 stage3

Examples:
  bash run.sh math-ta-01 status
  bash run.sh math-ta-01 next
  bash run.sh math-ta-01 stage3 math-ta-01_custom-suffix
EOF
    echo ""
    echo "Available experiments:"
    for f in "${PIPELINE_DIR}/../experiments/"*.yaml; do
        [[ -f "$f" ]] && echo "  $(basename "$f" .yaml)"
    done
}

[[ $# -ge 1 ]] || { usage; exit 1; }
[[ "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]] && { usage; exit 0; }

EXPERIMENT_NAME="$1"; shift
CMD="${1:-status}"; [[ $# -gt 0 ]] && shift

load_experiment "${EXPERIMENT_NAME}"

case "${CMD}" in
    status)
        print_pipeline_status "${1:-}"
        ;;
    stage1)
        exec bash "${PIPELINE_DIR}/stage1_compute_activations.sh" "${EXPERIMENT_NAME}"
        ;;
    stage2)
        exec bash "${PIPELINE_DIR}/stage2_add_experts.sh" "${EXPERIMENT_NAME}"
        ;;
    stage3)
        exec bash "${PIPELINE_DIR}/stage3_train_experts.sh" "${EXPERIMENT_NAME}" "${1:-}"
        ;;
    next)
        have_aws || die "'next' requires the aws CLI"
        if ! stage1_activation_exists; then
            log_note "Stage 1 missing, launching"
            exec bash "${PIPELINE_DIR}/stage1_compute_activations.sh" "${EXPERIMENT_NAME}"
        elif ! stage2_checkpoint_exists; then
            log_note "Stage 2 missing, launching"
            exec bash "${PIPELINE_DIR}/stage2_add_experts.sh" "${EXPERIMENT_NAME}"
        else
            log_note "Stages 1-2 present, launching Stage 3"
            exec bash "${PIPELINE_DIR}/stage3_train_experts.sh" "${EXPERIMENT_NAME}" "${1:-}"
        fi
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        die "Unknown command '${CMD}'"
        ;;
esac
