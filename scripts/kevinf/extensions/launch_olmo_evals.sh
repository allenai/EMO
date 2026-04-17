im c#!/bin/bash
# Launch olmo-eval-internal evaluations for FlexMoE extension checkpoints.
#
# Launches one Beaker experiment per (model, task-suite) pair so they run
# in parallel instead of serially through 127K instances.
#
# Prerequisites:
#   - olmo-eval-internal checkout (set OLMO_EVAL_DIR or default ../olmo-eval-internal)
#   - On the akshitab/hf-bug branch (or main once the HF fix merges)
#
# Usage:
#   bash scripts/kevinf/extensions/eval_extensions.sh [--dry-run]

set -euo pipefail

DRY_RUN=""
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="-d"

OLMO_EVAL_DIR="${OLMO_EVAL_DIR:-/Users/kevinfarhat/repos/olmo-eval-internal}"
GROUP="extensions"
GPUS="${GPUS:-4}"
TRANSFORMERS_FORK="git+https://github.com/ryanyxw/transformers.git@de05b34309baf1c1110a3371031767edba81a317"

# Models to evaluate
BASE="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995-hf"
MATH="/weka/oe-training-default/kevinf/FlexMoE/models/moereducedp512sharedexp1_132experts_4trained_math_init_top2_average_train_act_10B_lr_4e-4_20260407a/step2385-hf"
CODE="/weka/oe-training-default/kevinf/extension-experiments/code-ta-01/step30995/runs/code-ta-01_lr4e-4_10B_20260407-234403/step2385-hf"
CROISSANT="/weka/oe-training-default/kevinf/extension-experiments/croissant-ta-01/step30995/runs/croissant-ta-01_lr4e-4_10B_20260407-234459/step2385-hf"

MODEL_NAMES=(base-128exp math-ext code-ext croissant-ext)
MODEL_PATHS=("$BASE" "$MATH" "$CODE" "$CROISSANT")

# Task suites — each becomes a separate Beaker experiment per model
TASK_SUITES=(
    "olmo3:base_easy:code:bpb"    # 19 tasks: humaneval, mbpp, mt_mbpp
    "code_fresh:bpb"              # 42 language BPB
    "minerva_math_olmo3"          # 7 minerva math tasks
    "mmlu"                        # 57 subtasks
)

# Individual tasks that aren't in a suite — bundled into one experiment
INDIVIDUAL_TASKS=(
    "arc_easy:mc:olmo3base"
    "arc_challenge:mc:olmo3base"
    "csqa:mc:olmo3base"
    "hellaswag:mc:olmo3base"
    "piqa:mc:olmo3base"
    "socialiqa:mc:olmo3base"
    "coqa:gen:olmo3base"
    "squad:gen:olmo3base"
    "drop:gen:olmo3base"
    "medqa_en:mc:olmo3base"
    "medmcqa:mc:olmo3base"
)

launch() {
    local name="$1"
    local model="$2"
    shift 2
    # remaining args are -t task1 -t task2 ...

    cd "$OLMO_EVAL_DIR"
    uv run olmo-eval beaker launch \
        -n "$name" \
        -g "$GROUP" \
        -m "$model" \
        -H default \
        -o 'provider.kind=hf' \
        -o "provider.dependencies=[\"${TRANSFORMERS_FORK}\"]" \
        "$@" \
        -c h100 \
        -G "$GPUS" \
        -w ai2/flex2 \
        -B ai2/oceo \
        -p urgent \
        --preemptible \
        --no-follow \
        --store \
        -y \
        $DRY_RUN
}

# Count total experiments
TOTAL_SUITES=${#TASK_SUITES[@]}
HAS_INDIVIDUAL=0
[[ ${#INDIVIDUAL_TASKS[@]} -gt 0 ]] && HAS_INDIVIDUAL=1
TOTAL_EXPERIMENTS=$(( ${#MODEL_NAMES[@]} * (TOTAL_SUITES + HAS_INDIVIDUAL) ))

echo "Launching ${TOTAL_EXPERIMENTS} experiments (${#MODEL_NAMES[@]} models x $((TOTAL_SUITES + HAS_INDIVIDUAL)) task groups)"
echo "Group: $GROUP"
echo "GPUs per experiment: $GPUS"
echo ""

for i in "${!MODEL_NAMES[@]}"; do
    name="${MODEL_NAMES[$i]}"
    model="${MODEL_PATHS[$i]}"

    # Launch each suite as its own experiment
    for suite in "${TASK_SUITES[@]}"; do
        # Make a clean job name from the suite
        suite_tag=$(echo "$suite" | sed 's/[^a-zA-Z0-9]/-/g' | sed 's/--*/-/g' | sed 's/-$//')
        job_name="${name}-${suite_tag}"
        echo ">>> $job_name"
        launch "$job_name" "$model" -t "$suite"
        echo ""
    done

    # Bundle individual tasks into one experiment
    if [[ ${#INDIVIDUAL_TASKS[@]} -gt 0 ]]; then
        task_args=()
        for t in "${INDIVIDUAL_TASKS[@]}"; do
            task_args+=(-t "$t")
        done
        job_name="${name}-mc-gen-med"
        echo ">>> $job_name"
        launch "$job_name" "$model" "${task_args[@]}"
        echo ""
    fi
done

echo "All launches complete. Group: $GROUP"
echo "Query results with: cd $OLMO_EVAL_DIR && uv run olmo-eval results query -G $GROUP"
