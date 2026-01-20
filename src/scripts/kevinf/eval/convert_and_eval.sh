#!/bin/bash

# Script to convert OLMo Core checkpoint to HF format and launch Beaker evaluations
# Usage: bash src/scripts/kevinf/eval/convert_and_eval.sh <checkpoint_path> [output_path]
#
# Example:
#   bash src/scripts/kevinf/eval/convert_and_eval.sh /data/input/kevinf/checkpoints/my-run/step10000
#
# This will:
#   1. Convert checkpoint to HF format at <checkpoint_path>-hf
#   2. Launch Beaker evaluations for all configured tasks

set -e  # Exit on any error

# Activate virtual environment for uv/python access
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

# ============================================================================
# Configuration
# ============================================================================

BASE_OUTPUT_DIR="/data/input/kevinf/flexmoe/eval/results"
CLUSTER="ai2/saturn"
LIMIT=1000
MAX_SEQUENCE_LENGTH=65536

# Tasks to evaluate
TASKS=(
    # MC9 tasks
    arc_easy:mc::olmes
    arc_challenge:mc::olmes
    boolq:mc::olmes
    csqa:mc::olmes
    hellaswag:mc::olmes
    openbookqa:mc::olmes
    piqa:mc::olmes
    socialiqa:mc::olmes
    winogrande:mc::olmes
    
    # Gen5 tasks
    coqa::olmes
    squad::olmes
    naturalqs::olmes
    triviaqa::olmes
    drop::olmes

    # MMLU tasks
    mmlu:mc::olmes

    # AGI eval
    agi_eval_english:1shot::olmes

    # BBH
    bbh:cot-v1::olmes

    # Math2 tasks
    gsm8k::olmes
    minerva_math_algebra::olmes
    minerva_math_counting_and_probability::olmes
    minerva_math_geometry::olmes
    minerva_math_intermediate_algebra::olmes
    minerva_math_number_theory::olmes
    minerva_math_prealgebra::olmes
    minerva_math_precalculus::olmes

    # Code4 tasks
    codex_humaneval:temp0.8
    codex_humanevalplus:temp0.8
    mbpp::none
    mbppplus::none

    # ChemBench MC and generative tasks
    chembench:mc
    chembench:gen
)

# ============================================================================
# Parse arguments
# ============================================================================

if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path> [output_path]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path  Path to OLMo Core checkpoint or base run folder (required)"
    echo "                   If base folder given, automatically finds latest step*"
    echo "  output_path      Path for HF checkpoint output (optional, defaults to <checkpoint_path>-hf)"
    echo ""
    echo "Options (set via environment variables):"
    echo "  SKIP_CONVERSION=1   Skip conversion step (use existing HF checkpoint)"
    echo "  SKIP_EVAL=1         Skip evaluation step (only convert)"
    echo "  DRY_RUN=1           Print commands without executing"
    exit 1
fi

INPUT_PATH="${1%/}"  # Remove trailing slash if present

# Check if this is a step directory or a base folder
# Step directories match pattern like "step12345"
if [[ "$(basename "$INPUT_PATH")" =~ ^step[0-9]+$ ]]; then
    # Already a step directory
    MODEL_PATH="$INPUT_PATH"
else
    # Base folder - find the latest step directory
    echo ">>> Base folder detected, finding latest step..."
    
    # Find all step directories (excluding -hf converted ones) and sort by step number (descending)
    LATEST_STEP=$(find "$INPUT_PATH" -maxdepth 1 -type d -name "step*" ! -name "*-hf" 2>/dev/null | \
        sed 's/.*step//' | \
        sort -rn | \
        head -1)
    
    if [ -z "$LATEST_STEP" ]; then
        echo "ERROR: No step* directories found in $INPUT_PATH"
        exit 1
    fi
    
    MODEL_PATH="${INPUT_PATH}/step${LATEST_STEP}"
    echo ">>> Found latest step: step${LATEST_STEP}"
fi

OUTPUT_PATH="${2:-${MODEL_PATH}-hf}"

echo "=============================================="
echo "Convert and Evaluate Pipeline"
echo "=============================================="
echo "Input checkpoint:  $MODEL_PATH"
echo "Output HF path:    $OUTPUT_PATH"
echo "Eval output dir:   $BASE_OUTPUT_DIR"
echo "Cluster:           $CLUSTER"
echo "Tasks:             ${#TASKS[@]} tasks"
echo "=============================================="
echo ""

# ============================================================================
# Step 1: Convert checkpoint to HF format
# ============================================================================

if [ "${SKIP_CONVERSION:-0}" = "1" ]; then
    echo ">>> Skipping conversion (SKIP_CONVERSION=1)"
    if [ ! -d "$OUTPUT_PATH" ]; then
        echo "ERROR: Output path does not exist: $OUTPUT_PATH"
        exit 1
    fi
else
    echo ">>> Step 1: Converting checkpoint to HuggingFace format..."
    echo ""
    
    CONVERT_CMD="python src/examples/huggingface/convert_checkpoint_to_hf.py \
        -i $MODEL_PATH \
        -o $OUTPUT_PATH \
        --skip-validation \
        --max-sequence-length $MAX_SEQUENCE_LENGTH"
    
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY RUN] Would execute:"
        echo "$CONVERT_CMD"
    else
        eval $CONVERT_CMD
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Checkpoint conversion failed!"
            exit 1
        fi
    fi
    
    echo ""
    echo ">>> Conversion complete: $OUTPUT_PATH"
fi

echo ""

# ============================================================================
# Step 2: Launch Beaker evaluations
# ============================================================================

if [ "${SKIP_EVAL:-0}" = "1" ]; then
    echo ">>> Skipping evaluation (SKIP_EVAL=1)"
    exit 0
fi

echo ">>> Step 2: Launching Beaker evaluations..."
echo ""

# Function to get checkpoint name - extracts run name and step from path
function get_checkpoint_name {
    local path=$1
    local step_dir=$(basename "$path")
    local run_name=$(basename "$(dirname "$path")")
    echo "${run_name}_${step_dir}"
}

# Get model name for output directory
if [[ $OUTPUT_PATH == "/"* ]]; then
    model=$(get_checkpoint_name $OUTPUT_PATH)
else
    model=$(echo $OUTPUT_PATH | cut -d'/' -f2)
fi

EVAL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"

echo "Model name:     $model"
echo "Eval output:    $EVAL_OUTPUT_DIR"
echo ""

job_count=0

for TASK in "${TASKS[@]}"; do
    gpus=1
    
    # Batch size adjustment
    if [[ $TASK == *"cot"* || $TASK == "minerva_math_"* || $TASK == "mbpp"* || $TASK == "bigcodebench"* || $TASK == "ruler"* || $TASK == "sciriff"* ]]; then
        batch_size=1
    else
        batch_size=4
    fi
    
    # Create safe job name (Beaker limit: 128 chars)
    # Reserve ~5 for "eval-", ~1 for "-", leaves ~122 for model+task
    safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-80)
    safe_task_name=$(echo $TASK | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-40)
    job_name="eval-${safe_model_name}-${safe_task_name}"
    
    echo "Launching: $TASK (batch_size=$batch_size)"
    
    GANTRY_CMD="gantry run \
        --name $job_name \
        --weka oe-training-default:/data/input \
        --install \"pip install -e \\\".[eval]\\\"\" \
        --budget ai2/oceo \
        --workspace ai2/flex2 \
        --cluster $CLUSTER \
        --priority urgent \
        --gpus $gpus \
        --env-secret HF_TOKEN=KEVINF_HF_TOKEN \
        --env-secret AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID \
        --env-secret AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY \
        --allow-dirty \
        -- \
        bash -c \"PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
            --model $OUTPUT_PATH \
            --model-type hf \
            --task $TASK \
            --limit $LIMIT \
            --output-dir $EVAL_OUTPUT_DIR \
            --batch-size $batch_size \
            --gpus $gpus \
            \""
    
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY RUN] Would execute: gantry run ... --task $TASK"
    else
        eval $GANTRY_CMD
    fi
    
    ((++job_count))
done

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "=============================================="
echo "HF checkpoint:   $OUTPUT_PATH"
echo "Eval output:     $EVAL_OUTPUT_DIR"
echo "Jobs launched:   $job_count"
echo ""
echo "Check Beaker dashboard for job status."

