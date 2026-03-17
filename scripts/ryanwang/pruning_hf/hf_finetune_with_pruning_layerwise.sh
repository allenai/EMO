#!/bin/bash
#
# HuggingFace-Native Finetuning Pipeline with Greedy Layerwise Expert Pruning
#
# Identical to hf_finetune_with_pruning.sh except that steps 1+2
# (activation collection and pruning) are replaced by a single call to
# greedy_prune_layerwise.py, which prunes one layer at a time so that each
# layer's activation statistics are conditioned on the already-pruned
# earlier layers.
#
# Usage:
#   ./scripts/ryanwang/pruning_hf/hf_finetune_with_pruning_layerwise.sh \
#       --model /path/to/model \
#       --task arc_challenge \
#       --prune-keep-k 32 \
#       --num-shared-experts 1 \
#       --base-dir /path/to/prune_evals \
#       --relative-dir <run_subdir> \
#       --num-gpus 4 \
#       --run-name <job_name>
#
# For dense / already-pruned models pass --pruned-model and --skip-prune to
# jump straight to finetuning.
#

set -e

# Default values
MODEL=""
TASK=""
PRUNE_KEEP_K=4
NUM_SHARED_EXPERTS=0
RELATIVE_DIR=""
BASE_DIR=""
NUM_GPUS=1
NUM_EPOCHS=3
NUM_CHECKPOINTS=5
BATCH_SIZE=4
MICRO_BATCH_SIZE=1
SKIP_PRUNE=false
PRUNED_MODEL=""
LEARNING_RATE=5e-5
RUN_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --prune-keep-k)
            PRUNE_KEEP_K="$2"
            shift 2
            ;;
        --num-shared-experts)
            NUM_SHARED_EXPERTS="$2"
            shift 2
            ;;
        --relative-dir)
            RELATIVE_DIR="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --num-checkpoints)
            NUM_CHECKPOINTS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --micro-batch-size)
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --skip-prune)
            SKIP_PRUNE=true
            shift
            ;;
        --pruned-model)
            PRUNED_MODEL="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --model           Path to the full (unpruned) HF model"
            echo "  --task            Task name (arc_challenge, mmlu, etc.)"
            echo "  --relative-dir    Relative output subdirectory"
            echo "  --base-dir        Base output directory"
            echo "  --run-name        Run name for logging"
            echo ""
            echo "Optional:"
            echo "  --prune-keep-k       Number of experts to keep per layer (default: 4)"
            echo "  --num-shared-experts Number of shared experts to keep (default: 0)"
            echo "  --num-gpus           Number of GPUs for finetuning (default: 1)"
            echo "  --num-epochs         Number of finetuning epochs (default: 3)"
            echo "  --num-checkpoints    Number of checkpoints to save (default: 5)"
            echo "  --batch-size         Global batch size (default: 4)"
            echo "  --micro-batch-size   Per-device batch size for finetuning (default: 1)"
            echo "  --skip-prune         Skip pruning; requires --pruned-model"
            echo "  --pruned-model       Path to an already-pruned model (used with --skip-prune)"
            echo "  --learning-rate      Learning rate (default: 5e-5)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ] && [ "$SKIP_PRUNE" = false ]; then
    echo "Error: --model is required unless --skip-prune is set"
    exit 1
fi

if [ "$SKIP_PRUNE" = true ] && [ -z "$PRUNED_MODEL" ]; then
    echo "Error: --pruned-model is required when --skip-prune is set"
    exit 1
fi

if [ -z "$TASK" ]; then
    echo "Error: --task is required"
    exit 1
fi

if [ -z "$RELATIVE_DIR" ]; then
    echo "Error: --relative-dir is required"
    exit 1
fi

if [ -z "$BASE_DIR" ]; then
    echo "Error: --base-dir is required"
    exit 1
fi

if [[ "$RELATIVE_DIR" != *"$TASK"* ]]; then
    echo "ERROR: --relative-dir does not contain the task name '$TASK'"
    exit 1
fi

if [ -z "$RUN_NAME" ]; then
    echo "Error: --run-name is required"
    exit 1
fi

if (( BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "Error: --batch-size must be a multiple of --num-gpus"
    exit 1
fi
if (( BATCH_SIZE % MICRO_BATCH_SIZE != 0 )); then
    echo "Error: --batch-size must be a multiple of --micro-batch-size"
    exit 1
fi

OUTPUT_DIR="${BASE_DIR}/${RELATIVE_DIR}"
mkdir -p "$OUTPUT_DIR"

if [ -z "$PRUNED_MODEL" ]; then
    PRUNED_MODEL="${OUTPUT_DIR}/pruned_model"
fi

FINETUNED_MODEL="${OUTPUT_DIR}/finetuned_model"

echo "========================================"
echo "HuggingFace Finetuning Pipeline (Layerwise Pruning)"
echo "========================================"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Prune keep k: $PRUNE_KEEP_K"
echo "Output dir: $OUTPUT_DIR"
echo "Num GPUs: $NUM_GPUS"
echo "Num epochs: $NUM_EPOCHS"
echo "========================================"

# Steps 1+2: Greedy layerwise activation collection + pruning (fused into one script)
if [ "$SKIP_PRUNE" = false ]; then
    echo ""
    echo "Steps 1+2: Greedy layerwise pruning..."
    echo "========================================"

    python -m src.hf_training.greedy_prune_layerwise \
        --model "$MODEL" \
        --task "$TASK" \
        --split "validation" \
        --prune-keep-k "$PRUNE_KEEP_K" \
        --num-shared-experts "$NUM_SHARED_EXPERTS" \
        --save-path "$PRUNED_MODEL" \
        --batch-size 32

    echo "Pruned model saved to: $PRUNED_MODEL"
else
    echo ""
    echo "Steps 1+2: Skipping pruning (using existing pruned model)"
    echo "Pruned model: $PRUNED_MODEL"
fi

# Step 3: Finetune
echo ""
echo "Step 3: Finetuning..."
echo "========================================"

if [ "$NUM_GPUS" -gt 1 ]; then
    FSDP_FLAG=""
else
    FSDP_FLAG="--no-fsdp"
fi

export WANDB_PROJECT="olmoe-modular"
export WANDB_ENTITY="ryanyxw"
export WANDB_TAGS="finetune,${TASK:0:60},${PRUNED_MODEL: -60}"

gas=$(( BATCH_SIZE / (NUM_GPUS * MICRO_BATCH_SIZE) ))

torchrun --nproc_per_node="$NUM_GPUS" \
    -m src.hf_training.finetune \
    --model "$PRUNED_MODEL" \
    --task "$TASK" \
    --split "train" \
    --output-dir "$FINETUNED_MODEL" \
    --num-epochs "$NUM_EPOCHS" \
    --num-checkpoints "$NUM_CHECKPOINTS" \
    --learning-rate "$LEARNING_RATE" \
    --run-name "$RUN_NAME" \
    --per-device-batch-size "$MICRO_BATCH_SIZE" \
    --gradient-accumulation-steps "$gas" \
    $FSDP_FLAG

# Step 4: Evals
echo ""
echo "Step 4: Evals..."
echo "========================================"

# Datasets are already cached from steps 1-3; skip HF API calls to avoid rate limits
export HF_DATASETS_OFFLINE=1

all_checkpoints=("$FINETUNED_MODEL"/checkpoint-*/)

for checkpoint in "${all_checkpoints[@]}"; do
    echo "Evaluating checkpoint: $checkpoint"
    checkpoint_num=$(basename "$checkpoint" | sed 's/checkpoint-//')

    EVAL_BATCH_SIZE=32
    # prevent oom for mmlu_history
    if [[ $TASK == *"history"* ]]; then
      echo "Setting eval batch size to 4 for history task"
      EVAL_BATCH_SIZE=4
    fi

    python -m src.scripts.eval.launch_eval \
        --model "$checkpoint" \
        --model-type hf \
        --task "$TASK-pruned" \
        --pruned_split "test" \
        --remote-output-dir "s3://ai2-sewonm/ryanwang/prune_evals_0313/${RELATIVE_DIR}/results/checkpoint-${checkpoint_num}" \
        --batch-size $EVAL_BATCH_SIZE \
        --gpus "$NUM_GPUS"
done

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
echo "Pruned model: $PRUNED_MODEL"
echo "Finetuned model: $FINETUNED_MODEL"
