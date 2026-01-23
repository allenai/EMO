#!/bin/bash
#
# HuggingFace-Native Finetuning Pipeline with Expert Pruning
#
# This script orchestrates the full pipeline:
# 1. Compute router activations on validation set
# 2. Prune model to keep top-k experts
# 3. Finetune on training set
#
# Usage:
#   ./scripts/hf_finetune_with_pruning.sh \
#       --model allenai/OLMoE-1B-7B-0924 \
#       --task gsm8k \
#       --prune-keep-k 4 \
#       --output-dir ./experiments/gsm8k_pruned \
#       --num-gpus 4
#

set -e

# Default values
MODEL=""
TASK=""
PRUNE_KEEP_K=4
OUTPUT_DIR=""
NUM_GPUS=1
NUM_EPOCHS=3
NUM_CHECKPOINTS=5
BATCH_SIZE=4
SKIP_ACTIVATION=false
SKIP_PRUNE=false
ACTIVATION_FILE=""
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
        --output-dir)
            OUTPUT_DIR="$2"
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
        --skip-activation)
            SKIP_ACTIVATION=true
            shift
            ;;
        --skip-prune)
            SKIP_PRUNE=true
            shift
            ;;
        --activation-file)
            ACTIVATION_FILE="$2"
            shift 2
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
            echo "  --model           HuggingFace model name or path"
            echo "  --task            Task name (gsm8k, mmlu, squad, coqa)"
            echo "  --output-dir      Output directory for all artifacts"
            echo ""
            echo "Optional:"
            echo "  --prune-keep-k    Number of experts to keep (default: 4)"
            echo "  --num-gpus        Number of GPUs for training (default: 1)"
            echo "  --num-epochs      Number of training epochs (default: 3)"
            echo "  --num-checkpoints Number of checkpoints to save (default: 5)"
            echo "  --batch-size      Batch size for activation computation (default: 4)"
            echo "  --skip-activation Skip activation computation (requires --activation-file)"
            echo "  --skip-prune      Skip pruning (requires --pruned-model)"
            echo "  --activation-file Path to existing activation file"
            echo "  --pruned-model    Path to existing pruned model"
            echo "  --learning-rate   Learning rate (default: 5e-5)"
            echo "  --run-name        Run name for logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ] && [ "$SKIP_ACTIVATION" = false ]; then
    echo "Error: --model is required unless --skip-activation and --pruned-model are set"
    exit 1
fi

if [ -z "$TASK" ]; then
    echo "Error: --task is required"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output-dir is required"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set up paths
if [ -z "$ACTIVATION_FILE" ]; then
    ACTIVATION_FILE="${OUTPUT_DIR}/activations/${TASK}_val_activations.jsonl"
fi

if [ -z "$PRUNED_MODEL" ]; then
    PRUNED_MODEL="${OUTPUT_DIR}/pruned_model"
fi

FINETUNED_MODEL="${OUTPUT_DIR}/finetuned_model"

if [ -z "$RUN_NAME" ]; then
    RUN_NAME="${TASK}_k${PRUNE_KEEP_K}"
fi

echo "========================================"
echo "HuggingFace Finetuning Pipeline"
echo "========================================"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Prune keep k: $PRUNE_KEEP_K"
echo "Output dir: $OUTPUT_DIR"
echo "Num GPUs: $NUM_GPUS"
echo "Num epochs: $NUM_EPOCHS"
echo "========================================"

# Step 1: Compute router activations
if [ "$SKIP_ACTIVATION" = false ]; then
    echo ""
    echo "Step 1: Computing router activations..."
    echo "========================================"

    mkdir -p "$(dirname "$ACTIVATION_FILE")"

    python -m src.hf_training.compute_router_activations \
        --model "$MODEL" \
        --task "$TASK" \
        --split "validation" \
        --output-file "$ACTIVATION_FILE" \
        --batch-size "$BATCH_SIZE"

    echo "Activations saved to: $ACTIVATION_FILE"
else
    echo ""
    echo "Step 1: Skipping activation computation (using existing file)"
    echo "Activation file: $ACTIVATION_FILE"
fi

# Step 2: Prune model
if [ "$SKIP_PRUNE" = false ]; then
    echo ""
    echo "Step 2: Pruning model..."
    echo "========================================"

    python -m src.hf_training.prune_hf_model \
        --model "$MODEL" \
        --activation-file "$ACTIVATION_FILE" \
        --prune-keep-k "$PRUNE_KEEP_K" \
        --save-path "$PRUNED_MODEL"

    echo "Pruned model saved to: $PRUNED_MODEL"
else
    echo ""
    echo "Step 2: Skipping pruning (using existing pruned model)"
    echo "Pruned model: $PRUNED_MODEL"
fi
#
## Step 3: Finetune
#echo ""
#echo "Step 3: Finetuning..."
#echo "========================================"
#
## Determine FSDP setting
#if [ "$NUM_GPUS" -gt 1 ]; then
#    FSDP_FLAG=""
#else
#    FSDP_FLAG="--no-fsdp"
#fi
#
#torchrun --nproc_per_node="$NUM_GPUS" \
#    -m src.hf_training.finetune \
#    --model "$PRUNED_MODEL" \
#    --task "$TASK" \
#    --output-dir "$FINETUNED_MODEL" \
#    --num-epochs "$NUM_EPOCHS" \
#    --num-checkpoints "$NUM_CHECKPOINTS" \
#    --learning-rate "$LEARNING_RATE" \
#    --run-name "$RUN_NAME" \
#    $FSDP_FLAG
#
#echo ""
#echo "========================================"
#echo "Pipeline complete!"
#echo "========================================"
#echo "Activations: $ACTIVATION_FILE"
#echo "Pruned model: $PRUNED_MODEL"
#echo "Finetuned model: $FINETUNED_MODEL"
