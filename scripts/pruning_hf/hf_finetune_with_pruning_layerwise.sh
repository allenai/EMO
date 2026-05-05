#!/bin/bash
# Make src/ a top-level import root so bare imports like `offline_evals` and
# `scripts.eval.tasks` resolve. pip install -e . only registers olmo_core*.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
#
# HuggingFace-Native Finetuning Pipeline with Greedy Layerwise Expert Pruning
#
# Pipeline:
#   1. Greedy layerwise pruning via greedy_prune_layerwise.py — prunes one
#      layer at a time so each layer's activation statistics are conditioned
#      on the already-pruned earlier layers.
#   2. Finetune the pruned model on the task's training set.
#   3. Evaluate.
#
# Usage:
#   ./scripts/pruning_hf/hf_finetune_with_pruning_layerwise.sh \
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
NUM_PRUNE_EXAMPLES=""
NUM_PRUNE_SEED=""
NUM_SHOTS_PRUNE=""
NUM_SHOTS_EVAL=""
TRUST_REMOTE_CODE=false

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
        --num-prune-examples)
            NUM_PRUNE_EXAMPLES="$2"
            shift 2
            ;;
        --trust-remote-code)
            TRUST_REMOTE_CODE=true
            shift
            ;;
        --num-prune-seed)
            NUM_PRUNE_SEED="$2"
            shift 2
            ;;
        --num-shots-prune)
            NUM_SHOTS_PRUNE="$2"
            shift 2
            ;;
        --num-shots-eval)
            NUM_SHOTS_EVAL="$2"
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

# Per-stage --num-shots forwarding flags. Empty ⇒ downstream falls back to
# task config defaults.
NUM_SHOTS_PRUNE_FLAG=()
if [ -n "$NUM_SHOTS_PRUNE" ]; then
    NUM_SHOTS_PRUNE_FLAG=(--num-shots "$NUM_SHOTS_PRUNE")
fi
NUM_SHOTS_EVAL_FLAG=()
if [ -n "$NUM_SHOTS_EVAL" ]; then
    NUM_SHOTS_EVAL_FLAG=(--num-shots "$NUM_SHOTS_EVAL")
fi

# Steps 1+2: Greedy layerwise activation collection + pruning (fused into one script)
if [ "$SKIP_PRUNE" = false ]; then
    echo ""
    echo "Steps 1+2: Greedy layerwise pruning..."
    echo "========================================"

    NUM_CAL_FLAG=()
    if [ -n "$NUM_PRUNE_EXAMPLES" ]; then
        NUM_CAL_FLAG=(--num-calibration "$NUM_PRUNE_EXAMPLES")
    fi
    PRUNE_SEED_FLAG=()
    if [ -n "$NUM_PRUNE_SEED" ]; then
        PRUNE_SEED_FLAG=(--prune-seed "$NUM_PRUNE_SEED")
    fi

    TRC_FLAG=()
    if [ "$TRUST_REMOTE_CODE" = true ]; then
        TRC_FLAG=(--trust-remote-code)
    fi

    python -m src.hf_training.greedy_prune_layerwise \
        --model "$MODEL" \
        --task "$TASK" \
        --split "validation" \
        --prune-keep-k "$PRUNE_KEEP_K" \
        --num-shared-experts "$NUM_SHARED_EXPERTS" \
        --save-path "$PRUNED_MODEL" \
        --batch-size 32 \
        "${TRC_FLAG[@]}" \
        "${NUM_CAL_FLAG[@]}" \
        "${PRUNE_SEED_FLAG[@]}" \
        "${NUM_SHOTS_PRUNE_FLAG[@]}"

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
PM_TAG="${PRUNED_MODEL: -60}"
[ -z "$PM_TAG" ] && PM_TAG="$PRUNED_MODEL"
export WANDB_TAGS="finetune,${TASK:0:60},${PM_TAG}"

gas=$(( BATCH_SIZE / (NUM_GPUS * MICRO_BATCH_SIZE) ))

FT_TRC_FLAG=""
if [ "$TRUST_REMOTE_CODE" = true ]; then
    FT_TRC_FLAG="--trust-remote-code"
fi

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
    $FSDP_FLAG \
    $FT_TRC_FLAG \
    "${NUM_SHOTS_EVAL_FLAG[@]}"

# Step 4: Evals
echo ""
echo "Step 4: Evals..."
echo "========================================"

# Datasets are already cached from steps 1-3; skip HF API calls to avoid rate limits
export HF_DATASETS_OFFLINE=1

# Forward trust_remote_code to launch_eval via --model-args. Saved checkpoints
# inherit auto_map from the original HF model, so reloading still needs it.
EVAL_MA_FLAG=""
if [ "$TRUST_REMOTE_CODE" = true ]; then
    EVAL_MA_FLAG="--model-args trust_remote_code=true"
fi

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
    if [[ $TASK == *"gsm8k_generation_8shot"* ]]; then
      echo "Setting eval batch size to 16 for gsm8k_generation_8shot task"
      EVAL_BATCH_SIZE=16
    fi

    python -m src.scripts.eval.launch_eval \
        --model "$checkpoint" \
        --model-type hf \
        --task "$TASK-pruned" \
        --pruned_split "test" \
        --output-dir "${OUTPUT_DIR}/results/checkpoint-${checkpoint_num}" \
        --batch-size $EVAL_BATCH_SIZE \
        --gpus "$NUM_GPUS" \
        $EVAL_MA_FLAG \
        "${NUM_SHOTS_EVAL_FLAG[@]}"
done

# Step 5: Per-subject evals (for MMLU category/cluster tasks only)
# Evaluates the finetuned model on each individual subject within the category,
# enabling macro-average computation for fair comparison across grouping schemes.
MMLU_SUBJECTS=$(python -m src.scripts.eval.get_mmlu_subjects "$TASK" 2>/dev/null | grep -v "^Warning:" || true)

if [ -n "$MMLU_SUBJECTS" ]; then
    echo ""
    echo "Step 5: Per-subject MMLU evals..."
    echo "========================================"

    # If the parent task is a merged category (mmlu_merged_<cat>), route per-subject
    # evals to the merged per-subject task entries so naming stays consistent.
    if [[ $TASK == mmlu_merged_* ]]; then
        SUBJECT_TASK_PREFIX="mmlu_merged_"
    else
        SUBJECT_TASK_PREFIX="mmlu_"
    fi

    for checkpoint in "${all_checkpoints[@]}"; do
        checkpoint_num=$(basename "$checkpoint" | sed 's/checkpoint-//')
        echo "Per-subject evals for checkpoint: $checkpoint"

        while IFS= read -r subject; do
            echo "  Evaluating subject: $subject"

            EVAL_BATCH_SIZE=32
            if [[ $subject == *"history"* ]]; then
                EVAL_BATCH_SIZE=4
            fi

            python -m src.scripts.eval.launch_eval \
                --model "$checkpoint" \
                --model-type hf \
                --task "${SUBJECT_TASK_PREFIX}${subject}-pruned" \
                --pruned_split "test" \
                --output-dir "${OUTPUT_DIR}/results/checkpoint-${checkpoint_num}/per_subject/${subject}" \
                --batch-size $EVAL_BATCH_SIZE \
                --gpus "$NUM_GPUS" \
                $EVAL_MA_FLAG \
                "${NUM_SHOTS_EVAL_FLAG[@]}"
        done <<< "$MMLU_SUBJECTS"
    done
else
    echo ""
    echo "Skipping per-subject evals (task $TASK is not an MMLU category/cluster)"
fi

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
echo "Pruned model: $PRUNED_MODEL"
echo "Finetuned model: $FINETUNED_MODEL"

# Step 6: Cleanup — remove local output directory to save disk space (results are on S3)
echo ""
echo "Step 6: Cleaning up local output directory..."
echo "========================================"
echo "Removing: $OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"
echo "Cleanup complete."
