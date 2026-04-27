#!/bin/bash
# Make src/ a top-level import root so bare imports like `offline_evals` and
# `scripts.eval.tasks` resolve. pip install -e . only registers olmo_core*.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
#
# HuggingFace Extension Pipeline with Greedy Layerwise Expert Pruning
#
# Parallel of hf_finetune_with_pruning_layerwise.sh, with two extra steps:
#   5) merge the finetuned pruned model's experts back into the original parent
#   6) eval the merged full-sized model
#
# v0 defaults:
#   - Finetune is unfrozen (no expert freezing).
#   - Merge copies only routable expert MLPs (router rows / shared / non-MoE
#     stay as in parent).
#   - --num-checkpoints 1 (cap evals at the final checkpoint).
#
# Usage:
#   ./scripts/ryanwang/extensions_hf/hf_extension_with_pruning_layerwise.sh \
#       --model /path/to/model \
#       --task arc_challenge \
#       --prune-keep-k 32 \
#       --num-shared-experts 1 \
#       --base-dir /path/to/extension_evals \
#       --relative-dir <run_subdir> \
#       --num-gpus 4 \
#       --run-name <job_name>
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
NUM_CHECKPOINTS=1
BATCH_SIZE=4
MICRO_BATCH_SIZE=1
LEARNING_RATE=5e-5
RUN_NAME=""
NUM_PRUNE_EXAMPLES=""
NUM_SHOTS_PRUNE=""
NUM_SHOTS_EVAL=""
S3_BASE="s3://ai2-sewonm/ryanwang/extension_evals_hf_0426"

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
        --num-shots-prune)
            NUM_SHOTS_PRUNE="$2"
            shift 2
            ;;
        --num-shots-eval)
            NUM_SHOTS_EVAL="$2"
            shift 2
            ;;
        --s3-base)
            S3_BASE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo "Error: --model is required"
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

PRUNED_MODEL="${OUTPUT_DIR}/pruned_model"
FINETUNED_MODEL="${OUTPUT_DIR}/finetuned_model"
MERGED_MODEL="${OUTPUT_DIR}/merged_model"

echo "========================================"
echo "HuggingFace Extension Pipeline (Layerwise Pruning + Merge-Back)"
echo "========================================"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Prune keep k: $PRUNE_KEEP_K"
echo "Output dir: $OUTPUT_DIR"
echo "Num GPUs: $NUM_GPUS"
echo "Num epochs: $NUM_EPOCHS"
echo "Num checkpoints: $NUM_CHECKPOINTS"
echo "S3 base: $S3_BASE"
echo "========================================"

# Per-stage --num-shots forwarding flags. Empty ⇒ downstream falls back to task config defaults.
NUM_SHOTS_PRUNE_FLAG=()
if [ -n "$NUM_SHOTS_PRUNE" ]; then
    NUM_SHOTS_PRUNE_FLAG=(--num-shots "$NUM_SHOTS_PRUNE")
fi
NUM_SHOTS_EVAL_FLAG=()
if [ -n "$NUM_SHOTS_EVAL" ]; then
    NUM_SHOTS_EVAL_FLAG=(--num-shots "$NUM_SHOTS_EVAL")
fi

# Steps 1+2: Greedy layerwise activation collection + pruning
echo ""
echo "Steps 1+2: Greedy layerwise pruning..."
echo "========================================"

NUM_CAL_FLAG=()
if [ -n "$NUM_PRUNE_EXAMPLES" ]; then
    NUM_CAL_FLAG=(--num-calibration "$NUM_PRUNE_EXAMPLES")
fi

python -m src.hf_training.greedy_prune_layerwise \
    --model "$MODEL" \
    --task "$TASK" \
    --split "validation" \
    --prune-keep-k "$PRUNE_KEEP_K" \
    --num-shared-experts "$NUM_SHARED_EXPERTS" \
    --save-path "$PRUNED_MODEL" \
    --batch-size 32 \
    "${NUM_CAL_FLAG[@]}" \
    "${NUM_SHOTS_PRUNE_FLAG[@]}"

echo "Pruned model saved to: $PRUNED_MODEL"

# Step 3: Finetune the small pruned model
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
export WANDB_TAGS="extension,${TASK:0:60},${PRUNED_MODEL: -60}"

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
    $FSDP_FLAG \
    "${NUM_SHOTS_EVAL_FLAG[@]}"

# Identify the final (largest checkpoint number) checkpoint dir for downstream steps.
all_checkpoints=("$FINETUNED_MODEL"/checkpoint-*/)
# Sort by trailing checkpoint number to pick the latest deterministically.
final_checkpoint=$(ls -d "$FINETUNED_MODEL"/checkpoint-*/ | sed 's:/$::' | awk -F- '{print $NF, $0}' | sort -n | tail -1 | awk '{print $2}')
final_checkpoint_num=$(basename "$final_checkpoint" | sed 's/checkpoint-//')
echo "Final finetune checkpoint: $final_checkpoint (step $final_checkpoint_num)"

# ------------------------------------------------------------------------------
# Datasets are already cached from steps 1-3; skip HF API calls for evals
# ------------------------------------------------------------------------------
export HF_DATASETS_OFFLINE=1

EVAL_BATCH_SIZE=32
if [[ $TASK == *"history"* ]]; then
    echo "Setting eval batch size to 4 for history task"
    EVAL_BATCH_SIZE=4
fi
if [[ $TASK == *"gsm8k_generation_8shot"* ]]; then
    echo "Setting eval batch size to 16 for gsm8k_generation_8shot task"
    EVAL_BATCH_SIZE=16
fi

# Step 4: Eval the small finetuned model (final checkpoint only)
echo ""
echo "Step 4: Evaluating small finetuned model (final checkpoint)..."
echo "========================================"

python -m src.scripts.eval.launch_eval \
    --model "$final_checkpoint" \
    --model-type hf \
    --task "${TASK}-pruned" \
    --pruned_split "test" \
    --remote-output-dir "${S3_BASE}/${RELATIVE_DIR}/small/checkpoint-${final_checkpoint_num}" \
    --batch-size $EVAL_BATCH_SIZE \
    --gpus "$NUM_GPUS" \
    "${NUM_SHOTS_EVAL_FLAG[@]}"

# Step 5: Merge the finetuned small model's experts back into the parent
echo ""
echo "Step 5: Merging trained experts back into parent..."
echo "========================================"

merged_checkpoint="${MERGED_MODEL}/checkpoint-${final_checkpoint_num}"
python -m src.hf_training.merge_pruned_experts_back \
    --parent-model "$MODEL" \
    --pruned-trained-model "$final_checkpoint" \
    --pruning-metadata "${PRUNED_MODEL}/pruning_metadata.json" \
    --output-dir "$merged_checkpoint"

echo "Merged model saved to: $merged_checkpoint"

# Step 6: Eval the merged full-sized model
echo ""
echo "Step 6: Evaluating merged full-sized model..."
echo "========================================"

python -m src.scripts.eval.launch_eval \
    --model "$merged_checkpoint" \
    --model-type hf \
    --task "${TASK}-pruned" \
    --pruned_split "test" \
    --remote-output-dir "${S3_BASE}/${RELATIVE_DIR}/merged/checkpoint-${final_checkpoint_num}" \
    --batch-size $EVAL_BATCH_SIZE \
    --gpus "$NUM_GPUS" \
    "${NUM_SHOTS_EVAL_FLAG[@]}"

# Step 7: Per-subject MMLU evals (if task is an MMLU category/cluster)
# Mirrors pruning_hf step 5: enables macro-average computation across subjects.
# Run for both small and merged final checkpoints.
MMLU_SUBJECTS=$(python -m src.scripts.eval.get_mmlu_subjects "$TASK" 2>/dev/null | grep -v "^Warning:" || true)

if [ -n "$MMLU_SUBJECTS" ]; then
    echo ""
    echo "Step 7: Per-subject MMLU evals (small + merged, final checkpoint)..."
    echo "========================================"

    if [[ $TASK == mmlu_merged_* ]]; then
        SUBJECT_TASK_PREFIX="mmlu_merged_"
    else
        SUBJECT_TASK_PREFIX="mmlu_"
    fi

    for variant in small merged; do
        if [ "$variant" = "small" ]; then
            ckpt="$final_checkpoint"
        else
            ckpt="$merged_checkpoint"
        fi
        echo "Per-subject evals for $variant: $ckpt"

        while IFS= read -r subject; do
            echo "  Evaluating subject: $subject"

            SUBJECT_BATCH_SIZE=32
            if [[ $subject == *"history"* ]]; then
                SUBJECT_BATCH_SIZE=4
            fi

            python -m src.scripts.eval.launch_eval \
                --model "$ckpt" \
                --model-type hf \
                --task "${SUBJECT_TASK_PREFIX}${subject}-pruned" \
                --pruned_split "test" \
                --remote-output-dir "${S3_BASE}/${RELATIVE_DIR}/${variant}/checkpoint-${final_checkpoint_num}/per_subject/${subject}" \
                --batch-size $SUBJECT_BATCH_SIZE \
                --gpus "$NUM_GPUS" \
                "${NUM_SHOTS_EVAL_FLAG[@]}"
        done <<< "$MMLU_SUBJECTS"
    done
else
    echo ""
    echo "Skipping per-subject evals (task $TASK is not an MMLU category/cluster)"
fi

echo ""
echo "========================================"
echo "Extension pipeline complete!"
echo "========================================"
echo "Pruned model:   $PRUNED_MODEL"
echo "Finetuned model: $FINETUNED_MODEL"
echo "Merged model:   $MERGED_MODEL"

# Step 8: Cleanup — remove local output directory to save disk space (results are on S3)
echo ""
echo "Step 8: Cleaning up local output directory..."
echo "========================================"
echo "Removing: $OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"
echo "Cleanup complete."
