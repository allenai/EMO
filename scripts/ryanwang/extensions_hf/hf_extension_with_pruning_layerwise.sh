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
# Comma-separated list of merge variants to produce + eval. Each name maps to a flag
# combination (see flags_for_variant() below).
#
# Replace-mode variants (parent's params get OVERWRITTEN by the small model's):
#   default       — only routable expert MLPs
#   shared        — routable + shared expert MLP
#   router        — routable + router rows
#   shared_router — routable + shared expert + router rows
#   non_moe       — routable + attention/norms/embed/lm_head (full continual-pretrain ish)
#
# Average-mode variants (parent ← 0.5·parent + 0.5·small for the same params, --average):
#   default_avg, shared_avg, router_avg, shared_router_avg, non_moe_avg
MERGE_VARIANTS="default,shared,router,shared_router,default_avg,shared_avg,router_avg,shared_router_avg"

# Pipeline phase. Lets you split a slow run across multiple beaker jobs.
#   prune_finetune — only Steps 1-3 (greedy_prune_layerwise + finetune). Leaves
#                    pruned_model/ + finetuned_model/ on disk for a later phase.
#   merge_eval     — only Steps 4-7 (eval small + per-variant merge + eval merged
#                    + per-subject MMLU). Requires pruned_model + finetuned_model
#                    to already exist under ${OUTPUT_DIR}.
#   full           — runs everything end-to-end (default).
PHASE="full"

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
        --merge-variants)
            MERGE_VARIANTS="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
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
case "$PHASE" in
    full|prune_finetune|merge_eval) ;;
    *) echo "Error: --phase must be one of: full, prune_finetune, merge_eval (got '$PHASE')"; exit 1 ;;
esac

OUTPUT_DIR="${BASE_DIR}/${RELATIVE_DIR}"
mkdir -p "$OUTPUT_DIR"

PRUNED_MODEL="${OUTPUT_DIR}/pruned_model"
FINETUNED_MODEL="${OUTPUT_DIR}/finetuned_model"
# Merged outputs go to ${OUTPUT_DIR}/merged_<variant>/checkpoint-<N>/ — see Steps 5-6 below.

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
echo "Phase: $PHASE"
echo "Merge variants: $MERGE_VARIANTS"
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

if [[ "$PHASE" == "merge_eval" ]]; then
    echo ""
    echo "Phase=merge_eval: skipping Steps 1-3 (prune + finetune)."
    echo "Verifying that pre-existing artifacts are present under $OUTPUT_DIR ..."
    if [ ! -d "$PRUNED_MODEL" ]; then
        echo "Error: phase=merge_eval requires existing pruned model at $PRUNED_MODEL"
        exit 1
    fi
    if [ ! -f "${PRUNED_MODEL}/pruning_metadata.json" ]; then
        echo "Error: phase=merge_eval requires ${PRUNED_MODEL}/pruning_metadata.json"
        exit 1
    fi
    if [ ! -d "$FINETUNED_MODEL" ] || ! ls "$FINETUNED_MODEL"/checkpoint-*/ >/dev/null 2>&1; then
        echo "Error: phase=merge_eval requires existing finetuned checkpoints at $FINETUNED_MODEL/checkpoint-*"
        exit 1
    fi
    echo "Found pruned + finetuned artifacts; jumping to Step 4."
else

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

fi  # end: phase != merge_eval (Steps 1-3)

# --- Common setup for any eval-running phase (full, prune_finetune w/ small eval, merge_eval) ---
#
# Map a variant name -> extra flags for merge_pruned_experts_back.py.
# Names ending in "_avg" run in average mode (parent ← (1-α)·parent + α·small);
# names without that suffix run in replace mode (parent ← small).
flags_for_variant () {
    local v="$1"
    local avg_flag=""
    if [[ "$v" == *_avg ]]; then
        avg_flag="--average"
        v="${v%_avg}"
    fi
    case "$v" in
        default)        echo "$avg_flag" ;;
        shared)         echo "--also-copy-shared $avg_flag" ;;
        router)         echo "--also-copy-router-rows $avg_flag" ;;
        shared_router) echo "--also-copy-shared --also-copy-router-rows $avg_flag" ;;
        non_moe)        echo "--also-copy-non-moe $avg_flag" ;;
        *) echo "Unknown merge variant: $1" >&2; exit 1 ;;
    esac
}

# Identify the final (largest checkpoint number) checkpoint dir for downstream steps.
final_checkpoint=$(ls -d "$FINETUNED_MODEL"/checkpoint-*/ | sed 's:/$::' | awk -F- '{print $NF, $0}' | sort -n | tail -1 | awk '{print $2}')
final_checkpoint_num=$(basename "$final_checkpoint" | sed 's/checkpoint-//')
echo "Final finetune checkpoint: $final_checkpoint (step $final_checkpoint_num)"

# Datasets are already cached from steps 1-3 (or from a prior phase); skip HF API calls.
export HF_DATASETS_OFFLINE=1

EVAL_BATCH_SIZE=32
if [[ $TASK == *"history"* ]]; then
    EVAL_BATCH_SIZE=4
fi
if [[ $TASK == *"gsm8k_generation_8shot"* ]]; then
    EVAL_BATCH_SIZE=16
fi

# Per-subject MMLU eval helper (no-op for non-MMLU tasks). Reused for small + each variant.
MMLU_SUBJECTS=$(python -m src.scripts.eval.get_mmlu_subjects "$TASK" 2>/dev/null | grep -v "^Warning:" || true)
if [[ $TASK == mmlu_merged_* ]]; then
    SUBJECT_TASK_PREFIX="mmlu_merged_"
else
    SUBJECT_TASK_PREFIX="mmlu_"
fi

run_per_subject_evals () {
    local label="$1"
    local ckpt="$2"
    if [ -z "$MMLU_SUBJECTS" ]; then
        return 0
    fi
    echo ""
    echo "  Per-subject MMLU evals for $label: $ckpt"
    while IFS= read -r subject; do
        local SUBJECT_BATCH_SIZE=32
        if [[ $subject == *"history"* ]]; then
            SUBJECT_BATCH_SIZE=4
        fi
        python -m src.scripts.eval.launch_eval \
            --model "$ckpt" \
            --model-type hf \
            --task "${SUBJECT_TASK_PREFIX}${subject}-pruned" \
            --pruned_split "test" \
            --remote-output-dir "${S3_BASE}/${RELATIVE_DIR}/${label}/checkpoint-${final_checkpoint_num}/per_subject/${subject}" \
            --batch-size $SUBJECT_BATCH_SIZE \
            --gpus "$NUM_GPUS" \
            "${NUM_SHOTS_EVAL_FLAG[@]}"
    done <<< "$MMLU_SUBJECTS"
}

# Step 4: Eval the small finetuned model (only when we have fresh prune+finetune outputs;
# skipped in phase=merge_eval so we don't redundantly re-run it 8x across variant jobs).
if [[ "$PHASE" != "merge_eval" ]]; then
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

    run_per_subject_evals small "$final_checkpoint"
fi

# Exit early when phase=prune_finetune: we've done Steps 1-3 + small eval + small per-subject.
if [[ "$PHASE" == "prune_finetune" ]]; then
    echo ""
    echo "Phase=prune_finetune complete (Steps 1-3 + small-model eval). Artifacts left under:"
    echo "  $PRUNED_MODEL"
    echo "  $FINETUNED_MODEL"
    echo "Run again with --phase merge_eval (same --base-dir + --relative-dir) to do per-variant merge + eval."
    exit 0
fi

# --- Steps 5+6 (+ per-subject): per-variant merge + eval ---
IFS=',' read -ra MERGE_VARIANTS_ARR <<< "$MERGE_VARIANTS"
echo "Merge variants requested: ${MERGE_VARIANTS_ARR[*]}"

for variant in "${MERGE_VARIANTS_ARR[@]}"; do
    extra_flags=$(flags_for_variant "$variant")
    merged_ckpt="${OUTPUT_DIR}/merged_${variant}/checkpoint-${final_checkpoint_num}"

    echo ""
    echo "Step 5 [$variant]: merging (extra flags: ${extra_flags:-<none>})..."
    echo "========================================"
    python -m src.hf_training.merge_pruned_experts_back \
        --parent-model "$MODEL" \
        --pruned-trained-model "$final_checkpoint" \
        --pruning-metadata "${PRUNED_MODEL}/pruning_metadata.json" \
        --output-dir "$merged_ckpt" \
        $extra_flags

    echo ""
    echo "Step 6 [$variant]: evaluating merged_${variant}..."
    echo "========================================"
    python -m src.scripts.eval.launch_eval \
        --model "$merged_ckpt" \
        --model-type hf \
        --task "${TASK}-pruned" \
        --pruned_split "test" \
        --remote-output-dir "${S3_BASE}/${RELATIVE_DIR}/merged_${variant}/checkpoint-${final_checkpoint_num}" \
        --batch-size $EVAL_BATCH_SIZE \
        --gpus "$NUM_GPUS" \
        "${NUM_SHOTS_EVAL_FLAG[@]}"

    run_per_subject_evals "merged_${variant}" "$merged_ckpt"
done
#
echo ""
echo "========================================"
echo "Extension pipeline complete!"
echo "========================================"
echo "Pruned model:    $PRUNED_MODEL"
echo "Finetuned model: $FINETUNED_MODEL"
echo "Merged variants: ${MERGE_VARIANTS} (each at \${OUTPUT_DIR}/merged_<variant>/checkpoint-<N>)"
#
## Step 8: Cleanup — remove local output directory to save disk space (results are on S3)
#echo ""
#echo "Step 8: Cleaning up local output directory..."
#echo "========================================"
#echo "Removing: $OUTPUT_DIR"
#rm -rf "$OUTPUT_DIR"
#echo "Cleanup complete."
