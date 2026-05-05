#!/bin/bash
# Make src/ a top-level import root so bare imports like `offline_evals` and
# `scripts.eval.tasks` resolve. pip install -e . only registers olmo_core*.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
#
# Baseline eval wrapper for a HuggingFace-Hub model (or any unmodified HF
# checkpoint). No selection, no finetuning. Mirrors the tail of the selective
# wrappers (launch_eval + per-subject MMLU loop).
#
# Output layout (under s3://ai2-sewonm/ryanwang/selective_evals_final/):
#   ${RELATIVE_DIR}/results/checkpoint-0/
#   ${RELATIVE_DIR}/results/checkpoint-0/per_subject/<subject>/   (MMLU only)
#

set -e

MODEL=""
REVISION=""
TASK=""
RELATIVE_DIR=""
NUM_GPUS=1
RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)         MODEL="$2";        shift 2 ;;
        --revision)      REVISION="$2";     shift 2 ;;
        --task)          TASK="$2";         shift 2 ;;
        --relative-dir)  RELATIVE_DIR="$2"; shift 2 ;;
        --num-gpus)      NUM_GPUS="$2";     shift 2 ;;
        --run-name)      RUN_NAME="$2";     shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ];        then echo "Error: --model is required";        exit 1; fi
if [ -z "$TASK" ];         then echo "Error: --task is required";         exit 1; fi
if [ -z "$RELATIVE_DIR" ]; then echo "Error: --relative-dir is required"; exit 1; fi
if [ -z "$RUN_NAME" ];     then echo "Error: --run-name is required";     exit 1; fi

REVISION_ARGS=()
if [ -n "$REVISION" ] && [ "$REVISION" != "none" ]; then
    REVISION_ARGS=(--revision "$REVISION")
fi

EVAL_BATCH_SIZE=32
if [[ $TASK == *"history"* ]]; then
    EVAL_BATCH_SIZE=4
fi
if [[ $TASK == *"gsm8k_generation_8shot"* ]]; then
    EVAL_BATCH_SIZE=16
fi

# NOTE: unlike the selective wrappers we do NOT set HF_DATASETS_OFFLINE=1 here.
# The selective flow's offline flag is safe because the selection step runs
# first and warms the HF dataset cache. Baseline skips selection, so we need
# the online fallback path when the weka mount misses.
S3_BASE="s3://ai2-sewonm/ryanwang/selective_evals_final/${RELATIVE_DIR}"

echo "========================================"
echo "HF model baseline eval"
echo "Model:    $MODEL (revision: ${REVISION:-default})"
echo "Task:     $TASK"
echo "GPUs:     $NUM_GPUS"
echo "Batch:    $EVAL_BATCH_SIZE"
echo "Out:      $S3_BASE/results/checkpoint-0/"
echo "========================================"

python -m src.scripts.eval.launch_eval \
    --model "$MODEL" \
    --model-type hf \
    "${REVISION_ARGS[@]}" \
    --task "${TASK}-pruned" \
    --pruned_split "test" \
    --remote-output-dir "${S3_BASE}/results/checkpoint-0" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --gpus "$NUM_GPUS" \
    --model-args trust_remote_code=true

# Per-subject MMLU breakdown (only for MMLU category/cluster tasks)
MMLU_SUBJECTS=$(python -m src.scripts.eval.get_mmlu_subjects "$TASK" 2>/dev/null | grep -v "^Warning:" || true)

if [ -n "$MMLU_SUBJECTS" ]; then
    echo ""
    echo "Per-subject MMLU evals..."
    echo "========================================"

    if [[ $TASK == mmlu_merged_* ]]; then
        SUBJECT_TASK_PREFIX="mmlu_merged_"
    else
        SUBJECT_TASK_PREFIX="mmlu_"
    fi

    while IFS= read -r subject; do
        echo "  Evaluating subject: $subject"

        SUBJECT_BATCH_SIZE=32
        if [[ $subject == *"history"* ]]; then
            SUBJECT_BATCH_SIZE=4
        fi

        python -m src.scripts.eval.launch_eval \
            --model "$MODEL" \
            --model-type hf \
            "${REVISION_ARGS[@]}" \
            --task "${SUBJECT_TASK_PREFIX}${subject}-pruned" \
            --pruned_split "test" \
            --remote-output-dir "${S3_BASE}/results/checkpoint-0/per_subject/${subject}" \
            --batch-size "$SUBJECT_BATCH_SIZE" \
            --gpus "$NUM_GPUS" \
            --model-args trust_remote_code=true
    done <<< "$MMLU_SUBJECTS"
else
    echo ""
    echo "Skipping per-subject evals (task $TASK is not an MMLU category/cluster)"
fi

echo ""
echo "========================================"
echo "Baseline eval complete!"
echo "========================================"
