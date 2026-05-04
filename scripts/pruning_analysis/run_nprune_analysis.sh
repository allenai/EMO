#!/bin/bash
#
# Calibration-size sensitivity sweep for greedy layerwise MoE pruning.
# For a single model, loops over (task, keep_k, num_prune_examples) and records the
# kept-expert sets + avg router probabilities. Uses byte-identical calibration data
# as the eval pipeline (same validation split, same seed=0 subsample, same tokenizer
# kwargs). Results land at <OUTPUT_DIR>/<stringified_model>/<task>/keepk-<k>/nprune-<N>.json.

set -e

# Make src/ a top-level import root (same trick as hf_finetune_with_pruning_layerwise.sh).
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

BASE_DIR="/root/phdbrainstorm/FlexMoE"

MODEL="${MODEL:-moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419/step250339-hf}"

# Match the NUM_SHARED_EXPERTS auto-detection logic from launch_pruning_hf.sh:151-162.
if [[ $MODEL == *"twolevelbatchlbreducedp512sharedexp1"* ]]; then
    NUM_SHARED_EXPERTS=1
elif [[ $MODEL == *"twolevelbatchlbreducedp512sharedexp2"* ]]; then
    NUM_SHARED_EXPERTS=2
elif [[ $MODEL == *"twolevelbatchlbreducedp512sharedexp4c2"* ]]; then
    NUM_SHARED_EXPERTS=2
elif [[ $MODEL == *"moereducedp512sharedexp1"* ]]; then
    NUM_SHARED_EXPERTS=1
else
    NUM_SHARED_EXPERTS=0
fi

KEEP_K_VALUES=(8 16 32 64)
NPRUNE_VALUES=(1 5 10 100 all)

TASKS=(
    # GSM8K 8-shot generation (mirrors launch_pruning_hf.sh:87)
    gsm8k_generation_8shot_merged

    # 17 mmlu_merged categories (mirrors launch_pruning_hf.sh:102-118)
    mmlu_merged_biology
    mmlu_merged_business
    mmlu_merged_chemistry
    mmlu_merged_computer_science
    mmlu_merged_culture
    mmlu_merged_economics
    mmlu_merged_engineering
    mmlu_merged_geography
    mmlu_merged_health
    mmlu_merged_history
    mmlu_merged_law
    mmlu_merged_math
    mmlu_merged_other
    mmlu_merged_philosophy_cat
    mmlu_merged_physics
    mmlu_merged_politics
    mmlu_merged_psychology

    # 14 mmlu_pro_merged categories (mirrors launch_pruning_hf.sh:121-134)
    mmlu_pro_merged_math
    mmlu_pro_merged_health
    mmlu_pro_merged_physics
    mmlu_pro_merged_business
    mmlu_pro_merged_biology
    mmlu_pro_merged_chemistry
    mmlu_pro_merged_computer_science
    mmlu_pro_merged_economics
    mmlu_pro_merged_engineering
    mmlu_pro_merged_philosophy
    mmlu_pro_merged_other
    mmlu_pro_merged_history
    mmlu_pro_merged_psychology
    mmlu_pro_merged_law
)

OUTPUT_DIR="claude_outputs/prune_plots/nprune_analysis"
STRINGIFIED_MODEL=$(echo "$MODEL" | sed 's/[^a-zA-Z0-9_-]//g')

# Number of GPU shards to run in parallel. Tasks are assigned round-robin
# (task_idx %% NUM_SHARDS == shard_idx), so each shard processes ~len(TASKS)/NUM_SHARDS
# tasks on its own GPU. Set NUM_SHARDS=1 for single-GPU sequential mode.
NUM_SHARDS="${NUM_SHARDS:-4}"
LOG_DIR="$OUTPUT_DIR/$STRINGIFIED_MODEL/shard_logs"
mkdir -p "$LOG_DIR"

echo "Model: $MODEL"
echo "Num shared experts: $NUM_SHARED_EXPERTS"
echo "Tasks: ${#TASKS[@]}"
echo "Keep-k values: ${KEEP_K_VALUES[*]}"
echo "Nprune values: ${NPRUNE_VALUES[*]}"
echo "Output: $OUTPUT_DIR/$STRINGIFIED_MODEL"
echo "Shards: $NUM_SHARDS (logs: $LOG_DIR/shard_*.log)"
echo

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="$i" PYTHONUNBUFFERED=1 python -u \
        -m src.hf_training.analyze_nprune_keep_sets \
        --model "${BASE_DIR}/models/${MODEL}" \
        --num-shared-experts "$NUM_SHARED_EXPERTS" \
        --tasks "${TASKS[@]}" \
        --prune-keep-k-values "${KEEP_K_VALUES[@]}" \
        --num-prune-examples-values "${NPRUNE_VALUES[@]}" \
        --output-dir "$OUTPUT_DIR" \
        --shard-idx "$i" \
        --num-shards "$NUM_SHARDS" \
        --skip-existing \
        >"$LOG_DIR/shard_$i.log" 2>&1 &
    PIDS+=($!)
    echo "  launched shard $i (pid ${PIDS[$i]}) on CUDA_VISIBLE_DEVICES=$i"
done

echo
echo "Waiting for ${#PIDS[@]} shards to finish..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=1
    fi
done
if [ "$FAIL" -ne 0 ]; then
    echo "One or more shards failed. Check $LOG_DIR/shard_*.log"
    exit 1
fi

python -u -m src.hf_training.summarize_nprune_keep_sets \
    --output-dir "$OUTPUT_DIR" \
    --model "$STRINGIFIED_MODEL"

echo
echo "Done. Summary: $OUTPUT_DIR/$STRINGIFIED_MODEL/summary.md"
