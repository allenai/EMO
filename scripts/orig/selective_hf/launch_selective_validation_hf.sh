#!/bin/bash

# Output root. Each (model, task, config) gets its own subdir under here
# containing the selected-expert model, finetuned checkpoints, and eval results.
# Override via env var before invoking this script.
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/selective_evals_final}"

# Number of GPUs to use per worker run. Forwarded to torchrun inside each
# worker script. Override via env var.
NUM_GPUS="${NUM_GPUS:-1}"

MODELS=(
    # HF Hub entries: format "hf:<id>|shared=<N>|skip_selective=<true|false>"
    "hf:allenai/Emo_1b14b_1T|shared=1|skip_selective=false"
    )

# Selective mode: "layerwise"           -- greedy layer-by-layer expert selection
#                                          (each layer conditioned on already-selected
#                                          earlier layers)
#                 "easy_ep"             -- EASY-EP (arXiv 2504.06792): one-shot
#                                          domain-specific selection using
#                                          gating*||expert_out|| weighted by
#                                          (1 - cos_sim) of MoE in/out on few-shot
#                                          calibration
SELECTIVE_MODE="layerwise"

num_epochs=1
SELECTIVE_KEEP_K_VALUES=(8 16 32 128)
batch_size=32

# --- Selective calibration-set sizes to sweep ---
# Each entry: integer ⇒ deterministic subsample of that many prompts;
# empty string ⇒ full validation pool ("All").
NUM_SELECTIVE_EXAMPLES_VALUES=(1 5 10 100 "")

# --- Calibration-subsample seed ---
# Controls torch.Generator().manual_seed(...) in the calibration permutation.
# Default 0 reproduces historical behavior (the same single example is picked
# each run with NUM_SELECTIVE_EXAMPLES=1). Change to 1, 2, ... to draw a different
# calibration subset. Output dir gets a _pseed-<N> suffix when != 0 so different
# seeds don't collide on disk. Ignored when NUM_SELECTIVE_EXAMPLES is empty (no
# subsampling) or "random" (no calibration).
NUM_SELECTIVE_SEED=""

# --- Shot-count overrides (two orthogonal knobs) ---
# Each var: empty ⇒ each task's default num_shots (e.g. mmlu_merged_* = 5-shot,
# gsm8k_generation_8shot_merged = 8-shot). Set to an integer to force that
# shot count for the corresponding stage.
#
#   NUM_SHOTS_SELECTIVE: selection-calibration shots. Ignored for skip-selective
#     models (no selection stage) and NUM_SELECTIVE_EXAMPLES="random" (no calibration).
#   NUM_SHOTS_EVAL:      finetune + eval shots.
#
# Output dir suffix uses _pshots-{X} and/or _eshots-{Y}. Only stages that are
# actually overridden appear. Examples:
#   SELECTIVE="0" EVAL=""  → _pshots-0
#   SELECTIVE=""  EVAL="0" → _eshots-0
#   SELECTIVE="0" EVAL="0" → _pshots-0_eshots-0
# Three (NUM_SHOTS_SELECTIVE, NUM_SHOTS_EVAL) configurations to sweep, parallel arrays.
# Empty string ⇒ each task's default num_shots.
SHOT_CONFIGS_SELECTIVE=("" "" "0")
SHOT_CONFIGS_EVAL=(""    "0"  "0")

# Define grouped tasks
TASK_GROUPS_LIST=(
  # Merged variants for the MC9 + perplexity tasks (selection + finetuning share data)
  "arc_easy_merged"
  "arc_challenge_merged"
  "boolq_merged"
  "hellaswag_merged"
  "csqa_merged"
  "openbookqa_merged"
  "piqa_merged"
  "socialiqa_merged"
  "winogrande_merged"

  # GSM8K generation merged variant (selection + finetuning share data)
  "gsm8k_generation_8shot_merged"

  # Gen5 merged variants (selection + finetuning share data)
  "squad_merged"
  "coqa_merged"
  "naturalqs_merged"
  "triviaqa_merged"
  "drop_merged"

  # MMLU 17-category merged variants (selection + finetuning share data)
  "mmlu_merged_biology"
  "mmlu_merged_business"
  "mmlu_merged_chemistry"
  "mmlu_merged_computer_science"
  "mmlu_merged_culture"
  "mmlu_merged_economics"
  "mmlu_merged_engineering"
  "mmlu_merged_geography"
  "mmlu_merged_health"
  "mmlu_merged_history"
  "mmlu_merged_law"
  "mmlu_merged_math"
  "mmlu_merged_other"
  "mmlu_merged_philosophy_cat"
  "mmlu_merged_physics"
  "mmlu_merged_politics"
  "mmlu_merged_psychology"

  # MMLU-Pro merged variant (selection + finetuning use same data)
  "mmlu_pro_merged_math"
  "mmlu_pro_merged_health"
  "mmlu_pro_merged_physics"
  "mmlu_pro_merged_business"
  "mmlu_pro_merged_biology"
  "mmlu_pro_merged_chemistry"
  "mmlu_pro_merged_computer_science"
  "mmlu_pro_merged_economics"
  "mmlu_pro_merged_engineering"
  "mmlu_pro_merged_philosophy"
  "mmlu_pro_merged_other"
  "mmlu_pro_merged_history"
  "mmlu_pro_merged_psychology"
  "mmlu_pro_merged_law"

)

echo "Launching evals for ${#SHOT_CONFIGS_SELECTIVE[@]} shot configs, ${#NUM_SELECTIVE_EXAMPLES_VALUES[@]} validation sizes, ${#MODELS[@]} models, ${#SELECTIVE_KEEP_K_VALUES[@]} keep-k values, and ${#TASK_GROUPS_LIST[@]} task groups..."
echo "Models: ${MODELS[@]}"
echo "Keep-k values: ${SELECTIVE_KEEP_K_VALUES[@]}"
echo "Validation sizes: ${NUM_SELECTIVE_EXAMPLES_VALUES[@]}"
echo "GPUs: $NUM_GPUS"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Outer sweep over (NUM_SHOTS_SELECTIVE, NUM_SHOTS_EVAL) shot configurations.
# Each iteration sets both vars, then runs the full
# (model × keep-k × NUM_SELECTIVE_EXAMPLES × task) cross-product for that config.
for shot_idx in "${!SHOT_CONFIGS_SELECTIVE[@]}"; do
  NUM_SHOTS_SELECTIVE="${SHOT_CONFIGS_SELECTIVE[$shot_idx]}"
  NUM_SHOTS_EVAL="${SHOT_CONFIGS_EVAL[$shot_idx]}"
  echo ""
  echo "=========================================="
  echo "=== Shots: SELECTIVE=${NUM_SHOTS_SELECTIVE:-(empty)}, EVAL=${NUM_SHOTS_EVAL:-(empty)}"
  echo "=========================================="

# Launch evaluation for each model, keep-k, and task combination
for ENTRY in "${MODELS[@]}"; do
  # Parse entry: either a bare local-path (legacy) or an HF tag of the form
  #   "hf:<hf_id>|shared=<N>|skip_selective=<true|false>"
  # HF entries require explicit 'shared' and 'skip_selective' since the substring
  # heuristics below don't recognize the new short HF names (StdMoE, ModMoE, …).
  if [[ "$ENTRY" == hf:* ]]; then
    IS_HF=true
    rest="${ENTRY#hf:}"
    MODEL="${rest%%|*}"
    num_shared_experts_override=""
    skip_selective_override=""
    if [[ "$rest" == *"|"* ]]; then
      metadata="${rest#*|}"
      IFS='|' read -ra meta_parts <<< "$metadata"
      for kv in "${meta_parts[@]}"; do
        case "$kv" in
          shared=*) num_shared_experts_override="${kv#shared=}" ;;
          skip_selective=*) skip_selective_override="${kv#skip_selective=}" ;;
          *) echo "ERROR: unknown metadata '$kv' in HF entry '$ENTRY'"; exit 1 ;;
        esac
      done
    fi
    if [ -z "$num_shared_experts_override" ]; then
      echo "ERROR: HF entry '$ENTRY' missing required 'shared=N' metadata"
      exit 1
    fi
    if [ -z "$skip_selective_override" ]; then
      echo "ERROR: HF entry '$ENTRY' missing required 'skip_selective=true|false' metadata"
      exit 1
    fi
    MODEL_ARG="$MODEL"
    TRC_FLAG="--trust-remote-code"
    num_shared_experts="$num_shared_experts_override"
    SKIP_SELECTIVE_DECISION="$skip_selective_override"
  else
    IS_HF=false
    MODEL="$ENTRY"
    MODEL_ARG="${MODEL_ARG}"
    TRC_FLAG=""
    # Auto-detect num_shared_experts from substring on local path
    if [[ $MODEL == *"twolevelbatchlbreducedp512sharedexp1"* ]]; then
        num_shared_experts=1
    elif [[ $MODEL == *"twolevelbatchlbreducedp512sharedexp2"* ]]; then
        num_shared_experts=2
    elif [[ $MODEL == *"twolevelbatchlbreducedp512sharedexp4c2"* ]]; then
        num_shared_experts=2
    elif [[ $MODEL == *"moereducedp512sharedexp1"* ]]; then
        num_shared_experts=1
    else
        num_shared_experts=0
    fi
    # Auto-detect skip-selective from substring on local path
    if [[ $MODEL == *"dense_1b"* || $MODEL == *"1b4b"* ]]; then
        SKIP_SELECTIVE_DECISION=true
    else
        SKIP_SELECTIVE_DECISION=false
    fi
  fi

  for selective_keep_k in "${SELECTIVE_KEEP_K_VALUES[@]}"; do
    echo "Processing model: ${MODEL} (hf=${IS_HF}, shared=${num_shared_experts}, skip_selective=${SKIP_SELECTIVE_DECISION}), keep-k: ${selective_keep_k}"

    for NUM_SELECTIVE_EXAMPLES in "${NUM_SELECTIVE_EXAMPLES_VALUES[@]}"; do
      echo "  Validation size: ${NUM_SELECTIVE_EXAMPLES:-All}"

    for TASK in "${TASK_GROUPS_LIST[@]}"; do
        # Per-task micro-batch overrides (memory-bound: longer prompts / generation tasks
        # OOM at the default size). These are about model/task memory, not cluster
        # scheduling, so they stay even in local mode.
        micro_batch_size=8
        if [[ $TASK == *"mmlu_history"* ]]; then
            micro_batch_size=2
        fi
        if [[ $TASK == *"gsm8k_generation_8shot"* || $TASK == *"drop_merged"* || $TASK == *"squad_merged"* ]]; then
            micro_batch_size=2
        fi

        # choose the right learning rate based on task
        lr=5e-5

        # Create a shorter, valid job name
        # Remove invalid characters and truncate long names

        stringified_model=$(echo $MODEL | sed 's/[^a-zA-Z0-9_-]//g')

        # Random selection is mode-agnostic: override selectivemode in the output name
        # and skip the _nselective-... suffix (redundant with _selectivemode-random).
        if [[ $NUM_SELECTIVE_EXAMPLES == "random" ]]; then
          relative_dir="${stringified_model}/${TASK}_keepk_${selective_keep_k}_bs-${batch_size}_lr-${lr}_epoch-${num_epochs}_selectivemode-random"
        else
          relative_dir="${stringified_model}/${TASK}_keepk_${selective_keep_k}_bs-${batch_size}_lr-${lr}_epoch-${num_epochs}_selectivemode-${SELECTIVE_MODE}"
        fi

        # Append calibration-set-size suffix when overriding the default (use-all) behavior.
        # Skip when NUM_SELECTIVE_EXAMPLES=="random" since the selectivemode-random token
        # already conveys this.
        if [ -n "$NUM_SELECTIVE_EXAMPLES" ] && [[ $NUM_SELECTIVE_EXAMPLES != "random" ]]; then
            relative_dir="${relative_dir}_nselective-${NUM_SELECTIVE_EXAMPLES}"
        fi

        # Append calibration-seed suffix when overriding the default seed=0.
        # Only meaningful when calibration is actually subsampled (skip when no
        # NUM_SELECTIVE_EXAMPLES or NUM_SELECTIVE_EXAMPLES==random).
        if [ -n "$NUM_SELECTIVE_SEED" ] && [ "$NUM_SELECTIVE_SEED" != "0" ] \
             && [ -n "$NUM_SELECTIVE_EXAMPLES" ] && [[ $NUM_SELECTIVE_EXAMPLES != "random" ]]; then
            relative_dir="${relative_dir}_pseed-${NUM_SELECTIVE_SEED}"
        fi

        # Append per-stage shot-count suffixes when overriding task defaults.
        # _pshots-* is skipped for runs that don't have a selection-calibration
        # stage (skip-selective models or random selection) — avoids misleading names.
        selective_uses_calibration=true
        if [[ $NUM_SELECTIVE_EXAMPLES == "random" ]] || [ "$SKIP_SELECTIVE_DECISION" = true ]; then
            selective_uses_calibration=false
        fi
        if [ "$selective_uses_calibration" = true ] && [ -n "$NUM_SHOTS_SELECTIVE" ]; then
            relative_dir="${relative_dir}_pshots-${NUM_SHOTS_SELECTIVE}"
        fi
        if [ -n "$NUM_SHOTS_EVAL" ]; then
            relative_dir="${relative_dir}_eshots-${NUM_SHOTS_EVAL}"
        fi

        safe_relative_dir=$(printf '%s' "$relative_dir" | sed 's/[^a-zA-Z0-9_-]//g')
        job_name="eval-${safe_relative_dir}"

        # Optional calibration-size flag forwarded to the per-mode worker scripts.
        NSE_FLAG=""
        if [ -n "$NUM_SELECTIVE_EXAMPLES" ]; then
            NSE_FLAG="--num-selective-examples ${NUM_SELECTIVE_EXAMPLES}"
        fi

        # Optional calibration-seed flag forwarded to the per-mode worker scripts.
        # Skipped for random / skip-selective paths (no calibration step).
        NSEED_FLAG=""
        if [ -n "$NUM_SELECTIVE_SEED" ]; then
            NSEED_FLAG="--num-selective-seed ${NUM_SELECTIVE_SEED}"
        fi

        # Optional shot-count flags forwarded to the per-mode worker scripts.
        # --num-shots-selective goes to the selection Python call; --num-shots-eval
        # goes to finetune + eval. Worker scripts that skip selection (random,
        # skip-selective) ignore --num-shots-selective.
        NSHOTS_SELECTIVE_FLAG=""
        if [ -n "$NUM_SHOTS_SELECTIVE" ]; then
            NSHOTS_SELECTIVE_FLAG="--num-shots-selective ${NUM_SHOTS_SELECTIVE}"
        fi
        NSHOTS_EVAL_FLAG=""
        if [ -n "$NUM_SHOTS_EVAL" ]; then
            NSHOTS_EVAL_FLAG="--num-shots-eval ${NUM_SHOTS_EVAL}"
        fi

        echo "  model: ${MODEL_ARG}"
        echo "  task: ${TASK}"
        echo "  output-dir: ${OUTPUT_DIR}/${relative_dir}"
        echo "  job_name: ${job_name}"
        echo "  num_gpus: ${NUM_GPUS}"
        echo "  batch_size: ${batch_size} (micro=${micro_batch_size})"
        echo "  learning-rate: ${lr}"
        echo "  epochs: ${num_epochs}"
        echo "  num_shared_experts: ${num_shared_experts}"

        # Skip selection if the model was tagged skip_selective=true (HF) or matched
        # the legacy dense/1b4b substring heuristic (non-HF branch).
        if [ "$SKIP_SELECTIVE_DECISION" = true ]; then
            echo "  Skipping selection for model: $MODEL"
            bash scripts/selective_hf/hf_finetune_with_selective_layerwise.sh \
                --selected-model ${MODEL_ARG} \
                --task ${TASK} \
                --base-dir "${OUTPUT_DIR}" \
                --relative-dir ${relative_dir} \
                --num-gpus $NUM_GPUS \
                --run-name ${job_name} \
                --learning-rate ${lr} \
                --batch-size ${batch_size} \
                --micro-batch-size ${micro_batch_size} \
                --num-epochs ${num_epochs} \
                --num-checkpoints 1 \
                --num-shared-experts ${num_shared_experts} \
                --skip-selective \
                ${TRC_FLAG} \
                ${NSHOTS_EVAL_FLAG}
            echo "Ran evaluation for model: $MODEL, task: $TASK"
            echo "----------------------------------------"
            continue
        fi

        if [[ $NUM_SELECTIVE_EXAMPLES == "random" ]]; then
            bash scripts/selective_hf/hf_finetune_with_selective_random.sh \
                --model ${MODEL_ARG} \
                --task ${TASK} \
                --selective-keep-k ${selective_keep_k} \
                --base-dir "${OUTPUT_DIR}" \
                --relative-dir ${relative_dir} \
                --num-gpus $NUM_GPUS \
                --run-name ${job_name} \
                --learning-rate ${lr} \
                --batch-size ${batch_size} \
                --micro-batch-size ${micro_batch_size} \
                --num-epochs ${num_epochs} \
                --num-checkpoints 1 \
                --num-shared-experts ${num_shared_experts} \
                ${TRC_FLAG} \
                ${NSHOTS_EVAL_FLAG}
        elif [[ $SELECTIVE_MODE == "easy_ep" ]]; then
            bash scripts/selective_hf/hf_finetune_with_selective_easy_ep.sh \
                --model ${MODEL_ARG} \
                --task ${TASK} \
                --selective-keep-k ${selective_keep_k} \
                --base-dir "${OUTPUT_DIR}" \
                --relative-dir ${relative_dir} \
                --num-gpus $NUM_GPUS \
                --run-name ${job_name} \
                --learning-rate ${lr} \
                --batch-size ${batch_size} \
                --micro-batch-size ${micro_batch_size} \
                --num-epochs ${num_epochs} \
                --num-checkpoints 1 \
                --num-shared-experts ${num_shared_experts} \
                ${TRC_FLAG} \
                ${NSE_FLAG} \
                ${NSEED_FLAG} \
                ${NSHOTS_SELECTIVE_FLAG} \
                ${NSHOTS_EVAL_FLAG}
        elif [[ $SELECTIVE_MODE == "layerwise" ]]; then
            bash scripts/selective_hf/hf_finetune_with_selective_layerwise.sh \
                --model ${MODEL_ARG} \
                --task ${TASK} \
                --selective-keep-k ${selective_keep_k} \
                --base-dir "${OUTPUT_DIR}" \
                --relative-dir ${relative_dir} \
                --num-gpus $NUM_GPUS \
                --run-name ${job_name} \
                --learning-rate ${lr} \
                --batch-size ${batch_size} \
                --micro-batch-size ${micro_batch_size} \
                --num-epochs ${num_epochs} \
                --num-checkpoints 1 \
                --num-shared-experts ${num_shared_experts} \
                ${TRC_FLAG} \
                ${NSE_FLAG} \
                ${NSEED_FLAG} \
                ${NSHOTS_SELECTIVE_FLAG} \
                ${NSHOTS_EVAL_FLAG}
        else
            echo "ERROR: unsupported SELECTIVE_MODE='${SELECTIVE_MODE}' (valid: layerwise, easy_ep)"
            exit 1
        fi

        echo "Ran evaluation for model: $MODEL, task: $TASK"
        echo "----------------------------------------"
    done

    done  # end NUM_SELECTIVE_EXAMPLES loop

    echo "Completed all tasks for model: $MODEL, keep-k: $selective_keep_k"
    echo "========================================"
  done
done

done  # end SHOT_CONFIGS loop

echo "All evaluations have completed!"
echo "Total runs: $((${#SHOT_CONFIGS_SELECTIVE[@]} * ${#NUM_SELECTIVE_EXAMPLES_VALUES[@]} * ${#MODELS[@]} * ${#SELECTIVE_KEEP_K_VALUES[@]} * ${#TASK_GROUPS_LIST[@]}))"
