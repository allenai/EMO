#!/bin/bash

# Output root. Each (model, task, config) gets its own subdir under here
# containing the pruned model, finetuned checkpoints, and eval results.
# Override via env var before invoking this script.
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/prune_evals_final}"

# Number of GPUs to use per worker run. Forwarded to torchrun inside each
# worker script. Override via env var.
NUM_GPUS="${NUM_GPUS:-1}"

MODELS=(
    # HF Hub entries: format "hf:<id>|shared=<N>|skip_prune=<true|false>"
    "hf:allenai/Dense_1b_130B|shared=0|skip_prune=true"
#    "hf:allenai/StdMoE_1b4b_130B|shared=1|skip_prune=false"
#    "hf:allenai/StdMoE_1b14b_130B|shared=1|skip_prune=false"
#    "hf:allenai/StdMoE_1b14b_1T|shared=1|skip_prune=false"
#    "hf:allenai/Emo_1b14b_130B|shared=1|skip_prune=false"
#    "hf:allenai/Emo_1b14b_1T|shared=1|skip_prune=false"

    )

# Pruning mode: "layerwise"           -- greedy layer-by-layer pruning (each layer conditioned
#                                        on already-pruned earlier layers)
#               "easy_ep"             -- EASY-EP (arXiv 2504.06792): one-shot domain-specific
#                                        pruning using gating*||expert_out|| weighted by
#                                        (1 - cos_sim) of MoE in/out on few-shot calibration
PRUNING_MODE="layerwise"

num_epochs=1
#PRUNE_KEEP_K_VALUES=(8 16 32 64 128)
PRUNE_KEEP_K_VALUES=(16)
batch_size=32

# --- Pruning calibration-set size ---
# Leave empty to use the full validation pool for pruning (default).
# Set to an integer (e.g. 50) to subsample that many prompts (deterministic shuffle).
# Set to "random" to bypass calibration entirely and randomly select experts
# (seed=0, mode-agnostic — ignores PRUNING_MODE). Output dir uses _prunemode-random.
NUM_PRUNE_EXAMPLES=""

# --- Calibration-subsample seed ---
# Controls torch.Generator().manual_seed(...) in the calibration permutation.
# Default 0 reproduces historical behavior (the same single example is picked
# each run with NUM_PRUNE_EXAMPLES=1). Change to 1, 2, ... to draw a different
# calibration subset. Output dir gets a _pseed-<N> suffix when != 0 so different
# seeds don't collide on S3. Ignored when NUM_PRUNE_EXAMPLES is empty (no
# subsampling) or "random" (no calibration).
NUM_PRUNE_SEED=""

# --- Shot-count overrides (two orthogonal knobs) ---
# Each var: empty ⇒ each task's default num_shots (e.g. mmlu_merged_* = 5-shot,
# gsm8k_generation_8shot_merged = 8-shot). Set to an integer to force that
# shot count for the corresponding stage.
#
#   NUM_SHOTS_PRUNE: pruning-calibration shots. Ignored for skip-prune models
#     (no pruning stage) and NUM_PRUNE_EXAMPLES="random" (no calibration).
#   NUM_SHOTS_EVAL:  finetune + eval shots.
#
# Output dir suffix uses _pshots-{X} and/or _eshots-{Y}. Only stages that are
# actually overridden appear. Examples:
#   PRUNE="0" EVAL=""  → _pshots-0
#   PRUNE=""  EVAL="0" → _eshots-0
#   PRUNE="0" EVAL="0" → _pshots-0_eshots-0
NUM_SHOTS_PRUNE=""
NUM_SHOTS_EVAL=""

# Define grouped tasks
TASK_GROUPS_LIST=(
  # Merged variants for the MC9 + perplexity tasks (pruning + finetuning share data)
#  "arc_easy_merged"
#  "arc_challenge_merged"
#  "boolq_merged"
#  "hellaswag_merged"
#  "csqa_merged"
#  "openbookqa_merged"
#  "piqa_merged"
#  "socialiqa_merged"
#  "winogrande_merged"

  # GSM8K generation merged variants (pruning + finetuning share data)
#  "gsm8k_generation_0shot_merged"
#  "gsm8k_generation_8shot_merged"

  # SQuAD merged variants
#  "squad_merged"
#  "squad_0shot_merged"

  # CoQA merged variant (matches coqa::olmes)
#  "coqa_merged"

  # NaturalQS, TriviaQA, DROP merged variants
#  "naturalqs_merged"
#  "triviaqa_merged"
#  "drop_merged"

  # MMLU 17-category merged variants (pruning + finetuning share data)
  "mmlu_merged_biology"
#  "mmlu_merged_business"
#  "mmlu_merged_chemistry"
#  "mmlu_merged_computer_science"
#  "mmlu_merged_culture"
#  "mmlu_merged_economics"
#  "mmlu_merged_engineering"
#  "mmlu_merged_geography"
#  "mmlu_merged_health"
#  "mmlu_merged_history"
#  "mmlu_merged_law"
#  "mmlu_merged_math"
#  "mmlu_merged_other"
#  "mmlu_merged_philosophy_cat"
#  "mmlu_merged_physics"
#  "mmlu_merged_politics"
#  "mmlu_merged_psychology"

  # MMLU-Pro merged variant (pruning + finetuning use same data)
#  "mmlu_pro_merged_math"
#  "mmlu_pro_merged_health"
#  "mmlu_pro_merged_physics"
#  "mmlu_pro_merged_business"
#  "mmlu_pro_merged_biology"
#  "mmlu_pro_merged_chemistry"
#  "mmlu_pro_merged_computer_science"
#  "mmlu_pro_merged_economics"
#  "mmlu_pro_merged_engineering"
#  "mmlu_pro_merged_philosophy"
#  "mmlu_pro_merged_other"
#  "mmlu_pro_merged_history"
#  "mmlu_pro_merged_psychology"
#  "mmlu_pro_merged_law"

)

echo "Launching evals for ${#MODELS[@]} models, ${#PRUNE_KEEP_K_VALUES[@]} keep-k values, and ${#TASK_GROUPS_LIST[@]} task groups..."
echo "Models: ${MODELS[@]}"
echo "Keep-k values: ${PRUNE_KEEP_K_VALUES[@]}"
echo "GPUs: $NUM_GPUS"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Launch evaluation for each model, keep-k, and task combination
for ENTRY in "${MODELS[@]}"; do
  # Parse entry: either a bare local-path (legacy) or an HF tag of the form
  #   "hf:<hf_id>|shared=<N>|skip_prune=<true|false>"
  # HF entries require explicit 'shared' and 'skip_prune' since the substring
  # heuristics below don't recognize the new short HF names (StdMoE, ModMoE, …).
  if [[ "$ENTRY" == hf:* ]]; then
    IS_HF=true
    rest="${ENTRY#hf:}"
    MODEL="${rest%%|*}"
    num_shared_experts_override=""
    skip_prune_override=""
    if [[ "$rest" == *"|"* ]]; then
      metadata="${rest#*|}"
      IFS='|' read -ra meta_parts <<< "$metadata"
      for kv in "${meta_parts[@]}"; do
        case "$kv" in
          shared=*) num_shared_experts_override="${kv#shared=}" ;;
          skip_prune=*) skip_prune_override="${kv#skip_prune=}" ;;
          *) echo "ERROR: unknown metadata '$kv' in HF entry '$ENTRY'"; exit 1 ;;
        esac
      done
    fi
    if [ -z "$num_shared_experts_override" ]; then
      echo "ERROR: HF entry '$ENTRY' missing required 'shared=N' metadata"
      exit 1
    fi
    if [ -z "$skip_prune_override" ]; then
      echo "ERROR: HF entry '$ENTRY' missing required 'skip_prune=true|false' metadata"
      exit 1
    fi
    MODEL_ARG="$MODEL"
    TRC_FLAG="--trust-remote-code"
    num_shared_experts="$num_shared_experts_override"
    SKIP_PRUNE_DECISION="$skip_prune_override"
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
    # Auto-detect skip-prune from substring on local path
    if [[ $MODEL == *"dense_1b"* || $MODEL == *"1b4b"* ]]; then
        SKIP_PRUNE_DECISION=true
    else
        SKIP_PRUNE_DECISION=false
    fi
  fi

  for prune_keep_k in "${PRUNE_KEEP_K_VALUES[@]}"; do
    echo "Processing model: ${MODEL} (hf=${IS_HF}, shared=${num_shared_experts}, skip_prune=${SKIP_PRUNE_DECISION}), keep-k: ${prune_keep_k}"

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

        # Random pruning is mode-agnostic: override prunemode in the output name
        # and skip the _nprune-... suffix (redundant with _prunemode-random).
        if [[ $NUM_PRUNE_EXAMPLES == "random" ]]; then
          relative_dir="${stringified_model}/${TASK}_keepk_${prune_keep_k}_bs-${batch_size}_lr-${lr}_epoch-${num_epochs}_prunemode-random"
        else
          relative_dir="${stringified_model}/${TASK}_keepk_${prune_keep_k}_bs-${batch_size}_lr-${lr}_epoch-${num_epochs}_prunemode-${PRUNING_MODE}"
        fi

        # Append calibration-set-size suffix when overriding the default (use-all) behavior.
        # Skip when NUM_PRUNE_EXAMPLES=="random" since the prunemode-random token
        # already conveys this.
        if [ -n "$NUM_PRUNE_EXAMPLES" ] && [[ $NUM_PRUNE_EXAMPLES != "random" ]]; then
            relative_dir="${relative_dir}_nprune-${NUM_PRUNE_EXAMPLES}"
        fi

        # Append calibration-seed suffix when overriding the default seed=0.
        # Only meaningful when calibration is actually subsampled (skip when no
        # NUM_PRUNE_EXAMPLES or NUM_PRUNE_EXAMPLES==random).
        if [ -n "$NUM_PRUNE_SEED" ] && [ "$NUM_PRUNE_SEED" != "0" ] \
             && [ -n "$NUM_PRUNE_EXAMPLES" ] && [[ $NUM_PRUNE_EXAMPLES != "random" ]]; then
            relative_dir="${relative_dir}_pseed-${NUM_PRUNE_SEED}"
        fi

        # Append per-stage shot-count suffixes when overriding task defaults.
        # _pshots-* is skipped for runs that don't have a pruning-calibration
        # stage (skip-prune models or random pruning) — avoids misleading names.
        pruning_uses_calibration=true
        if [[ $NUM_PRUNE_EXAMPLES == "random" ]] || [ "$SKIP_PRUNE_DECISION" = true ]; then
            pruning_uses_calibration=false
        fi
        if [ "$pruning_uses_calibration" = true ] && [ -n "$NUM_SHOTS_PRUNE" ]; then
            relative_dir="${relative_dir}_pshots-${NUM_SHOTS_PRUNE}"
        fi
        if [ -n "$NUM_SHOTS_EVAL" ]; then
            relative_dir="${relative_dir}_eshots-${NUM_SHOTS_EVAL}"
        fi

        safe_relative_dir=$(printf '%s' "$relative_dir" | sed 's/[^a-zA-Z0-9_-]//g')
        job_name="eval-${safe_relative_dir}"

        # Optional calibration-size flag forwarded to the per-mode worker scripts.
        NPE_FLAG=""
        if [ -n "$NUM_PRUNE_EXAMPLES" ]; then
            NPE_FLAG="--num-prune-examples ${NUM_PRUNE_EXAMPLES}"
        fi

        # Optional calibration-seed flag forwarded to the per-mode worker scripts.
        # Skipped for random / skip-prune paths (no calibration step).
        NSEED_FLAG=""
        if [ -n "$NUM_PRUNE_SEED" ]; then
            NSEED_FLAG="--num-prune-seed ${NUM_PRUNE_SEED}"
        fi

        # Optional shot-count flags forwarded to the per-mode worker scripts.
        # --num-shots-prune goes to the pruning Python call; --num-shots-eval
        # goes to finetune + eval. Worker scripts that skip pruning (random,
        # skip-prune) ignore --num-shots-prune.
        NSHOTS_PRUNE_FLAG=""
        if [ -n "$NUM_SHOTS_PRUNE" ]; then
            NSHOTS_PRUNE_FLAG="--num-shots-prune ${NUM_SHOTS_PRUNE}"
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

        # Skip pruning if the model was tagged skip_prune=true (HF) or matched
        # the legacy dense/1b4b substring heuristic (non-HF branch).
        if [ "$SKIP_PRUNE_DECISION" = true ]; then
            echo "  Skipping pruning for model: $MODEL"
            bash scripts/pruning_hf/hf_finetune_with_pruning_layerwise.sh \
                --pruned-model ${MODEL_ARG} \
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
                --skip-prune \
                ${TRC_FLAG} \
                ${NSHOTS_EVAL_FLAG}
            echo "Ran evaluation for model: $MODEL, task: $TASK"
            echo "----------------------------------------"
            continue
        fi

        if [[ $NUM_PRUNE_EXAMPLES == "random" ]]; then
            bash scripts/pruning_hf/hf_finetune_with_pruning_random.sh \
                --model ${MODEL_ARG} \
                --task ${TASK} \
                --prune-keep-k ${prune_keep_k} \
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
        elif [[ $PRUNING_MODE == "easy_ep" ]]; then
            bash scripts/pruning_hf/hf_finetune_with_pruning_easy_ep.sh \
                --model ${MODEL_ARG} \
                --task ${TASK} \
                --prune-keep-k ${prune_keep_k} \
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
                ${NPE_FLAG} \
                ${NSEED_FLAG} \
                ${NSHOTS_PRUNE_FLAG} \
                ${NSHOTS_EVAL_FLAG}
        elif [[ $PRUNING_MODE == "layerwise" ]]; then
            bash scripts/pruning_hf/hf_finetune_with_pruning_layerwise.sh \
                --model ${MODEL_ARG} \
                --task ${TASK} \
                --prune-keep-k ${prune_keep_k} \
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
                ${NPE_FLAG} \
                ${NSEED_FLAG} \
                ${NSHOTS_PRUNE_FLAG} \
                ${NSHOTS_EVAL_FLAG}
        else
            echo "ERROR: unsupported PRUNING_MODE='${PRUNING_MODE}' (valid: layerwise, easy_ep)"
            exit 1
        fi

        echo "Ran evaluation for model: $MODEL, task: $TASK"
        echo "----------------------------------------"
    done

    echo "Completed all tasks for model: $MODEL, keep-k: $prune_keep_k"
    echo "========================================"
  done
done

echo "All evaluations have completed!"
echo "Total runs: $((${#MODELS[@]} * ${#PRUNE_KEEP_K_VALUES[@]} * ${#TASK_GROUPS_LIST[@]}))"
