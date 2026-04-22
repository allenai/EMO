#!/bin/bash

# Configuration
BASE_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE
#BASE_DIR="/root/phdbrainstorm/FlexMoE"
MODELS=(
#    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419/step250339-hf"
#    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf"
#    "dense_1b_lr-4e-3_0213/step30995-hf"
#    "moereducedp512sharedexp1_1b4b_lr-4e-3_lb-1e-1_0308/step30995-hf"
#    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995-hf"
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419/step250339-hf"
#    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_twolevel_randpool-8-128_from_step238419/step250339-hf"

#    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313/step238419-hf"
#    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322/step238419-hf"


#    "moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf"

#    "moereducedp256_1b4b_lr-4e-3_lb-1e-1_0212/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp2randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0305/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp1densefirst-32_1b14b_lr-4e-3_lb-1e-1_0227/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-2_0213/step30995-hf"

#    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995-hf"
#    "moe_1b14b_128experts_lb-1e-1_1217/step30995-hf"

#    "dense_1b_olmoe-mix_prenorm_noqknorm_1123/step30995-hf"
#    "moe_1b4b_32experts_1224/step30995-hf"
#    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995-hf"
#    "twolevelbatchlb-32_1b14b_lr-4e-3_lb-1e-1_0119/step30995-hf"
#    "twolevelbatchlb-32_1b14b_lr-4e-3_lb-1e-2_0118/step30995-hf"
#    "twolevelbatchlbreducedp512-32_1b14b_lr-4e-3_lb-1e-2_0207/step30995-hf"
#    "twolevelbatchlbreducedp512-32_1b14b_lr-4e-3_lb-1e-1_0119/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp4c2-32_1b14b_lr-4e-3_lb-1e-1_sharelb-1e-1_0214/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp4c2-32_1b14b_lr-4e-3_lb-1e-2_sharelb-1e-2_0214/step30995-hf"

    )

CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf

# Pruning mode: "global"              -- original single-pass activation collection + prune
#               "layerwise"           -- greedy layer-by-layer pruning (each layer conditioned
#                                        on already-pruned earlier layers)
#               "layerwise_variable"  -- greedy layerwise with per-layer keep-k schedule
#               "easy_ep"             -- EASY-EP (arXiv 2504.06792): one-shot domain-specific
#                                        pruning using gating*||expert_out|| weighted by
#                                        (1 - cos_sim) of MoE in/out on few-shot calibration
PRUNING_MODE="easy_ep"

num_epochs=1
#PRUNE_KEEP_K_VALUES=(8 16 32 64 128)
PRUNE_KEEP_K_VALUES=(8 16 32 64)
batch_size=32

# --- Pruning calibration-set size ---
# Leave empty to use the full validation pool for pruning (default).
# Set to an integer (e.g. 50) to subsample that many prompts (deterministic shuffle, seed=0).
# Set to "random" to bypass calibration entirely and randomly select experts
# (seed=0, mode-agnostic — ignores PRUNING_MODE). Output dir uses _prunemode-random.
NUM_PRUNE_EXAMPLES="5"

# --- Layerwise-variable settings (only used when PRUNING_MODE="layerwise_variable") ---
# Schedule name (used in output directory naming)
PRUNE_SCHEDULE_NAME="first2_unpruned"
# Per-layer keep-k: layers 0-1 keep all 128 experts, layers 2-15 pruned to prune_keep_k
KEEP_K_PER_LAYER="128,128,32,32,32,32,32,32,32,32,32,32,32,32,32,32"

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
  "gsm8k_generation_8shot_merged"

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

  # MMLU-Pro merged variant (pruning + finetuning use same data)
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

#  "synthea_zeroshot"

)

echo "Launching evals for ${#MODELS[@]} models, ${#PRUNE_KEEP_K_VALUES[@]} keep-k values, and ${#TASK_GROUPS_LIST[@]} task groups..."
echo "Models: ${MODELS[@]}"
echo "Keep-k values: ${PRUNE_KEEP_K_VALUES[@]}"
echo "Cluster: $CLUSTER"
echo ""

# Launch evaluation for each model, keep-k, and task combination
for MODEL in "${MODELS[@]}"; do
  for prune_keep_k in "${PRUNE_KEEP_K_VALUES[@]}"; do
    echo "Processing model: ${MODEL}, keep-k: ${prune_keep_k}"

    # choose the number of pruned down shared experts
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

    for TASK in "${TASK_GROUPS_LIST[@]}"; do
        # TODO: choose the right batch size based on the task
#        # Batch size adjustment (matching original script)
#        if [[ $TASK == *"mmlu_high_school_european_history"* || $TASK == *"mmlu_high_school_us_history"* || $TASK == *"mmlu_history"* || $TASK == *"mmlu_philosophy"* || $TASK == *"cot"* || $TASK == *"minerva_math_"* || $TASK == *"mbpp"* || $TASK == *"bigcodebench"* || $TASK == *"ruler"* || $TASK == *"sciriff"* || $TASK == *"boolq"* || $TASK == *"synthea"* || $MODEL == *"1b35b"* ]]; then
#            micro_batch_size=$((micro_batch_size / 4))
#        else
#            micro_batch_size=$micro_batch_size
#        fi
        micro_batch_size=8
        if [[ $TASK == *"mmlu_history"* ]]; then
            micro_batch_size=2
        fi
        if [[ $TASK == *"gsm8k_generation_8shot"* || $TASK == *"drop_merged"* || $TASK == *"squad_merged"* ]]; then
            micro_batch_size=2
        fi

        # TODO choose the right number of gpus based on task (so that it doesn't oom)
#        # adjust number of gpus requested if its agi_eval, bbh, gsm8k, minerva, codex, mbpp
#        if [[ $TASK == *agi_eval* || $TASK == *bbh* || $TASK == *gsm8k* || $TASK == *minerva_math_* || $TASK == *codex* || $TASK == *mbpp* || $MODEL == *"1b35b"* ]]; then
#            gpus=4
#        else
#            gpus=1
#        fi
        gpus=4
        if [[ $TASK == *"mmlu_history"* || $TASK == *"gsm8k_generation_8shot"* || $TASK == *"drop_merged"* || $TASK == *"squad_merged"* ]]; then
            gpus=8
        fi

        # TODO: choose the right learning rate based on task
        lr=5e-5

        # Create a shorter, valid job name
        # Remove invalid characters and truncate long names

        stringified_model=$(echo $MODEL | sed 's/[^a-zA-Z0-9_-]//g')

        # Random pruning is mode-agnostic: override prunemode in the output name
        # and skip the _nprune-... suffix (redundant with _prunemode-random).
        if [[ $NUM_PRUNE_EXAMPLES == "random" ]]; then
          relative_dir="${stringified_model}/${TASK}_keepk_${prune_keep_k}_bs-${batch_size}_lr-${lr}_epoch-${num_epochs}_prunemode-random"
        elif [[ $PRUNING_MODE == "global" ]]; then
          relative_dir="${stringified_model}/${TASK}_keepk_${prune_keep_k}_bs-${batch_size}_lr-${lr}_epoch-${num_epochs}"
        elif [[ $PRUNING_MODE == "layerwise_variable" ]]; then
          relative_dir="${stringified_model}/${TASK}_keepk_${prune_keep_k}_bs-${batch_size}_lr-${lr}_epoch-${num_epochs}_prunemode-${PRUNING_MODE}_${PRUNE_SCHEDULE_NAME}"
        else
          relative_dir="${stringified_model}/${TASK}_keepk_${prune_keep_k}_bs-${batch_size}_lr-${lr}_epoch-${num_epochs}_prunemode-${PRUNING_MODE}"
        fi

        # Append calibration-set-size suffix when overriding the default (use-all) behavior.
        # Skip when NUM_PRUNE_EXAMPLES=="random" since the prunemode-random token
        # already conveys this.
        if [ -n "$NUM_PRUNE_EXAMPLES" ] && [[ $NUM_PRUNE_EXAMPLES != "random" ]]; then
            relative_dir="${relative_dir}_nprune-${NUM_PRUNE_EXAMPLES}"
        fi

        safe_relative_dir=$(printf '%s' "$relative_dir" | sed 's/[^a-zA-Z0-9_-]//g' | tail -c 100)
        job_name="eval-${safe_relative_dir}"

        # Optional calibration-size flag forwarded to the per-mode worker scripts.
        NPE_FLAG=""
        if [ -n "$NUM_PRUNE_EXAMPLES" ]; then
            NPE_FLAG="--num-prune-examples ${NUM_PRUNE_EXAMPLES}"
        fi

        # Clean any previous results for this exact (model, keep-k, task, prune-mode)
        # combination on S3 so re-runs never mix new metrics with stale ones.
        # aws s3 rm on a non-existent prefix is a no-op (exit 0).
        s3_clean_prefix="s3://ai2-sewonm/ryanwang/prune_evals_final/${relative_dir}/"
        echo "  Cleaning stale S3 results: ${s3_clean_prefix}"
        aws s3 rm --recursive --quiet "${s3_clean_prefix}" || true

        echo "  Model name: ${BASE_DIR}/${MODEL}"
        echo "  GPUs: $gpus"
        echo "  Batch size: $batch_size"
        echo "  Job name: $job_name"

        # debug what will be passed
        echo "  model: ${BASE_DIR}/${MODEL}"
        echo "  task: ${TASK}"
        echo "  relative-dir: ${relative_dir}"
        echo "  base-dir: ${BASE_DIR}/prune_evals"
        echo "  num-gpus: $gpus"
        echo "  run_name: ${job_name}"
        echo "  learning-rate: ${lr}"
        echo "  batch_size: ${batch_size}"
        echo "  epochs: ${num_epochs}"
        echo "  num_shared_experts: ${num_shared_experts}"

        # if the model is dense or 1b4b, we skip activation and pruning
        if [[ $MODEL == *"dense_1b"* || $MODEL == *"1b4b"* ]]; then
            echo "  Skipping activation computation and pruning for model: $MODEL"
#            bash scripts/ryanwang/pruning_hf/hf_finetune_with_pruning.sh \
#                --pruned-model ${BASE_DIR}/models/${MODEL} \
#                --task ${TASK} \
#                --base-dir "${BASE_DIR}/prune_evals" \
#                --relative-dir ${relative_dir} \
#                --num-gpus $gpus \
#                --run-name ${job_name} \
#                --learning-rate ${lr} \
#                --batch-size ${batch_size} \
#                --micro-batch-size ${micro_batch_size} \
#                --num-epochs ${num_epochs} \
#               --num-shared-experts ${num_shared_experts} \
#                --skip-activation \
#                --skip-prune
            python -m olmo_core.launch.beaker \
                --name $job_name \
                --gpus $gpus \
                --nodes 1 \
                --weka=oe-training-default \
                --shared-filesystem \
                --workspace ai2/flex2 \
                --cluster ai2/jupiter \
                --preemptible \
                --allow-dirty \
                --priority urgent \
                --no-follow \
                --no-torchrun \
                --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
                -- bash -c "scripts/ryanwang/pruning_hf/hf_finetune_with_pruning.sh \
                --pruned-model ${BASE_DIR}/models/${MODEL} \
                --task ${TASK} \
                --base-dir "${BASE_DIR}/prune_evals" \
                --relative-dir ${relative_dir} \
                --num-gpus $gpus \
                --run-name ${job_name} \
                --learning-rate ${lr} \
                --batch-size ${batch_size} \
                --micro-batch-size ${micro_batch_size} \
                --num-epochs ${num_epochs} \
                --num-checkpoints 1 \
                --num-shared-experts ${num_shared_experts} \
                --skip-activation \
                --skip-prune
                "
            echo "Launched evaluation for model: $model, task: $TASK"
            echo "----------------------------------------"
#            sleep 500 # brief pause to avoid overwhelming huggingface
            continue
        fi

        if [[ $NUM_PRUNE_EXAMPLES == "random" ]]; then
#            bash scripts/ryanwang/pruning_hf/hf_finetune_with_pruning_random.sh \
#                --model ${BASE_DIR}/models/${MODEL} \
#                --task ${TASK} \
#                --prune-keep-k ${prune_keep_k} \
#                --base-dir "${BASE_DIR}/prune_evals" \
#                --relative-dir ${relative_dir} \
#                --num-gpus $gpus \
#                --run-name ${job_name} \
#                --learning-rate ${lr} \
#                --batch-size ${batch_size} \
#                --micro-batch-size ${micro_batch_size} \
#                --num-epochs ${num_epochs} \
#                --num-checkpoints 1 \
#                --num-shared-experts ${num_shared_experts}

            python -m olmo_core.launch.beaker \
                --name $job_name \
                --gpus $gpus \
                --nodes 1 \
                --weka=oe-training-default \
                --shared-filesystem \
                --workspace ai2/flex2 \
                --cluster ai2/jupiter \
                --preemptible \
                --allow-dirty \
                --priority urgent \
                --no-follow \
                --no-torchrun \
                --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
                -- bash -c "scripts/ryanwang/pruning_hf/hf_finetune_with_pruning_random.sh \
                    --model ${BASE_DIR}/models/${MODEL} \
                    --task ${TASK} \
                    --prune-keep-k ${prune_keep_k} \
                    --base-dir "${BASE_DIR}/prune_evals" \
                    --relative-dir ${relative_dir} \
                    --num-gpus $gpus \
                    --run-name ${job_name} \
                    --learning-rate ${lr} \
                    --batch-size ${batch_size} \
                    --micro-batch-size ${micro_batch_size} \
                    --num-epochs ${num_epochs} \
                    --num-checkpoints 1 \
                    --num-shared-experts ${num_shared_experts}
                "
        elif [[ $PRUNING_MODE == "layerwise_variable" ]]; then
#            bash scripts/ryanwang/pruning_hf/hf_finetune_with_pruning_layerwise_variable.sh \
#                --model ${BASE_DIR}/models/${MODEL} \
#                --task ${TASK} \
#                --keep-k-per-layer "${KEEP_K_PER_LAYER}" \
#                --base-dir "${BASE_DIR}/prune_evals" \
#                --relative-dir ${relative_dir} \
#                --num-gpus $gpus \
#                --run-name ${job_name} \
#                --learning-rate ${lr} \
#                --batch-size ${batch_size} \
#                --micro-batch-size ${micro_batch_size} \
#                --num-epochs ${num_epochs} \
#                --num-shared-experts ${num_shared_experts} \
#                --prune-mode ${PRUNE_SCHEDULE_NAME}

            python -m olmo_core.launch.beaker \
                --name $job_name \
                --gpus $gpus \
                --nodes 1 \
                --weka=oe-training-default \
                --shared-filesystem \
                --workspace ai2/flex2 \
                --cluster ai2/jupiter \
                --preemptible \
                --allow-dirty \
                --priority urgent \
                --no-follow \
                --no-torchrun \
                --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
                -- bash -c "scripts/ryanwang/pruning_hf/hf_finetune_with_pruning_layerwise_variable.sh \
                    --model ${BASE_DIR}/models/${MODEL} \
                    --task ${TASK} \
                    --keep-k-per-layer '${KEEP_K_PER_LAYER}' \
                    --base-dir "${BASE_DIR}/prune_evals" \
                    --relative-dir ${relative_dir} \
                    --num-gpus $gpus \
                    --run-name ${job_name} \
                    --learning-rate ${lr} \
                    --batch-size ${batch_size} \
                    --micro-batch-size ${micro_batch_size} \
                    --num-epochs ${num_epochs} \
                    --num-checkpoints 1 \
                    --num-shared-experts ${num_shared_experts} \
                    --prune-mode ${PRUNE_SCHEDULE_NAME} \
                    ${NPE_FLAG}
                "
        elif [[ $PRUNING_MODE == "easy_ep" ]]; then
#            bash scripts/ryanwang/pruning_hf/hf_finetune_with_pruning_easy_ep.sh \
#                --model ${BASE_DIR}/models/${MODEL} \
#                --task ${TASK} \
#                --prune-keep-k ${prune_keep_k} \
#                --base-dir "${BASE_DIR}/prune_evals" \
#                --relative-dir ${relative_dir} \
#                --num-gpus $gpus \
#                --run-name ${job_name} \
#                --learning-rate ${lr} \
#                --batch-size ${batch_size} \
#                --micro-batch-size ${micro_batch_size} \
#                --num-epochs ${num_epochs} \
#                --num-shared-experts ${num_shared_experts}

            python -m olmo_core.launch.beaker \
                --name $job_name \
                --gpus $gpus \
                --nodes 1 \
                --weka=oe-training-default \
                --shared-filesystem \
                --workspace ai2/flex2 \
                --cluster ai2/jupiter \
                --preemptible \
                --allow-dirty \
                --priority urgent \
                --no-follow \
                --no-torchrun \
                --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
                -- bash -c "scripts/ryanwang/pruning_hf/hf_finetune_with_pruning_easy_ep.sh \
                    --model ${BASE_DIR}/models/${MODEL} \
                    --task ${TASK} \
                    --prune-keep-k ${prune_keep_k} \
                    --base-dir "${BASE_DIR}/prune_evals" \
                    --relative-dir ${relative_dir} \
                    --num-gpus $gpus \
                    --run-name ${job_name} \
                    --learning-rate ${lr} \
                    --batch-size ${batch_size} \
                    --micro-batch-size ${micro_batch_size} \
                    --num-epochs ${num_epochs} \
                    --num-checkpoints 1 \
                    --num-shared-experts ${num_shared_experts} \
                    ${NPE_FLAG}
                "
        elif [[ $PRUNING_MODE == "layerwise" ]]; then
#            bash scripts/ryanwang/pruning_hf/hf_finetune_with_pruning_layerwise.sh \
#                --model ${BASE_DIR}/models/${MODEL} \
#                --task ${TASK} \
#                --prune-keep-k ${prune_keep_k} \
#                --base-dir "${BASE_DIR}/prune_evals" \
#                --relative-dir ${relative_dir} \
#                --num-gpus $gpus \
#                --run-name ${job_name} \
#                --learning-rate ${lr} \
#                --batch-size ${batch_size} \
#                --micro-batch-size ${micro_batch_size} \
#                --num-epochs ${num_epochs} \
#                --num-shared-experts ${num_shared_experts}

            python -m olmo_core.launch.beaker \
                --name $job_name \
                --gpus $gpus \
                --nodes 1 \
                --weka=oe-training-default \
                --shared-filesystem \
                --workspace ai2/flex2 \
                --cluster ai2/jupiter \
                --preemptible \
                --allow-dirty \
                --priority urgent \
                --no-follow \
                --no-torchrun \
                --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
                -- bash -c "scripts/ryanwang/pruning_hf/hf_finetune_with_pruning_layerwise.sh \
                    --model ${BASE_DIR}/models/${MODEL} \
                    --task ${TASK} \
                    --prune-keep-k ${prune_keep_k} \
                    --base-dir "${BASE_DIR}/prune_evals" \
                    --relative-dir ${relative_dir} \
                    --num-gpus $gpus \
                    --run-name ${job_name} \
                    --learning-rate ${lr} \
                    --batch-size ${batch_size} \
                    --micro-batch-size ${micro_batch_size} \
                    --num-epochs ${num_epochs} \
                    --num-checkpoints 1 \
                    --num-shared-experts ${num_shared_experts} \
                    ${NPE_FLAG}
                "
        else
#            bash scripts/ryanwang/pruning_hf/hf_finetune_with_pruning.sh \
#                --model ${BASE_DIR}/models/${MODEL} \
#                --task ${TASK} \
#                --prune-keep-k ${prune_keep_k} \
#                --base-dir "${BASE_DIR}/prune_evals" \
#                --relative-dir ${relative_dir} \
#                --num-gpus $gpus \
#                --run-name ${job_name} \
#                --learning-rate ${lr} \
#                --batch-size ${batch_size} \
#                --micro-batch-size ${micro_batch_size} \
#                --num-epochs ${num_epochs} \
#                --num-shared-experts ${num_shared_experts}

            python -m olmo_core.launch.beaker \
                --name $job_name \
                --gpus $gpus \
                --nodes 1 \
                --weka=oe-training-default \
                --shared-filesystem \
                --workspace ai2/flex2 \
                --cluster ai2/jupiter \
                --preemptible \
                --allow-dirty \
                --priority urgent \
                --no-follow \
                --no-torchrun \
                --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
                -- bash -c "scripts/ryanwang/pruning_hf/hf_finetune_with_pruning.sh \
                    --model ${BASE_DIR}/models/${MODEL} \
                    --task ${TASK} \
                    --prune-keep-k ${prune_keep_k} \
                    --base-dir "${BASE_DIR}/prune_evals" \
                    --relative-dir ${relative_dir} \
                    --num-gpus $gpus \
                    --run-name ${job_name} \
                    --learning-rate ${lr} \
                    --batch-size ${batch_size} \
                    --micro-batch-size ${micro_batch_size} \
                    --num-epochs ${num_epochs} \
                    --num-checkpoints 1 \
                    --num-shared-experts ${num_shared_experts} \
                    ${NPE_FLAG}
                "
        fi

        echo "Launched evaluation for model: $MODEL, task: $TASK"
        echo "----------------------------------------"

#        sleep 500 # brief pause to avoid overwhelming huggingface
    done

    echo "Completed all tasks for model: $MODEL, keep-k: $prune_keep_k"
    echo "========================================"
  done
done

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#PRUNE_KEEP_K_VALUES[@]} * ${#TASK_GROUPS_LIST[@]}))"
echo "Check the beaker dashboard for job status."
