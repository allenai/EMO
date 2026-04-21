#!/bin/bash
#
# Launch baseline evaluations for models hosted on HuggingFace Hub.
#
# Results are stored under the same S3 prefix as launch_original_model_eval.sh:
#   s3://ai2-sewonm/ryanwang/prune_evals_final/<sanitized_model>/original_model/<task>/results/checkpoint-0/
#
# This uses the same beaker-vs-bash pattern as the other launch scripts:
# commented-out local blocks for running locally, active beaker blocks for cluster.
#

# Configuration
BASE_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE
#BASE_DIR="/root/phdbrainstorm/FlexMoE"

# Each entry: "hf_model_id|revision|sanitized_dir_name"
# Use "none" for revision to use the default branch.
MODEL_ENTRIES=(
    "allenai/OLMoE-1B-7B-0924|none|allenaiOLMoE-1B-7B-0924"
#    "allenai/OLMoE-1B-7B-0924|step240000-tokens1006B|allenaiOLMoE-1B-7B-0924_step240000-tokens1006B"
)

CLUSTER="ai2/jupiter-cirrascale-2"

# Define grouped tasks
TASK_GROUPS_LIST=(
  # Merged variants for the MC9 + perplexity tasks (pruning + finetuning share data)
  "arc_easy_merged"
  "arc_challenge_merged"
  "boolq_merged"
  "hellaswag_merged"
  "csqa_merged"
  "openbookqa_merged"
  "piqa_merged"
  "socialiqa_merged"
  "winogrande_merged"

  # GSM8K generation merged variants (pruning + finetuning share data)
  "gsm8k_generation_0shot_merged"
  "gsm8k_generation_8shot_merged"

  # SQuAD merged variants
  "squad_merged"
  "squad_0shot_merged"

  # CoQA merged variant (matches coqa::olmes)
  "coqa_merged"

  # NaturalQS, TriviaQA, DROP merged variants
  "naturalqs_merged"
  "triviaqa_merged"
  "drop_merged"

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

gpus=4
EVAL_BATCH_SIZE=32

echo "Launching HF model baseline evals for ${#MODEL_ENTRIES[@]} models and ${#TASK_GROUPS_LIST[@]} tasks..."
echo ""

for ENTRY in "${MODEL_ENTRIES[@]}"; do
    IFS='|' read -r HF_MODEL REVISION SANITIZED_MODEL <<< "$ENTRY"

    echo "Model: $HF_MODEL (revision: $REVISION)"
    echo "  Directory name: $SANITIZED_MODEL"

    for TASK in "${TASK_GROUPS_LIST[@]}"; do
        # Batch size adjustment for memory-hungry tasks
        batch_size=$EVAL_BATCH_SIZE
        if [[ $TASK == *"history"* ]]; then
            batch_size=4
        fi

        safe_name=$(printf '%s' "${SANITIZED_MODEL}_${TASK}" | sed 's/[^a-zA-Z0-9_-]//g' | tail -c 100)
        job_name="baseline-${safe_name}"

        relative_dir="${SANITIZED_MODEL}/original_model/${TASK}"

        # Build revision args
        if [[ $REVISION == "none" ]]; then
            revision_args=""
        else
            revision_args="--revision ${REVISION}"
        fi

        echo "  Launching: task=$TASK, batch_size=$batch_size"
        echo "    Job name: $job_name"
        echo "    Remote dir: s3://ai2-sewonm/ryanwang/prune_evals_final/${relative_dir}/results/checkpoint-0"

        # Local version (uncomment for local runs):
#        python -m src.scripts.eval.launch_eval \
#            --model "${HF_MODEL}" \
#            --model-type hf \
#            ${revision_args} \
#            --task "${TASK}-pruned" \
#            --pruned_split "test" \
#            --remote-output-dir "s3://ai2-sewonm/ryanwang/prune_evals_final/${relative_dir}/results/checkpoint-0" \
#            --batch-size $batch_size \
#            --gpus $gpus

        # Beaker version:
        python -m olmo_core.launch.beaker \
            --name $job_name \
            --gpus $gpus \
            --nodes 1 \
            --is_private_repo \
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
            -- python -m src.scripts.eval.launch_eval \
                --model "${HF_MODEL}" \
                --model-type hf \
                ${revision_args} \
                --task "${TASK}-pruned" \
                --pruned_split "test" \
                --remote-output-dir "s3://ai2-sewonm/ryanwang/prune_evals_final/${relative_dir}/results/checkpoint-0" \
                --batch-size $batch_size \
                --gpus $gpus

        echo "    Launched: $job_name"
        echo "    ----------------------------------------"
#        sleep 60
    done
done

echo "All HF model baseline evaluations launched!"
echo "Total jobs: $((${#MODEL_ENTRIES[@]} * ${#TASK_GROUPS_LIST[@]}))"
