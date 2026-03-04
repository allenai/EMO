#!/bin/bash
#
# Launch baseline evaluations (no pruning, no finetuning) for original models.
#
# Results are stored under:
#   s3://ai2-sewonm/ryanwang/prune_evals/<model>/original_model/<task>/results/checkpoint-0/
#
# This uses the same beaker-vs-bash pattern as launch_pruning_hf.sh:
# commented-out local blocks for running locally, active beaker blocks for cluster.
#

# Configuration
BASE_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE
#BASE_DIR="/root/phdbrainstorm/FlexMoE"

MODELS=(
    "twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp1densefirst-32_1b14b_lr-4e-3_lb-1e-1_0227/step30995-hf"
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf"
    "moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf"
    "moereducedp256_1b4b_lr-4e-3_lb-1e-1_0212/step30995-hf"
    "dense_1b_lr-4e-3_0213/step30995-hf"
)

CLUSTER="ai2/jupiter-cirrascale-2"

TASK_GROUPS_LIST=(
  "arc_easy"
  "arc_challenge"
  "boolq"
  "csqa"
  "hellaswag"
  "openbookqa"
  "piqa"
  "socialiqa"
  "winogrande"
  "gsm8k_generation_0shot"
  "gsm8k_perplexity_0shot"
  "coqa_0shot"
  "coqa_full_0shot"
  "squad_0shot"

  "mmlu_biology"
  "mmlu_business"
  "mmlu_chemistry"
  "mmlu_computer_science"
  "mmlu_culture"
  "mmlu_economics"
  "mmlu_engineering"
  "mmlu_geography"
  "mmlu_health"
  "mmlu_history"
  "mmlu_law"
  "mmlu_math"
  "mmlu_other"
  "mmlu_philosophy_cat"
  "mmlu_physics"
  "mmlu_politics"
  "mmlu_psychology"
)

gpus=4
EVAL_BATCH_SIZE=32

echo "Launching baseline evals for ${#MODELS[@]} models and ${#TASK_GROUPS_LIST[@]} tasks..."
echo "Models: ${MODELS[@]}"
echo ""

for MODEL in "${MODELS[@]}"; do
    stringified_model=$(echo $MODEL | sed 's/[^a-zA-Z0-9_-]//g')

    for TASK in "${TASK_GROUPS_LIST[@]}"; do
        # Batch size adjustment for memory-hungry tasks
        batch_size=$EVAL_BATCH_SIZE
        if [[ $TASK == *"history"* ]]; then
            batch_size=4
        fi

        safe_name=$(printf '%s' "${stringified_model}_${TASK}" | sed 's/[^a-zA-Z0-9_-]//g' | tail -c 100)
        job_name="baseline-${safe_name}"

        relative_dir="${stringified_model}/original_model/${TASK}"

        echo "Launching baseline eval: model=$MODEL, task=$TASK"
        echo "  Job name: $job_name"
        echo "  Remote dir: s3://ai2-sewonm/ryanwang/prune_evals/${relative_dir}/results/checkpoint-0"

        # Local version (uncomment for local runs):
#        python -m src.scripts.eval.launch_eval \
#            --model "${BASE_DIR}/models/${MODEL}" \
#            --model-type hf \
#            --task "${TASK}-pruned" \
#            --pruned_split "test" \
#            --remote-output-dir "s3://ai2-sewonm/ryanwang/prune_evals/${relative_dir}/results/checkpoint-0" \
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
                --model "${BASE_DIR}/models/${MODEL}" \
                --model-type hf \
                --task "${TASK}-pruned" \
                --pruned_split "test" \
                --remote-output-dir "s3://ai2-sewonm/ryanwang/prune_evals/${relative_dir}/results/checkpoint-0" \
                --batch-size $batch_size \
                --gpus $gpus

        echo "Launched: $job_name"
        echo "----------------------------------------"
        sleep 60
    done
done

echo "All baseline evaluations launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASK_GROUPS_LIST[@]}))"
