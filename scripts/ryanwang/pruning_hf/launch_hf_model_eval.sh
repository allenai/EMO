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
#    "allenai/OLMoE-1B-7B-0924|none|allenaiOLMoE-1B-7B-0924"
#    "allenai/OLMoE-1B-7B-0924|step240000-tokens1006B|allenaiOLMoE-1B-7B-0924_step240000-tokens1006B"
#    "allenai/StdMoE_1b14b_1T|none|allenai_StdMoE_1b14b_1T"
#    "allenai/StdMoE_1b14b_140B|none|allenai_StdMoE_1b14b_140B"
#    "allenai/StdMoE_1b4b_130B|none|allenai_StdMoE_1b4b_130B"
#    "allenai/ModMoE_1b14b_130B|none|allenai_ModMoE_1b14b_130B"
#    "allenai/ModMoE_1b14b_1T|none|allenai_ModMoE_1b14b_1T"
    "allenai/Dense_1b_130B|none|allenai_Dense_1b_130B"

    allenai/StdMoE_1b14b_1T
    allenai/StdMoE_1b14b_140B
    allenai/StdMoE_1b4b_130B
    allenai/ModMoE_1b14b_130B
    allenai/ModMoE_1b14b_1T
    allenai/Dense_1b_130B
)

CLUSTER="ai2/jupiter-cirrascale-2"

TASK_GROUPS_LIST=(
  "mmlu_merged_biology"

#  "arc_easy"
#  "arc_challenge"
#  "boolq"
#  "csqa"
#  "hellaswag"
#  "openbookqa"
#  "piqa"
#  "socialiqa"
#  "winogrande"
#  "gsm8k_generation_0shot"
#  "gsm8k_perplexity_0shot"
#  "coqa_0shot"
#  "coqa_full_0shot"
#  "squad_0shot"

#  "mmlu_biology"
#  "mmlu_business"
#  "mmlu_chemistry"
#  "mmlu_computer_science"
#  "mmlu_culture"
#  "mmlu_economics"
#  "mmlu_engineering"
#  "mmlu_geography"
#  "mmlu_health"
#  "mmlu_history"
#  "mmlu_law"
#  "mmlu_math"
#  "mmlu_other"
#  "mmlu_philosophy_cat"
#  "mmlu_physics"
#  "mmlu_politics"
#  "mmlu_psychology"
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
#        export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
#        python -m src.scripts.eval.launch_eval \
#            --model "${HF_MODEL}" \
#            --model-type hf \
#            ${revision_args} \
#            --task "${TASK}-pruned" \
#            --pruned_split "test" \
#            --remote-output-dir "s3://ai2-sewonm/ryanwang/prune_evals_final/${relative_dir}/results/checkpoint-0" \
#            --batch-size $batch_size \
#            --model-args trust_remote_code=true \
#            --gpus $gpus

        # Beaker version:
        python -m olmo_core.launch.beaker \
            --name $job_name \
            --gpus $gpus \
            --nodes 1 \
            --beaker-image tylerr/olmo-core-tch280cu128-2025-11-25 \
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
            -- bash -c "export PYTHONPATH=\"\$(pwd)/src\${PYTHONPATH:+:\${PYTHONPATH}}\" && python -m src.scripts.eval.launch_eval \
                --model \"${HF_MODEL}\" \
                --model-type hf \
                ${revision_args} \
                --task \"${TASK}-pruned\" \
                --pruned_split \"test\" \
                --remote-output-dir \"s3://ai2-sewonm/ryanwang/prune_evals_final/${relative_dir}/results/checkpoint-0\" \
                --batch-size $batch_size \
                --model-args trust_remote_code=true \
                --gpus $gpus
            "

        echo "    Launched: $job_name"
        echo "    ----------------------------------------"
#        sleep 60
    done
done

echo "All HF model baseline evaluations launched!"
echo "Total jobs: $((${#MODEL_ENTRIES[@]} * ${#TASK_GROUPS_LIST[@]}))"
