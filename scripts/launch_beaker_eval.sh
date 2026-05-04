#!/bin/bash

# Script to launch beaker evaluations for all available tasks
# Usage: bash src/scripts/eval/launch_beaker_eval.sh

# Configuration
MODEL_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/Emo/models
#MODEL_DIR="/root/ryanwang/phdbrainstorm/Emo/models"
MODELS=(
    "dense_1b_olmoe-mix_prenorm_noqknorm_1123/step30995-hf"
    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995-hf"
    "moe_1b4b_32experts_1224/step30995-hf"
    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995-hf"

#    "twolevelbatchlb-32_1b14b_stability_lr-6e-4_1203/step30995-hf"
#    "twolevelsamplingnolb-32_1b10b_stability_1127/step30995-hf"
#    "twolevelsamplingnolb-32_1b14b_stability_1127/step30995-hf"

#    "moe_1b35b_320experts_lb-1e-1_1214/step30995-hf"
#    "twolevelbatchlb-128_1b35b_320experts_lb-1e-1_poolsched-lineardecay2000_1217/step30995-hf"

#    "twolevelsamplingnolb-32_1b14b_stability_filter-true_zlossweight-1e-3_1116/step30995-hf"
#    "twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115/step30995-hf"
#    "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995-hf"
#    "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995-hf"
#    "dense_1b_olmoe-mix_300B_1030/step71526-hf"
#    "dense_1b_olmoe-mix_1028/step30995-hf"
#    "moe_1b7b_olmoe-mix_300B_1030/step71526-hf"
#    "moe_1b7b_olmoe-mix/step30995-hf"
)


BASE_OUTPUT_DIR="s3://ai2-sewonm/ryanwang/evals"
#BASE_OUTPUT_DIR="/root/ryanwang/phdbrainstorm/Emo/evals"
BATCH_SIZE=16
CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf

# Define grouped tasks
TASK_GROUPS_LIST=(
  ######### TEST-only ##########
  # MC9 tasks
#  "arc_easy|arc_easy:mc_test::olmes arc_easy:rc_test::olmes"
#  "arc_challenge|arc_challenge:mc_test::olmes arc_challenge:rc_test::olmes"
#  "boolq|boolq:mc_test::olmes boolq:rc_test::olmes"
#  "csqa|csqa:mc_test::olmes csqa:rc_test::olmes"
#  "hellaswag|hellaswag:mc_test::olmes hellaswag:rc_test::olmes"
#  "openbookqa|openbookqa:mc_test::olmes openbookqa:rc_test::olmes"
#  "piqa|piqa:mc_test::olmes piqa:rc_test::olmes"
#  "socialiqa|socialiqa:mc_test::olmes socialiqa:rc_test::olmes"
#  "winogrande|winogrande:mc_test::olmes winogrande:rc_test::olmes"

#   MMLU
#  "mmlu_mc_test|mmlu:mc_test::olmes"
#  "mmlu_rc_test|mmlu:rc_test::olmes"

#   Gen5 tasks
#  "gen5|coqa::olmes squad::olmes naturalqs::olmes triviaqa::olmes"
#  "drop|drop::olmes"

#  "synthea|synthea:rc_test_0shot::olmes"
#  "gsm8k|gsm8k_generation:test_0shot::olmes"
#  "coqa|coqa:test_0shot::olmes"
#  "squad|squad:test_0shot::olmes"

#   GSM8K
  "mmlu_biology|mmlu_biology:rc_test::olmes"
  "mmlu_business|mmlu_business:rc_test::olmes"
  "mmlu_chemistry|mmlu_chemistry:rc_test::olmes"
  "mmlu_computer_science|mmlu_computer_science:rc_test::olmes"
  "mmlu_culture|mmlu_culture:rc_test::olmes"
  "mmlu_economics|mmlu_economics:rc_test::olmes"
  "mmlu_engineering|mmlu_engineering:rc_test::olmes"
  "mmlu_geography|mmlu_geography:rc_test::olmes"
  "mmlu_health|mmlu_health:rc_test::olmes"
  "mmlu_history|mmlu_history:rc_test::olmes"
  "mmlu_law|mmlu_law:rc_test::olmes"
  "mmlu_math|mmlu_math:rc_test::olmes"
  "mmlu_other|mmlu_other:rc_test::olmes"
  "mmlu_philosophy_cat|mmlu_philosophy_cat:rc_test::olmes"
  "mmlu_physics|mmlu_physics:rc_test::olmes"
  "mmlu_politics|mmlu_politics:rc_test::olmes"
  "mmlu_psychology|mmlu_psychology:rc_test::olmes"

#  "mmlu_abstract_algebra|mmlu_abstract_algebra:rc_test::olmes"
#  "mmlu_anatomy|mmlu_anatomy:rc_test::olmes"
#  "mmlu_astronomy|mmlu_astronomy:rc_test::olmes"
#  "mmlu_business_ethics|mmlu_business_ethics:rc_test::olmes"
#  "mmlu_clinical_knowledge|mmlu_clinical_knowledge:rc_test::olmes"
#  "mmlu_college_biology|mmlu_college_biology:rc_test::olmes"
#  "mmlu_college_chemistry|mmlu_college_chemistry:rc_test::olmes"
#  "mmlu_college_computer_science|mmlu_college_computer_science:rc_test::olmes"
#  "mmlu_college_mathematics|mmlu_college_mathematics:rc_test::olmes"
#  "mmlu_college_medicine|mmlu_college_medicine:rc_test::olmes"
#  "mmlu_college_physics|mmlu_college_physics:rc_test::olmes"
#
#  "gsm8k_test|gsm8k:perplexity_test::olmes"

  ######### TRAIN-VAL-TEST ##########
#  # MC9 tasks
#  "arc_easy|arc_easy:mc_train::olmes arc_easy:mc_validation::olmes arc_easy:mc_test::olmes arc_easy:rc_train::olmes arc_easy:rc_validation::olmes arc_easy:rc_test::olmes"
#  "arc_challenge|arc_challenge:mc_train::olmes arc_challenge:mc_validation::olmes arc_challenge:mc_test::olmes arc_challenge:rc_train::olmes arc_challenge:rc_validation::olmes arc_challenge:rc_test::olmes"
#  "boolq|boolq:mc_train::olmes boolq:mc_validation::olmes boolq:mc_test::olmes boolq:rc_train::olmes boolq:rc_validation::olmes boolq:rc_test::olmes"
#  "csqa|csqa:mc_train::olmes csqa:mc_validation::olmes csqa:mc_test::olmes csqa:rc_train::olmes csqa:rc_validation::olmes csqa:rc_test::olmes"
#  "hellaswag|hellaswag:mc_train::olmes hellaswag:mc_validation::olmes hellaswag:mc_test::olmes hellaswag:rc_train::olmes hellaswag:rc_validation::olmes hellaswag:rc_test::olmes"
#  "openbookqa|openbookqa:mc_train::olmes openbookqa:mc_validation::olmes openbookqa:mc_test::olmes openbookqa:rc_train::olmes openbookqa:rc_validation::olmes openbookqa:rc_test::olmes"
#  "piqa|piqa:mc_train::olmes piqa:mc_validation::olmes piqa:mc_test::olmes piqa:rc_train::olmes piqa:rc_validation::olmes piqa:rc_test::olmes"
#  "socialiqa|socialiqa:mc_train::olmes socialiqa:mc_validation::olmes socialiqa:mc_test::olmes socialiqa:rc_train::olmes socialiqa:rc_validation::olmes socialiqa:rc_test::olmes"
#  "winogrande|winogrande:mc_train::olmes winogrande:mc_validation::olmes winogrande:mc_test::olmes winogrande:rc_train::olmes winogrande:rc_validation::olmes winogrande:rc_test::olmes"
#
#  # MMLU
#  "mmlu_mc_train|mmlu:mc_train::olmes"
#  "mmlu_mc_valid_test|mmlu:mc_validation::olmes mmlu:mc_test::olmes"
#  "mmlu_rc_train|mmlu:rc_train::olmes"
#  "mmlu_rc_valid_test|mmlu:rc_validation::olmes mmlu:rc_test::olmes"
#
#  # Gen5 tasks
#  "gen5|coqa::olmes squad::olmes naturalqs::olmes triviaqa::olmes drop::olmes"

)

# Function to get checkpoint name (matching the original script)
function get_checkpoint_name {
    local path=$1
    local split_path=${path#*OLMo2-7B-}
    local modified_path=${split_path//\//_}
    modified_path=$(echo $modified_path | sed 's/^_//;s/_$//')
    echo "${modified_path}"
}

echo "Launching beaker evaluations for ${#MODELS[@]} models and ${#TASK_GROUPS[@]} task groups..."
echo "Models: ${MODELS[@]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Cluster: $CLUSTER"
echo ""

# Launch evaluation for each model and task combination
for MODEL_NAME in "${MODELS[@]}"; do
    echo "Processing model: $MODEL_NAME"

    # For setting the output_dir (matching original script logic)
#    if [[ $MODEL_NAME == "/"* ]]; then
#        # internal model
#        model=$(get_checkpoint_name $MODEL_NAME)
#    else
#        # HF model
#        model=$(echo $MODEL_NAME | cut -d'/' -f2)
#    fi
    model=$(get_checkpoint_name $MODEL_NAME)

    echo "Model name for output dir: $model"

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"

    for entry in "${TASK_GROUPS_LIST[@]}"; do
        GROUP_NAME="${entry%%|*}"                # text before '|'
        TASK="${entry#*|}"            # text after '|'

        # Batch size adjustment (matching original script)
        if [[ $TASK == *"mmlu_high_school_european_history"* || $TASK == *"mmlu_high_school_us_history"* || $TASK == *"cot"* || $TASK == *"minerva_math_"* || $TASK == *"mbpp"* || $TASK == *"bigcodebench"* || $TASK == *"ruler"* || $TASK == *"sciriff"* || $TASK == *"boolq"* || $TASK == *"drop"* ]]; then
            batch_size=$((BATCH_SIZE / 4))
        else
            batch_size=$BATCH_SIZE
        fi

        # adjust number of gpus requested if its mmlu, agi_eval, bbh, gsm8k, minerva, codex, mbpp
        if [[ $TASK == *agi_eval* || $TASK == *bbh* || $TASK == *gsm8k* || $TASK == *minerva_math_* || $TASK == *codex* || $TASK == *mbpp* || $TASK == *synthea* ]]; then
            gpus=4
        else
            gpus=1
        fi

        # if the model is a 35b model, further reduce batch size by half
        if [[ $MODEL_NAME == *"1b35b"* ]]; then
            batch_size=$((batch_size / 4))
            gpus=$((gpus * 2))
        fi

        # Create a shorter, valid job name
        # Remove invalid characters and truncate long names
        safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g')
        safe_group_name=$(echo $GROUP_NAME | sed 's/[^a-zA-Z0-9_-]//g')
        job_name="eval-${safe_model_name}-${safe_group_name}"

        echo "  Model name: $model"
        echo "  Output dir: $OUTPUT_DIR"
        echo "  GPUs: $gpus"
        echo "  Batch size: $batch_size"
        echo "  Job name: $job_name"

#        PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
#                --model "${MODEL_DIR}/${MODEL_NAME}" \
#                --model-type hf \
#                --task $TASK \
#                --output-dir $OUTPUT_DIR \
#                --batch-size $batch_size \
#                --gpus $gpus \

        gantry run \
            --name $job_name \
            --weka oe-training-default:/weka/oe-training-default \
            --install "pip install -e \".[all]\"" \
            --budget ai2/oceo \
            --workspace ai2/flex2 \
            --cluster $CLUSTER \
            --priority urgent \
            --allow-dirty \
            --gpus $gpus \
            --env-secret HF_TOKEN=RYAN_HF_TOKEN \
            --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
            --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
            -- \
            bash -c "PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
                --model "${MODEL_DIR}/${MODEL_NAME}" \
                --model-type hf \
                --task $TASK \
                --remote-output-dir $OUTPUT_DIR \
                --batch-size $batch_size \
                --gpus $gpus \
                "

        sleep 20

        echo "Launched evaluation for model: $model, group: $GROUP_NAME"
        echo "----------------------------------------"
    done

    echo "Completed all groups for model: $model"
    echo "========================================"
done

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASK_GROUPS_LIST[@]}))"
echo "Check the beaker dashboard for job status."