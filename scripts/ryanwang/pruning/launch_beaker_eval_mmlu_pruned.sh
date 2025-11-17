#!/bin/bash

# Script to launch beaker evaluations for all available tasks
# Usage: bash src/scripts/eval/launch_beaker_eval.sh

# Configuration
BASE_DIR="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE"
MODEL_DIR="${BASE_DIR}/models"
PRUNE_DIR="${BASE_DIR}/prune"
#MODEL_DIR="/root/ryanwang/phdbrainstorm/FlexMoE/models"

PARENT_MODELS=(
#    "twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115/step30995"
    "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995"
)

BASE_OUTPUT_DIR="s3://ai2-sewonm/ryanwang/evals"
#BASE_OUTPUT_DIR="/root/ryanwang/phdbrainstorm/FlexMoE/evals"
BATCH_SIZE=16
prune_keep_k=32
CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf


# Define grouped tasks
TASK_GROUPS_LIST=(
  ######### TEST-only ##########
#   MMLU
#  "abstract_algebra|mmlu_abstract_algebra:rc_test::olmes"
#  "anatomy|mmlu_anatomy:rc_test::olmes"
#  "astronomy|mmlu_astronomy:rc_test::olmes"
#  "business_ethics|mmlu_business_ethics:rc_test::olmes"
#  "clinical_knowledge|mmlu_clinical_knowledge:rc_test::olmes"
#  "college_biology|mmlu_college_biology:rc_test::olmes"
#  "college_chemistry|mmlu_college_chemistry:rc_test::olmes"
  "college_computer_science|mmlu_college_computer_science:rc_test::olmes"
  "college_mathematics|mmlu_college_mathematics:rc_test::olmes"
  "college_medicine|mmlu_college_medicine:rc_test::olmes"
  "college_physics|mmlu_college_physics:rc_test::olmes"
  "computer_security|mmlu_computer_security:rc_test::olmes"
  "conceptual_physics|mmlu_conceptual_physics:rc_test::olmes"
  "econometrics|mmlu_econometrics:rc_test::olmes"
  "electrical_engineering|mmlu_electrical_engineering:rc_test::olmes"
  "elementary_mathematics|mmlu_elementary_mathematics:rc_test::olmes"
  "formal_logic|mmlu_formal_logic:rc_test::olmes"
  "global_facts|mmlu_global_facts:rc_test::olmes"
  "high_school_biology|mmlu_high_school_biology:rc_test::olmes"
  "high_school_chemistry|mmlu_high_school_chemistry:rc_test::olmes"
  "high_school_computer_science|mmlu_high_school_computer_science:rc_test::olmes"
  "high_school_european_history|mmlu_high_school_european_history:rc_test::olmes"
  "high_school_geography|mmlu_high_school_geography:rc_test::olmes"
  "high_school_government_and_politics|mmlu_high_school_government_and_politics:rc_test::olmes"
  "high_school_macroeconomics|mmlu_high_school_macroeconomics:rc_test::olmes"
  "high_school_mathematics|mmlu_high_school_mathematics:rc_test::olmes"
  "high_school_microeconomics|mmlu_high_school_microeconomics:rc_test::olmes"
  "high_school_physics|mmlu_high_school_physics:rc_test::olmes"
  "high_school_psychology|mmlu_high_school_psychology:rc_test::olmes"
  "high_school_statistics|mmlu_high_school_statistics:rc_test::olmes"
  "high_school_us_history|mmlu_high_school_us_history:rc_test::olmes"
  "high_school_world_history|mmlu_high_school_world_history:rc_test::olmes"
  "human_aging|mmlu_human_aging:rc_test::olmes"
  "human_sexuality|mmlu_human_sexuality:rc_test::olmes"
  "international_law|mmlu_international_law:rc_test::olmes"
  "jurisprudence|mmlu_jurisprudence:rc_test::olmes"
  "logical_fallacies|mmlu_logical_fallacies:rc_test::olmes"
  "machine_learning|mmlu_machine_learning:rc_test::olmes"
  "management|mmlu_management:rc_test::olmes"
  "marketing|mmlu_marketing:rc_test::olmes"
  "medical_genetics|mmlu_medical_genetics:rc_test::olmes"
  "miscellaneous|mmlu_miscellaneous:rc_test::olmes"
  "moral_disputes|mmlu_moral_disputes:rc_test::olmes"
  "moral_scenarios|mmlu_moral_scenarios:rc_test::olmes"
  "nutrition|mmlu_nutrition:rc_test::olmes"
  "philosophy|mmlu_philosophy:rc_test::olmes"
  "prehistory|mmlu_prehistory:rc_test::olmes"
  "professional_accounting|mmlu_professional_accounting:rc_test::olmes"
  "professional_law|mmlu_professional_law:rc_test::olmes"
  "professional_medicine|mmlu_professional_medicine:rc_test::olmes"
  "professional_psychology|mmlu_professional_psychology:rc_test::olmes"
  "public_relations|mmlu_public_relations:rc_test::olmes"
  "security_studies|mmlu_security_studies:rc_test::olmes"
  "sociology|mmlu_sociology:rc_test::olmes"
  "us_foreign_policy|mmlu_us_foreign_policy:rc_test::olmes"
  "virology|mmlu_virology:rc_test::olmes"
  "world_religions|mmlu_world_religions:rc_test::olmes"

)

# Function to get checkpoint name (matching the original script)
function get_checkpoint_name {
    local path=$1
    local split_path=${path#*OLMo2-7B-}
    local modified_path=${split_path//\//_}
    modified_path=$(echo $modified_path | sed 's/^_//;s/_$//')
    echo "${modified_path}"
}

echo "Launching beaker evaluations for ${#PARENT_MODELS[@]} parent models and ${#TASK_GROUPS_LIST[@]} task groups..."
echo "Parent models: ${PARENT_MODELS[@]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Cluster: $CLUSTER"
echo ""

# Launch evaluation for each combination of parent model and finetune task
for PARENT_MODEL in "${PARENT_MODELS[@]}"; do
      # Construct the full model path by combining parent model and finetune task
      MODEL_NAME="${PARENT_MODEL}-hf"
      echo "Processing model: $MODEL_NAME"

      model=$(get_checkpoint_name $MODEL_NAME)

      echo "Model name for output dir: $model"

      OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model-keep32"

      for entry in "${TASK_GROUPS_LIST[@]}"; do
          GROUP_NAME="${entry%%|*}"                # text before '|'
          TASK="${entry#*|}"            # text after '|'

          # Batch size adjustment (matching original script)
          if [[ $TASK == *"cot"* || $TASK == *"minerva_math_"* || $TASK == *"mbpp"* || $TASK == *"bigcodebench"* || $TASK == *"ruler"* || $TASK == *"sciriff"* || $TASK == *"boolq"* ]]; then
              batch_size=$((BATCH_SIZE / 4))
          else
              batch_size=$BATCH_SIZE
          fi

          # fix gpu to 1 since we are doing one mmlu task at a time
          gpus=1

          # Create a shorter, valid job name
          # Remove invalid characters and truncate long names
          safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g')
          safe_group_name=$(echo $GROUP_NAME | sed 's/[^a-zA-Z0-9_-]//g')
          job_name="eval-${safe_model_name}-${safe_group_name}"
          # limit job_name to be at most 128 characters
          job_name=${job_name:0:120}

          # find the activation file
          parent_model_name=$(get_checkpoint_name $PARENT_MODEL)
          # get the corresponding activation file
          task_name="${TASK%%::olmes*}"
          task_name="${task_name/test/validation_0shot}"
          task_name="${task_name/:/_}"
          activation_file="${PRUNE_DIR}/${parent_model_name}-hf/task-${task_name}-router.jsonl"


          echo "  Model name: $model"
          echo "  Output dir: $OUTPUT_DIR"
          echo "  GPUs: $gpus"
          echo "  Batch size: $batch_size"
          echo "  Job name: $job_name"
          echo "  Activation file: $activation_file"


#            PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
#                    --model "${MODEL_DIR}/${MODEL_NAME}" \
#                    --model-type hf \
#                    --task $TASK \
#                    --output-dir $OUTPUT_DIR \
#                    --batch-size $batch_size \
#                    --gpus $gpus \

          gantry run \
          --name $job_name \
          --weka oe-training-default:/weka/oe-training-default \
          --install "pip install -e \".[all]\"" \
          --budget ai2/oceo \
          --workspace ai2/flex2 \
          --cluster $CLUSTER \
          --priority urgent \
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
              --do_prune \
              --activation_file $activation_file \
              --prune_keep_k $prune_keep_k \
              "

          echo "Launched evaluation for model: $model, group: $GROUP_NAME"
          echo "----------------------------------------"
      done

      echo "Completed all groups for model: $model"
      echo "========================================"
done

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#PARENT_MODELS[@]} * ${#TASK_GROUPS_LIST[@]}))"
echo "Check the beaker dashboard for job status."