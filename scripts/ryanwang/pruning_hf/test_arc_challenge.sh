#!/bin/bash

# Configuration
#BASE_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE
BASE_DIR="/root/ryanwang/phdbrainstorm/FlexMoE"
MODELS=(
    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995-hf"
#    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995-hf"
    )

CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf

num_epochs=3
prune_keep_k=16

# Define grouped tasks
TASK_GROUPS_LIST=(
  ######### few-shot ##########
  # MC9 tasks
#  "arc_easy"
  "arc_challenge"
#  "boolq"
#  "csqa"
#  "hellaswag"
#  "openbookqa"
#  "piqa"
#  "socialiqa"
#  "winogrande"
#  "synthea_zeroshot"
#  "gsm8k_generation_zeroshot"
#  "coqa_zeroshot"
#  "coqa_full_zeroshot"
#  "squad_zeroshot"

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

#  "mmlu_abstract_algebra"
#  "mmlu_anatomy"
#  "mmlu_astronomy"
#  "mmlu_business_ethics"
#  "mmlu_clinical_knowledge"
#  "mmlu_college_biology"
#  "mmlu_college_chemistry"
#  "mmlu_college_computer_science"
#  "mmlu_college_mathematics"
#  "mmlu_college_medicine"
#  "mmlu_college_physics"
#  "mmlu_computer_security"
#  "mmlu_conceptual_physics"
#  "mmlu_econometrics"
#  "mmlu_electrical_engineering"
#  "mmlu_elementary_mathematics"
#  "mmlu_formal_logic"
#  "mmlu_global_facts"
#  "mmlu_high_school_biology"
#  "mmlu_high_school_chemistry"
#  "mmlu_high_school_computer_science"
#  "mmlu_high_school_european_history"
#  "mmlu_high_school_geography"
#  "mmlu_high_school_government_and_politics"
#  "mmlu_high_school_macroeconomics"
#  "mmlu_high_school_mathematics"
#  "mmlu_high_school_microeconomics"
#  "mmlu_high_school_physics"
#  "mmlu_high_school_psychology"
#  "mmlu_high_school_statistics"
#  "mmlu_high_school_us_history"
#  "mmlu_high_school_world_history"
#  "mmlu_human_aging"
#  "mmlu_human_sexuality"
#  "mmlu_international_law"
#  "mmlu_jurisprudence"
#  "mmlu_logical_fallacies"
#  "mmlu_machine_learning"
#  "mmlu_management"
#  "mmlu_marketing"
#  "mmlu_medical_genetics"
#  "mmlu_miscellaneous"
#  "mmlu_moral_disputes"
#  "mmlu_moral_scenarios"
#  "mmlu_nutrition"
#  "mmlu_philosophy"
#  "mmlu_prehistory"
#  "mmlu_professional_accounting"
#  "mmlu_professional_law"
#  "mmlu_professional_medicine"
#  "mmlu_professional_psychology"
#  "mmlu_public_relations"
#  "mmlu_security_studies"
#  "mmlu_sociology"
#  "mmlu_us_foreign_policy"
#  "mmlu_virology"
#  "mmlu_world_religions"

#   MMLU
#  "mmlu"

#   GSM8K
#  "gsm8k"

  ######### ZERO-SHOT only ##########
  # MC9 tasks
#  "arc_easy_zeroshot"
#  "arc_challenge_zeroshot"
#  "boolq_zeroshot"
#  "csqa_zeroshot"
#  "hellaswag_zeroshot"
#  "openbookqa_zeroshot"
#  "piqa_zeroshot"
#  "socialiqa_zeroshot"
#  "winogrande_zeroshot"

#   MMLU
#  "mmlu_zeroshot"

#   GSM8K
#  "gsm8k_zeroshot"
)


echo "Launching evals for ${#MODELS[@]} models and ${#TASK_GROUPS[@]} task groups..."
echo "Models: ${MODELS[@]}"
echo "Cluster: $CLUSTER"
echo ""

# Launch evaluation for each model and task combination
for MODEL in "${MODELS[@]}"; do
    echo "Processing model: ${MODEL}"

    for TASK in "${TASK_GROUPS_LIST[@]}"; do
        # TODO: choose the right batch size based on the task
#        # Batch size adjustment (matching original script)
#        if [[ $TASK == *"mmlu_high_school_european_history"* || $TASK == *"mmlu_high_school_us_history"* || $TASK == *"mmlu_history"* || $TASK == *"mmlu_philosophy"* || $TASK == *"cot"* || $TASK == *"minerva_math_"* || $TASK == *"mbpp"* || $TASK == *"bigcodebench"* || $TASK == *"ruler"* || $TASK == *"sciriff"* || $TASK == *"boolq"* || $TASK == *"synthea"* || $MODEL == *"1b35b"* ]]; then
#            batch_size=$((BATCH_SIZE / 4))
#        else
#            batch_size=$BATCH_SIZE
#        fi
        batch_size=32

        # TODO choose the right number of gpus based on task (so that it doesn't oom)
#        # adjust number of gpus requested if its agi_eval, bbh, gsm8k, minerva, codex, mbpp
#        if [[ $TASK == *agi_eval* || $TASK == *bbh* || $TASK == *gsm8k* || $TASK == *minerva_math_* || $TASK == *codex* || $TASK == *mbpp* || $MODEL == *"1b35b"* ]]; then
#            gpus=4
#        else
#            gpus=1
#        fi
        gpus=2

        # TODO: choose the right learning rate based on task
        lr=4e-5

        # Create a shorter, valid job name
        # Remove invalid characters and truncate long names

        stringified_model=$(echo $MODEL | sed 's/[^a-zA-Z0-9_-]//g')
        relative_dir="${stringified_model}/${TASK}_keepk_${prune_keep_k}_bs-${batch_size}_lr-${lr}_epoch-${num_epochs}"
        job_name="eval-$(echo $relative_dir | sed 's/[^a-zA-Z0-9_-]//g')"

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

        bash scripts/hf_finetune_with_pruning.sh \
                --model ${BASE_DIR}/models/${MODEL} \
                --task ${TASK} \
                --prune-keep-k ${prune_keep_k} \
                --base-dir "${BASE_DIR}/prune_evals" \
                --relative-dir ${relative_dir} \
                --num-gpus $gpus \
                --run-name ${job_name} \
                --learning-rate ${lr} \
                --batch-size ${batch_size} \
                --num-epochs ${num_epochs}

#        gantry run \
#            --name $job_name \
#            --weka oe-training-default:/weka/oe-training-default \
#            --install "pip install -e \".[all]\"" \
#            --budget ai2/oceo \
#            --workspace ai2/flex2 \
#            --cluster $CLUSTER \
#            --priority urgent \
#            --gpus $gpus \
#            --allow-dirty \
#            --env-secret HF_TOKEN=RYAN_HF_TOKEN \
#            --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
#            --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
#            -- \
#            bash -c "bash scripts/hf_finetune_with_pruning.sh \
#                --model ${BASE_DIR}/${MODEL} \
#                --task ${TASK} \
#                --prune-keep-k 16 \
#                --base-dir ${BASE_DIR}/evals \
#                --relative-dir ${relative_dir} \
#                --num-gpus $gpus \
#                --skip-activation \
#                --skip-prune \
#                --run-name ${job_name}
#            "

        echo "Launched evaluation for model: $model, task: $TASK"
        echo "----------------------------------------"
    done

    echo "Completed all groups for model: $model"
    echo "========================================"
done

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASK_GROUPS_LIST[@]}))"
echo "Check the beaker dashboard for job status."
