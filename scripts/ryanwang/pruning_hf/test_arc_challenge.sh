#!/bin/bash

# Configuration
BASE_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE
#BASE_DIR="/root/ryanwang/phdbrainstorm/FlexMoE"
MODELS=(
    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995-hf"
#    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995-hf"
#    "dense_1b_olmoe-mix_prenorm_noqknorm_1123/step30995-hf"
#    "moe_1b4b_32experts_1224/step30995-hf"
    )

CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf

num_epochs=1
prune_keep_k=32
batch_size=32

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
#  "gsm8k_generation_0shot"
#  "coqa_0shot"
##  "coqa_full_0shot"
#  "squad_0shot"

#   TO BE IMPLEMENTED
#  "mmlu"

#  "synthea_zeroshot"

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
#            micro_batch_size=$((micro_batch_size / 4))
#        else
#            micro_batch_size=$micro_batch_size
#        fi
        micro_batch_size=8

        # TODO choose the right number of gpus based on task (so that it doesn't oom)
#        # adjust number of gpus requested if its agi_eval, bbh, gsm8k, minerva, codex, mbpp
#        if [[ $TASK == *agi_eval* || $TASK == *bbh* || $TASK == *gsm8k* || $TASK == *minerva_math_* || $TASK == *codex* || $TASK == *mbpp* || $MODEL == *"1b35b"* ]]; then
#            gpus=4
#        else
#            gpus=1
#        fi
        gpus=4

        # TODO: choose the right learning rate based on task
        lr=5e-5

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

        # if the model is dense or 1b4b, we skip activation and pruning
        if [[ $MODEL == *"dense"* || $MODEL == *"1b4b"* ]]; then
            echo "  Skipping activation computation and pruning for model: $MODEL"
            bash scripts/hf_finetune_with_pruning.sh \
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
                --skip-activation \
                --skip-prune
            echo "Launched evaluation for model: $model, task: $TASK"
            echo "----------------------------------------"
            continue
        fi

#        bash scripts/hf_finetune_with_pruning.sh \
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
#                --num-epochs ${num_epochs}

        gantry run \
            --name $job_name \
            --weka oe-training-default:/weka/oe-training-default \
            --install "pip install -e \".[all]\"" \
            --budget ai2/oceo \
            --workspace ai2/flex2 \
            --cluster $CLUSTER \
            --priority urgent \
            --gpus $gpus \
            --allow-dirty \
            --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" \
            -- \
            bash -c "bash scripts/hf_finetune_with_pruning.sh \
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
                --num-epochs ${num_epochs}
            "

        echo "Launched evaluation for model: $model, task: $TASK"
        echo "----------------------------------------"
    done

    echo "Completed all groups for model: $model"
    echo "========================================"
done

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASK_GROUPS_LIST[@]}))"
echo "Check the beaker dashboard for job status."
