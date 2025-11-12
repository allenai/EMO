#!/bin/bash

# Script to launch beaker evaluations for all available tasks
# Usage: bash src/scripts/eval/launch_beaker_eval.sh

# Configuration
MODELS=("/weka/oe-training-default/ryanwang/phdbrainstorm/models/dense_1b_olmoe-mix_1028/step30995/olmoe-finetune-arc_easy-mc/step1686-hf")
BASE_OUTPUT_DIR="/weka/oe-training-default/ryanwang/phdbrainstorm/evals"
activation_file="/weka/oe-training-default/ryanwang/phdbrainstorm/evals/weka_oe-training-default_ryanwang_phdbrainstorm_models_olmoe-pretrain-mose-natural-1022_step30995-hf/arc_easy:mc-router.jsonl"
prune_keep_k=32
BATCH_SIZE=4
CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf



# Function to get checkpoint name (matching the original script)
function get_checkpoint_name {
    local path=$1
    local split_path=${path#*OLMo2-7B-}
    local modified_path=${split_path//\//_}
    modified_path=$(echo $modified_path | sed 's/^_//;s/_$//')
    echo "${modified_path//hf/${model_type}}"
}

echo "Launching beaker evaluations for ${#MODELS[@]} models and ${#TASKS[@]} tasks..."
echo "Models: ${MODELS[@]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Cluster: $CLUSTER"
echo ""

# Launch evaluation for each model and task combination
for MODEL_PATH in "${MODELS[@]}"; do
    echo "Processing model: $MODEL_PATH"

    # For setting the output_dir (matching original script logic)
    if [[ $MODEL_PATH == "/"* ]]; then
        # internal model
        model=$(get_checkpoint_name $MODEL_PATH)
    else
        # HF model
        model=$(echo $MODEL_PATH | cut -d'/' -f2)
    fi

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"

    for TASK in "${TASKS[@]}"; do
        echo "Launching evaluation for model: $model, task: $TASK"

    gpus=1

    # Batch size adjustment (matching original script)
    if [[ $TASK == *"cot"* || $TASK == "minerva_math_"* || $TASK == "mbpp"* || $TASK == "bigcodebench"* || $TASK == "ruler"* || $TASK == "sciriff"* ]]; then
        batch_size=$((BATCH_SIZE / 4))
    else
        batch_size=$BATCH_SIZE
    fi

    # Create a shorter, valid job name
    # Remove invalid characters and truncate long names
    safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-20)
    safe_task_name=$(echo $TASK | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-15)
    job_name="eval-${safe_model_name}-${safe_task_name}"

    echo "  Model name: $model"
    echo "  Output dir: $OUTPUT_DIR"
    echo "  GPUs: $gpus"
    echo "  Batch size: $batch_size"
    echo "  Job name: $job_name"

    gantry run \
        --name $job_name \
        --weka oe-training-default:/weka/oe-training-default \
        --install "pip install -e \".[all]\"" \
        --budget ai2/oe-base \
        --workspace ai2/flex2 \
        --cluster $CLUSTER \
        --priority urgent \
        --gpus $gpus \
        --env-secret HF_TOKEN=RYAN_HF_TOKEN \
        --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
        --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
        -- \
        bash -c "PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
            --model $MODEL_PATH \
            --model-type hf \
            --task $TASK \
            --output-dir $OUTPUT_DIR \
            --batch-size $batch_size \
            --gpus $gpus \
            --do_prune \
            --activation_file $activation_file \
            --prune_keep_k $prune_keep_k \
            "

        echo "Launched evaluation for model: $model, task: $TASK"
        echo "----------------------------------------"
    done

    echo "Completed all tasks for model: $model"
    echo "========================================"
done

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASKS[@]}))"
echo "Check the beaker dashboard for job status."