#!/bin/bash
# Configuration
MODEL_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models
#MODEL_DIR="/root/ryanwang/phdbrainstorm/FlexMoE/models"
MODELS=(
    "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995-hf"
    )
#BASE_OUTPUT_DIR="s3://ai2-sewonm/ryanwang/prune"
#BASE_OUTPUT_DIR="/root/ryanwang/phdbrainstorm/FlexMoE/prune"
BASE_OUTPUT_DIR="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/prune"
BATCH_SIZE=16
CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf

# Define grouped tasks
TASK_GROUPS_LIST=(
  ######### TEST-only ##########
  # MC9 tasks
#  "arc_easy"
#  "arc_challenge"
#  "boolq"
#  "csqa"
#  "hellaswag"
#  "openbookqa"
#  "piqa"
#  "socialiqa"
#  "winogrande"

#   MMLU
#  "mmlu"

#   GSM8K
  "gsm8k"
)

# Function to get checkpoint name (matching the original script)
function get_checkpoint_name {
    local path=$1
    local split_path=${path#*OLMo2-7B-}
    local modified_path=${split_path//\//_}
    modified_path=$(echo $modified_path | sed 's/^_//;s/_$//')
    echo "${modified_path//hf/${model_type}}"
}

echo "Launching beaker evaluations for ${#MODELS[@]} models and ${#TASK_GROUPS[@]} task groups..."
echo "Models: ${MODELS[@]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Cluster: $CLUSTER"
echo ""

# Launch evaluation for each model and task combination
for MODEL_PATH in "${MODELS[@]}"; do
    echo "Processing model: $MODEL_PATH"

    # For setting the output_dir (matching original script logic)
#    if [[ $MODEL_PATH == "/"* ]]; then
#        # internal model
#        model=$(get_checkpoint_name $MODEL_PATH)
#    else
#        # HF model
#        model=$(echo $MODEL_PATH | cut -d'/' -f2)
#    fi
    model=$(get_checkpoint_name $MODEL_PATH)

    echo "Model name for output dir: $model"

    for entry in "${TASK_GROUPS_LIST[@]}"; do
        GROUP_NAME="$entry"
        TASK="$GROUP_NAME"

        # Batch size adjustment (matching original script)
        if [[ $TASK == *"cot"* || $TASK == *"minerva_math_"* || $TASK == *"mbpp"* || $TASK == *"bigcodebench"* || $TASK == *"ruler"* || $TASK == *"sciriff"* || $TASK == *"boolq"* ]]; then
            batch_size=$((BATCH_SIZE / 4))
        else
            batch_size=$BATCH_SIZE
        fi

        # adjust number of gpus requested if its mmlu, agi_eval, bbh, gsm8k, minerva, codex, mbpp
        if [[ $TASK == *mmlu* || $TASK == *agi_eval* || $TASK == *bbh* || $TASK == *gsm8k* || $TASK == *minerva_math_* || $TASK == *codex* || $TASK == *mbpp* ]]; then
            gpus=4
        else
            gpus=1
        fi

        # Create a shorter, valid job name
        # Remove invalid characters and truncate long names
        safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-20)
        safe_group_name=$(echo $GROUP_NAME | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-15)
        job_name="pruneprep-${safe_model_name}-${safe_group_name}"

        echo "  Model name: $model"
        echo "  Base output dir: $BASE_OUTPUT_DIR"
        echo "  GPUs: $gpus"
        echo "  Batch size: $batch_size"
        echo "  Job name: $job_name"

#        bash src/scripts/eval/prepare_pruning.sh \
#            --GROUP_NAME "$GROUP_NAME" \
#            --BASE_OUTPUT_DIR "$BASE_OUTPUT_DIR" \
#            --BATCH_SIZE "$batch_size" \
#            --MODEL_PATH "${MODEL_DIR}/${MODEL_PATH}" \
#            --GPUS "$gpus"

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
            bash -c "bash src/scripts/eval/prepare_pruning.sh \
                --GROUP_NAME "$GROUP_NAME" \
                --BASE_OUTPUT_DIR "$BASE_OUTPUT_DIR" \
                --BATCH_SIZE "$batch_size" \
                --MODEL_PATH "${MODEL_DIR}/${MODEL_PATH}" \
                --GPUS "$gpus"
            "

        echo "Launched evaluation for model: $model, group: $GROUP_NAME"
        echo "----------------------------------------"
    done

    echo "Completed all groups for model: $model"
    echo "========================================"
done

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASK_GROUPS_LIST[@]}))"
echo "Check the beaker dashboard for job status."