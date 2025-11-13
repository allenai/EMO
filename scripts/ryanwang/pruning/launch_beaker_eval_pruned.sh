#!/bin/bash

# Script to launch beaker evaluations for all available tasks
# Usage: bash src/scripts/eval/launch_beaker_eval.sh

# Configuration
BASE_DIR="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE"
MODEL_DIR="${BASE_DIR}/models"
PRUNE_DIR="${BASE_DIR}/prune"
#MODEL_DIR="/root/ryanwang/phdbrainstorm/FlexMoE/models"

PARENT_MODELS=(
    "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995"
)

FINETUNE_TASKS=(
    "task-arc_easy_rc_train_0shot_finetune-keepk32/step84-hf"
    "task-arc_easy_rc_train_0shot_finetune-keepk32/step168-hf"
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
  # MC9 tasks
  "arc_easy|arc_easy:mc_test::olmes arc_easy:rc_test::olmes"
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
#  "gen5|coqa::olmes squad::olmes naturalqs::olmes triviaqa::olmes drop::olmes"

#   GSM8K
#  "gsm8k_test|gsm8k:perplexity_test::olmes"
)

# Function to get checkpoint name (matching the original script)
function get_checkpoint_name {
    local path=$1
    local split_path=${path#*OLMo2-7B-}
    local modified_path=${split_path//\//_}
    modified_path=$(echo $modified_path | sed 's/^_//;s/_$//')
    echo "${modified_path}"
}

echo "Launching beaker evaluations for ${#PARENT_MODELS[@]} parent models, ${#FINETUNE_TASKS[@]} finetune tasks, and ${#TASK_GROUPS_LIST[@]} task groups..."
echo "Parent models: ${PARENT_MODELS[@]}"
echo "Finetune tasks: ${FINETUNE_TASKS[@]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Cluster: $CLUSTER"
echo ""

# Launch evaluation for each combination of parent model and finetune task
for PARENT_MODEL in "${PARENT_MODELS[@]}"; do
    for FINETUNE_TASK in "${FINETUNE_TASKS[@]}"; do
        # Construct the full model path by combining parent model and finetune task
        MODEL_NAME="${PARENT_MODEL}/${FINETUNE_TASK}"
        echo "Processing model: $MODEL_NAME"

        model=$(get_checkpoint_name $MODEL_NAME)

        echo "Model name for output dir: $model"

        OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"

        for entry in "${TASK_GROUPS_LIST[@]}"; do
            GROUP_NAME="${entry%%|*}"                # text before '|'
            TASK="${entry#*|}"            # text after '|'

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
            safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g')
            safe_group_name=$(echo $GROUP_NAME | sed 's/[^a-zA-Z0-9_-]//g')
            job_name="eval-${safe_model_name}-${safe_group_name}"

            # find the activation file
            parent_model_name=$(get_checkpoint_name $PARENT_MODEL)
            # get finetune_task_name by getting what comes before _finetune, and also swapping out all "train" with "validation"
            finetune_task_name="${FINETUNE_TASK%%_finetune*}"
            finetune_task_name="${finetune_task_name/train/validation}"
            activation_file="${PRUNE_DIR}/${parent_model_name}-hf/${finetune_task_name}-router.jsonl"


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

#            gantry run \
#            --name $job_name \
#            --weka oe-training-default:/weka/oe-training-default \
#            --install "pip install -e \".[all]\"" \
#            --budget ai2/oe-oceo \
#            --workspace ai2/flex2 \
#            --cluster $CLUSTER \
#            --priority urgent \
#            --gpus $gpus \
#            --env-secret HF_TOKEN=RYAN_HF_TOKEN \
#            --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
#            --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
#            -- \
#            bash -c "PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
#                --model "${MODEL_DIR}/${MODEL_NAME}" \
#                --model-type hf \
#                --task $TASK \
#                --remote-output-dir $OUTPUT_DIR \
#                --batch-size $batch_size \
#                --gpus $gpus \
#                --do_prune \
#                --activation_file $activation_file \
#                --prune_keep_k $prune_keep_k \
#                "

            echo "Launched evaluation for model: $model, group: $GROUP_NAME"
            echo "----------------------------------------"
        done

        echo "Completed all groups for model: $model"
        echo "========================================"
    done
done

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#PARENT_MODELS[@]} * ${#FINETUNE_TASKS[@]} * ${#TASK_GROUPS_LIST[@]}))"
echo "Check the beaker dashboard for job status."