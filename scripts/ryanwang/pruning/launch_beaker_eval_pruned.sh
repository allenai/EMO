#!/bin/bash

# Script to launch beaker evaluations for pruned models. Requires changing "postfix" accordingly

# Configuration
BASE_DIR="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE"
MODEL_DIR="${BASE_DIR}/models"
PRUNE_DIR="${BASE_DIR}/prune"
#MODEL_DIR="/root/ryanwang/phdbrainstorm/FlexMoE/models"

PARENT_MODELS=(
#    "dense_1b_olmoe-mix_1028/step30995"
    "twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115/step30995"
#    "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995"
#    "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995"
)

#postfix="-keepk8"
postfix="-keepk32"
#postfix=""

#taskvariant="_0shot"
taskvariant=""

FINETUNE_TASKS=(
    "task-arc_easy_rc_train${taskvariant}_finetune${postfix}/step0-hf"
#    "task-arc_easy_rc_train${taskvariant}_finetune${postfix}/step84-hf"
#    "task-arc_easy_rc_train${taskvariant}_finetune${postfix}/step168-hf"
#    "task-arc_easy_rc_train${taskvariant}_finetune${postfix}/step252-hf"
#    "task-arc_easy_rc_train${taskvariant}_finetune${postfix}/step336-hf"
#    "task-arc_easy_rc_train${taskvariant}_finetune${postfix}/step420-hf"
#
#    "task-arc_challenge_rc_train${taskvariant}_finetune${postfix}/step0-hf"
#    "task-arc_challenge_rc_train${taskvariant}_finetune${postfix}/step41-hf"
#    "task-arc_challenge_rc_train${taskvariant}_finetune${postfix}/step82-hf"
#    "task-arc_challenge_rc_train${taskvariant}_finetune${postfix}/step123-hf"
#    "task-arc_challenge_rc_train${taskvariant}_finetune${postfix}/step164-hf"
#    "task-arc_challenge_rc_train${taskvariant}_finetune${postfix}/step207-hf"
#
#    "task-boolq_rc_train${taskvariant}_finetune${postfix}/step0-hf"
#    "task-boolq_rc_train${taskvariant}_finetune${postfix}/step315-hf"
#    "task-boolq_rc_train${taskvariant}_finetune${postfix}/step630-hf"
#    "task-boolq_rc_train${taskvariant}_finetune${postfix}/step945-hf"
#    "task-boolq_rc_train${taskvariant}_finetune${postfix}/step1260-hf"
#    "task-boolq_rc_train${taskvariant}_finetune${postfix}/step1578-hf"

#    "task-csqa_rc_train${taskvariant}_finetune${postfix}/step0-hf"
#    "task-csqa_rc_train${taskvariant}_finetune${postfix}/step327-hf"
#    "task-csqa_rc_train${taskvariant}_finetune${postfix}/step654-hf"
#    "task-csqa_rc_train${taskvariant}_finetune${postfix}/step981-hf"
#    "task-csqa_rc_train${taskvariant}_finetune${postfix}/step1308-hf"
#    "task-csqa_rc_train${taskvariant}_finetune${postfix}/step1638-hf"
#
#    "task-hellaswag_rc_train${taskvariant}_finetune${postfix}/step0-hf"
#    "task-hellaswag_rc_train${taskvariant}_finetune${postfix}/step1458-hf"
#    "task-hellaswag_rc_train${taskvariant}_finetune${postfix}/step2916-hf"
#    "task-hellaswag_rc_train${taskvariant}_finetune${postfix}/step4374-hf"
#    "task-hellaswag_rc_train${taskvariant}_finetune${postfix}/step5832-hf"
#    "task-hellaswag_rc_train${taskvariant}_finetune${postfix}/step7293-hf"
#
#    "task-openbookqa_rc_train${taskvariant}_finetune${postfix}/step0-hf"
#    "task-openbookqa_rc_train${taskvariant}_finetune${postfix}/step185-hf"
#    "task-openbookqa_rc_train${taskvariant}_finetune${postfix}/step370-hf"
#    "task-openbookqa_rc_train${taskvariant}_finetune${postfix}/step555-hf"
#    "task-openbookqa_rc_train${taskvariant}_finetune${postfix}/step740-hf"
#    "task-openbookqa_rc_train${taskvariant}_finetune${postfix}/step927-hf"
#
#    "task-piqa_rc_train${taskvariant}_finetune${postfix}/step0-hf"
#    "task-piqa_rc_train${taskvariant}_finetune${postfix}/step566-hf"
#    "task-piqa_rc_train${taskvariant}_finetune${postfix}/step1132-hf"
#    "task-piqa_rc_train${taskvariant}_finetune${postfix}/step1698-hf"
#    "task-piqa_rc_train${taskvariant}_finetune${postfix}/step2264-hf"
#    "task-piqa_rc_train${taskvariant}_finetune${postfix}/step2832-hf"
#
#    "task-socialiqa_rc_train${taskvariant}_finetune${postfix}/step0-hf"
#    "task-socialiqa_rc_train${taskvariant}_finetune${postfix}/step1215-hf"
#    "task-socialiqa_rc_train${taskvariant}_finetune${postfix}/step2430-hf"
#    "task-socialiqa_rc_train${taskvariant}_finetune${postfix}/step3645-hf"
#    "task-socialiqa_rc_train${taskvariant}_finetune${postfix}/step4860-hf"
#    "task-socialiqa_rc_train${taskvariant}_finetune${postfix}/step6075-hf"
#
#    "task-winogrande_rc_train${taskvariant}_finetune${postfix}/step0-hf"
#    "task-winogrande_rc_train${taskvariant}_finetune${postfix}/step1477-hf"
#    "task-winogrande_rc_train${taskvariant}_finetune${postfix}/step2954-hf"
#    "task-winogrande_rc_train${taskvariant}_finetune${postfix}/step4431-hf"
#    "task-winogrande_rc_train${taskvariant}_finetune${postfix}/step5908-hf"
#    "task-winogrande_rc_train${taskvariant}_finetune${postfix}/step7386-hf"

)

BASE_OUTPUT_DIR="s3://ai2-sewonm/ryanwang/evals"
#BASE_OUTPUT_DIR="/root/ryanwang/phdbrainstorm/FlexMoE/evals"
BATCH_SIZE=16
CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf


# Define grouped tasks
TASK_GROUPS_LIST=(
  ######### TEST-only ##########
  # MC9 tasks
  "arc_easy|arc_easy:rc_test::olmes"
  "arc_challenge|arc_challenge:rc_test::olmes"
  "boolq|boolq:rc_test::olmes"
  "csqa|csqa:rc_test::olmes"
  "hellaswag|hellaswag:rc_test::olmes"
  "openbookqa|openbookqa:rc_test::olmes"
  "piqa|piqa:rc_test::olmes"
  "socialiqa|socialiqa:rc_test::olmes"
  "winogrande|winogrande:rc_test::olmes"

#   MMLU
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
        for entry in "${TASK_GROUPS_LIST[@]}"; do
            GROUP_NAME="${entry%%|*}"                # text before '|'
            TASK="${entry#*|}"            # text after '|'

            # if GROUP_NAME is not in FINETUNE_TASK, skip
            if [[ $FINETUNE_TASK != *"$GROUP_NAME"* ]]; then
                echo "Skipping group $GROUP_NAME for finetune task $FINETUNE_TASK"
                continue
            fi

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

            # find the activation file
            parent_model_name=$(get_checkpoint_name $PARENT_MODEL)
            # get validation_task_prefix by getting what comes before _finetune, and also swapping out all "train" with "validation"
            validation_task_prefix="${FINETUNE_TASK%%_finetune*}"
            validation_task_prefix="${validation_task_prefix/train/validation}"
            activation_file="${PRUNE_DIR}/${parent_model_name}-hf/${validation_task_prefix}-router.jsonl"

            # Construct the full model path
            MODEL_NAME="${PARENT_MODEL}_${validation_task_prefix}/${FINETUNE_TASK}"
            echo "Processing model: $MODEL_NAME"
            model=$(get_checkpoint_name $MODEL_NAME)
            echo "Model name for output dir: $model"
            OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"

            # Create a shorter, valid job name
            # Remove invalid characters and truncate long names
            safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g')
            safe_group_name=$(echo $GROUP_NAME | sed 's/[^a-zA-Z0-9_-]//g')
            job_name="eval-${safe_model_name}-${safe_group_name}"
            # limit job_name to be at most 128 characters
            job_name=${job_name:0:120}


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
                --prune_keep_k $keepk \
                "

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