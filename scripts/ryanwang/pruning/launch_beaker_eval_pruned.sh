#!/bin/bash

# Script to launch beaker evaluations for pruned models. Requires changing "postfix" accordingly

# Configuration
BASE_DIR="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE"
MODEL_DIR="${BASE_DIR}/models"
PRUNE_DIR="${BASE_DIR}/prune"
#MODEL_DIR="/root/ryanwang/phdbrainstorm/FlexMoE/models"

PARENT_MODELS=(
#    "moe_1b14b_128experts_olmoe-mix_130B_1117/step30995"
#    "dense_1b_olmoe-mix_1119/step30995"
#    "dense_1b_olmoe-mix_1119/step30995/noloadoptim"
#    "twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115/step30995"

    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995"
#    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995"
#    "twolevelsamplingnolb-32_1b14b_stability_1127/step30995"
)

postfix="_keepk32"
#postfix="_keepk32/lr-7e-5_warmup-0.1"


FINETUNE_TASKS=(
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step0-hf"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step84-hf"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step168-hf"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step252-hf"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step336-hf"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step420-hf"
#
#    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step0-hf"
#    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step41-hf"
#    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step82-hf"
#    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step123-hf"
#    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step164-hf"
#    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step207-hf"
#
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step0-hf"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step315-hf"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step630-hf"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step945-hf"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step1260-hf"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step1578-hf"

    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step0-hf"
    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step327-hf"
    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step654-hf"
    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step981-hf"
    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step1308-hf"
    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step1638-hf"

    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step0-hf"
    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step1458-hf"
    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step2916-hf"
    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step4374-hf"
    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step5832-hf"
    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step7293-hf"

    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step0-hf"
    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step185-hf"
    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step370-hf"
    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step555-hf"
    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step740-hf"
    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step927-hf"
#
#    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step0-hf"
#    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step566-hf"
#    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step1132-hf"
#    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step1698-hf"
#    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step2264-hf"
#    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step2832-hf"
#
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step0-hf"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step1215-hf"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step2430-hf"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step3645-hf"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step4860-hf"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step6075-hf"
#
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step0-hf"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step1477-hf"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step2954-hf"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step4431-hf"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step5908-hf"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step7386-hf"

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

            # check if "dense" appears in BASE, if so then change dir structure (dense did not go through pruning)
            if [[ "$PARENT_MODEL" == *"dense"* ]]; then
              # remove everything before the first "/" in FINETUNE
                FINETUNE_TASK="${FINETUNE_TASK#*/}"
                MODEL_NAME="${PARENT_MODEL}/${FINETUNE_TASK}"
            else
                MODEL_NAME="${PARENT_MODEL}_${FINETUNE_TASK}"
            fi

            # Construct the full model path
            echo "Processing model: $MODEL_NAME"
            model=$(get_checkpoint_name $MODEL_NAME)
            echo "Model name for output dir: $model"
            OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"

            # Create a shorter, valid job name
            # Remove invalid characters and truncate long names
            safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g')
            safe_group_name=$(echo $GROUP_NAME | sed 's/[^a-zA-Z0-9_-]//g')
            job_name="eval-${safe_model_name}-${safe_group_name}"
            # limit runname to 128 characters, take first 25 and last 75
            job_name=$(echo $job_name | cut -c1-35)_$(echo $job_name | rev | cut -c1-65 | rev)


            echo "  Model name: $model"
            echo "  Output dir: $OUTPUT_DIR"
            echo "  GPUs: $gpus"
            echo "  Batch size: $batch_size"
            echo "  Job name: $job_name"


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