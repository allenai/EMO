#!/bin/bash

# Script to launch beaker evaluations for all available tasks
# Usage: bash src/scripts/eval/launch_beaker_eval.sh

# Configuration
MODEL_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models
MODELS=("dense_1b_olmoe-mix_1028/step30995-hf")
BASE_OUTPUT_DIR="s3://ai2-sewonm/ryanwang/evals"
BATCH_SIZE=16
CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf

# Define grouped tasks
declare -A TASK_GROUPS

#TASK_GROUPS["arc_easy"]="arc_easy:mc_train::olmes arc_easy:mc_validation::olmes arc_easy:mc_test::olmes arc_easy:rc_train::olmes arc_easy:rc_validation::olmes arc_easy:rc_test::olmes"
#TASK_GROUPS["arc_challenge"]="arc_challenge:mc_train::olmes arc_challenge:mc_validation::olmes arc_challenge:mc_test::olmes arc_challenge:rc_train::olmes arc_challenge:rc_validation::olmes arc_challenge:rc_test::olmes"
#TASK_GROUPS["boolq"]="boolq:mc_train::olmes boolq:mc_validation::olmes boolq:mc_test::olmes boolq:rc_train::olmes boolq:rc_validation::olmes boolq:rc_test::olmes"
#TASK_GROUPS["csqa"]="csqa:mc_train::olmes csqa:mc_validation::olmes csqa:mc_test::olmes csqa:rc_train::olmes csqa:rc_validation::olmes csqa:rc_test::olmes"
#TASK_GROUPS["hellaswag"]="hellaswag:mc_train::olmes hellaswag:mc_validation::olmes hellaswag:mc_test::olmes hellaswag:rc_train::olmes hellaswag:rc_validation::olmes hellaswag:rc_test::olmes"

TASK_GROUPS_LIST=(
#  "arc_easy|arc_easy:mc_train::olmes arc_easy:mc_validation::olmes arc_easy:mc_test::olmes arc_easy:rc_train::olmes arc_easy:rc_validation::olmes arc_easy:rc_test::olmes"
#  "arc_challenge|arc_challenge:mc_train::olmes arc_challenge:mc_validation::olmes arc_challenge:mc_test::olmes arc_challenge:rc_train::olmes arc_challenge:rc_validation::olmes arc_challenge:rc_test::olmes"
#  "boolq|boolq:mc_train::olmes boolq:mc_validation::olmes boolq:mc_test::olmes boolq:rc_train::olmes boolq:rc_validation::olmes boolq:rc_test::olmes"
#  "csqa|csqa:mc_train::olmes csqa:mc_validation::olmes csqa:mc_test::olmes csqa:rc_train::olmes csqa:rc_validation::olmes csqa:rc_test::olmes"
  "hellaswag|hellaswag:mc_train::olmes hellaswag:mc_validation::olmes hellaswag:mc_test::olmes hellaswag:rc_train::olmes hellaswag:rc_validation::olmes hellaswag:rc_test::olmes"
#  "openbookqa|openbookqa:mc_train::olmes openbookqa:mc_validation::olmes openbookqa:mc_test::olmes openbookqa:rc_train::olmes openbookqa:rc_validation::olmes openbookqa:rc_test::olmes"
#  "piqa|piqa:mc_train::olmes piqa:mc_validation::olmes piqa:mc_test::olmes piqa:rc_train::olmes piqa:rc_validation::olmes piqa:rc_test::olmes"
  "socialiqa|socialiqa:mc_train::olmes socialiqa:mc_validation::olmes socialiqa:mc_test::olmes socialiqa:rc_train::olmes socialiqa:rc_validation::olmes socialiqa:rc_test::olmes"
#  "winogrande|winogrande:mc_train::olmes winogrande:mc_validation::olmes winogrande:mc_test::olmes winogrande:rc_train::olmes winogrande:rc_validation::olmes winogrande:rc_test::olmes"
)

# Define all available tasks from run_eval.sh (ALL tasks from all groups)
TASKS=(
    # MC9 tasks
#    arc_easy:mc::olmes
#    arc_challenge:mc::olmes
#    boolq:mc::olmes
#    csqa:mc::olmes
#    hellaswag:mc::olmes
#    openbookqa:mc::olmes
#    piqa:mc::olmes
#    socialiqa:mc::olmes
#    winogrande:mc::olmes
#
#    arc_easy:rc::olmes
#    arc_challenge:rc::olmes
#    boolq:rc::olmes
#    csqa:rc::olmes
#    hellaswag:rc::olmes
#    openbookqa:rc::olmes
#    piqa:rc::olmes
#    socialiqa:rc::olmes
#    winogrande:rc::olmes
#
#    # Gen5 tasks
#    coqa::olmes
#    squad::olmes
#    naturalqs::olmes
#    triviaqa::olmes
#    drop::olmes

    # MMLU tasks
#    mmlu:mc::olmes
#    mmlu_pro:mc::none
#
#    mmlu:rc::olmes
#
##    # AGI eval
#    agi_eval_english:1shot::olmes
##
##    # BBH
#    bbh:cot-v1::olmes
##
##    # Math2 tasks
#    gsm8k::olmes
#    minerva_math_algebra::olmes
#    minerva_math_counting_and_probability::olmes
#    minerva_math_geometry::olmes
#    minerva_math_intermediate_algebra::olmes
#    minerva_math_number_theory::olmes
#    minerva_math_prealgebra::olmes
#    minerva_math_precalculus::olmes
#
##    # Code4 tasks
#    codex_humaneval:temp0.8
#    codex_humanevalplus:temp0.8
#    mbpp::none
#    mbppplus::none

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

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"

    for entry in "${TASK_GROUPS_LIST[@]}"; do
        GROUP_NAME="${entry%%|*}"                # text before '|'
        TASK="${entry#*|}"            # text after '|'

        # Batch size adjustment (matching original script)
        if [[ $TASK == *"cot"* || $TASK == *"minerva_math_"* || $TASK == *"mbpp"* || $TASK == *"bigcodebench"* || $TASK == *"ruler"* || $TASK == *"sciriff"* ]]; then
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
        job_name="eval-${safe_model_name}-${safe_group_name}"

        echo "  Model name: $model"
        echo "  Output dir: $OUTPUT_DIR"
        echo "  GPUs: $gpus"
        echo "  Batch size: $batch_size"
        echo "  Job name: $job_name"

        gantry run \
            --name $job_name \
            --weka oe-training-default:/weka/oe-training-default \
            --install "bash src/scripts/eval/setup_eval_env.sh;" \
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
                --model "${MODEL_DIR}/${MODEL_PATH}" \
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

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASK_GROUPS[@]}))"
echo "Check the beaker dashboard for job status."