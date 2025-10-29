#!/bin/bash

# Script to launch beaker evaluations for all available tasks
# Usage: bash src/scripts/eval/launch_beaker_eval.sh

# Configuration
#MODEL_DIRS=("/weka/oe-training-default/ryanwang/phdbrainstorm/models/olmoe-pretrain-mose-natural-1022/step30995-hf")
#EVAL_DIRS="/weka/oe-training-default/ryanwang/phdbrainstorm/evals/weka_oe-training-default_ryanwang_phdbrainstorm_models_olmoe-pretrain-mose-natural-1022_step30995-hf"

MODEL_DIRS=("/root/ryanwang/phdbrainstorm/models/olmoe-pretrain-mose-natural-1022/step30995-hf")
EVAL_DIRS=("/root/ryanwang/phdbrainstorm/evals/weka_oe-training-default_ryanwang_phdbrainstorm_models_olmoe-pretrain-mose-natural-1022_step30995-hf")
BATCH_SIZE=16
GPUS=1
CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf

# Define all available tasks from run_eval.sh (ALL tasks from all groups)
TASKS=(
    # MC9 tasks
    arc_easy:mc_validation::olmes
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

    # Gen5 tasks
#    coqa::olmes
#    squad::olmes
#    naturalqs::olmes
#    triviaqa::olmes
#    drop::olmes
#
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

echo "Launching beaker logits evaluations for ${#MODEL_DIRS[@]} models and ${#TASKS[@]} tasks..."
echo "Models: ${MODEL_DIRS[@]}"
echo "Eval dirs: ${EVAL_DIRS[@]}"
echo "Cluster: $CLUSTER"
echo ""


# Loop through both arrays together
for i in "${!MODEL_DIRS[@]}"; do
    MODEL_PATH="${MODEL_DIRS[$i]}"
    EVAL_DIR="${EVAL_DIRS[$i]}"

    echo "Processing model: $MODEL_PATH"
    echo "Evaluation dir: $EVAL_DIR"

    model="$(basename "$(dirname "$MODEL_PATH")")"

    for TASK in "${TASKS[@]}"; do
        echo "Launching evaluation for model: $model, task: $TASK"

        safe_model_name=$(echo "$model" | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-20)
        safe_task_name=$(echo "$TASK" | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-15)
        job_name="eval_logits-${safe_model_name}-${safe_task_name}"

        echo "  Model name: $model"
        echo "  Eval dir: $EVAL_DIR"
        echo "  GPUs: $GPUS"
        echo "  Batch size: $BATCH_SIZE"
        echo "  Job name: $job_name"

        PYTHONPATH=. python -u src/scripts/eval/launch_logits.py \
            --model "$MODEL_PATH" \
            --task "$TASK" \
            --eval-dir "$EVAL_DIR" \
            --batch-size "$BATCH_SIZE" \
            --gpus "$GPUS" \
            --use_correct_only

#        gantry run \
#            --name $job_name \
#            --weka oe-training-default:/weka/oe-training-default \
#            --install "bash src/scripts/eval/setup_eval_env.sh;" \
#            --budget ai2/oceo \
#            --workspace ai2/flex2 \
#            --cluster $CLUSTER \
#            --priority urgent \
#            --gpus $GPUS \
#            --env-secret HF_TOKEN=RYAN_HF_TOKEN \
#            --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
#            --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
#            -- \
#            bash -c "PYTHONPATH=. python -u src/scripts/eval/launch_logits.py \
#                --model $MODEL_PATH \
#                --task $TASK \
#                --eval-dir "$EVAL_DIR" \
#                --batch-size "$BATCH_SIZE" \
#                --gpus "$GPUS" \
#                --use_correct_only
#                "
        echo "Launched evaluation for model: $model, task: $TASK"
        echo "----------------------------------------"
    done

    echo "Completed all tasks for model: $model"
    echo "========================================"
done

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASKS[@]}))"
echo "Check the beaker dashboard for job status."