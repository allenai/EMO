#!/bin/bash

# Script to launch beaker evaluations for all available tasks
# Usage: bash src/scripts/eval/launch_beaker_eval.sh

# Configuration
#OUT_DIR="/weka/oe-training-default/ryanwang/phdbrainstorm/data"
OUT_DIR="/root/ryanwang/phdbrainstorm/data/finetune"
BATCH_SIZE=16
CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf

# Define all available tasks from run_eval.sh (ALL tasks from all groups)
TASKS=(
    # MC9 tasks
    arc_easy:mc_train::olmes
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

echo "Launching finetuning data preparation for ${#TASKS[@]} tasks..."
echo "out dirs: $OUT_DIR"
echo "Cluster: $CLUSTER"
echo ""

for TASK in "${TASKS[@]}"; do
    echo "Launching dataset creation for task: $TASK"

    clean_task=${TASK%%::*}
    echo "$clean_task"

    job_name="prepare_data_finetuning_${clean_task}"

    base_dir="${OUT_DIR}/${clean_task}"

    echo "  base dir: $base_dir"
    echo "  Job name: $job_name"

    # this formats the data and saves raw requests
    PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
        --task "$TASK" \
        --output-dir "${base_dir}/raw" \
        --save-raw-requests true

    # this gets the correct requests and saves them into dolma format (jsonl
    PYTHONPATH=. python -u src/scripts/eval/extract_finetuning_examples.py \
        --task "$TASK" \
        --input-dir "${base_dir}/raw" \
        --output-dir "${base_dir}/processed" \


#    gantry run \
#        --name $job_name \
#        --weka oe-training-default:/weka/oe-training-default \
#        --install "bash src/scripts/eval/setup_eval_env.sh;" \
#        --budget ai2/oceo \
#        --workspace ai2/flex2 \
#        --cluster $CLUSTER \
#        --priority urgent \
#        --env-secret HF_TOKEN=RYAN_HF_TOKEN \
#        --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
#        --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
#        -- \
#        bash -c "PYTHONPATH=. python -u src/scripts/eval/launch_evals.py \
#            --task $TASK \
#            --output-dir $OUT_DIR \
#            --save-raw-requests
#            "
    echo "Launched evaluation for task: $TASK"
    echo "----------------------------------------"
done

echo "Completed all tasks"
echo "========================================"

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#TASKS[@]}))"
echo "Check the beaker dashboard for job status."