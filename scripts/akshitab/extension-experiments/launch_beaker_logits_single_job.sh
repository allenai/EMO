#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/launch_common.sh"

# Usage:
#   ./launch_beaker_logits_single_job.sh              # run locally (sequentially)
#   ./launch_beaker_logits_single_job.sh --gantry      # submit one gantry job per model

# Parse flags
USE_GANTRY=false
for arg in "$@"; do
    case $arg in
        --gantry) USE_GANTRY=true ;;
    esac
done

# Configuration

MODEL_DIR="${MODELS}"

MODELS=(
    # moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995-hf
    # moe1b14b_129experts_1trained_math_init_random_expert_5B/step1193-hf
    # moe1b14b_129experts_1trained_math_init_average_5B/step1193-hf
    # twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995-hf

    # freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_10B_lr_4e-4/step2385-hf
    # ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4/step2385-hf

    # ff-moe_1b14b_128base_4math_10B_4code_init_top2_code_mix_average_noise_10B_lr_4e-4/step2385-hf
    # merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf

    ## math extension before training:
    extensions/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc-hf
    ## code extension before training:
    extensions/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_code_average_noise-hf

    ## merged model
    # merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf


    # moe1b14b_128experts_76_5_122_126_trained_math_10B_lr_4e-4/step2385-hf
    # moe1b14b_128experts_76_41_120_3_trained_code_10B_lr_4e-4/step2385-hf
)

BASE_OUTPUT_DIR="${EVALS_S3_BASE}"
BATCH_SIZE=16
CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf


# Define tasks (launch_logits.py already loops over tasks internally, so we pass them all at once)
TASKS=(
  ######### TEST-only ##########
  # MC9 tasks
  "arc_easy:rc_test::olmes"
  "arc_challenge:rc_test::olmes"
  "boolq:rc_test::olmes"
  "csqa:rc_test::olmes"
#  "hellaswag:rc_test::olmes"
  "openbookqa:rc_test::olmes"
  "piqa:rc_test::olmes"
  "socialiqa:rc_test::olmes"
  "winogrande:rc_test::olmes"

  "gsm8k::olmes"
  "mbpp"
  "codex_humaneval"

  "squad::olmes"
)

# Function to get checkpoint name (matching the original script)
function get_checkpoint_name {
    local path=$1
    local split_path=${path#*OLMo2-7B-}
    local modified_path=${split_path//\//_}
    modified_path=$(echo $modified_path | sed 's/^_//;s/_$//')
    echo "${modified_path}"
}

# Compute the max number of GPUs needed across all tasks
max_gpus=1
for TASK in "${TASKS[@]}"; do
    if [[ $TASK == *mmlu* || $TASK == *agi_eval* || $TASK == *bbh* || $TASK == *gsm8k* || $TASK == *minerva_math_* || $TASK == *codex* || $TASK == *mbpp* || $TASK == *synthea* ]]; then
        gpus=4
    else
        gpus=1
    fi
    if [[ $gpus -gt $max_gpus ]]; then
        max_gpus=$gpus
    fi
done

echo "Running logits for ${#MODELS[@]} models and ${#TASKS[@]} tasks..."
echo "Mode: $(if $USE_GANTRY; then echo 'gantry'; else echo 'local'; fi)"
echo "Models: ${MODELS[@]}"
echo "Tasks: ${TASKS[@]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
if $USE_GANTRY; then
    echo "Cluster: $CLUSTER"
fi
echo ""

FAILED_MODELS=()
NUM_PASSED=0

for MODEL_NAME in "${MODELS[@]}"; do
    model=$(get_checkpoint_name $MODEL_NAME)
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"

    gpus=$max_gpus
    batch_size=$BATCH_SIZE
    if [[ $MODEL_NAME == *"1b35b"* ]]; then
        batch_size=$((batch_size / 4))
        gpus=$((gpus * 2))
    fi

    safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g')
    job_name="logits-${safe_model_name}"

    echo "Processing model: $MODEL_NAME"
    echo "  Output dir: $OUTPUT_DIR"
    echo "  GPUs: $gpus"
    echo "  Batch size: $batch_size"

    CMD="PYTHONPATH=. python -u src/scripts/eval/launch_logits.py \
    --model ${MODEL_DIR}/${MODEL_NAME} \
    --task ${TASKS[*]} \
    --eval-dir ${OUTPUT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${batch_size} \
    --gpus ${gpus}"

    if $USE_GANTRY; then
        gantry run \
            --name $job_name \
            --weka oe-training-default:/weka/oe-training-default \
            --install "uv pip install -e \".[all]\"" \
            --budget ai2/oceo \
            --workspace ai2/flex2 \
            --cluster $CLUSTER \
            --priority urgent \
            --allow-dirty \
            --gpus $gpus \
            --env-secret HF_TOKEN=RYAN_HF_TOKEN \
            --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
            --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
            -- \
            bash -c "$CMD"

        echo "Launched gantry job: $job_name"
    else
        echo "=== Computing logits for model: ${MODEL_NAME} ==="
        if bash -c "$CMD"; then
            NUM_PASSED=$((NUM_PASSED + 1))
            echo "=== Done: ${MODEL_NAME} ==="
        else
            echo "=== FAILED: ${MODEL_NAME} ==="
            FAILED_MODELS+=("${MODEL_NAME}")
        fi
    fi

    echo "----------------------------------------"
done

if $USE_GANTRY; then
    echo ""
    echo "All ${#MODELS[@]} gantry jobs launched!"
    echo "Check the beaker dashboard for job status."
else
    echo ""
    echo "========================================"
    echo "           LOGITS SUMMARY"
    echo "========================================"
    echo "Passed: ${NUM_PASSED}/${#MODELS[@]} models"
    echo "Failed: ${#FAILED_MODELS[@]}/${#MODELS[@]} models"
    if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
        echo ""
        echo "Failed models:"
        for m in "${FAILED_MODELS[@]}"; do
            echo "  - ${m}"
        done
        exit 1
    fi
fi
