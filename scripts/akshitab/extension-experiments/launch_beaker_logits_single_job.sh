#!/bin/bash

# Usage:
#   ./launch_beaker_logits_single_job.sh              # run locally
#   ./launch_beaker_logits_single_job.sh --gantry      # submit as a gantry job

# Parse flags
USE_GANTRY=false
for arg in "$@"; do
    case $arg in
        --gantry) USE_GANTRY=true ;;
    esac
done

# Configuration

# MODEL_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models
MODEL_DIR=/weka/oe-training-default/akshitab/FlexMoE/models

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

BASE_OUTPUT_DIR="s3://ai2-sewonm/akshitab/mose/evals/extensions"
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
#  "hellaswag|hellaswag:rc_test::olmes"
 "openbookqa|openbookqa:rc_test::olmes"
 "piqa|piqa:rc_test::olmes"
 "socialiqa|socialiqa:rc_test::olmes"
 "winogrande|winogrande:rc_test::olmes"

#     # mbpp:3shot:bpb::none
#     # codex_humaneval:3shot:bpb::none

    "gsm8k::olmes"
#     # # "gsm8k_generation|gsm8k_generation:test_0shot::olmes"
#     # # "minerva_math_500::olmes"
    "mbpp"
    "codex_humaneval"

    "squad|squad::olmes"
)

# Function to get checkpoint name (matching the original script)
function get_checkpoint_name {
    local path=$1
    local split_path=${path#*OLMo2-7B-}
    local modified_path=${split_path//\//_}
    modified_path=$(echo $modified_path | sed 's/^_//;s/_$//')
    echo "${modified_path}"
}

# Compute the max number of GPUs needed across all (model, task) combos
max_gpus=1
for MODEL_NAME in "${MODELS[@]}"; do
    for entry in "${TASK_GROUPS_LIST[@]}"; do
        TASK="${entry#*|}"

        if [[ $TASK == *mmlu* || $TASK == *agi_eval* || $TASK == *bbh* || $TASK == *gsm8k* || $TASK == *minerva_math_* || $TASK == *codex* || $TASK == *mbpp* || $TASK == *synthea* ]]; then
            gpus=4
        else
            gpus=1
        fi

        if [[ $MODEL_NAME == *"1b35b"* ]]; then
            gpus=$((gpus * 2))
        fi

        if [[ $gpus -gt $max_gpus ]]; then
            max_gpus=$gpus
        fi
    done
done

echo "Running logits for ${#MODELS[@]} models and ${#TASK_GROUPS_LIST[@]} task groups..."
echo "Mode: $(if $USE_GANTRY; then echo 'gantry'; else echo 'local'; fi)"
echo "Models: ${MODELS[@]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
if $USE_GANTRY; then
    echo "Cluster: $CLUSTER"
    echo "Max GPUs needed: $max_gpus"
fi
echo ""

# Build the eval script with all commands
EVAL_SCRIPT="FAILED_JOBS=()"$'\n'
EVAL_SCRIPT+="NUM_PASSED=0"$'\n'

for MODEL_NAME in "${MODELS[@]}"; do
    model=$(get_checkpoint_name $MODEL_NAME)
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"

    for entry in "${TASK_GROUPS_LIST[@]}"; do
        GROUP_NAME="${entry%%|*}"
        TASK="${entry#*|}"

        # Batch size adjustment
        if [[ $TASK == *"cot"* || $TASK == *"minerva_math_"* || $TASK == *"mbpp"* || $TASK == *"bigcodebench"* || $TASK == *"ruler"* || $TASK == *"sciriff"* || $TASK == *"boolq"* || $TASK == *"drop"* ]]; then
            batch_size=$((BATCH_SIZE / 4))
        else
            batch_size=$BATCH_SIZE
        fi

        # GPU adjustment
        if [[ $TASK == *mmlu* || $TASK == *agi_eval* || $TASK == *bbh* || $TASK == *gsm8k* || $TASK == *minerva_math_* || $TASK == *codex* || $TASK == *mbpp* || $TASK == *synthea* ]]; then
            gpus=4
        else
            gpus=1
        fi

        if [[ $MODEL_NAME == *"1b35b"* ]]; then
            batch_size=$((batch_size / 4))
            gpus=$((gpus * 2))
        fi

        JOB_LABEL="${MODEL_NAME} | ${GROUP_NAME}"

        EVAL_SCRIPT+="echo '=== Computing logits for model: ${MODEL_NAME}, task: ${GROUP_NAME} ==='"$'\n'
        EVAL_SCRIPT+="if PYTHONPATH=. python -u src/scripts/eval/launch_logits.py \
    --model ${MODEL_DIR}/${MODEL_NAME} \
    --task ${TASK} \
    --eval-dir ${OUTPUT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${batch_size} \
    --gpus ${gpus}; then"$'\n'
        EVAL_SCRIPT+="  NUM_PASSED=\$((NUM_PASSED + 1))"$'\n'
        EVAL_SCRIPT+="  echo '=== Done: ${MODEL_NAME}, task: ${GROUP_NAME} ==='"$'\n'
        EVAL_SCRIPT+="else"$'\n'
        EVAL_SCRIPT+="  echo '=== FAILED: ${MODEL_NAME}, task: ${GROUP_NAME} ==='"$'\n'
        EVAL_SCRIPT+="  FAILED_JOBS+=('${JOB_LABEL}')"$'\n'
        EVAL_SCRIPT+="fi"$'\n'

        echo "  Added logits: model=$model, task=$GROUP_NAME, batch_size=$batch_size, gpus=$gpus"
    done
done

# Add summary to the script
EVAL_SCRIPT+="echo ''"$'\n'
EVAL_SCRIPT+="echo '========================================'"$'\n'
EVAL_SCRIPT+="echo '           LOGITS SUMMARY'"$'\n'
EVAL_SCRIPT+="echo '========================================'"$'\n'
EVAL_SCRIPT+="echo \"Passed: \${NUM_PASSED}\""$'\n'
EVAL_SCRIPT+="echo \"Failed: \${#FAILED_JOBS[@]}\""$'\n'
EVAL_SCRIPT+="if [[ \${#FAILED_JOBS[@]} -gt 0 ]]; then"$'\n'
EVAL_SCRIPT+="  echo ''"$'\n'
EVAL_SCRIPT+="  echo 'Failed jobs:'"$'\n'
EVAL_SCRIPT+="  for job in \"\${FAILED_JOBS[@]}\"; do"$'\n'
EVAL_SCRIPT+="    echo \"  - \${job}\""$'\n'
EVAL_SCRIPT+="  done"$'\n'
EVAL_SCRIPT+="  exit 1"$'\n'
EVAL_SCRIPT+="fi"$'\n'

echo ""

if $USE_GANTRY; then
    echo "Launching single gantry job with $max_gpus GPUs..."

    gantry run \
        --name "logits-all-models" \
        --weka oe-training-default:/weka/oe-training-default \
        --install "uv pip install -e \".[all]\"" \
        --budget ai2/oceo \
        --workspace ai2/flex2 \
        --cluster $CLUSTER \
        --priority urgent \
        --allow-dirty \
        --gpus $max_gpus \
        --env-secret HF_TOKEN=RYAN_HF_TOKEN \
        --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
        --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
        -- \
        bash -c "$EVAL_SCRIPT"

    echo "Single beaker logits job launched!"
    echo "Total logits runs inside the job: $((${#MODELS[@]} * ${#TASK_GROUPS_LIST[@]}))"
    echo "Check the beaker dashboard for job status."
else
    echo "Running logits locally..."
    bash -c "$EVAL_SCRIPT"
fi
