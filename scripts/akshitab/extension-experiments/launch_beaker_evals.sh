#!/bin/bash

# Configuration

# MODEL_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models
MODEL_DIR=/weka/oe-training-default/akshitab/FlexMoE/models

MODELS=(
    # moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995-hf
    # moe1b14b_129experts_1trained_math_init_random_expert_5B/step1193-hf
    # moe1b14b_129experts_1trained_math_init_average_5B/step1193-hf
    # moe1b14b_129experts_1trained_math_init_top2_average_5B/step1193-hf
    # moe1b14b_129experts_1trained_math_init_top2_average_10B/step2385-hf
    # moe1b14b_129experts_1trained_math_init_top2_average_20B/step4769-hf

    # moe1b14b_130experts_2trained_math_init_average_5B/step1193-hf
    # moe1b14b_130experts_2trained_math_init_average_noise_5B/step1193-hf
    # moe1b14b_130experts_2trained_math_init_average_noise_10perc_5B/step1193-hf
    # moe1b14b_130experts_2trained_math_init_top2_5B/step1193-hf
    # moe1b14b_130experts_2trained_math_init_top2_average_noise_5B/step1193-hf
    # moe1b14b_128experts_1trained_math_5B/step1193-hf
    # moe1b14b_132experts_4trained_math_init_average_noise_10perc_5B/step1193-hf
    # moe1b14b_136experts_8trained_math_init_average_noise_10perc_5B/step1193-hf
    # twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995-hf

    # moe1b14b_129experts_1trained_math_init_average_10B/step2385-hf
    # moe1b14b_129experts_1trained_math_init_average_20B/step4769-hf
    # moe1b14b_130experts_2trained_math_init_average_noise_10perc_10B/step2385-hf
    # moe1b14b_132experts_4trained_math_init_average_noise_10perc_10B/step2385-hf
    # moe1b14b_136experts_8trained_math_init_average_noise_10perc_10B/step2385-hf

    # moe1b14b_130experts_2trained_math_init_average_noise_10perc_20B/step4769-hf
    # moe1b14b_132experts_4trained_math_init_average_noise_10perc_20B/step4769-hf
    # moe1b14b_136experts_8trained_math_init_average_noise_10perc_20B/step4769-hf

    # twolevelbatchlb-32_1b14b_129experts_1trained_math_init_top2_5B/step1193-hf
    # twolevelbatchlb-32_1b14b_129experts_1trained_math_init_average_5B/step1193-hf

    # freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_5B_lr_4e-3/step1193-hf
    # freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_5B_lr_4e-4/step1193-hf
    # freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_5B_lr_4e-5/step1193-hf

    # freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_10B_lr_4e-4/step2385-hf
    # freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_10B_lr_4e-3/step2385-hf
    # freeze-fix-moe1b14b_129experts_1trained_math_init_average_5B_lr_4e-4/step1193-hf

    # freeze-fix-moe1b14b_129experts_1trained_math_init_random_expert_5B_lr_4e-4/step1193-hf
    # freeze-fix-moe1b14b_130experts_2trained_math_init_average_noise_10pc_5B_lr_4e-4/step1193-hf
    # freeze-fix-moe1b14b_experts_2trained_math_init_top2_5B_lr_4e-4/step1193-hf
    # freeze-fix-twolevel_129experts_1trained_math_init_average_5B_lr_4e-4/step1193-hf
    # freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_20B_lr_4e-4/step4769-hf
    # freeze-fix-twolevel_129experts_1trained_math_init_top2_average_5B_lr_4e-4/step1193-hf

    # freeze-fix-moe1b14b_130experts_2trained_math_init_top2_average_noise_5B_lr_4e-4/step1193-hf
    # freeze-fix-moe1b14b_130experts_2trained_math_init_top2_average_noise_10B_lr_4e-4/step2385-hf
    # freeze-fix-moe1b14b_130experts_2trained_math_init_top2_average_noise_20B_lr_4e-4/step4769-hf

    # freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_5B_lr_4e-4/step1193-hf
    # freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_10B_lr_4e-4/step2385-hf

    # freeze-fix-moe1b14b_136experts_8trained_math_init_top2_average_noise_5B_lr_4e-4/step1193-hf
    # freeze-fix-moe1b14b_136experts_8trained_math_init_top2_average_noise_10B_lr_4e-4/step2385-hf

    # freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_20B_lr_4e-4/step4769-hf
    # freeze-fix-twolevel_130experts_2trained_math_init_average_noise_10pc_5B_lr_4e-4/step1193-hf
    # freeze-fix-twolevel_130experts_2trained_math_init_top2_average_noise_5B_lr_4e-4/step1193-hf
    # freeze-fix-moe1b14b_136experts_8trained_math_init_top2_average_noise_20B_lr_4e-4/step4769-hf

    # freeze-fix-moe1b14b_132experts_4trained_math_init_average_noise_10pc_10B_lr_4e-4/step2385-hf
    # freeze-fix-moe1b14b_132experts_4trained_starcoder_init_average_noise_10pc_10B_lr_4e-4/step2385-hf
    # ff-moe1b14b_132experts_4trained_starcoder_init_top2_average_noise_10B_lr_4e-4/step2385-hf
    # ff-moe_1b14b_128base_4math_10B_4code_init_top2_starcoder_average_noise_10B_lr_4e-4/step2385-hf

    # merged_moe_1b14b_128base_4math_10B_4starcoder_10B_init_top2_average_noise-hf
    # ff-moe_1b14b_128base_4math_10B_4code_init_top2_code_mix_average_noise_10B_lr_4e-4/step2385-hf
    # freeze-fix-moe1b14b_132experts_4trained_code_mix_init_average_noise_10pc_10B_lr_4e-4/step2385-hf
    # ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4/step2385-hf
    # merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf

    # ff-moe1b14b_132experts_4trained_croissant_init_average_noise_10pc_10B_lr_4e-4/step2385-hf

    # extensions/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc-hf
    # extensions/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_code_average_noise-hf

    # rt-merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise_1B_lr_4e-4/step239-hf

    merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf

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
    "hellaswag|hellaswag:rc_test::olmes"
    "openbookqa|openbookqa:rc_test::olmes"
    "piqa|piqa:rc_test::olmes"
    "socialiqa|socialiqa:rc_test::olmes"
    "winogrande|winogrande:rc_test::olmes"

    # math tasks
    "gsm8k::olmes"
    "gsm8k_generation|gsm8k_generation:test_0shot::olmes"
    "minerva_math_500::olmes"
    "basic_skills::olmes"

    # code tasks
    "mbpp:3shot:bpb::none"
    "codex_humaneval:3shot:bpb::none"
)

# Function to get checkpoint name (matching the original script)
function get_checkpoint_name {
    local path=$1
    local split_path=${path#*OLMo2-7B-}
    local modified_path=${split_path//\//_}
    modified_path=$(echo $modified_path | sed 's/^_//;s/_$//')
    echo "${modified_path}"
}

echo "Launching beaker evaluations for ${#MODELS[@]} models and ${#TASK_GROUPS[@]} task groups..."
echo "Models: ${MODELS[@]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Cluster: $CLUSTER"
echo ""

# Launch evaluation for each model and task combination
for MODEL_NAME in "${MODELS[@]}"; do
    echo "Processing model: $MODEL_NAME"

    # For setting the output_dir (matching original script logic)
#    if [[ $MODEL_NAME == "/"* ]]; then
#        # internal model
#        model=$(get_checkpoint_name $MODEL_NAME)
#    else
#        # HF model
#        model=$(echo $MODEL_NAME | cut -d'/' -f2)
#    fi
    model=$(get_checkpoint_name $MODEL_NAME)

    echo "Model name for output dir: $model"

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"

    for entry in "${TASK_GROUPS_LIST[@]}"; do
        GROUP_NAME="${entry%%|*}"                # text before '|'
        TASK="${entry#*|}"            # text after '|'

        # Batch size adjustment (matching original script)
        if [[ $TASK == *"cot"* || $TASK == *"minerva_math_"* || $TASK == *"mbpp"* || $TASK == *"bigcodebench"* || $TASK == *"ruler"* || $TASK == *"sciriff"* || $TASK == *"boolq"* || $TASK == *"drop"* ]]; then
            batch_size=$((BATCH_SIZE / 4))
        else
            batch_size=$BATCH_SIZE
        fi

        # adjust number of gpus requested if its mmlu, agi_eval, bbh, gsm8k, minerva, codex, mbpp
        if [[ $TASK == *mmlu* || $TASK == *agi_eval* || $TASK == *bbh* || $TASK == *gsm8k* || $TASK == *minerva_math_* || $TASK == *codex* || $TASK == *mbpp* || $TASK == *synthea* ]]; then
            gpus=4
        else
            gpus=1
        fi

        # if the model is a 35b model, further reduce batch size by half
        if [[ $MODEL_NAME == *"1b35b"* ]]; then
            batch_size=$((batch_size / 4))
            gpus=$((gpus * 2))
        fi

        # Create a shorter, valid job name
        # Remove invalid characters and truncate long names
        safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g')
        safe_group_name=$(echo $GROUP_NAME | sed 's/[^a-zA-Z0-9_-]//g')

        safe_model_name=${safe_model_name/freeze-fix/ff}
        # job_name="eval-${safe_model_name}-${safe_group_name}"
        job_name="eval-${safe_group_name}"

        echo "  Model name: $model"
        echo "  Output dir: $OUTPUT_DIR"
        echo "  GPUs: $gpus"
        echo "  Batch size: $batch_size"
        echo "  Job name: $job_name"

#        PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
#                --model "${MODEL_DIR}/${MODEL_NAME}" \
#                --model-type hf \
#                --task $TASK \
#                --output-dir $OUTPUT_DIR \
#                --batch-size $batch_size \
#                --gpus $gpus \

        gantry run \
            --name $job_name \
            --weka oe-training-default:/weka/oe-training-default \
            --install "pip install -e \".[all]\"" \
            --budget ai2/oceo \
            --workspace ai2/flex2 \
            --cluster $CLUSTER \
            --priority urgent \
            --allow-dirty \
            --gpus $gpus \
            --env-secret HF_TOKEN=AKSHITAB_HF_TOKEN \
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

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASK_GROUPS_LIST[@]}))"
echo "Check the beaker dashboard for job status."