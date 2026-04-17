#!/bin/bash
# Launch evals using Akshita's task set (core9 RC, squad, triviaqa, gsm8k, basic_skills, code BPB).
# Uses the same eval framework as scripts/akshitab/extension-experiments/launch_beaker_evals.sh

MODEL_DIR=""
MODELS=(
    # Merged + router trained (1B tokens)
    # "/weka/oe-training-default/kevinf/extension-experiments/rt-merged_math_code_croissant_train_act_1B_lr_4e-4/step239-hf"

    # Merged with router norm matching (no extra training)
    "/weka/oe-training-default/kevinf/extension-experiments/merged-math-code-croissant-train-act-norm-matched-hf"
)

BASE_OUTPUT_DIR="s3://ai2-kevinf/FlexMoE/evals/extensions"
BATCH_SIZE=16
CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf

TASK_GROUPS_LIST=(
    "arc_easy|arc_easy:rc_test::olmes"
    "arc_challenge|arc_challenge:rc_test::olmes"
    "boolq|boolq:rc_test::olmes"
    "csqa|csqa:rc_test::olmes"
    "hellaswag|hellaswag:rc_test::olmes"
    "openbookqa|openbookqa:rc_test::olmes"
    "piqa|piqa:rc_test::olmes"
    "socialiqa|socialiqa:rc_test::olmes"
    "winogrande|winogrande:rc_test::olmes"
    "squad|squad::olmes"
    "triviaqa|triviaqa::olmes"
    "gsm8k::olmes"
    "gsm8k_generation|gsm8k_generation_0shot:test::olmes"
    "basic_skills::olmes"
    "mbpp:3shot:bpb::none"
    "codex_humaneval:3shot:bpb::none"
)

function get_checkpoint_name {
    local path=$1
    local step_dir=$(basename "$path")
    local run_name=$(basename "$(dirname "$path")")
    echo "${run_name}_${step_dir}"
}

echo "Launching evals for ${#MODELS[@]} models and ${#TASK_GROUPS_LIST[@]} task groups..."

for MODEL_NAME in "${MODELS[@]}"; do
    echo "Processing model: $MODEL_NAME"
    model=$(get_checkpoint_name $MODEL_NAME)
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"

    for entry in "${TASK_GROUPS_LIST[@]}"; do
        GROUP_NAME="${entry%%|*}"
        TASK="${entry#*|}"

        if [[ $TASK == *"cot"* || $TASK == *"minerva_math_"* || $TASK == *"mbpp"* || $TASK == *"bigcodebench"* || $TASK == *"ruler"* || $TASK == *"sciriff"* || $TASK == *"boolq"* || $TASK == *"drop"* ]]; then
            batch_size=$((BATCH_SIZE / 4))
        else
            batch_size=$BATCH_SIZE
        fi

        if [[ $TASK == *mmlu* || $TASK == *agi_eval* || $TASK == *bbh* || $TASK == *gsm8k* || $TASK == *minerva_math_* || $TASK == *codex* || $TASK == *mbpp* || $TASK == *synthea* ]]; then
            gpus=4
        else
            gpus=1
        fi

        safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-80)
        safe_group_name=$(echo $GROUP_NAME | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-40)
        job_name="eval-${safe_group_name}"

        echo "  Task: $TASK (batch=$batch_size, gpus=$gpus)"

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
            --preemptible \
            --env-secret GITHUB_TOKEN=KEVINF_GITHUB_TOKEN \
            --env-secret HF_TOKEN=KEVINF_HF_TOKEN \
            --env-secret AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID \
            --env-secret AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY \
            -- \
            bash -c "PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
                --model ${MODEL_NAME} \
                --model-type hf \
                --task $TASK \
                --remote-output-dir $OUTPUT_DIR \
                --batch-size $batch_size \
                --gpus $gpus \
                "

        echo "  Launched: $GROUP_NAME"
        echo "  ----------------------------------------"
    done
done

echo "All evals launched!"
