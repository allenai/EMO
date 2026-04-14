#!/bin/bash
#
# Eval-only launcher for extension/continual-pretraining checkpoints.
#
# Strip-down of scripts/ryanwang/pruning_hf/launch_pruning_hf.sh — no pruning,
# no finetuning. For each (MODEL, TASK) pair, launches a beaker job that runs
# `python -m src.scripts.eval.launch_eval` on the HF checkpoint and pushes
# results to s3://ai2-sewonm/ryanwang/extension_evals_0414/.
#
# Task strings are passed through as-is (no train/val/test-split splitting).
# Task library lives in src/scripts/eval/tasks.py + upstream olmes.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE
S3_BASE=s3://ai2-sewonm/ryanwang/extension_evals_0414
CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf

# Models (relative to ${BASE_DIR}/models/ — all paths must end in -hf).
MODELS=(
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419_ct-m8_lb0/step2385-hf"
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419_ct-math_8/step2385-hf"
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238_ct-m8_lb0_wd/step2385-hf"
)

# Tasks — pass-through strings. Uncomment the ones you want to run.
TASK_GROUPS_LIST=(
    # MC9 (test split, rc, olmes default)
    "arc_easy:rc_test::olmes"
    "arc_challenge:rc_test::olmes"
    "boolq:rc_test::olmes"
    "csqa:rc_test::olmes"
    "hellaswag:rc_test::olmes"
    "openbookqa:rc_test::olmes"
    "piqa:rc_test::olmes"
    "socialiqa:rc_test::olmes"
    "winogrande:rc_test::olmes"

    # QA / generation
    "squad::olmes"
    "triviaqa::olmes"

    # Math
    "gsm8k::olmes"
    "gsm8k_generation_0shot:test::olmes"
    "basic_skills::olmes"
    "minerva_math_500::olmes"

    # Code
    "mbpp:3shot:bpb::none"
    "codex_humaneval:3shot:bpb::none"
)

echo "Launching extension evals for ${#MODELS[@]} models and ${#TASK_GROUPS_LIST[@]} tasks..."
echo "Models: ${MODELS[@]}"
echo "S3 base: ${S3_BASE}"
echo "Cluster: $CLUSTER"
echo ""

# ---------------------------------------------------------------------------
# Launch loop
# ---------------------------------------------------------------------------
for MODEL in "${MODELS[@]}"; do
    echo "Processing model: ${MODEL}"

    # Sanitize model name for use in job names / S3 prefixes
    stringified_model=$(echo $MODEL | sed 's/[^a-zA-Z0-9_-]//g')

    for TASK in "${TASK_GROUPS_LIST[@]}"; do
        # Per-task GPU / batch-size heuristics (mirroring the pruning_hf script's tuning)
        batch_size=16
        gpus=1
        if [[ $TASK == *mmlu* || $TASK == *agi_eval* || $TASK == *bbh* || $TASK == *gsm8k* || $TASK == *minerva_math_* || $TASK == *codex* || $TASK == *mbpp* ]]; then
            gpus=4
        fi
        if [[ $TASK == *"cot"* || $TASK == *"minerva_math_"* || $TASK == *"mbpp"* || $TASK == *"bigcodebench"* || $TASK == *"boolq"* || $TASK == *"drop"* ]]; then
            batch_size=$((batch_size / 4))
        fi

        # Sanitize task name for job name / path components
        safe_task=$(echo "$TASK" | sed 's/[^a-zA-Z0-9_-]//g')
        relative_dir="${stringified_model}/${safe_task}"
        safe_relative_dir=$(printf '%s' "$relative_dir" | sed 's/[^a-zA-Z0-9_-]//g' | tail -c 100)
        job_name="eval-${safe_relative_dir}"

        s3_output_dir="${S3_BASE}/${relative_dir}"

        # Clean any previous results for this (model, task) combo so re-runs don't
        # interleave stale metrics. `aws s3 rm` on a missing prefix is a no-op.
        echo "  Cleaning stale S3 results: ${s3_output_dir}/"
        aws s3 rm --recursive --quiet "${s3_output_dir}/" || true

        echo "  Model: ${BASE_DIR}/models/${MODEL}"
        echo "  Task: ${TASK}"
        echo "  GPUs: $gpus"
        echo "  Batch size: $batch_size"
        echo "  S3 out: ${s3_output_dir}"
        echo "  Job name: $job_name"

        python -m olmo_core.launch.beaker \
            --name "$job_name" \
            --gpus "$gpus" \
            --nodes 1 \
            --weka=oe-training-default \
            --shared-filesystem \
            --workspace ai2/flex2 \
            --cluster ai2/jupiter \
            --beaker-image tylerr/olmo-core-tch280cu128-2025-11-25 \
            --preemptible \
            --allow-dirty \
            --priority urgent \
            --no-follow \
            --no-torchrun \
            --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
            -- bash -c "export PYTHONPATH=\"\$(pwd)/src:\${PYTHONPATH}\" && \
                export HF_DATASETS_OFFLINE=0 && \
                python -m src.scripts.eval.launch_eval \
                    --model '${BASE_DIR}/models/${MODEL}' \
                    --model-type ${model_type} \
                    --task '${TASK}' \
                    --remote-output-dir '${s3_output_dir}' \
                    --batch-size ${batch_size} \
                    --gpus ${gpus}
                "

        echo "Launched eval for model: $MODEL, task: $TASK"
        echo "----------------------------------------"
    done

    echo "Completed all tasks for model: $MODEL"
    echo "========================================"
done

echo "All beaker evals have been launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASK_GROUPS_LIST[@]}))"
echo "Check the beaker dashboard for job status."
