#!/bin/bash
#
# TEMPORARY one-off launcher: eval the small (pruned + finetuned) model only.
#
# Use case: you ran prune + finetune before the pipeline supported a separate
# small-eval step (or before phase=prune_finetune existed) and now you just
# want the small-model test-split eval pushed to S3 — without re-running prune
# or doing any merge.
#
# Mirrors the structure of launch_extensions_hf.sh: same MODELS list, same
# TASK_GROUPS_LIST, same relative_dir naming convention. Per (model, task) it:
#   1) locates ${BASE_DIR}/extension_evals_hf/${relative_dir}/finetuned_model/checkpoint-*
#   2) picks the largest-numbered checkpoint
#   3) launches a beaker job that runs `python -m src.scripts.eval.launch_eval`
#      against that checkpoint with --task ${TASK}-pruned --pruned_split test
#   4) writes results to ${S3_BASE}/${relative_dir}/small/checkpoint-<N>/
#   5) does the per-subject MMLU loop too if TASK is an MMLU category/cluster
#
# Once you've run this once, you can delete this script — the equivalent step
# now lives inside hf_extension_with_pruning_layerwise.sh under phase=prune_finetune.

# Configuration (kept in sync with launch_extensions_hf.sh)
BASE_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE
#BASE_DIR="/root/phdbrainstorm/FlexMoE"
S3_BASE="s3://ai2-sewonm/ryanwang/extension_evals_hf_0426"

MODELS=(
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419/step250339-hf"
)

CLUSTER="ai2/jupiter-cirrascale-2"

# These four go into relative_dir construction. They must match what was used
# during the prune + finetune run that produced the artifacts on disk.
num_epochs=1
PRUNE_KEEP_K_VALUES=(32)
batch_size=32
lr=5e-5

# Optional suffixes (must match the prune+finetune run).
NUM_PRUNE_EXAMPLES=""
NUM_SHOTS_PRUNE=""
NUM_SHOTS_EVAL=""
# Freeze mode used during finetune; "none" means no suffix (= unfrozen run).
FREEZE_MODE="none"

TASK_GROUPS_LIST=(
  "gsm8k_generation_8shot_merged"
)

echo "Launching small-eval-only jobs for ${#MODELS[@]} models, ${#PRUNE_KEEP_K_VALUES[@]} keep-k values, and ${#TASK_GROUPS_LIST[@]} tasks..."
echo "Models: ${MODELS[@]}"
echo "S3 base: $S3_BASE"
echo ""

for MODEL in "${MODELS[@]}"; do
  for prune_keep_k in "${PRUNE_KEEP_K_VALUES[@]}"; do
    for TASK in "${TASK_GROUPS_LIST[@]}"; do
        # Per-task GPU + batch heuristics (mirror launch_extensions_hf.sh)
        eval_batch_size=32
        if [[ $TASK == *"history"* ]]; then
            eval_batch_size=4
        fi
        if [[ $TASK == *"gsm8k_generation_8shot"* ]]; then
            eval_batch_size=16
        fi

        gpus=4
        if [[ $TASK == *"mmlu_history"* || $TASK == *"gsm8k_generation_8shot"* || $TASK == *"drop_merged"* || $TASK == *"squad_merged"* ]]; then
            gpus=8
        fi

        stringified_model=$(echo $MODEL | sed 's/[^a-zA-Z0-9_-]//g')

        relative_dir="${stringified_model}/${TASK}_keepk_${prune_keep_k}_bs-${batch_size}_lr-${lr}_epoch-${num_epochs}_prunemode-layerwise"
        if [ -n "$NUM_PRUNE_EXAMPLES" ]; then
            relative_dir="${relative_dir}_nprune-${NUM_PRUNE_EXAMPLES}"
        fi
        if [ -n "$NUM_SHOTS_PRUNE" ]; then
            relative_dir="${relative_dir}_pshots-${NUM_SHOTS_PRUNE}"
        fi
        if [ -n "$NUM_SHOTS_EVAL" ]; then
            relative_dir="${relative_dir}_eshots-${NUM_SHOTS_EVAL}"
        fi
        if [ "$FREEZE_MODE" != "none" ]; then
            relative_dir="${relative_dir}_fz-${FREEZE_MODE}"
        fi

        safe_relative_dir=$(printf '%s' "$relative_dir" | sed 's/[^a-zA-Z0-9_-]//g' | tail -c 100)
        job_name="ext-small-${safe_relative_dir:0:90}"

        NSHOTS_EVAL_FLAG=""
        if [ -n "$NUM_SHOTS_EVAL" ]; then
            NSHOTS_EVAL_FLAG="--num-shots ${NUM_SHOTS_EVAL}"
        fi

        # Wipe any stale small/ S3 results for this (model, task, keep-k).
        s3_clean_prefix="${S3_BASE}/${relative_dir}/small/"
        echo "  Cleaning stale S3 results: ${s3_clean_prefix}"
        aws s3 rm --recursive --quiet "${s3_clean_prefix}" || true

        echo "  Model:        ${BASE_DIR}/models/${MODEL}"
        echo "  Task:         ${TASK}"
        echo "  Relative dir: ${relative_dir}"
        echo "  GPUs:         $gpus"
        echo "  Job name:     $job_name"

#        bash -c "
#            export PYTHONPATH=\"\$(pwd)/src:\${PYTHONPATH}\"
#            export HF_DATASETS_OFFLINE=0
#            export HF_HUB_OFFLINE=0
#            OUTPUT_DIR=\"${BASE_DIR}/extension_evals_hf/${relative_dir}\"
#            final_checkpoint=\$(ls -d \"\$OUTPUT_DIR/finetuned_model\"/checkpoint-*/ | sed 's:/\$::' | awk -F- '{print \$NF, \$0}' | sort -n | tail -1 | awk '{print \$2}')
#            final_checkpoint_num=\$(basename \"\$final_checkpoint\" | sed 's/checkpoint-//')
#            echo \"Evaluating final checkpoint: \$final_checkpoint (step \$final_checkpoint_num)\"
#            python -m src.scripts.eval.launch_eval \
#                --model \"\$final_checkpoint\" \
#                --model-type hf \
#                --task \"${TASK}-pruned\" \
#                --pruned_split \"test\" \
#                --remote-output-dir \"${S3_BASE}/${relative_dir}/small/checkpoint-\${final_checkpoint_num}\" \
#                --batch-size ${eval_batch_size} \
#                --gpus ${gpus} \
#                ${NSHOTS_EVAL_FLAG}
#
#            # Per-subject MMLU evals if applicable
#            MMLU_SUBJECTS=\$(python -m src.scripts.eval.get_mmlu_subjects \"${TASK}\" 2>/dev/null | grep -v '^Warning:' || true)
#            if [ -n \"\$MMLU_SUBJECTS\" ]; then
#                if [[ \"${TASK}\" == mmlu_merged_* ]]; then
#                    SUBJECT_TASK_PREFIX=\"mmlu_merged_\"
#                else
#                    SUBJECT_TASK_PREFIX=\"mmlu_\"
#                fi
#                while IFS= read -r subject; do
#                    SUBJECT_BATCH_SIZE=32
#                    [[ \$subject == *history* ]] && SUBJECT_BATCH_SIZE=4
#                    python -m src.scripts.eval.launch_eval \
#                        --model \"\$final_checkpoint\" \
#                        --model-type hf \
#                        --task \"\${SUBJECT_TASK_PREFIX}\${subject}-pruned\" \
#                        --pruned_split \"test\" \
#                        --remote-output-dir \"${S3_BASE}/${relative_dir}/small/checkpoint-\${final_checkpoint_num}/per_subject/\${subject}\" \
#                        --batch-size \$SUBJECT_BATCH_SIZE \
#                        --gpus ${gpus} \
#                        ${NSHOTS_EVAL_FLAG}
#                done <<< \"\$MMLU_SUBJECTS\"
#            fi
#        "

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
            -- bash -c "
                export PYTHONPATH=\"\$(pwd)/src:\${PYTHONPATH}\"
                export HF_DATASETS_OFFLINE=0
                export HF_HUB_OFFLINE=0
                OUTPUT_DIR=\"${BASE_DIR}/extension_evals_hf/${relative_dir}\"
                final_checkpoint=\$(ls -d \"\$OUTPUT_DIR/finetuned_model\"/checkpoint-*/ | sed 's:/\$::' | awk -F- '{print \$NF, \$0}' | sort -n | tail -1 | awk '{print \$2}')
                final_checkpoint_num=\$(basename \"\$final_checkpoint\" | sed 's/checkpoint-//')
                echo \"Evaluating final checkpoint: \$final_checkpoint (step \$final_checkpoint_num)\"
                python -m src.scripts.eval.launch_eval \
                    --model \"\$final_checkpoint\" \
                    --model-type hf \
                    --task \"${TASK}-pruned\" \
                    --pruned_split \"test\" \
                    --remote-output-dir \"${S3_BASE}/${relative_dir}/small/checkpoint-\${final_checkpoint_num}\" \
                    --batch-size ${eval_batch_size} \
                    --gpus ${gpus} \
                    ${NSHOTS_EVAL_FLAG}

                MMLU_SUBJECTS=\$(python -m src.scripts.eval.get_mmlu_subjects \"${TASK}\" 2>/dev/null | grep -v '^Warning:' || true)
                if [ -n \"\$MMLU_SUBJECTS\" ]; then
                    if [[ \"${TASK}\" == mmlu_merged_* ]]; then
                        SUBJECT_TASK_PREFIX=\"mmlu_merged_\"
                    else
                        SUBJECT_TASK_PREFIX=\"mmlu_\"
                    fi
                    while IFS= read -r subject; do
                        SUBJECT_BATCH_SIZE=32
                        [[ \$subject == *history* ]] && SUBJECT_BATCH_SIZE=4
                        python -m src.scripts.eval.launch_eval \
                            --model \"\$final_checkpoint\" \
                            --model-type hf \
                            --task \"\${SUBJECT_TASK_PREFIX}\${subject}-pruned\" \
                            --pruned_split \"test\" \
                            --remote-output-dir \"${S3_BASE}/${relative_dir}/small/checkpoint-\${final_checkpoint_num}/per_subject/\${subject}\" \
                            --batch-size \$SUBJECT_BATCH_SIZE \
                            --gpus ${gpus} \
                            ${NSHOTS_EVAL_FLAG}
                    done <<< \"\$MMLU_SUBJECTS\"
                fi
            "

        echo "Launched: $job_name"
        echo "----------------------------------------"
    done
  done
done

echo "All small-eval jobs launched. Total jobs: $((${#MODELS[@]} * ${#PRUNE_KEEP_K_VALUES[@]} * ${#TASK_GROUPS_LIST[@]}))"
echo "Check the beaker dashboard for job status."
