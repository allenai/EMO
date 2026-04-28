#!/bin/bash
#
# Extensions-HF launcher: parallel of launch_pruning_hf.sh.
#
# For each (MODEL, prune_keep_k, TASK) tuple, launches a beaker job that runs
# scripts/ryanwang/extensions_hf/hf_extension_with_pruning_layerwise.sh, which:
#   1) prunes (greedy layerwise) on TASK validation
#   2) finetunes the small pruned model on TASK train
#   3) evals the small finetuned model (final checkpoint only)
#   4) merges the trained experts back into the original full model
#   5) evals the merged full-sized model
#   6) (optional) per-subject MMLU evals on both small + merged
#
# v0 defaults:
#   - layerwise pruning only (random / global / easy_ep / layerwise_variable not yet wired)
#   - finetune unfrozen
#   - merge copies only routable expert MLPs
#   - --num-checkpoints 1
#   - S3 prefix: extension_evals_hf_0426/

# Configuration
BASE_DIR=/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE
#BASE_DIR="/root/phdbrainstorm/FlexMoE"
S3_BASE="s3://ai2-sewonm/ryanwang/extension_evals_hf_0426"
MODELS=(
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419/step250339-hf"
#    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf"
#    "dense_1b_lr-4e-3_0213/step30995-hf"
#    "moereducedp512sharedexp1_1b4b_lr-4e-3_lb-1e-1_0308/step30995-hf"
#    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995-hf"
#    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419/step250339-hf"
#    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_twolevel_randpool-8-128_from_step238419/step250339-hf"

#    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313/step238419-hf"
#    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322/step238419-hf"


#    "moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf"

#    "moereducedp256_1b4b_lr-4e-3_lb-1e-1_0212/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp2randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0305/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp1densefirst-32_1b14b_lr-4e-3_lb-1e-1_0227/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-2_0213/step30995-hf"

#    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995-hf"
#    "moe_1b14b_128experts_lb-1e-1_1217/step30995-hf"

#    "dense_1b_olmoe-mix_prenorm_noqknorm_1123/step30995-hf"
#    "moe_1b4b_32experts_1224/step30995-hf"
#    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995-hf"
#    "twolevelbatchlb-32_1b14b_lr-4e-3_lb-1e-1_0119/step30995-hf"
#    "twolevelbatchlb-32_1b14b_lr-4e-3_lb-1e-2_0118/step30995-hf"
#    "twolevelbatchlbreducedp512-32_1b14b_lr-4e-3_lb-1e-2_0207/step30995-hf"
#    "twolevelbatchlbreducedp512-32_1b14b_lr-4e-3_lb-1e-1_0119/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp4c2-32_1b14b_lr-4e-3_lb-1e-1_sharelb-1e-1_0214/step30995-hf"
#    "twolevelbatchlbreducedp512sharedexp4c2-32_1b14b_lr-4e-3_lb-1e-2_sharelb-1e-2_0214/step30995-hf"

    )

CLUSTER="ai2/jupiter-cirrascale-2"
model_type=hf

num_epochs=1
#PRUNE_KEEP_K_VALUES=(8 16 32 64 128)
PRUNE_KEEP_K_VALUES=(8)
batch_size=32

# --- Pruning calibration-set size ---
# Leave empty to use the full validation pool for pruning (default).
# Set to an integer (e.g. 50) to subsample that many prompts (deterministic shuffle, seed=0).
# Set to "random" to bypass calibration entirely and randomly select experts
# (seed=0, mode-agnostic — ignores PRUNING_MODE). Output dir uses _prunemode-random.
NUM_PRUNE_EXAMPLES=""

# --- Shot-count overrides (two orthogonal knobs) ---
# Each var: empty ⇒ each task's default num_shots (e.g. mmlu_merged_* = 5-shot,
# gsm8k_generation_8shot_merged = 8-shot). Set to an integer to force that
# shot count for the corresponding stage.
#
#   NUM_SHOTS_PRUNE: pruning-calibration shots. Ignored for dense_1b / 1b4b
#     (no pruning stage) and NUM_PRUNE_EXAMPLES="random" (no calibration).
#   NUM_SHOTS_EVAL:  finetune + eval shots.
#
# Output dir suffix uses _pshots-{X} and/or _eshots-{Y}. Only stages that are
# actually overridden appear. Examples:
#   PRUNE="0" EVAL=""  → _pshots-0
#   PRUNE=""  EVAL="0" → _eshots-0
#   PRUNE="0" EVAL="0" → _pshots-0_eshots-0
NUM_SHOTS_PRUNE=""
NUM_SHOTS_EVAL=""

# --- Merge variants ---
# Comma-separated list of merge-back variants to produce + eval. Each name maps to a
# flag combination passed to merge_pruned_experts_back.py (see flags_for_variant in
# the worker script).
#
# Replace-mode (parent's params get overwritten with small's):
#   default       — only routable expert MLPs
#   shared        — routable + shared expert MLP
#   router        — routable + router rows
#   shared_router — routable + shared expert + router rows
#   non_moe       — routable + attention/norms/embed/lm_head (≈ full continual-pretrain)
#
# Average-mode (parent ← 0.5·parent + 0.5·small for the same params, --average):
#   default_avg, shared_avg, router_avg, shared_router_avg, non_moe
MERGE_VARIANTS="shared"

# --- Selective-finetune freeze pattern ---
# Forwarded to the worker (and from there to finetune.py --freeze-mode). The relative_dir
# (and thus S3 prefix + on-disk path) gets a _fz-<mode> suffix when not "none", so freeze
# variants don't clobber each other or the unfrozen baseline.
#   none                  — train everything (default; current behavior).
#   routed                — only routable expert MLPs trainable.
#   routed_shared         — routable + shared expert MLPs trainable; router frozen.
#   routed_shared_router  — routable + shared experts + router trainable.
FREEZE_MODE="routed_shared"

# --- Pipeline phase ---
# Lets you split a slow run across multiple beaker jobs:
#   full           — one beaker job per (model, task) doing prune+finetune+small-eval+
#                    per-variant merge+eval.
#   prune_finetune — one beaker job per (model, task) doing Steps 1-3 + small-model eval
#                    (+ small per-subject MMLU if the task is MMLU). Pruned + finetuned
#                    artifacts and small/ S3 results are left for a later phase.
#                    MERGE_VARIANTS is ignored.
#   merge_eval     — ONE beaker job per (model, task, variant) doing only the per-variant
#                    merge + eval (+ per-subject MMLU). Requires that prune_finetune (or
#                    full) has previously run with the same --base-dir + --relative-dir.
PHASE="merge_eval"

# Define grouped tasks
TASK_GROUPS_LIST=(
  # Merged variants for the MC9 + perplexity tasks (pruning + finetuning share data)
#  "arc_easy_merged"
#  "arc_challenge_merged"
#  "boolq_merged"
#  "hellaswag_merged"
#  "csqa_merged"
#  "openbookqa_merged"
#  "piqa_merged"
#  "socialiqa_merged"
#  "winogrande_merged"

  # GSM8K generation merged variants (pruning + finetuning share data)
#  "gsm8k_generation_0shot_merged"
  "gsm8k_generation_8shot_merged"

  # SQuAD merged variants
#  "squad_merged"
#  "squad_0shot_merged"

  # CoQA merged variant (matches coqa::olmes)
#  "coqa_merged"

  # NaturalQS, TriviaQA, DROP merged variants
#  "naturalqs_merged"
#  "triviaqa_merged"
#  "drop_merged"

  # MMLU 17-category merged variants (pruning + finetuning share data)
#  "mmlu_merged_biology"
#  "mmlu_merged_business"
#  "mmlu_merged_chemistry"
#  "mmlu_merged_computer_science"
#  "mmlu_merged_culture"
#  "mmlu_merged_economics"
#  "mmlu_merged_engineering"
#  "mmlu_merged_geography"
#  "mmlu_merged_health"
#  "mmlu_merged_history"
#  "mmlu_merged_law"
#  "mmlu_merged_math"
#  "mmlu_merged_other"
#  "mmlu_merged_philosophy_cat"
#  "mmlu_merged_physics"
#  "mmlu_merged_politics"
#  "mmlu_merged_psychology"

  # MMLU-Pro merged variant (pruning + finetuning use same data)
#  "mmlu_pro_merged_math"
#  "mmlu_pro_merged_health"
#  "mmlu_pro_merged_physics"
#  "mmlu_pro_merged_business"
#  "mmlu_pro_merged_biology"
#  "mmlu_pro_merged_chemistry"
#  "mmlu_pro_merged_computer_science"
#  "mmlu_pro_merged_economics"
#  "mmlu_pro_merged_engineering"
#  "mmlu_pro_merged_philosophy"
#  "mmlu_pro_merged_other"
#  "mmlu_pro_merged_history"
#  "mmlu_pro_merged_psychology"
#  "mmlu_pro_merged_law"

#  "synthea_zeroshot"

)

echo "Launching extension-hf jobs for ${#MODELS[@]} models, ${#PRUNE_KEEP_K_VALUES[@]} keep-k values, and ${#TASK_GROUPS_LIST[@]} task groups..."
echo "Models: ${MODELS[@]}"
echo "Keep-k values: ${PRUNE_KEEP_K_VALUES[@]}"
echo "Cluster: $CLUSTER"
echo "S3 base: $S3_BASE"
echo ""

# Launch evaluation for each model, keep-k, and task combination
for MODEL in "${MODELS[@]}"; do
  for prune_keep_k in "${PRUNE_KEEP_K_VALUES[@]}"; do
    echo "Processing model: ${MODEL}, keep-k: ${prune_keep_k}"

    # choose the number of pruned down shared experts
    if [[ $MODEL == *"twolevelbatchlbreducedp512sharedexp1"* ]]; then
        num_shared_experts=1
    elif [[ $MODEL == *"twolevelbatchlbreducedp512sharedexp2"* ]]; then
        num_shared_experts=2
    elif [[ $MODEL == *"twolevelbatchlbreducedp512sharedexp4c2"* ]]; then
        num_shared_experts=2
    elif [[ $MODEL == *"moereducedp512sharedexp1"* ]]; then
        num_shared_experts=1
    else
        num_shared_experts=0
    fi

    for TASK in "${TASK_GROUPS_LIST[@]}"; do
        # TODO: choose the right batch size based on the task
#        # Batch size adjustment (matching original script)
#        if [[ $TASK == *"mmlu_high_school_european_history"* || $TASK == *"mmlu_high_school_us_history"* || $TASK == *"mmlu_history"* || $TASK == *"mmlu_philosophy"* || $TASK == *"cot"* || $TASK == *"minerva_math_"* || $TASK == *"mbpp"* || $TASK == *"bigcodebench"* || $TASK == *"ruler"* || $TASK == *"sciriff"* || $TASK == *"boolq"* || $TASK == *"synthea"* || $MODEL == *"1b35b"* ]]; then
#            micro_batch_size=$((micro_batch_size / 4))
#        else
#            micro_batch_size=$micro_batch_size
#        fi
        micro_batch_size=8
        if [[ $TASK == *"mmlu_history"* ]]; then
            micro_batch_size=2
        fi
        if [[ $TASK == *"gsm8k_generation_8shot"* || $TASK == *"drop_merged"* || $TASK == *"squad_merged"* ]]; then
            micro_batch_size=2
        fi

        # TODO choose the right number of gpus based on task (so that it doesn't oom)
#        # adjust number of gpus requested if its agi_eval, bbh, gsm8k, minerva, codex, mbpp
#        if [[ $TASK == *agi_eval* || $TASK == *bbh* || $TASK == *gsm8k* || $TASK == *minerva_math_* || $TASK == *codex* || $TASK == *mbpp* || $MODEL == *"1b35b"* ]]; then
#            gpus=4
#        else
#            gpus=1
#        fi
        gpus=4
        if [[ $TASK == *"mmlu_history"* || $TASK == *"gsm8k_generation_8shot"* || $TASK == *"drop_merged"* || $TASK == *"squad_merged"* ]]; then
            gpus=8
        fi

        # TODO: choose the right learning rate based on task
        lr=5e-5

        # Create a shorter, valid job name
        # Remove invalid characters and truncate long names

        stringified_model=$(echo $MODEL | sed 's/[^a-zA-Z0-9_-]//g')

        # Same naming convention as pruning_hf, but lives under a separate S3 prefix.
        relative_dir="${stringified_model}/${TASK}_keepk_${prune_keep_k}_bs-${batch_size}_lr-${lr}_epoch-${num_epochs}_prunemode-layerwise"

        # Append calibration-set-size suffix when overriding the default (use-all) behavior.
        if [ -n "$NUM_PRUNE_EXAMPLES" ]; then
            relative_dir="${relative_dir}_nprune-${NUM_PRUNE_EXAMPLES}"
        fi

        # Append per-stage shot-count suffixes when overriding task defaults.
        if [ -n "$NUM_SHOTS_PRUNE" ]; then
            relative_dir="${relative_dir}_pshots-${NUM_SHOTS_PRUNE}"
        fi
        if [ -n "$NUM_SHOTS_EVAL" ]; then
            relative_dir="${relative_dir}_eshots-${NUM_SHOTS_EVAL}"
        fi
        # Append freeze-mode suffix when not the "none" default.
        if [ "$FREEZE_MODE" != "none" ]; then
            relative_dir="${relative_dir}_fz-${FREEZE_MODE}"
        fi

        safe_relative_dir=$(printf '%s' "$relative_dir" | sed 's/[^a-zA-Z0-9_-]//g' | tail -c 100)

        # Optional calibration-size flag forwarded to the worker.
        NPE_FLAG=""
        if [ -n "$NUM_PRUNE_EXAMPLES" ]; then
            NPE_FLAG="--num-prune-examples ${NUM_PRUNE_EXAMPLES}"
        fi

        # Optional shot-count flags forwarded to the worker.
        NSHOTS_PRUNE_FLAG=""
        if [ -n "$NUM_SHOTS_PRUNE" ]; then
            NSHOTS_PRUNE_FLAG="--num-shots-prune ${NUM_SHOTS_PRUNE}"
        fi
        NSHOTS_EVAL_FLAG=""
        if [ -n "$NUM_SHOTS_EVAL" ]; then
            NSHOTS_EVAL_FLAG="--num-shots-eval ${NUM_SHOTS_EVAL}"
        fi

        # Decide what to launch based on PHASE:
        #   merge_eval ⇒ one beaker job per merge variant (each gets the full eval pipeline
        #                for just that variant + the small-model eval).
        #   full / prune_finetune ⇒ one beaker job, passing the entire MERGE_VARIANTS list.
        if [[ "$PHASE" == "merge_eval" ]]; then
            IFS=',' read -ra JOB_VARIANT_LIST <<< "$MERGE_VARIANTS"
        else
            JOB_VARIANT_LIST=("$MERGE_VARIANTS")
        fi

        for variant_arg in "${JOB_VARIANT_LIST[@]}"; do
            # Per-job naming + S3 cleanup: scope by phase.
            if [[ "$PHASE" == "merge_eval" ]]; then
                job_name="ext-${safe_relative_dir:0:90}-${variant_arg}"
                # Only wipe this variant's prefix. The small/ prefix is owned by the
                # prune_finetune phase — leave it alone so concurrent merge_eval jobs
                # don't clobber it (or each other's variants).
                s3_clean_prefixes=("${S3_BASE}/${relative_dir}/merged_${variant_arg}/")
            elif [[ "$PHASE" == "prune_finetune" ]]; then
                job_name="ext-pf-${safe_relative_dir}"
                # Wipe only the small/ prefix (this phase produces small + small per-subject).
                s3_clean_prefixes=("${S3_BASE}/${relative_dir}/small/")
            else
                job_name="ext-${safe_relative_dir}"
                s3_clean_prefixes=("${S3_BASE}/${relative_dir}/")
            fi

            for prefix in "${s3_clean_prefixes[@]}"; do
                echo "  Cleaning stale S3 results: ${prefix}"
                aws s3 rm --recursive --quiet "${prefix}" || true
            done

            echo "  Model name: ${BASE_DIR}/models/${MODEL}"
            echo "  GPUs: $gpus"
            echo "  Batch size: $batch_size"
            echo "  Phase: $PHASE"
            echo "  Variants for this job: $variant_arg"
            echo "  Job name: $job_name"

            # debug what will be passed
            echo "  model: ${BASE_DIR}/models/${MODEL}"
            echo "  task: ${TASK}"
            echo "  relative-dir: ${relative_dir}"
            echo "  base-dir: ${BASE_DIR}/extension_evals_hf"
            echo "  num-gpus: $gpus"
            echo "  run_name: ${job_name}"
            echo "  learning-rate: ${lr}"
            echo "  batch_size: ${batch_size}"
            echo "  epochs: ${num_epochs}"
            echo "  num_shared_experts: ${num_shared_experts}"

#            bash scripts/ryanwang/extensions_hf/hf_extension_with_pruning_layerwise.sh \
#                --model ${BASE_DIR}/models/${MODEL} \
#                --task ${TASK} \
#                --prune-keep-k ${prune_keep_k} \
#                --base-dir "${BASE_DIR}/extension_evals_hf" \
#                --relative-dir ${relative_dir} \
#                --num-gpus $gpus \
#                --run-name ${job_name} \
#                --learning-rate ${lr} \
#                --batch-size ${batch_size} \
#                --micro-batch-size ${micro_batch_size} \
#                --num-epochs ${num_epochs} \
#                --num-checkpoints 1 \
#                --num-shared-experts ${num_shared_experts} \
#                --s3-base ${S3_BASE} \
#                --merge-variants ${variant_arg} \
#                --phase ${PHASE} \
#                --freeze-mode ${FREEZE_MODE} \
#                ${NPE_FLAG} \
#                ${NSHOTS_PRUNE_FLAG} \
#                ${NSHOTS_EVAL_FLAG}

            python -m olmo_core.launch.beaker \
                --name $job_name \
                --gpus $gpus \
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
                -- bash -c "scripts/ryanwang/extensions_hf/hf_extension_with_pruning_layerwise.sh \
                    --model ${BASE_DIR}/models/${MODEL} \
                    --task ${TASK} \
                    --prune-keep-k ${prune_keep_k} \
                    --base-dir \"${BASE_DIR}/extension_evals_hf\" \
                    --relative-dir ${relative_dir} \
                    --num-gpus $gpus \
                    --run-name ${job_name} \
                    --learning-rate ${lr} \
                    --batch-size ${batch_size} \
                    --micro-batch-size ${micro_batch_size} \
                    --num-epochs ${num_epochs} \
                    --num-checkpoints 1 \
                    --num-shared-experts ${num_shared_experts} \
                    --s3-base ${S3_BASE} \
                    --merge-variants ${variant_arg} \
                    --phase ${PHASE} \
                    --freeze-mode ${FREEZE_MODE} \
                    ${NPE_FLAG} \
                    ${NSHOTS_PRUNE_FLAG} \
                    ${NSHOTS_EVAL_FLAG}
                "

            echo "Launched extension job: $job_name"
            echo "----------------------------------------"
        done  # variant_arg loop

#        sleep 500 # brief pause to avoid overwhelming huggingface
    done

    echo "Completed all tasks for model: $MODEL, keep-k: $prune_keep_k"
    echo "========================================"
  done
done

echo "All beaker extension jobs have been launched!"
if [[ "$PHASE" == "merge_eval" ]]; then
    n_variants=$(echo "$MERGE_VARIANTS" | tr ',' '\n' | wc -l)
    echo "Phase=merge_eval: total jobs = ${#MODELS[@]} * ${#PRUNE_KEEP_K_VALUES[@]} * ${#TASK_GROUPS_LIST[@]} * ${n_variants} variants = $((${#MODELS[@]} * ${#PRUNE_KEEP_K_VALUES[@]} * ${#TASK_GROUPS_LIST[@]} * n_variants))"
else
    echo "Phase=$PHASE: total jobs = $((${#MODELS[@]} * ${#PRUNE_KEEP_K_VALUES[@]} * ${#TASK_GROUPS_LIST[@]}))"
fi
echo "Check the beaker dashboard for job status."
