#!/bin/bash
# Merged-test-split eval for the models_v2 base checkpoints (plain base-model eval: NO expert
# selection, NO finetuning). Scores each model on the same MC9 / GSM8K / Gen5 / MMLU tasks the
# selective project reports, i.e. the custom `*_merged` train/valid/test re-splits, evaluated on
# the held-out *test* split. This mirrors the selective pipeline's Step-4 eval
# (scripts/selective_hf/hf_finetune_with_selective_layerwise.sh) which calls
#   launch_eval.py --task "<task>_merged-pruned" --pruned_split test
# The `-pruned` suffix routes through get_oe_task_name(task, "test") -> `*_merged:rc_test::olmes`
# (rc) or `*_merged:test::olmes` (generation). We just skip selection/finetuning and run the base
# model on that test split, so the numbers are comparable to the selective base scores.
#
# Tasks are grouped into 8 Beaker jobs per model (size-balanced, not per-suite): heavy tasks
# (hellaswag + the big generation tasks) get their own job; the cheap small-rc MC9 tasks share
# one. launch_eval.py --task takes multiple tasks (nargs="+"); on the worker run_eval runs them
# sequentially on that job's GPUs, amortizing the long per-job setup. Each task in a group writes
# <output-dir>/task-<task_name>-metrics.json.
#
# Prereq: HF checkpoints. This script auto-converts the OLMo-core step dir to <step>-hf via
# scripts/convert_emo_to_hf.py (GPU-validated) when missing; it HARD-FAILS if the source step dir
# is absent. Conversion runs locally in this GPU-attached session; the Beaker eval workers read
# the same bytes at the absolute weka path.
#
#   bash scripts/models_v2/launch_merged_eval.sh                       # convert (if needed) + launch
#   DRY_RUN=1 bash scripts/models_v2/launch_merged_eval.sh             # print only (no convert/launch)
#   ONLY=emo_1b14b GROUP_FILTER=mc9_light bash scripts/models_v2/...sh       # one model, one group
#   FORCE=1 bash scripts/models_v2/launch_merged_eval.sh              # re-run already-scored groups
#
# NOTE: commit AND push this script before launching — gantry clones source from origin on each
# worker, so unpushed commits crash all replicas.
set -euo pipefail
cd "$(dirname "$0")/../.."

# olmo_core.launch.beaker pulls in a gRPC client whose epoll event-engine corrupts file
# descriptors inherited by forked `git` subprocesses (gantry runs `git diff` for the dirty check).
# Enable gRPC fork support + the poll strategy so the forked git survives.
export GRPC_ENABLE_FORK_SUPPORT=1
export GRPC_POLL_STRATEGY=poll

WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
LOCAL_ROOT="${HOME}/EMO"
EXP="models_v2"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WEKA_ROOT}/${EXP}/merged_evals}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tylerr/olmo-core-tch280cu128-2025-11-25}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
ONLY="${ONLY:-}"      # optional: restrict to models whose runname contains this substring
GROUP_FILTER="${GROUP_FILTER:-}"  # optional: restrict to groups whose name contains this substring

# --- Model table: "runname|target_step" ---
# The two 1b14b runs are symlinks into models_fullextend; the three native stdMoE runs live under
# models_v2/. Target step = the run's final/headline checkpoint (50B = step11921, 25B = step5961).
# stdmoe_128exp_50b may still be training: ensure_hf hard-fails until step11921 exists.
MODELS=(
    "emo_1b14b_50bof130b|11921"
    "stdmoe_1b14b_50bof130b|11921"
    "stdmoe_64exp_50b|11921"
    "stdmoe_64exp_25b|5961"
    "stdmoe_128exp_50b|11921"
    # WSD family (run keys may be hierarchical -- slashes resolve into paths/output dirs and are
    # sanitized out of the Beaker job name). Both reach 50B and are comparable to the 50B baselines:
    #   the trunk's own final = WSD with a 5B end-of-run decay (45B->50B);
    #   the branch = forked at 37.5B (step8941) and decayed 12.5B to 50B.
    "stdmoe_64exp_50b_wsd|11921"
    "stdmoe_64exp_50b_wsd/anneals/s8941_12p5b|11921"
    # lr2e-3 stable trunk's decay branches (both end at 50B): 5B decay from 45B, 10B decay from 40B.
    "stdmoe_64exp_50b_wsd_lr2e-3/anneals/s10729_5b|11921"
    "stdmoe_64exp_50b_wsd_lr2e-3/anneals/s9537_10b|11921"
)

# --- MMLU 17 categories (match mmlu_merged_<cat>:rc_test::olmes in src/scripts/eval/tasks.py) ---
MMLU_CATS="biology business chemistry computer_science culture economics engineering geography health history law math other philosophy_cat physics politics psychology"
MMLU_TASKS=""
for _c in $MMLU_CATS; do MMLU_TASKS="${MMLU_TASKS} mmlu_merged_${_c}"; done
MMLU_TASKS="${MMLU_TASKS# }"

# --- Group table: "name|gpus|batch|space-separated base-merged tasks" ---
# Tasks are base `*_merged` names; `-pruned` is appended at launch (resolves to the :*_test::olmes
# variant). gpus/batch follow the reference scripts that eval these exact 1B/14B fp32 MoE models
# (launch_mc9_eval.sh: rc -> gpus=2 bs=4; launch_extension_eval.sh: gsm8k -> gpus=8, gen/mmlu -> gpus=4).
GROUP_TABLE=(
    "mc9_light|2|4|arc_easy_merged arc_challenge_merged boolq_merged csqa_merged openbookqa_merged piqa_merged socialiqa_merged winogrande_merged"
    "hellaswag|2|4|hellaswag_merged"
    "mmlu|4|4|${MMLU_TASKS}"
    "gsm8k|8|16|gsm8k_generation_8shot_merged"
    "squad|4|4|squad_merged"
    "drop|4|4|drop_merged"
    "coqa|4|4|coqa_merged"
    "triviaqa_naturalqs|4|4|triviaqa_merged naturalqs_merged"
)

# Map an absolute weka path to its local mount in this GPU session (/weka/.../ryanwang -> $HOME).
weka_to_local() { echo "${1/\/weka\/oe-training-default\/ryanwang/${HOME}}"; }

# Ensure the HF checkpoint exists for a (run, step); convert from the OLMo-core step dir if not.
# Hard-fail if the source step dir is missing.
ensure_hf() {
    local run="$1" step="$2"
    local src="${LOCAL_ROOT}/${EXP}/${run}/step${step}"
    local hf="${LOCAL_ROOT}/${EXP}/${run}/step${step}-hf"
    if [ -f "${hf}/config.json" ]; then
        echo "=== hf exists: ${hf} ==="
        return 0
    fi
    if [ ! -d "${src}" ]; then
        echo "ERROR: checkpoint ${src} not found (cannot convert ${run} step${step})." >&2
        exit 1
    fi
    echo "=== converting ${src} -> ${hf} ==="
    python scripts/convert_emo_to_hf.py \
        --checkpoint-input-path "${src}" \
        --huggingface-output-dir "${hf}" \
        --max-sequence-length 4096 \
        --dtype float32 \
        --validation-device cuda
}

# Launch one Beaker job for (run, step, group): a single launch_eval.py call over all the group's
# tasks. Idempotent: skips when every task already has a task-*-metrics.json (unless FORCE=1).
launch_group() {
    local run="$1" step="$2" gname="$3" gpus="$4" batch="$5"; shift 5
    local tasks=("$@")
    local hf_weka="${WEKA_ROOT}/${EXP}/${run}/step${step}-hf"
    local out="${OUTPUT_ROOT}/${run}/${gname}"
    local local_out; local_out="$(weka_to_local "$out")"

    # Build -pruned task args (each resolves via get_oe_task_name(..., "test")).
    local task_args=()
    local t
    for t in "${tasks[@]}"; do task_args+=("${t}-pruned"); done

    # Idempotency: done when #(task-*-metrics.json) >= #tasks in the group.
    if [ "${FORCE}" != "1" ]; then
        local have
        have=$( { ls -1 "${local_out}"/task-*-metrics.json 2>/dev/null || true; } | wc -l | tr -d ' ')
        if [ "${have}" -ge "${#tasks[@]}" ]; then
            echo "=== skip ${run} | ${gname} (${have}/${#tasks[@]} metrics present) ==="
            return 0
        fi
    fi

    local job="v2eval-${run}-${gname}"
    job=$(echo "$job" | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-80)

    echo ">>> ${run} | ${gname} | ${#tasks[@]} tasks (gpus=${gpus} bs=${batch})  ->  ${out}"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "    model: ${hf_weka}"
        echo "    tasks: ${task_args[*]}"
        return 0
    fi

    python -m olmo_core.launch.beaker \
        --name "$job" \
        --gpus "$gpus" \
        --nodes 1 \
        --weka=oe-training-default \
        --shared-filesystem \
        --workspace ai2/flex2 \
        --beaker-image "$BEAKER_IMAGE" \
        --cluster "$CLUSTER" \
        --preemptible \
        --allow-dirty \
        --priority urgent \
        --no-follow \
        --no-torchrun \
        --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
        -- bash -c "PYTHONPATH=.:src python -u src/scripts/eval/launch_eval.py --model ${hf_weka} --model-type hf --task ${task_args[*]} --pruned_split test --output-dir ${out} --batch-size ${batch} --gpus ${gpus} --model-args trust_remote_code=true"
}

n=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r run step <<< "$entry"
    if [ -n "$ONLY" ] && [[ "$run" != *"$ONLY"* ]]; then continue; fi

    # Convert (unless dry-run, which must not touch GPUs / write checkpoints).
    if [ "${DRY_RUN}" != "1" ]; then
        ensure_hf "$run" "$step"
    fi

    for row in "${GROUP_TABLE[@]}"; do
        IFS='|' read -r gname gpus batch tasks_str <<< "$row"
        if [ -n "$GROUP_FILTER" ] && [[ "$gname" != *"$GROUP_FILTER"* ]]; then continue; fi
        read -ra gtasks <<< "$tasks_str"
        launch_group "$run" "$step" "$gname" "$gpus" "$batch" "${gtasks[@]}"
        n=$((n+1))
    done
    echo "======================================== ${run} done"
done
echo "Total eval jobs: $n  (DRY_RUN=${DRY_RUN}, FORCE=${FORCE}, ONLY='${ONLY}', GROUP_FILTER='${GROUP_FILTER}')"
