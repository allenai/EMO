#!/bin/bash
# MC9 evaluation for models_fullextend ghost models, launched on Beaker (one job per
# model x mode x task). Two modes:
#   standard  -- stock inference (ghost OFF): does the ghost-trained model still work
#                without the ghost it trained with?
#   ghost     -- ghost ON: each MoE layer adds a per-sequence ghost expert = blend of
#                ALL standard experts (pool = all; no document-pool masking), via the
#                EmoConfig.ghost_extend_eval toggle in the trust_remote_code modeling.
# This is a preliminary distribution-shift probe (not finetuning, not selective).
#
# Prereq: HF checkpoints must already exist (run scripts/models_fullextend/convert_to_hf.sh).
# Paths are the absolute weka paths the Beaker workers see; in this GPU session the same
# bytes live at ~/EMO/... (= /weka/oe-training-default/ryanwang/EMO/...).
#
#   bash scripts/models_fullextend/launch_mc9_eval.sh            # launch
#   DRY_RUN=1 bash scripts/models_fullextend/launch_mc9_eval.sh  # print commands only
set -euo pipefail
cd "$(dirname "$0")/../.."

# olmo_core.launch.beaker pulls in a gRPC client whose epoll event-engine corrupts file
# descriptors inherited by forked `git` subprocesses (gantry runs `git diff` for the dirty
# check), crashing git with SIGABRT ("epoll_wait error: Bad file descriptor"). Enable gRPC
# fork support + the poll strategy so the forked git survives.
export GRPC_ENABLE_FORK_SUPPORT=1
export GRPC_POLL_STRATEGY=poll

WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WEKA_ROOT}/models_fullextend/mc9_evals}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
LIMIT="${LIMIT:-1000}"
GPUS="${GPUS:-2}"
COEFF_MODE="${COEFF_MODE:-usage}"   # ghost blend mode at eval (matches config #1 training)
DRY_RUN="${DRY_RUN:-0}"

# "name|hf_path|modes"  (modes = space-separated subset of {standard, ghost})
MODELS=(
  "ghost_usage_50b|${WEKA_ROOT}/models_fullextend/emo_1b14b_130b_ghost_usage_always_detachF/step11921-hf|standard ghost"
  "no_ghost_baseline_130b|${WEKA_ROOT}/models_sizescaling/emo_1b14b_130b/step30995-hf|standard"
)

# MC9, OLMES "rc" (rank-classification / cloze) formulation — the base-model-appropriate
# metric. The "mc" (letter-picking) variant scores base models near chance, so use rc.
# Override with TASK_SUFFIX=:mc::olmes if you specifically want the mc variant.
TASK_SUFFIX="${TASK_SUFFIX:-:rc::olmes}"
MC9_TASKS=(
  "arc_easy${TASK_SUFFIX}"
  "arc_challenge${TASK_SUFFIX}"
  "boolq${TASK_SUFFIX}"
  "csqa${TASK_SUFFIX}"
  "hellaswag${TASK_SUFFIX}"
  "openbookqa${TASK_SUFFIX}"
  "piqa${TASK_SUFFIX}"
  "socialiqa${TASK_SUFFIX}"
  "winogrande${TASK_SUFFIX}"
)

launch_one() {
  local name="$1" hf_path="$2" mode="$3" task="$4"
  local model_args job out
  # The ghost toggle lives in the variant's config.json (see make_ghost_hf_variant.py),
  # so --model-args only needs trust_remote_code for the custom modeling code.
  model_args="trust_remote_code=true"
  if [ "$mode" = "ghost" ]; then
    hf_path="${hf_path}-ghost"
  fi
  out="${OUTPUT_ROOT}/${name}/${mode}/${task}"
  local safe_task; safe_task=$(echo "$task" | sed 's/[^a-zA-Z0-9]//g' | cut -c1-14)
  job="mc9-${name}-${mode}-${safe_task}"
  job=$(echo "$job" | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-80)

  echo ">>> ${name} | ${mode} | ${task}  ->  ${out}"
  if [ "$DRY_RUN" = "1" ]; then
    echo "    model: ${hf_path}"
    echo "    model-args: ${model_args}"
    return
  fi

  # Launch via olmo_core.launch.beaker (workers git-clone from origin, so the huge
  # untracked checkpoint tree is never uploaded). --no-torchrun: launch_eval manages
  # its own GPUs. --no-follow: fire-and-forget. The olmo-core beaker image already has
  # oe-eval + a working flash-attn; do NOT re-run setup_eval_env (it reinstalls deps and
  # breaks flash-attn's ABI against the image's torch).
  python -m olmo_core.launch.beaker \
    --name "$job" \
    --gpus "$GPUS" \
    --nodes 1 \
    --weka=oe-training-default \
    --shared-filesystem \
    --workspace ai2/flex2 \
    --beaker-image tylerr/olmo-core-tch280cu128-2025-11-25 \
    --cluster "$CLUSTER" \
    --preemptible \
    --allow-dirty \
    --priority urgent \
    --no-follow \
    --no-torchrun \
    --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
    -- bash -c "PYTHONPATH=.:src python -u src/scripts/eval/launch_eval.py --model ${hf_path} --model-type hf --task ${task} --limit ${LIMIT} --output-dir ${out} --batch-size 4 --gpus ${GPUS} --model-args ${model_args}"
}

n=0
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name hf_path modes <<< "$entry"
  for mode in $modes; do
    for task in "${MC9_TASKS[@]}"; do
      launch_one "$name" "$hf_path" "$mode" "$task"
      n=$((n+1))
    done
  done
done
echo "Total eval jobs: $n  (DRY_RUN=${DRY_RUN})"
