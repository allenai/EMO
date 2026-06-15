#!/bin/bash
# Math + general evaluation for the models_fullextend EXTENSION experiment (one Beaker job
# per model x task). Evaluates each ghost model *after* a new expert was added and the model
# was continually pretrained on FineMath with everything but the new expert frozen
# (extend_finemath_frz.sh), against the same ghost model *before* extension (step11921-hf).
#
# The new expert is a real instantiated expert now, so eval is plain standard inference
# (no ghost toggle). The question: did adding+training one expert improve math (and how much
# was forgotten on general tasks), and does a ghost-trained base absorb it better?
#
# Prereq: HF checkpoints exist (convert_to_hf.sh for the *-pre models; convert_extension_to_hf.sh
# for the *-ext models). Paths are the absolute weka paths the Beaker workers see; in this GPU
# session the same bytes live at ~/EMO/... (= /weka/oe-training-default/ryanwang/EMO/...).
#
#   bash scripts/models_fullextend/launch_extension_eval.sh            # launch
#   DRY_RUN=1 bash scripts/models_fullextend/launch_extension_eval.sh  # print only
set -euo pipefail
cd "$(dirname "$0")/../.."

# See launch_mc9_eval.sh: keep gRPC fork-safe so gantry's forked git survives.
export GRPC_ENABLE_FORK_SUPPORT=1
export GRPC_POLL_STRATEGY=poll

WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
LOCAL_ROOT="${HOME}/EMO"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WEKA_ROOT}/models_fullextend/extension_evals}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
NUM_NEW_EXPERTS="${NUM_NEW_EXPERTS:-1}"
DRY_RUN="${DRY_RUN:-0}"
VARIANTS=(uniform usage random)

# Build the model list. For each variant: the pre-extension ghost model (step11921-hf) and
# the extended model (latest step*-hf under the extend run). Entries are "name|weka_hf_path";
# missing checkpoints are skipped with a warning.
MODELS=()
for v in "${VARIANTS[@]}"; do
    pre_local="${LOCAL_ROOT}/models_fullextend/emo_1b14b_130b_ghost_${v}_always_detachF/step11921-hf"
    if [ -f "${pre_local}/config.json" ]; then
        MODELS+=("ghost_${v}_pre|${WEKA_ROOT}/models_fullextend/emo_1b14b_130b_ghost_${v}_always_detachF/step11921-hf")
    else
        echo "!!! skip ghost_${v}_pre: ${pre_local} not found"
    fi
    ext_run="${LOCAL_ROOT}/models_fullextend/emo_1b14b_130b_ghost_${v}_extend${NUM_NEW_EXPERTS}_finemath_frz"
    # `|| true`: empty/absent run dir => `ls` glob fails => pipefail+`set -e` would abort.
    ext_hf=$(ls -d "${ext_run}"/step*-hf 2>/dev/null | sed 's#.*/step##; s#-hf$##' | sort -n | tail -1 || true)
    if [ -n "${ext_hf}" ]; then
        MODELS+=("ghost_${v}_ext|${WEKA_ROOT}/models_fullextend/emo_1b14b_130b_ghost_${v}_extend${NUM_NEW_EXPERTS}_finemath_frz/step${ext_hf}-hf")
    else
        echo "!!! skip ghost_${v}_ext: no step*-hf under ${ext_run} (run convert_extension_to_hf.sh)"
    fi
done

# Tasks: MC9 (general, rc) + QA + math + code. Math is the headline (FineMath continual-pretrain).
TASKS=(
    # MC9 general (rc, olmes default)
    "arc_easy:rc::olmes" "arc_challenge:rc::olmes" "boolq:rc::olmes" "csqa:rc::olmes"
    "hellaswag:rc::olmes" "openbookqa:rc::olmes" "piqa:rc::olmes" "socialiqa:rc::olmes" "winogrande:rc::olmes"
    # QA
    "squad::olmes" "triviaqa::olmes"
    # Math (headline)
    "gsm8k::olmes" "minerva_math_500::olmes" "basic_skills::olmes"
    # Code
    "mbpp:3shot:bpb::none" "codex_humaneval:3shot:bpb::none"
)

launch_one() {
    local name="$1" hf_path="$2" task="$3"
    local out job safe_task gpus batch_size
    out="${OUTPUT_ROOT}/${name}/$(echo "$task" | sed 's/[^a-zA-Z0-9]//g')"
    safe_task=$(echo "$task" | sed 's/[^a-zA-Z0-9]//g' | cut -c1-16)
    job="ext-${name}-${safe_task}"; job=$(echo "$job" | sed 's/[^a-zA-Z0-9_-]//g' | cut -c1-80)

    # Idempotent: skip configs already scored (map weka->local for the existence check).
    local local_out="${out/\/weka\/oe-training-default\/ryanwang/${HOME}}"
    if [ "${FORCE:-0}" != "1" ] && [ -f "${local_out}/metrics.json" ]; then
        echo "=== skip ${name} | ${task} (metrics.json exists) ==="; return
    fi

    # Per-task GPU/batch heuristics (mirror the reference extension launch_eval.sh).
    gpus=1; batch_size=16
    if [[ $task == *mmlu* || $task == *minerva_math_* || $task == *codex* || $task == *mbpp* ]]; then gpus=4; fi
    if [[ $task == *gsm8k* ]]; then gpus=8; fi
    if [[ $task == *"minerva_math_"* || $task == *"mbpp"* || $task == *"boolq"* ]]; then batch_size=$((batch_size / 4)); fi

    echo ">>> ${name} | ${task}  ->  ${out}  (gpus=${gpus} bs=${batch_size})"
    if [ "$DRY_RUN" = "1" ]; then echo "    model: ${hf_path}"; return; fi

    python -m olmo_core.launch.beaker \
        --name "$job" --gpus "$gpus" --nodes 1 \
        --weka=oe-training-default --shared-filesystem --workspace ai2/flex2 \
        --beaker-image tylerr/olmo-core-tch280cu128-2025-11-25 \
        --cluster "$CLUSTER" --preemptible --allow-dirty --priority urgent --no-follow --no-torchrun \
        --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
        -- bash -c "PYTHONPATH=.:src python -u src/scripts/eval/launch_eval.py --model ${hf_path} --model-type hf --task ${task} --output-dir ${out} --batch-size ${batch_size} --gpus ${gpus} --model-args trust_remote_code=true"
}

n=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name hf_path <<< "$entry"
    for task in "${TASKS[@]}"; do
        launch_one "$name" "$hf_path" "$task"; n=$((n+1))
    done
done
echo "Total eval jobs: $n  (DRY_RUN=${DRY_RUN})"
