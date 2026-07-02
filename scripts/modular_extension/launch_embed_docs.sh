#!/usr/bin/env bash
# Launch Beaker GPU jobs that compute document-level router embeddings (doc_probs +
# doc_topk_freq) for the extracted 100B-110B doc window, using the EMO 100B checkpoint
# (step23842-hf). Template mirrors scripts/models_v2/launch_merged_eval.sh.
#
# Sharding: docs i::NUM_SHARDS (global enumeration; see extract_doc_window.py). SHARDS
# selects which shards this invocation computes; they are split round-robin across JOBS
# Beaker jobs x 8 GPUs, each GPU process handling its share sequentially (one model load).
# Shard outputs are idempotent (skip-if-exists), so relaunching after preemption is safe
# and a future full sweep (all 128 shards) reuses everything already computed.
#
#   bash scripts/modular_extension/launch_embed_docs.sh                  # default: shards 0-15, 2 jobs
#   SHARDS="$(seq -s, 0 127)" JOBS=4 bash scripts/modular_extension/launch_embed_docs.sh   # full sweep
#   DRY_RUN=1 bash scripts/modular_extension/launch_embed_docs.sh
#
# NOTE: commit AND push before launching -- gantry clones source from origin on each worker.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# gRPC fork-safety for olmo_core.launch.beaker (see launch_merged_eval.sh).
export GRPC_ENABLE_FORK_SUPPORT=1
export GRPC_POLL_STRATEGY=poll

WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
EXP="modular_extension"
RUN="emo_64exp_50b_wsd_lr2e-3"
STEP=23842

MODEL_HF="${WEKA_ROOT}/models_v2/${RUN}/step${STEP}-hf"
DOCS_GLOB="${WEKA_ROOT}/${EXP}/data/${RUN}_100B-110B/docs-*.jsonl.gz"
OUTPUT_DIR="${WEKA_ROOT}/${EXP}/cluster/emo100b_step${STEP}/embeddings"

NUM_SHARDS="${NUM_SHARDS:-128}"
SHARDS="${SHARDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"   # small run default (~12.5%)
JOBS="${JOBS:-2}"
GPUS=8
CLUSTER="${CLUSTER:-ai2/jupiter}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tylerr/olmo-core-tch280cu128-2025-11-25}"
DRY_RUN="${DRY_RUN:-0}"

# Split SHARDS round-robin across JOBS*GPUS worker slots; each slot's list is passed as
# a single --shards argument so the worker loads the model once.
IFS=',' read -ra ALL_SHARDS <<< "$SHARDS"
declare -a SLOT_SHARDS
for i in "${!ALL_SHARDS[@]}"; do
    slot=$((i % (JOBS * GPUS)))
    SLOT_SHARDS[$slot]="${SLOT_SHARDS[$slot]:+${SLOT_SHARDS[$slot]},}${ALL_SHARDS[$i]}"
done

for j in $(seq 0 $((JOBS - 1))); do
    # Build the per-job worker command: 8 per-GPU processes, then wait for all. A final
    # existence check on every shard's info json makes the Beaker job FAIL if any shard
    # process died (plain `wait` would swallow child exit codes).
    inner=""
    job_infos=""
    for g in $(seq 0 $((GPUS - 1))); do
        slot=$((j * GPUS + g))
        slist="${SLOT_SHARDS[$slot]:-}"
        [ -z "$slist" ] && continue
        IFS=',' read -ra ss <<< "$slist"
        for s in "${ss[@]}"; do
            job_infos+="${OUTPUT_DIR}/info-$(printf '%03d' "$s").json "
        done
        inner+="CUDA_VISIBLE_DEVICES=${g} PYTHONPATH=.:src python -u -m src.scripts.clustering.extract_doc_window --docs-glob '${DOCS_GLOB}' --model-path ${MODEL_HF} --output-dir ${OUTPUT_DIR} --shards ${slist} --num-shards ${NUM_SHARDS} > /results/shard_gpu${g}.log 2>&1 & "
    done
    [ -z "$inner" ] && continue
    inner+="wait; ok=1; for f in ${job_infos}; do [ -f \"\$f\" ] || { echo \"MISSING \$f\"; ok=0; }; done; tail -n2 /results/shard_gpu*.log; [ \"\$ok\" -eq 1 ]"

    job="modext-embed-docs-j${j}"
    echo ">>> job ${job}: $(echo "$inner" | grep -o '\--shards [0-9,]*' | tr '\n' ' ')"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "    ${inner}"
        continue
    fi

    python -m olmo_core.launch.beaker \
        --name "$job" \
        --gpus "$GPUS" \
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
        -- bash -c "$inner"
done
echo "Launched (or printed) ${JOBS} jobs for shards: ${SHARDS} (NUM_SHARDS=${NUM_SHARDS}, DRY_RUN=${DRY_RUN})"
