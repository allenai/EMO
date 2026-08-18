#!/usr/bin/env bash
# PARENT: "scripts/modular_extension/launch_embed_docs.sh"
# DESCRIPTION:
#     Launch Beaker GPU jobs that compute document-level router embeddings (doc_probs +
#     doc_topk_freq) for the meta_learning 20B-40B doc window, using ONE ARM's 20B
#     checkpoint (step4768-hf). Phase-2 clustering is per-arm (each model clusters the
#     docs with its OWN router), so this runs once per arm with MODEL=<arm>.
#
# Sharding: docs i::NUM_SHARDS (global enumeration). Shard outputs are idempotent
# (skip-if-exists), so relaunching after preemption is safe and a pilot subset counts
# toward the full sweep.
#
#   MODEL=sametok_ws_lam05 bash scripts/meta_learning/launch_embed_docs.sh          # pilot: shards 0-15
#   MODEL=sametok_ws_lam05 SHARDS="$(seq -s, 0 127)" JOBS=4 bash scripts/meta_learning/launch_embed_docs.sh
#   DRY_RUN=1 MODEL=... bash scripts/meta_learning/launch_embed_docs.sh
#
# NOTE: commit AND push before launching -- gantry clones source from origin on each worker.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# gRPC fork-safety for olmo_core.launch.beaker (see launch_merged_eval.sh).
export GRPC_ENABLE_FORK_SUPPORT=1
export GRPC_POLL_STRATEGY=poll

WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
EXP="meta_learning"
MODEL="${MODEL:?set MODEL=vanilla|sametok_ws_lam05|...}"
STEP="${STEP:-4768}"

MODEL_HF="${WEKA_ROOT}/${EXP}/meta128_${MODEL}/step${STEP}-hf"
DOCS_GLOB="${DOCS_GLOB:-${WEKA_ROOT}/${EXP}/data/meta128_20B-40B/docs-*.jsonl.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-${WEKA_ROOT}/${EXP}/cluster/${MODEL}_step${STEP}/embeddings}"
JOB_PREFIX="${JOB_PREFIX:-metal-embed-${MODEL//_/-}}"

NUM_SHARDS="${NUM_SHARDS:-128}"
SHARDS="${SHARDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"   # pilot default (~12.5%)
JOBS="${JOBS:-2}"
GPUS="${GPUS:-8}"   # GPUs per job (one shard-worker process per GPU)
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
    # Per-job worker command: 8 per-GPU processes, then wait for all. A final existence
    # check on every shard's info json makes the Beaker job FAIL if any shard process
    # died (plain `wait` would swallow child exit codes).
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

    job="${JOB_PREFIX}-j${j}"
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
