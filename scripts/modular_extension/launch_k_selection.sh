#!/usr/bin/env bash
# Launch Beaker CPU jobs for the k-selection experiment (see
# src/scripts/clustering/k_selection.py): sweep k over {4,8,16,32,48,64,96,128} on the
# frozen preprocessed cache of the 27.5M-doc joint window.
#   - 16 global jobs: full-data fit, one per (k, seed in {1,2})  -> seed-pair stability vs k
#   - 3 subsample jobs: one per 1M-doc draw (seed in {0,1,2}), each sweeping all k
#     sequentially -> do subsamples choose the same k / recover the global fit at each k
# Outputs are idempotent (skip-if-exists); GPUs only size the CPU/RAM share.
#
#   bash scripts/modular_extension/launch_k_selection.sh
#   DRY_RUN=1 bash scripts/modular_extension/launch_k_selection.sh
#
# NOTE: commit AND push before launching -- gantry clones source from origin on each worker.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export GRPC_ENABLE_FORK_SUPPORT=1
export GRPC_POLL_STRATEGY=poll

WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
DATA_DIR="${WEKA_ROOT}/modular_extension/cluster/emo100b_step23842_100B-130B"

CLUSTER="${CLUSTER:-ai2/jupiter}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tylerr/olmo-core-tch280cu128-2025-11-25}"
DRY_RUN="${DRY_RUN:-0}"
KS="${KS:-4 8 16 32 48 64 96 128}"

launch_job() {  # name gpus inner
    local name=$1 gpus=$2 inner=$3
    echo ">>> job ${name} (gpus=${gpus})"
    if [ "${DRY_RUN}" = "1" ]; then echo "    ${inner}"; return; fi
    python -m olmo_core.launch.beaker \
        --name "$name" \
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
        -- bash -c "$inner"
}

PIP="python -c 'import sklearn, scipy' 2>/dev/null || pip install -q scikit-learn scipy; "

# global fits: one job per (k, seed)
for k in $KS; do
    for seed in 1 2; do
        inner="${PIP}OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 PYTHONPATH=.:src "
        inner+="python -u -m src.scripts.clustering.k_selection fit "
        inner+="--data-dir ${DATA_DIR} --k ${k} --seed ${seed} --fit-rows all "
        inner+="> /results/ksel.log 2>&1; rc=\$?; tail -n3 /results/ksel.log; "
        inner+="[ \$rc -eq 0 ] && [ -f ${DATA_DIR}/k_selection/global_k${k}_seed${seed}/labels.npy ]"
        launch_job "modext-ksel-global-k${k}-s${seed}" 2 "$inner"
    done
done

# subsample fits: one job per draw, sweeping all k sequentially
for seed in 0 1 2; do
    inner="${PIP}"
    for k in $KS; do
        inner+="OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONPATH=.:src "
        inner+="python -u -m src.scripts.clustering.k_selection fit "
        inner+="--data-dir ${DATA_DIR} --k ${k} --seed ${seed} --fit-rows 1000000 "
        inner+=">> /results/ksel.log 2>&1 && "
    done
    inner+="tail -n3 /results/ksel.log"
    for k in $KS; do
        inner+=" && [ -f ${DATA_DIR}/k_selection/sub1M_k${k}_seed${seed}/labels.npy ]"
    done
    launch_job "modext-ksel-sub1m-s${seed}" 1 "$inner"
done
echo "Launched (or printed) k-selection jobs (DRY_RUN=${DRY_RUN})"
