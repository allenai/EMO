#!/usr/bin/env bash
# Launch one Beaker CPU job per subsample-stability arm (see
# src/scripts/clustering/subsample_stability.py): can a clustering fit on a subsample
# recover the full 27.5M-doc oracle partition? Arms:
#   honest  n in {100K, 1M, 5M}  -- re-fit mean/PCA/L2 + k-means on the subsample only
#   frozen  n in {100K, 1M, 5M}  -- full-fit transform frozen, only k-means re-fit
#   (1M runs twice, seeds 0/1, to gauge draw-to-draw variance)
#   fullseed seeds 1,2           -- full-data k-means re-fit = seed-noise ceiling
# GPUs are unused (sklearn on CPU); the --gpus request just sizes the node share
# (cores/RAM). Outputs are idempotent (skip-if-exists), safe to relaunch.
#
#   bash scripts/modular_extension/launch_subsample_stability.sh
#   DRY_RUN=1 bash scripts/modular_extension/launch_subsample_stability.sh
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

# mode:n_sample:seed:gpus  (n_sample empty for fullseed; gpus sizes the CPU/RAM share)
ARMS=(
    "honest:100000:0:1"
    "honest:1000000:0:1"
    "honest:1000000:1:1"
    "honest:5000000:0:2"
    "frozen:100000:0:1"
    "frozen:1000000:0:1"
    "frozen:1000000:1:1"
    "frozen:5000000:0:1"
    "fullseed::1:2"
    "fullseed::2:2"
)

for arm in "${ARMS[@]}"; do
    IFS=':' read -r mode n seed gpus <<< "$arm"
    if [ "$mode" = "fullseed" ]; then
        arm_dir="fullseed_seed${seed}"
        job="modext-substab-fullseed-s${seed}"
        n_flag=""
    else
        arm_dir="${mode}_n${n}_seed${seed}"
        job="modext-substab-${mode}-n${n}-s${seed}"
        n_flag="--n-sample ${n}"
    fi
    threads=$((gpus * 16))

    inner="python -c 'import sklearn, scipy' 2>/dev/null || pip install -q scikit-learn scipy; "
    inner+="OMP_NUM_THREADS=${threads} OPENBLAS_NUM_THREADS=${threads} PYTHONPATH=.:src "
    inner+="python -u -m src.scripts.clustering.subsample_stability run "
    inner+="--data-dir ${DATA_DIR} --mode ${mode} ${n_flag} --seed ${seed} "
    inner+="> /results/substab.log 2>&1; rc=\$?; tail -n3 /results/substab.log; "
    inner+="[ \$rc -eq 0 ] && [ -f ${DATA_DIR}/subsample_stability/${arm_dir}/labels.npy ]"

    echo ">>> job ${job} (gpus=${gpus})"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "    ${inner}"
        continue
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
        -- bash -c "$inner"
done
echo "Launched (or printed) ${#ARMS[@]} arms (DRY_RUN=${DRY_RUN})"
