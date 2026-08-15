#!/usr/bin/env bash
# One 8-GPU Beaker job computing per-cluster 32-expert SUBSET held-out CE for one pool
# state (eval_k32cpt_subset_ce.py): 8 per-GPU processes, 4 target clusters each.
# Each target cluster's subset is sliced from the snapshot with that cluster's frozen
# selection and evaluated on that cluster's own held-out data (25M-token prefix).
#
#   SNAPSHOT=/weka/.../pool/snapshots/pool_after_c05.pt CONFIG_FROM=/weka/.../pool \
#       TAG=subset_carry_after_c05 bash scripts/modular_extension/launch_eval_k32cpt_subset.sh
#
# NOTE: commit AND push before launching -- gantry clones source from origin.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export GRPC_ENABLE_FORK_SUPPORT=1
export GRPC_POLL_STRATEGY=poll

: "${TAG:?set TAG}"
: "${SNAPSHOT:?set SNAPSHOT (bf16 pool .pt)}"
: "${CONFIG_FROM:?set CONFIG_FROM (dir with 64-expert config.json)}"
WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
OUT_DIR="${WEKA_ROOT}/modular_extension/k32_cpt_runs/evals"
TOKENS_ROOT="${WEKA_ROOT}/modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-130B/k32_cpt_tokens"
SELECTION="${WEKA_ROOT}/modular_extension/cluster/emo100b_step23842_100B-130B/k32_cpt/expert_concentration.json"
BEAKER_IMAGE="${BEAKER_IMAGE:-tylerr/olmo-core-tch280cu128-2025-11-25}"
CLUSTER="${CLUSTER:-ai2/jupiter}"

inner=""
for g in $(seq 0 7); do
    lo=$((g * 4)); hi=$((g * 4 + 3))
    out="${OUT_DIR}/ce_${TAG}_shard${g}.json"
    inner+="[ -f ${out} ] || CUDA_VISIBLE_DEVICES=${g} PYTHONPATH=.:src python -u scripts/modular_extension/eval_k32cpt_subset_ce.py --snapshot ${SNAPSHOT} --config-from ${CONFIG_FROM} --selection-json ${SELECTION} --tokens-root ${TOKENS_ROOT} --targets ${lo}-${hi} --out ${out} > /results/eval_gpu${g}.log 2>&1 & "
done
inner+="wait; ok=1; for g in 0 1 2 3 4 5 6 7; do [ -f ${OUT_DIR}/ce_${TAG}_shard\${g}.json ] || { echo MISSING shard \$g; tail -n5 /results/eval_gpu\${g}.log; ok=0; }; done; [ \"\$ok\" -eq 1 ]"

python -m olmo_core.launch.beaker \
    --name "modext-k32sub-${TAG}" \
    --gpus 8 \
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
echo "Launched subset eval job modext-k32sub-${TAG}"
