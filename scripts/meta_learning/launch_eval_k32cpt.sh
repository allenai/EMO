#!/usr/bin/env bash
# PARENT: "scripts/modular_extension/launch_eval_k32cpt.sh"
# Launch one 8-GPU Beaker job that computes per-cluster held-out CE (FULL 128-expert
# routing) for ONE meta_learning phase-2 pool snapshot, via the generic
# scripts/modular_extension/eval_k32cpt_ce.py: 8 per-GPU processes, 4 clusters each.
# Idempotent per output file. The heldout tokens root is per-arm (each arm has its own
# partition), so MODEL is required.
#
#   MODEL=sametok_ws_lam05 SNAPSHOT=/weka/.../pool/snapshots/pool_after_c05.pt \
#       CONFIG_FROM=/weka/.../pool TAG=sametok_ws_lam05_after_c05 \
#       bash scripts/meta_learning/launch_eval_k32cpt.sh
#
# NOTE: commit AND push before launching -- gantry clones source from origin.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export GRPC_ENABLE_FORK_SUPPORT=1
export GRPC_POLL_STRATEGY=poll

: "${MODEL:?set MODEL=vanilla|sametok_ws_lam05|...}"
: "${TAG:?set TAG (output name)}"
WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
OUT_DIR="${WEKA_ROOT}/meta_learning/k32_cpt_runs/evals"

if [ -n "${SNAPSHOT:-}" ]; then
    : "${CONFIG_FROM:?set CONFIG_FROM (dir with config.json) when using SNAPSHOT}"
    SRC_ARGS="--snapshot ${SNAPSHOT} --config-from ${CONFIG_FROM}"
else
    : "${CKPT:?set CKPT (step dir) or SNAPSHOT}"
    SRC_ARGS="--checkpoint ${CKPT}"
fi

BEAKER_IMAGE="${BEAKER_IMAGE:-tylerr/olmo-core-tch280cu128-2025-11-25}"
CLUSTER="${CLUSTER:-ai2/jupiter}"

TOKENS_ROOT="${WEKA_ROOT}/meta_learning/data/meta128_20B-40B/k32_cpt_tokens_${MODEL}"
MAX_TOK_ARG=""
[ -n "${MAX_TOKENS:-}" ] && MAX_TOK_ARG="--max-tokens-per-cluster ${MAX_TOKENS}"

inner=""
for g in $(seq 0 7); do
    lo=$((g * 4)); hi=$((g * 4 + 3))
    out="${OUT_DIR}/ce_${TAG}_shard${g}.json"
    inner+="[ -f ${out} ] || CUDA_VISIBLE_DEVICES=${g} PYTHONPATH=.:src python -u scripts/modular_extension/eval_k32cpt_ce.py ${SRC_ARGS} --model-dtype bfloat16 --batch-size 4 --tokens-root ${TOKENS_ROOT} ${MAX_TOK_ARG} --clusters ${lo}-${hi} --out ${out} > /results/eval_gpu${g}.log 2>&1 & "
done
inner+="wait; ok=1; for g in 0 1 2 3 4 5 6 7; do [ -f ${OUT_DIR}/ce_${TAG}_shard\${g}.json ] || { echo MISSING shard \$g; tail -n5 /results/eval_gpu\${g}.log; ok=0; }; done; [ \"\$ok\" -eq 1 ]"

python -m olmo_core.launch.beaker \
    --name "metal-k32ce-${TAG}" \
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
