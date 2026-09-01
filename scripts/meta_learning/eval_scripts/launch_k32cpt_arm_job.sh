#!/usr/bin/env bash
# DESCRIPTION:
#     Submit ONE 4-node Beaker experiment that runs an arm's ENTIRE phase-2 k=32 CPT
#     sweep (all 32 sequential extract/train/writeback stages) inside the job via
#     k32cpt_arm_worker.sh -- replacing the local per-stage driver so the sweep queues
#     on the cluster ONCE instead of 32 times. --no-torchrun: the worker runs torchrun
#     itself per stage (the launcher still enables leader selection + host networking
#     for --nodes>1, providing BEAKER_LEADER_REPLICA_HOSTNAME etc.). Resumable: markers
#     + trainer auto-resume make preemption/relaunch safe.
#
#     Per-stage heatmap evals stay OUTSIDE this job: the local run_snapshot_evals.sh
#     detects each new pool snapshot and launches its 8-GPU eval job.
#
#   MODEL=sametok_ws_lam05 bash scripts/meta_learning/eval_scripts/launch_k32cpt_arm_job.sh
#   MODEL=vanilla          bash scripts/meta_learning/eval_scripts/launch_k32cpt_arm_job.sh
#
# NOTE: commit AND push before launching -- gantry clones source from origin.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

export GRPC_ENABLE_FORK_SUPPORT=1
export GRPC_POLL_STRATEGY=poll

: "${MODEL:?set MODEL=vanilla|sametok_ws_lam05|...}"
NODES="${NODES:-4}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tylerr/olmo-core-tch280cu128-2025-11-25}"
CLUSTER="${CLUSTER:-ai2/jupiter}"

python -m olmo_core.launch.beaker \
    --name "k32cpt-${MODEL//_/-}-sweep" \
    --gpus 8 \
    --nodes "$NODES" \
    --weka=oe-training-default \
    --shared-filesystem \
    --workspace ai2/flex2 \
    --cluster "$CLUSTER" \
    --beaker-image "$BEAKER_IMAGE" \
    --preemptible \
    --allow-dirty \
    --no-follow \
    --no-torchrun \
    --priority urgent \
    --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
    --env "S3_PROFILE=" "MODEL=${MODEL}" \
    -- bash scripts/meta_learning/eval_scripts/k32cpt_arm_worker.sh
