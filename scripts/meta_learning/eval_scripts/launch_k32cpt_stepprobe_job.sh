#!/usr/bin/env bash
# PARENT: "scripts/meta_learning/eval_scripts/launch_k32cpt_arm_job.sh"
# DESCRIPTION:
#     Submit ONE 4-node Beaker experiment running the step-granularity CPT probe
#     (k32cpt_stepprobe_worker.sh): 3 independent clusters x 10 single training steps,
#     full-model CE on all 32 cluster heldouts after every step, all artifacts except
#     the eval JSONs deleted at the end.
#
#   MODEL=sametok_ws_lam05 bash scripts/meta_learning/eval_scripts/launch_k32cpt_stepprobe_job.sh
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
    --name "k32cpt-stepprobe-${MODEL//_/-}" \
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
    -- bash scripts/meta_learning/eval_scripts/k32cpt_stepprobe_worker.sh
