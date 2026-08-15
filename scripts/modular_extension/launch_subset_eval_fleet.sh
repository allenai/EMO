#!/usr/bin/env bash
# Launch NJOBS long-running Beaker jobs that together cover ALL subset-CE eval units
# (2 anchors + every pool snapshot of every arm). Each job processes units with
# index % NJOBS == its job id, SEQUENTIALLY, without resubmission -- amortizing queue
# latency over ~15 units per job instead of one job per unit.
#
# The unit list is written to weka at launch time (arms are paused, so snapshots are
# static). Per unit: 8 per-GPU worker processes (4 target clusters each), with
# per-shard skip-if-exists, so completed units are skipped instantly on relaunch.
#
#   bash scripts/modular_extension/launch_subset_eval_fleet.sh
#   NJOBS=5 DRY_RUN=1 bash scripts/modular_extension/launch_subset_eval_fleet.sh
#
# NOTE: commit AND push before launching -- gantry clones source from origin.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export GRPC_ENABLE_FORK_SUPPORT=1
export GRPC_POLL_STRATEGY=poll

RUNS="/root/EMO/modular_extension/k32_cpt_runs"
WEKA_RUNS="/weka/oe-training-default/ryanwang/EMO/modular_extension/k32_cpt_runs"
WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
WEKA_TRUNK="${WEKA_ROOT}/models_v2/emo_64exp_50b_wsd_lr2e-3/step23842"
OUT_DIR="${WEKA_RUNS}/evals"
TOKENS_ROOT="${WEKA_ROOT}/modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-130B/k32_cpt_tokens"
SELECTION="${WEKA_ROOT}/modular_extension/cluster/emo100b_step23842_100B-130B/k32_cpt/expert_concentration.json"
NJOBS="${NJOBS:-5}"
DRY_RUN="${DRY_RUN:-0}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tylerr/olmo-core-tch280cu128-2025-11-25}"
CLUSTER="${CLUSTER:-ai2/jupiter}"

# --- build the static unit list: "tag snapshot config_from" per line ---
UNITS_LOCAL="${RUNS}/evals/subset_units.txt"
UNITS_WEKA="${OUT_DIR}/subset_units.txt"
{
    echo "subset_100B ${WEKA_RUNS}/anchors/step23842_100B.pt ${WEKA_TRUNK}"
    echo "subset_baseline130B ${WEKA_RUNS}/anchors/baseline_130B.pt ${WEKA_TRUNK}"
    for arm in carry carry_shuf fresh; do
        for snap in "${RUNS}/${arm}"/pool/snapshots/pool_after_c*.pt; do
            [ -e "$snap" ] || continue
            base=$(basename "$snap" .pt)
            echo "subset_${arm}_${base#pool_} ${WEKA_RUNS}/${arm}/pool/snapshots/$(basename "$snap") ${WEKA_RUNS}/${arm}/pool"
        done
    done
} > "$UNITS_LOCAL"
n_units=$(wc -l < "$UNITS_LOCAL")
echo "unit list: ${n_units} units -> ${UNITS_LOCAL}"

for j in $(seq 0 $((NJOBS - 1))); do
    # Per job: iterate its units sequentially; per unit run 8 per-GPU shard processes.
    inner="i=0; while read -r tag snap cfg; do i=\$((i+1)); [ \$(( (i-1) % ${NJOBS} )) -ne ${j} ] && continue; "
    inner+="echo \"=== unit \$tag\"; "
    inner+="for g in 0 1 2 3 4 5 6 7; do lo=\$((g*4)); hi=\$((g*4+3)); out=${OUT_DIR}/ce_\${tag}_shard\${g}.json; "
    inner+="[ -f \$out ] || CUDA_VISIBLE_DEVICES=\$g PYTHONPATH=.:src python -u scripts/modular_extension/eval_k32cpt_subset_ce.py --snapshot \$snap --config-from \$cfg --selection-json ${SELECTION} --tokens-root ${TOKENS_ROOT} --targets \${lo}-\${hi} --out \$out > /results/\${tag}_gpu\${g}.log 2>&1 & done; wait; "
    inner+="done < ${UNITS_WEKA}; "
    inner+="ok=1; i=0; while read -r tag snap cfg; do i=\$((i+1)); [ \$(( (i-1) % ${NJOBS} )) -ne ${j} ] && continue; for g in 0 1 2 3 4 5 6 7; do [ -f ${OUT_DIR}/ce_\${tag}_shard\${g}.json ] || { echo \"MISSING \$tag shard \$g\"; tail -n5 /results/\${tag}_gpu\${g}.log 2>/dev/null; ok=0; }; done; done < ${UNITS_WEKA}; [ \"\$ok\" -eq 1 ]"

    echo ">>> fleet job ${j}/${NJOBS}"
    if [ "$DRY_RUN" = "1" ]; then echo "    $inner" | head -c 500; echo; continue; fi
    python -m olmo_core.launch.beaker \
        --name "modext-k32sub-fleet-${j}" \
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
done
echo "Launched ${NJOBS} fleet jobs over ${n_units} units (DRY_RUN=${DRY_RUN})"
