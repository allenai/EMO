#!/usr/bin/env bash
# Submit subset-CE eval jobs (launch_eval_k32cpt_subset.sh) for EVERY existing pool
# snapshot of every arm, plus the two anchors (100B start, 130B baseline), with an
# in-flight cap. Idempotent like run_snapshot_evals.sh (8/8 shard outputs = complete;
# submit markers go stale after 3h). Exits when everything is submitted and complete.
#
#   bash scripts/modular_extension/run_subset_evals.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUNS="/root/EMO/modular_extension/k32_cpt_runs"
WEKA_RUNS="/weka/oe-training-default/ryanwang/EMO/modular_extension/k32_cpt_runs"
WEKA_TRUNK="/weka/oe-training-default/ryanwang/EMO/models_v2/emo_64exp_50b_wsd_lr2e-3/step23842"
EV="${RUNS}/evals"
MAX_INFLIGHT="${MAX_INFLIGHT:-5}"
mkdir -p "$EV"

complete() { [ "$(ls "${EV}"/ce_$1_shard*.json 2>/dev/null | wc -l)" -eq 8 ]; }

submit() {  # tag snapshot_weka config_from_weka
    local tag=$1 snap=$2 cfg=$3
    local marker="${EV}/submitted_${tag}"
    complete "$tag" && { rm -f "$marker"; return 1; }
    if [ -e "$marker" ]; then
        local age=$(( $(date +%s) - $(stat -c %Y "$marker") ))
        [ "$age" -lt 10800 ] && return 1
    fi
    echo "=== submitting ${tag}"
    if SNAPSHOT="$snap" CONFIG_FROM="$cfg" TAG="$tag" \
       bash scripts/modular_extension/launch_eval_k32cpt_subset.sh > "${EV}/launch_${tag}.log" 2>&1; then
        touch "$marker"
        return 0
    fi
    echo "!!! submit ${tag} failed (see ${EV}/launch_${tag}.log)"
    return 1
}

while true; do
    inflight=0
    for m in "${EV}"/submitted_subset_*; do
        [ -e "$m" ] || continue
        tag="${m##*/submitted_}"
        if complete "$tag"; then rm -f "$m"; else inflight=$((inflight + 1)); fi
    done

    pending=0
    # anchors first (they are the reference for every delta)
    for a in "subset_100B ${WEKA_RUNS}/anchors/step23842_100B.pt" \
             "subset_baseline130B ${WEKA_RUNS}/anchors/baseline_130B.pt"; do
        set -- $a
        complete "$1" && continue
        pending=$((pending + 1))
        [ "$inflight" -lt "$MAX_INFLIGHT" ] && [ -e "${2/#$WEKA_RUNS/$RUNS}" ] && \
            submit "$1" "$2" "$WEKA_TRUNK" && inflight=$((inflight + 1))
    done
    # every snapshot of every arm, round-robin by index
    for i in $(seq 0 31); do
        for arm in carry carry_shuf fresh; do
            snap_local=$(ls "${RUNS}/${arm}"/pool/snapshots/pool_after_c*.pt 2>/dev/null | sed -n "$((i + 1))p")
            [ -n "$snap_local" ] || continue
            base=$(basename "$snap_local" .pt)   # pool_after_cXX
            tag="subset_${arm}_${base#pool_}"    # subset_<arm>_after_cXX
            complete "$tag" && continue
            pending=$((pending + 1))
            [ "$inflight" -lt "$MAX_INFLIGHT" ] && \
                submit "$tag" "${WEKA_RUNS}/${arm}/pool/snapshots/$(basename "$snap_local")" \
                       "${WEKA_RUNS}/${arm}/pool" && inflight=$((inflight + 1))
        done
    done

    n_done=$(ls "${EV}"/ce_subset_*_shard0.json 2>/dev/null | wc -l)
    echo "sweep: ${n_done} subset evals started/complete, ${pending} pending, ${inflight} in flight"
    [ "$pending" -eq 0 ] && [ "$inflight" -eq 0 ] && { echo "ALL SUBSET EVALS COMPLETE"; break; }
    sleep 600
done
