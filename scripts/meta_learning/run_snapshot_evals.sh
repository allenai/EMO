#!/usr/bin/env bash
# PARENT: "scripts/modular_extension/run_snapshot_evals.sh"
# Persistent local sweep: as the phase-2 k=32 CPT arms write per-stage pool snapshots,
# submit a Beaker eval job (launch_eval_k32cpt.sh) for each snapshot -> per-cluster
# held-out CE of the FULL 128-expert model after every stage of every arm (the heatmap).
# Idempotent: skips snapshots whose 8/8 shard outputs exist; a submit marker
# (evals/submitted_<tag>) suppresses duplicates, going stale after 3h so lost jobs get
# resubmitted. At most MAX_INFLIGHT submissions outstanding.
#
#   bash scripts/meta_learning/run_snapshot_evals.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUNS="/root/EMO/meta_learning/k32_cpt_runs"
WEKA_RUNS="/weka/oe-training-default/ryanwang/EMO/meta_learning/k32_cpt_runs"
EV="${RUNS}/evals"
MAX_INFLIGHT="${MAX_INFLIGHT:-4}"
MAX_TOKENS="${MAX_TOKENS:-25000000}"   # per cluster; SE(CE) well under effect sizes
ARMS=(sametok_ws_lam05 vanilla)
mkdir -p "$EV"

complete() {  # tag -> 0 if all 8 shard outputs exist
    [ "$(ls "${EV}"/ce_$1_shard*.json 2>/dev/null | wc -l)" -eq 8 ]
}

while true; do
    inflight=0
    for m in "${EV}"/submitted_*; do
        [ -e "$m" ] || continue
        tag="${m##*/submitted_}"
        if complete "$tag"; then rm -f "$m"; else inflight=$((inflight + 1)); fi
    done

    # Collect pending snapshots per arm, then submit ROUND-ROBIN across arms so no arm
    # starves the other while its backlog drains.
    declare -A pending=()
    for arm in "${ARMS[@]}"; do
        list=()
        for snap in "${RUNS}/${arm}"/pool/snapshots/pool_after_c*.pt; do
            [ -e "$snap" ] || continue
            c=$(basename "$snap" | grep -oE 'c[0-9]+' | tr -d c)
            tag="${arm}_after_c${c}"
            complete "$tag" && continue
            marker="${EV}/submitted_${tag}"
            if [ -e "$marker" ]; then
                age=$(( $(date +%s) - $(stat -c %Y "$marker") ))   # stale after 3h
                [ "$age" -lt 10800 ] && continue
            fi
            list+=("$(basename "$snap")|$tag")
        done
        pending[$arm]="${list[*]:-}"
    done
    i=0
    while [ "$inflight" -lt "$MAX_INFLIGHT" ]; do
        any=""
        for arm in "${ARMS[@]}"; do
            read -ra list <<< "${pending[$arm]:-}"
            [ "$i" -lt "${#list[@]}" ] || continue
            entry="${list[$i]}"
            snap_name="${entry%|*}"; tag="${entry#*|}"
            any=1
            echo "=== submitting eval ${tag}"
            if MODEL="$arm" \
               SNAPSHOT="${WEKA_RUNS}/${arm}/pool/snapshots/${snap_name}" \
               CONFIG_FROM="${WEKA_RUNS}/${arm}/pool" \
               TAG="$tag" MAX_TOKENS="$MAX_TOKENS" \
               bash scripts/meta_learning/launch_eval_k32cpt.sh > "${EV}/launch_${tag}.log" 2>&1; then
                touch "${EV}/submitted_${tag}"
                inflight=$((inflight + 1))
            else
                echo "!!! submit ${tag} failed (see ${EV}/launch_${tag}.log); will retry next sweep"
            fi
            [ "$inflight" -ge "$MAX_INFLIGHT" ] && break
        done
        i=$((i + 1))
        { [ -z "$any" ] || [ "$i" -gt 40 ]; } && break   # exhausted pending lists this sweep
    done
    unset pending
    echo "sweep done: $(ls "${EV}"/ce_*_shard0.json 2>/dev/null | wc -l) evals started/complete, ${inflight} in flight"
    sleep 600
done
