#!/usr/bin/env bash
# Persistent local sweep: as the k=32 CPT arms write per-stage pool snapshots, submit a
# Beaker eval job (launch_eval_k32cpt.sh) for each snapshot -> per-cluster held-out CE
# after every stage of every arm. Idempotent: skips snapshots whose 8/8 shard outputs
# exist; a submit marker (evals/submitted_<tag>) suppresses duplicates, going stale after
# 3h so lost jobs get resubmitted. At most MAX_INFLIGHT submissions outstanding.
#
#   bash scripts/modular_extension/run_snapshot_evals.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUNS="/root/EMO/modular_extension/k32_cpt_runs"
WEKA_RUNS="/weka/oe-training-default/ryanwang/EMO/modular_extension/k32_cpt_runs"
EV="${RUNS}/evals"
MAX_INFLIGHT="${MAX_INFLIGHT:-3}"
MAX_TOKENS="${MAX_TOKENS:-25000000}"   # per cluster; SE(CE) well under effect sizes
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

    for arm in carry carry_shuf fresh; do
        for snap in "${RUNS}/${arm}"/pool/snapshots/pool_after_c*.pt; do
            [ -e "$snap" ] || continue
            c=$(basename "$snap" | grep -oE 'c[0-9]+' | tr -d c)
            tag="${arm}_after_c${c}"
            complete "$tag" && continue
            marker="${EV}/submitted_${tag}"
            if [ -e "$marker" ]; then
                # stale (job lost) after 3h -> allow resubmit
                age=$(( $(date +%s) - $(stat -c %Y "$marker") ))
                [ "$age" -lt 10800 ] && continue
            fi
            [ "$inflight" -ge "$MAX_INFLIGHT" ] && continue
            echo "=== submitting eval ${tag}"
            if SNAPSHOT="${WEKA_RUNS}/${arm}/pool/snapshots/$(basename "$snap")" \
               CONFIG_FROM="${WEKA_RUNS}/${arm}/pool" \
               TAG="$tag" MAX_TOKENS="$MAX_TOKENS" \
               bash scripts/modular_extension/launch_eval_k32cpt.sh > "${EV}/launch_${tag}.log" 2>&1; then
                touch "$marker"
                inflight=$((inflight + 1))
            else
                echo "!!! submit ${tag} failed (see ${EV}/launch_${tag}.log); will retry next sweep"
            fi
        done
    done
    echo "sweep done: $(ls "${EV}"/ce_*_shard0.json 2>/dev/null | wc -l) evals started/complete, ${inflight} in flight"
    sleep 600
done
