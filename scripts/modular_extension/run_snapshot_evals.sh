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
MAX_INFLIGHT="${MAX_INFLIGHT:-4}"
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

    # Collect pending snapshots per arm, then submit ROUND-ROBIN across arms so no arm
    # starves the others while its backlog drains.
    declare -a p_carry=() p_carry_shuf=() p_fresh=()
    for arm in carry carry_shuf fresh; do
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
            case "$arm" in
                carry) p_carry+=("$(basename "$snap")|$tag");;
                carry_shuf) p_carry_shuf+=("$(basename "$snap")|$tag");;
                fresh) p_fresh+=("$(basename "$snap")|$tag");;
            esac
        done
    done
    i=0
    while [ "$inflight" -lt "$MAX_INFLIGHT" ]; do
        picked=""
        for arm in carry carry_shuf fresh; do
            case "$arm" in
                carry) [ "$i" -lt "${#p_carry[@]}" ] && picked="${p_carry[$i]}";;
                carry_shuf) [ "$i" -lt "${#p_carry_shuf[@]}" ] && picked="${p_carry_shuf[$i]}";;
                fresh) [ "$i" -lt "${#p_fresh[@]}" ] && picked="${p_fresh[$i]}";;
            esac
            [ -z "$picked" ] && continue
            snap_name="${picked%|*}"; tag="${picked#*|}"; picked=""
            echo "=== submitting eval ${tag}"
            if SNAPSHOT="${WEKA_RUNS}/${arm}/pool/snapshots/${snap_name}" \
               CONFIG_FROM="${WEKA_RUNS}/${arm}/pool" \
               TAG="$tag" MAX_TOKENS="$MAX_TOKENS" \
               bash scripts/modular_extension/launch_eval_k32cpt.sh > "${EV}/launch_${tag}.log" 2>&1; then
                touch "${EV}/submitted_${tag}"
                inflight=$((inflight + 1))
            else
                echo "!!! submit ${tag} failed (see ${EV}/launch_${tag}.log); will retry next sweep"
            fi
            [ "$inflight" -ge "$MAX_INFLIGHT" ] && break
        done
        i=$((i + 1))
        [ "$i" -gt 40 ] && break   # exhausted all pending lists this sweep
    done
    unset p_carry p_carry_shuf p_fresh
    echo "sweep done: $(ls "${EV}"/ce_*_shard0.json 2>/dev/null | wc -l) evals started/complete, ${inflight} in flight"
    sleep 600
done
