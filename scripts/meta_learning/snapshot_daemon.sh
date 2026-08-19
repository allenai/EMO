#!/usr/bin/env bash
# DESCRIPTION:
#     Out-of-band per-stage pool snapshotter for the phase-2 k=32 CPT sweeps. Every
#     POLL seconds, for each arm pool: find the NEWEST wroteback marker; if its bf16
#     snapshot (snapshots/pool_after_cNN.pt) is missing and no surgery is currently
#     rewriting that pool, capture it via snapshot_pool.py. Only the newest marker is
#     capturable -- the pool is rewritten in place, so older missed states are gone.
#     Snapshots feed the per-stage x per-cluster heldout-CE heatmap evals.
#
#   setsid nohup bash scripts/meta_learning/snapshot_daemon.sh > <log> 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

POLL="${POLL:-120}"
POOLS=(meta_learning/k32_cpt_runs/sametok_ws_lam05/pool meta_learning/k32_cpt_runs/vanilla/pool)

while true; do
    for pool in "${POOLS[@]}"; do
        [ -d "$pool" ] || continue
        latest=$(find "$pool" -maxdepth 1 -name 'wroteback_cluster*.json' 2>/dev/null \
                 | sed 's/.*wroteback_cluster\([0-9]*\)\.json/\1/' | sort -n | tail -1)
        [ -n "$latest" ] || continue
        c=$((10#$latest))
        snap="$pool/snapshots/pool_after_c$(printf '%02d' "$c").pt"
        [ -f "$snap" ] && continue
        # skip while a surgery process is rewriting any pool (cheap + safe)
        if pgrep -f "expert_subset_surgery[.]py" >/dev/null; then
            echo "$(date +%H:%M:%S) surgery active; deferring snapshot of $pool c$c"
            continue
        fi
        echo "$(date +%H:%M:%S) snapshotting $pool after_c$c"
        PYTHONPATH=src python scripts/meta_learning/snapshot_pool.py --pool "$pool" --cluster "$c" \
            || echo "$(date +%H:%M:%S) SNAPSHOT FAILED: $pool c$c"
    done
    sleep "$POLL"
done
