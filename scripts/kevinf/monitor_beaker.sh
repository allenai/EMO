#!/bin/bash
# Monitor a Beaker experiment until it reaches a terminal state.
# Usage: bash scripts/kevinf/monitor_beaker.sh <experiment_id> [poll_interval_seconds]
#
# Prints status and latest logs every poll_interval (default 60s).

set -euo pipefail

EXP_ID="${1:?Usage: monitor_beaker.sh <experiment_id> [poll_interval_seconds]}"
POLL="${2:-60}"

echo "Monitoring Beaker experiment: ${EXP_ID} (polling every ${POLL}s)"
echo "Dashboard: https://beaker.org/ex/${EXP_ID}"
echo ""

while true; do
    STATUS=$(beaker experiment get "$EXP_ID" 2>&1 | awk 'NR==2 {for(i=6;i<=NF-1;i++) printf "%s ", $i; print ""}' | xargs)
    TIMESTAMP=$(date '+%H:%M:%S')
    echo "============================================"
    echo "[${TIMESTAMP}] Status: ${STATUS}"
    echo "--- Latest logs (last 50 lines) ---"
    beaker experiment logs "$EXP_ID" 2>&1 | grep -v '\[A' | tail -50
    echo "============================================"
    echo ""

    case "$STATUS" in
        *succeeded*|*finalized*)
            echo "Job succeeded!"
            exit 0
            ;;
        *failed*|*stopped*|*canceled*)
            echo "Job FAILED with status: ${STATUS}"
            echo "--- Full tail of logs (last 100 lines) ---"
            beaker experiment logs "$EXP_ID" 2>&1 | tail -100
            exit 1
            ;;
    esac

    sleep "$POLL"
done
