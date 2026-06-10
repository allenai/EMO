#!/bin/bash
# Phase 4: cross-model expert matching on per-doc usage fingerprints.
# Consecutive pairs + the 32<->128 end-to-end pair. CPU-only, ~a minute per
# pair. Requires analysis_1 extractions (identical doc set across models).
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_DIR="claude_outputs/clustering/weborganizer"
OUT_BASE="claude_outputs/clustering/sizescaling/matching"

# "<smaller>|<larger>" by expert count
PAIRS=(
    "emo_1b4b_130b|emo_1b7b_130b"
    "emo_1b7b_130b|emo_1b11b_130b"
    "emo_1b11b_130b|emo_1b14b_130b"
    "emo_1b4b_130b|emo_1b14b_130b"
)

declare -A NICK=(
    [emo_1b4b_130b]=32e [emo_1b7b_130b]=64e [emo_1b11b_130b]=96e [emo_1b14b_130b]=128e
)

for pair in "${PAIRS[@]}"; do
    a="${pair%%|*}"
    b="${pair##*|}"
    out="${OUT_BASE}/${NICK[$a]}_vs_${NICK[$b]}"
    echo "=== matching ${NICK[$a]} vs ${NICK[$b]} ==="
    PYTHONUNBUFFERED=1 python -u -m src.scripts.clustering.match_experts \
        --dir-a "${BASE_DIR}/${a}" \
        --dir-b "${BASE_DIR}/${b}" \
        --label-a "${NICK[$a]}" \
        --label-b "${NICK[$b]}" \
        --output-dir "${out}"
done
