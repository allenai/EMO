#!/bin/bash
# Phase 4c: cross-model expert matching on per-TOKEN usage fingerprints.
# Identical to analysis_4 but points match_experts at the token-level
# extractions (analysis_4b) instead of the doc-level ones (analysis_1). Same
# four pairs, same tool, no code change — the "doc axis" is now a token axis.
# CPU-only, ~a minute per pair. Requires analysis_4b extractions.
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_DIR="claude_outputs/models_sizescaling/weborganizer_tokens"
OUT_BASE="claude_outputs/models_sizescaling/matching_tokens"

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
    echo "=== matching (token-level) ${NICK[$a]} vs ${NICK[$b]} ==="
    PYTHONUNBUFFERED=1 python -u -m src.scripts.clustering.match_experts \
        --dir-a "${BASE_DIR}/${a}" \
        --dir-b "${BASE_DIR}/${b}" \
        --label-a "${NICK[$a]}" \
        --label-b "${NICK[$b]}" \
        --output-dir "${out}"
done

echo ""
echo "=== Done. Token-level matching under ${OUT_BASE}/ ==="
