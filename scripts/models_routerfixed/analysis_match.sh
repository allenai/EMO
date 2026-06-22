#!/bin/bash
# models_routerfixed expert analysis -- matching phase (mirror of models_sizescaling analysis_4 +
# analysis_4c). Runs the cross-model expert-matching tool on the noaux vs baseline pair, at both
# doc-level and token-level. Requires analysis_extract.sh outputs. CPU-only, ~a minute per level.
#
# NB unlike sizescaling (different expert counts, no index alignment), noaux and baseline have the
# SAME 128 experts, the SAME (frozen=baseline-final) router, and started from byte-identical
# non-router init -- so experts are INDEX-ALIGNED. The match tool still applies (Hungarian, splits,
# novelty), and crucially the DIAGONAL of its saved correlation matrix (corr[i,i]) is the
# index-aligned same-expert functional similarity. The report extracts that diagonal from
# corr_matrices.npz; here we just produce the standard match outputs.
set -euo pipefail
cd "$(dirname "$0")/../.."

DOC_BASE="claude_outputs/models_routerfixed/weborganizer"
TOK_BASE="claude_outputs/models_routerfixed/weborganizer_tokens"
OUT_DOC="claude_outputs/models_routerfixed/matching/noaux_vs_baseline"
OUT_TOK="claude_outputs/models_routerfixed/matching_tokens/noaux_vs_baseline"

echo "=== doc-level matching: noaux vs baseline ==="
PYTHONUNBUFFERED=1 python -u -m src.scripts.clustering.match_experts \
    --dir-a "${DOC_BASE}/noaux" \
    --dir-b "${DOC_BASE}/baseline" \
    --label-a "noaux" \
    --label-b "baseline" \
    --output-dir "${OUT_DOC}"

echo "=== token-level matching: noaux vs baseline ==="
PYTHONUNBUFFERED=1 python -u -m src.scripts.clustering.match_experts \
    --dir-a "${TOK_BASE}/noaux" \
    --dir-b "${TOK_BASE}/baseline" \
    --label-a "noaux" \
    --label-b "baseline" \
    --output-dir "${OUT_TOK}"

echo ""
echo "=== Matching done. doc -> ${OUT_DOC}/  token -> ${OUT_TOK}/ ==="
