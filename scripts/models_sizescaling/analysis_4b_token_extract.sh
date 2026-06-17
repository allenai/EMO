#!/bin/bash
# Phase 4b: TOKEN-LEVEL expert-usage extraction for the four models_sizescaling
# checkpoints. Token-level analogue of analysis_1 — instead of one mean vector
# per document, every token becomes its own row, so analysis_4c can repeat the
# expert-matching analysis on a per-token fingerprint.
#
# Reuses the SAME mix_composition.json + shuffle seed as analysis_1, so the
# token rows are drawn from the identical document pool; all four models select
# the identical documents and the identical first --tokens-per-doc tokens, so
# token row t is the same token for every model (required for cross-model
# correlation; enforced by assert_same_doc_set in match_experts).
#
# Sampling: first 100 tokens of ~10k documents spread evenly across the 24
# topics (~1M token-rows) — maximizes document coverage at a fixed budget.
#
# GPU, ~10 min/model. Idempotent: skips models whose embeddings already exist.
# Usage:  bash scripts/models_sizescaling/analysis_4b_token_extract.sh
# Output: claude_outputs/models_sizescaling/weborganizer_tokens/<run>/
set -euo pipefail
cd "$(dirname "$0")/../.."

MODELS=(emo_1b4b_130b emo_1b7b_130b emo_1b11b_130b emo_1b14b_130b)
STEP="${STEP:-step30995}"
TOKENS_PER_DOC="${TOKENS_PER_DOC:-100}"
MAX_TOKENS="${MAX_TOKENS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
# Same composition as the doc-level extraction (analysis_1) → identical doc pool.
COMPOSITION_FILE="claude_outputs/models_sizescaling/weborganizer/mix_composition.json"
BASE_DIR="claude_outputs/models_sizescaling/weborganizer_tokens"

for m in "${MODELS[@]}"; do
    ckpt="models_sizescaling/${m}/${STEP}-hf"
    out="${BASE_DIR}/${m}"
    if [ -f "${out}/embeddings_doc_probs.npy" ]; then
        echo "=== ${m}: token extraction exists, skipping ==="
        continue
    fi
    echo "=== ${m}: token extraction (${ckpt}) ==="
    mkdir -p "${out}"
    PYTHONUNBUFFERED=1 python -u -m src.scripts.clustering.extract_document_tokens \
        --model-path "${ckpt}" \
        --output-dir "${out}" \
        --composition-file "${COMPOSITION_FILE}" \
        --tokens-per-doc "${TOKENS_PER_DOC}" \
        --max-tokens "${MAX_TOKENS}" \
        --batch-size "${BATCH_SIZE}" \
        2>&1 | tee "${out}/extraction.log"
done

echo ""
echo "=== All done. Per-model token outputs under ${BASE_DIR}/ ==="
