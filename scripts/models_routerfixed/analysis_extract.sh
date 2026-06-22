#!/bin/bash
# models_routerfixed expert analysis -- extraction phase (mirror of models_sizescaling analysis_1
# + analysis_4b). Extracts weborganizer expert-usage fingerprints for the noaux run and its
# baseline, at BOTH doc-level and token-level, so analysis_match.sh can run the same cross-model
# expert-matching as the sizescaling experiment.
#
# Both models share ONE mix_composition.json (same docs, shuffle seed 42) so per-doc / per-token
# cross-model comparison is valid (enforced by assert_same_doc_set in match_experts). The noaux
# extraction runs first and creates the composition; the baseline reuses it.
#
# Inputs: HF checkpoints
#   noaux    = models_routerfixed/emo_1b14b_50bof130b_routerfixed_noaux/step11921-hf  (convert_to_hf.sh)
#   baseline = models_fullextend/emo_1b14b_50bof130b/step11921-hf                     (already converted)
#
# GPU, ~40 min/model doc-level + ~10 min/model token-level on one A100-80GB. Idempotent.
# Usage:  bash scripts/models_routerfixed/analysis_extract.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

TARGET_TOKENS="${TARGET_TOKENS:-20000000}"
TOKENS_PER_DOC="${TOKENS_PER_DOC:-100}"
MAX_TOKENS="${MAX_TOKENS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

DOC_BASE="claude_outputs/models_routerfixed/weborganizer"
TOK_BASE="claude_outputs/models_routerfixed/weborganizer_tokens"
COMPOSITION_FILE="${DOC_BASE}/mix_composition.json"

# "<nickname>|<HF checkpoint>". noaux first so it creates the shared composition file.
MODELS=(
    "noaux|models_routerfixed/emo_1b14b_50bof130b_routerfixed_noaux/step11921-hf"
    "baseline|models_fullextend/emo_1b14b_50bof130b/step11921-hf"
)

for PAIR in "${MODELS[@]}"; do
    name="${PAIR%%|*}"
    ckpt="${PAIR##*|}"

    # ── doc-level ──────────────────────────────────────────────────────────────
    out="${DOC_BASE}/${name}"
    if [ -f "${out}/embeddings_doc_probs.npy" ]; then
        echo "=== ${name}: doc extraction exists, skipping ==="
    else
        echo "=== ${name}: doc-level extraction (${ckpt}) ==="
        MODEL_NAME="${name}" BASE_DIR="${DOC_BASE}" COMPOSITION_FILE="${COMPOSITION_FILE}" \
            bash scripts/clustering/weborganizer/extract.sh "${ckpt}" "${TARGET_TOKENS}"
    fi

    # ── token-level ────────────────────────────────────────────────────────────
    tout="${TOK_BASE}/${name}"
    if [ -f "${tout}/embeddings_doc_probs.npy" ]; then
        echo "=== ${name}: token extraction exists, skipping ==="
    else
        echo "=== ${name}: token-level extraction (${ckpt}) ==="
        mkdir -p "${tout}"
        PYTHONUNBUFFERED=1 python -u -m src.scripts.clustering.extract_document_tokens \
            --model-path "${ckpt}" \
            --output-dir "${tout}" \
            --composition-file "${COMPOSITION_FILE}" \
            --tokens-per-doc "${TOKENS_PER_DOC}" \
            --max-tokens "${MAX_TOKENS}" \
            --batch-size "${BATCH_SIZE}" \
            2>&1 | tee "${tout}/extraction.log"
    fi
done

echo ""
echo "=== Extraction done. doc -> ${DOC_BASE}/  token -> ${TOK_BASE}/ ==="
