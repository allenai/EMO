#!/bin/bash
# Token-level router embedding extraction (shuffled sampling).
#
# Extracts per-token router activations (probs, logits, topk_binary) instead of
# per-document aggregated embeddings. Uses ~100K tokens (vs ~20M for document-level).
# Also saves full document token arrays for context recovery.
#
# Output: claude_outputs/analysis/router_clustering_pretraining_shuffled_token/<model_name>/
#   embeddings_token_probs.npy      — softmax probs per token (float16)
#   embeddings_token_logits.npy     — raw logits per token (float16)
#   embeddings_token_topk_binary.npy — binary top-k mask per token (uint8)
#   documents.npy                   — all document tokens concatenated (int32)
#   doc_boundaries.npy              — document start indices
#   metadata_tokens.jsonl.gz        — per-token metadata (source, doc_index, token_position, token_id)
#   metadata_docs.jsonl.gz          — per-document metadata (source, doc_len)
#   info.json                       — run config and stats

set -euo pipefail

MODEL_NAME="twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301"
MODEL_PATH="models/${MODEL_NAME}/step30995-hf"
COMPOSITION="claude_outputs/analysis/router_clustering_pretraining/mix_composition.json"
OUTPUT_DIR="claude_outputs/analysis/router_clustering_pretraining_shuffled_token/${MODEL_NAME}"

mkdir -p "$OUTPUT_DIR"

echo "=== Token-level extraction: ${MODEL_NAME} ==="
echo "Output: ${OUTPUT_DIR}"

PYTHONUNBUFFERED=1 python -u \
    -m src.scripts.analysis.extract_router_embeddings \
    --model-path "$MODEL_PATH" \
    --composition-file "$COMPOSITION" \
    --output-dir "$OUTPUT_DIR" \
    --granularity token \
    --target-tokens 100000 \
    --batch-size 32 \
    --shuffle \
    2>&1 | tee "${OUTPUT_DIR}/extraction.log"

echo "=== Done ==="
