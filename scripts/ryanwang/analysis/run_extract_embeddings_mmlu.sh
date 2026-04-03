#!/bin/bash
# Extract router embeddings from MMLU validation data.
#
# Uses all 57 MMLU subjects as sources, with each subject treated as a
# separate data source. The output format matches extract_router_embeddings.py
# so that transform_and_cluster.py works unchanged.
#
# Usage:
#   bash scripts/ryanwang/analysis/run_extract_embeddings_mmlu.sh [MODEL_PATH]
#
# Example:
#   bash scripts/ryanwang/analysis/run_extract_embeddings_mmlu.sh \
#       models/moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf
set -e

MODEL_PATH="${1:-models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf}"
MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")"
BASE_DIR="claude_outputs/analysis/router_clustering_mmlu_val"
OUTPUT_DIR="${BASE_DIR}/${MODEL_NAME}"
BATCH_SIZE=32

echo "============================================"
echo "MMLU Validation Router Embedding Extraction"
echo "============================================"
echo "Model:  $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Extract embeddings from MMLU validation data (GPU)
echo "=== Step 1: Extracting MMLU router embeddings ==="
python -u -m src.scripts.analysis.extract_router_embeddings_mmlu \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    2>&1 | tee "$OUTPUT_DIR/extraction.log"

# Step 2: Derive sparse variants (CPU, ~10 sec)
echo ""
echo "=== Step 2: Deriving sparse embeddings ==="
python -u -m src.scripts.analysis.sparsify_embeddings \
    --data-dir "$OUTPUT_DIR"

echo ""
echo "=== Done ==="
echo "Outputs in ${OUTPUT_DIR}/:"
ls -lh "${OUTPUT_DIR}"/embeddings_*.npy
echo ""
echo "Next: run clustering sweep with:"
echo "  bash scripts/ryanwang/analysis/run_sweep_focused.sh ${OUTPUT_DIR}"
