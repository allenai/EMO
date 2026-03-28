#!/bin/bash
# Extract router embeddings from HellaSwag data (all splits).
#
# Extracts train (38,905), validation (1,000), and test (10,042) examples.
# Output: claude_outputs/analysis/router_clustering_hellaswag/<model_name>/
#
# Usage:
#   bash scripts/ryanwang/analysis/run_extract_embeddings_hellaswag.sh [MODEL_PATH]
set -e

MODEL_PATH="${1:-models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf}"
MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")"
BASE_DIR="claude_outputs/analysis/router_clustering_hellaswag"
OUTPUT_DIR="${BASE_DIR}/${MODEL_NAME}"
BATCH_SIZE=32

echo "============================================"
echo "HellaSwag Router Embedding Extraction"
echo "============================================"
echo "Model:  $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Extract embeddings from all HellaSwag splits (GPU)
echo "=== Step 1: Extracting HellaSwag router embeddings ==="
python -u -m src.scripts.analysis.extract_router_embeddings_hellaswag \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    2>&1 | tee "$OUTPUT_DIR/extraction.log"

# Step 2: Derive sparse variants (CPU, ~10 sec)
echo ""
echo "=== Step 2: Deriving sparse embeddings ==="
python -u -m src.scripts.analysis.sparsify_embeddings \
    --data-dir "$OUTPUT_DIR"

# Step 3: Split into train+val subset for clustering (excludes test)
echo ""
echo "=== Step 3: Creating train+val subset for clustering ==="
python scripts/ryanwang/analysis/split_hellaswag_train_val.py "$OUTPUT_DIR"

echo ""
echo "=== Done ==="
echo "Outputs in ${OUTPUT_DIR}/:"
ls -lh "${OUTPUT_DIR}"/embeddings_*.npy
echo ""
echo "Train+val subset for clustering: ${OUTPUT_DIR}/train_val/"
echo "Next: run clustering sweep on train_val/:"
echo "  OPENBLAS_NUM_THREADS=16 python -u -m src.scripts.analysis.transform_and_cluster \\"
echo "    --data-dir ${OUTPUT_DIR}/train_val --embedding topk_freq --transform mean_pca_l2 \\"
echo "    --cluster kmeans --k 8 16 32"
