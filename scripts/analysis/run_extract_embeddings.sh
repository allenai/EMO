#!/bin/bash
# Extract router embeddings from a trained MoE model.
# Produces dense logits + probs embeddings, then derives sparse variants.
#
# Prerequisites: mix_composition.json must exist (run run_analyze_data_mix.sh first)
#
# Usage:
#   bash scripts/analysis/run_extract_embeddings.sh [MODEL_PATH]
#
# If MODEL_PATH is not provided, defaults to the sharedexp1 model.
# Output dir is auto-derived: claude_outputs/analysis/router_clustering_pretraining/<model_name>/
set -e

MODEL_PATH="${1:-models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf}"

# Derive model name (parent directory of the checkpoint)
MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")"
BASE_DIR="claude_outputs/analysis/router_clustering_pretraining"
OUTPUT_DIR="${BASE_DIR}/${MODEL_NAME}"
COMPOSITION_FILE="${BASE_DIR}/mix_composition.json"
TARGET_TOKENS=20000000
BATCH_SIZE=32

echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

if [ ! -f "$COMPOSITION_FILE" ]; then
    echo "ERROR: ${COMPOSITION_FILE} not found. Run run_analyze_data_mix.sh first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Step 1: Extract dense embeddings (GPU, ~35 min for 20M tokens)
echo "=== Extracting router embeddings ==="
python -u -m src.scripts.analysis.extract_router_embeddings \
    --model-path "$MODEL_PATH" \
    --composition-file "$COMPOSITION_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --target-tokens $TARGET_TOKENS \
    --batch-size $BATCH_SIZE

# Step 2: Derive sparse variants (CPU, ~10 sec)
echo ""
echo "=== Deriving sparse embeddings ==="
python -u -m src.scripts.analysis.sparsify_embeddings \
    --data-dir "$OUTPUT_DIR"

echo ""
echo "=== Done. Outputs in ${OUTPUT_DIR}/ ==="
ls -lh "${OUTPUT_DIR}"/embeddings_*.npy
