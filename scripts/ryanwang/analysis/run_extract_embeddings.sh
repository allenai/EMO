#!/bin/bash
# Extract router embeddings from a trained MoE model.
# Produces dense logits + probs embeddings, then derives sparse variants.
#
# Prerequisites: mix_composition.json must exist (run run_analyze_data_mix.sh first)
set -e

MODEL_PATH="models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf"
OUTPUT_DIR="claude_outputs/analysis/router_clustering_pretraining"
COMPOSITION_FILE="${OUTPUT_DIR}/mix_composition.json"
TARGET_TOKENS=20000000
BATCH_SIZE=32

if [ ! -f "$COMPOSITION_FILE" ]; then
    echo "ERROR: ${COMPOSITION_FILE} not found. Run run_analyze_data_mix.sh first."
    exit 1
fi

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
