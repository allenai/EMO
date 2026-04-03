#!/bin/bash
# Extract router embeddings with shuffled (random) document sampling.
#
# Same as run_extract_embeddings.sh, but uses --shuffle to randomly sample
# documents from across all files in each source, instead of reading
# sequentially from the start. This avoids bias from unshuffled .npy files.
#
# Prerequisites: mix_composition.json must exist (run run_analyze_data_mix.sh first)
#
# Usage:
#   bash scripts/ryanwang/analysis/run_extract_embeddings_shuffled.sh [MODEL_PATH]
#
# Output goes to: claude_outputs/analysis/router_clustering_pretraining_shuffled/<model_name>/
set -e

MODEL_PATH="${1:-models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf}"

# Derive model name (parent directory of the checkpoint)
MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")"
BASE_DIR="claude_outputs/analysis/router_clustering_pretraining_shuffled"
# Reuse the same composition file from the non-shuffled version
COMPOSITION_FILE="claude_outputs/analysis/router_clustering_pretraining/mix_composition.json"
OUTPUT_DIR="${BASE_DIR}/${MODEL_NAME}"
TARGET_TOKENS=20000000
BATCH_SIZE=32

echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Sampling: SHUFFLED (random offsets across all files)"

if [ ! -f "$COMPOSITION_FILE" ]; then
    echo "ERROR: ${COMPOSITION_FILE} not found. Run run_analyze_data_mix.sh first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Step 1: Extract dense embeddings with shuffled sampling (GPU, ~35 min for 20M tokens)
echo "=== Extracting router embeddings (shuffled) ==="
python -u -m src.scripts.analysis.extract_router_embeddings \
    --model-path "$MODEL_PATH" \
    --composition-file "$COMPOSITION_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --target-tokens $TARGET_TOKENS \
    --batch-size $BATCH_SIZE \
    --shuffle

# Step 2: Derive sparse variants (CPU, ~10 sec)
echo ""
echo "=== Deriving sparse embeddings ==="
python -u -m src.scripts.analysis.sparsify_embeddings \
    --data-dir "$OUTPUT_DIR"

echo ""
echo "=== Done. Outputs in ${OUTPUT_DIR}/ ==="
ls -lh "${OUTPUT_DIR}"/embeddings_*.npy
