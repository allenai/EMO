#!/bin/bash
# Pipeline: analyze all-dressed topic distribution → extract router embeddings → cluster
#
# Uses cc_weborganizer vigintile_0020 data with uniform mixing across 24 topics.
#
# Step 1 (analyze_weborganizer): Fast S3 scan, only needs to run once.
# Step 2 (extract_router_embeddings): Long step (~2-4 hrs for 500M tokens).
# Step 3 (cluster_embeddings): Run interactively after consulting on k.
#
# All outputs go to claude_outputs/analysis/router_clustering_weborganizer/

set -e

MODEL_PATH="models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf"
OUTPUT_DIR="claude_outputs/analysis/router_clustering_weborganizer"
COMPOSITION_FILE="${OUTPUT_DIR}/mix_composition.json"
TARGET_TOKENS=20000000
BATCH_SIZE=8

# ---------------------------------------------------------------------------
# Step 1: Analyze all-dressed data distribution (run once; skip if done)
# ---------------------------------------------------------------------------
if [ ! -f "$COMPOSITION_FILE" ]; then
    echo "=== Step 1: Analyzing all-dressed data distribution ==="
    python -u -m src.scripts.analysis.analyze_weborganizer \
        --output-dir "$OUTPUT_DIR" \
        --num-preview-docs 0
else
    echo "=== Step 1: Composition file already exists, skipping ==="
    echo "    (delete ${COMPOSITION_FILE} to re-run)"
fi

# ---------------------------------------------------------------------------
# Step 2: Extract router embeddings (uniform sampling, ~500M tokens)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Extracting router embeddings ==="
python -u -m src.scripts.analysis.extract_router_embeddings \
    --model-path "$MODEL_PATH" \
    --composition-file "$COMPOSITION_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --target-tokens $TARGET_TOKENS \
    --batch-size $BATCH_SIZE \
    --min-doc-len 32 \
    --max-doc-len 2048

echo ""
echo "=== Extraction complete. Review info.json then run cluster_embeddings.py ==="
echo "    Output: ${OUTPUT_DIR}/"
