#!/bin/bash
# Pipeline: analyze data composition → extract router embeddings → cluster
#
# Step 1 (analyze_data_mix) is fast (~2 min) and only needs to be run once.
# Step 2 (extract_router_embeddings) is the long step (~2-4 hrs for 500M tokens).
# Step 3 (cluster_embeddings) will be run interactively after consulting on k.
#
# All outputs go to claude_outputs/analysis/router_clustering/

set -e

MODEL_PATH="models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf"
OUTPUT_DIR="claude_outputs/analysis/router_clustering_pretraining"
COMPOSITION_FILE="${OUTPUT_DIR}/mix_composition.json"
TARGET_TOKENS=20000000
BATCH_SIZE=8

# ---------------------------------------------------------------------------
# Step 1: Analyze data mix composition (run once; skip if already done)
# ---------------------------------------------------------------------------
if [ ! -f "$COMPOSITION_FILE" ]; then
    echo "=== Step 1: Analyzing data mix composition ==="
    python -u -m src.scripts.analysis.analyze_data_mix \
        --mix-file src/olmo_core/data/mixes/OLMoE-mix-0824.txt \
        --output-dir "$OUTPUT_DIR" \
        --num-preview-docs 2 \
        --stream-bytes 3000000
else
    echo "=== Step 1: Composition file already exists, skipping ==="
    echo "    (delete ${COMPOSITION_FILE} to re-run)"
fi

# ---------------------------------------------------------------------------
# Step 2: Extract router embeddings (proportional sampling, ~500M tokens)
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
