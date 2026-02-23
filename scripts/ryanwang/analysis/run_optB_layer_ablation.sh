#!/bin/bash
# Ablation: test how different layer subsets affect optB clustering quality.
#
# Variants:
#   L1-15  — exclude layer 0 (early prefix layer)
#   L6-10  — middle layers only (low diversity, high silhouette)
#   L15    — last layer only (highest diversity)
#
# Also runs per-layer pattern analysis to understand routing diversity.
#
# Requires: embeddings_optB_binary.npy already extracted in DATA_DIR.

set -e

DATA_DIR="claude_outputs/analysis/router_clustering_pretraining"
EMB_FILE="${DATA_DIR}/embeddings_optB_binary.npy"
NUM_LAYERS=16
NUM_EXPERTS=127

if [ ! -f "$EMB_FILE" ]; then
    echo "ERROR: $EMB_FILE not found. Run extraction first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 0: Per-layer pattern analysis
# ---------------------------------------------------------------------------
echo "=== Phase 0: Per-layer routing pattern analysis ==="
python -u -m src.scripts.analysis.layer_pattern_analysis \
    --emb-file "$EMB_FILE" \
    --num-layers $NUM_LAYERS --num-experts $NUM_EXPERTS

# ---------------------------------------------------------------------------
# Phase 1: Generate layer-subset variants
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 1: Generating layer-subset embedding variants ==="

declare -A VARIANTS
VARIANTS["L1-15"]="1-15"
VARIANTS["L6-10"]="6-10"
VARIANTS["L15"]="15"

for NAME in L1-15 L6-10 L15; do
    KEEP=${VARIANTS[$NAME]}
    echo ""
    echo "--- Keeping layers: $KEEP → $NAME ---"
    python -u -m src.scripts.analysis.exclude_layers \
        --emb-file "$EMB_FILE" \
        --keep-layers "$KEEP" \
        --num-layers $NUM_LAYERS --num-experts $NUM_EXPERTS
done

# ---------------------------------------------------------------------------
# Phase 2: K-means sweep for each variant
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 2: K-means sweep for each variant ==="

declare -A VARIANT_NLAYERS
VARIANT_NLAYERS["L1-15"]=15
VARIANT_NLAYERS["L6-10"]=5
VARIANT_NLAYERS["L15"]=1

for NAME in L1-15 L6-10 L15; do
    REMAINING=${VARIANT_NLAYERS[$NAME]}
    VARIANT_EMB="${DATA_DIR}/embeddings_optB_binary_${NAME}.npy"
    VARIANT_OUT="${DATA_DIR}/optB_binary_${NAME}/pca_l2"

    echo ""
    echo "--- Sweep: $NAME ($REMAINING layers) ---"
    mkdir -p "$VARIANT_OUT"

    OPENBLAS_NUM_THREADS=16 python -u \
        -m src.scripts.analysis.cluster_embeddings \
        --emb-file "$VARIANT_EMB" \
        --output-dir "$VARIANT_OUT" \
        --data-dir "$DATA_DIR" \
        --num-layers "$REMAINING" --num-experts $NUM_EXPERTS \
        --mode sweep --k-values 8 16 32 64 128 \
        --transform pca_l2 \
        2>&1 | tee "${VARIANT_OUT}/sweep.log"
done

# ---------------------------------------------------------------------------
# Phase 3: Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Done. Compare silhouette scores across variants ==="
echo ""
echo "Results:"
for NAME in L1-15 L6-10 L15; do
    SWEEP_JSON="${DATA_DIR}/optB_binary_${NAME}/pca_l2/kmeans_sweep.json"
    if [ -f "$SWEEP_JSON" ]; then
        echo "  $NAME: $(python3 -c "
import json
d = json.load(open('$SWEEP_JSON'))
best_idx = max(range(len(d['silhouettes'])), key=lambda i: d['silhouettes'][i])
print(f\"best silhouette={d['silhouettes'][best_idx]:.4f} @ k={d['k_values'][best_idx]}\")
")"
    fi
done
echo ""
echo "Layer pattern analysis saved to: ${DATA_DIR}/layer_pattern_analysis.json"
