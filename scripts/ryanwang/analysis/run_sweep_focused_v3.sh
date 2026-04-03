#!/bin/bash
# Focused sweep v3: hierarchical clustering with precomputed distance matrices.
#
# v1 found: topk_freq/l2/k=32 was best (sil_cos=0.2287)
# v2 found: topk_freq/mean_pca_l2/spherical_kmeans/k=32 improved to sil_cos=0.2490
# v3 tests: hierarchical clustering with different distance metrics and modes
#
# Grid:
#   Embeddings:   topk_freq, probs
#   Transform:    identity (distances are computed directly on raw embeddings)
#   Distance:     {cosine, euclidean, jensenshannon} × {flat, per_layer}
#   Linkage:      average
#   Clustering:   hierarchical
#   k:            16, 32, 64, 128
#
# Usage:
#   bash scripts/ryanwang/analysis/run_sweep_focused_v3.sh <DATA_DIR>
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <DATA_DIR>"
    echo "  DATA_DIR: per-model directory containing embeddings_*.npy, metadata.jsonl.gz, info.json"
    exit 1
fi

DATA_DIR="$1"
OUTPUT_TSV="${DATA_DIR}/sweep_results_focused_v3.tsv"

EMBEDDINGS="topk_freq probs"
DIST_METRICS="cosine euclidean jensenshannon"
DIST_MODES="flat per_layer"
LINKAGE="average"
K_VALUES="16 32 64 128"

# Back up existing TSV if present
if [ -f "$OUTPUT_TSV" ]; then
    BACKUP="${OUTPUT_TSV%.tsv}_$(date +%Y%m%d_%H%M%S).tsv"
    cp "$OUTPUT_TSV" "$BACKUP"
    echo "Backed up existing results → $BACKUP"
fi

# Write TSV header
echo -e "embedding\tdist_metric\tdist_mode\tlinkage\tk\tsilhouette\tcluster_size_min\tcluster_size_max\tcluster_size_median\tcluster_size_std\tavg_source_entropy" > "$OUTPUT_TSV"

COUNT=0
for EMB in $EMBEDDINGS; do
for DIST_METRIC in $DIST_METRICS; do
for DIST_MODE in $DIST_MODES; do

    echo ""
    echo "================================================================"
    echo "  Embedding: $EMB | Metric: $DIST_METRIC | Mode: $DIST_MODE | Linkage: $LINKAGE"
    echo "================================================================"

    LOGFILE=$(mktemp)
    OPENBLAS_NUM_THREADS=16 python -u -m src.scripts.analysis.transform_and_cluster \
        --data-dir "$DATA_DIR" \
        --embedding "$EMB" \
        --transform identity \
        --cluster hierarchical \
        --dist-metric "$DIST_METRIC" \
        --dist-mode "$DIST_MODE" \
        --linkage-method "$LINKAGE" \
        --k $K_VALUES 2>&1 | tee "$LOGFILE" || {
        echo "  FAILED: $EMB / $DIST_METRIC / $DIST_MODE / $LINKAGE"
        for K in $K_VALUES; do
            COUNT=$((COUNT + 1))
            echo -e "${EMB}\t${DIST_METRIC}\t${DIST_MODE}\t${LINKAGE}\t${K}\tERROR\t\t\t\t\t" >> "$OUTPUT_TSV"
        done
        rm -f "$LOGFILE"
        continue
    }

    for K in $K_VALUES; do
        COUNT=$((COUNT + 1))

        BLOCK=$(sed -n "/--- k=${K} ---/,/--- k=\|=== SWEEP/p" "$LOGFILE")

        SIL=$(echo "$BLOCK" | grep "silhouette:" | head -1 | awk -F'silhouette: *' '{print $2}' | awk '{print $1}')
        SZ_MIN=$(echo "$BLOCK" | grep "cluster sizes:" | head -1 | sed 's/.*min=\([0-9]*\).*/\1/')
        SZ_MAX=$(echo "$BLOCK" | grep "cluster sizes:" | head -1 | sed 's/.*max=\([0-9]*\).*/\1/')
        SZ_MED=$(echo "$BLOCK" | grep "cluster sizes:" | head -1 | sed 's/.*median=\([0-9]*\).*/\1/')
        SZ_STD=$(echo "$BLOCK" | grep "cluster sizes:" | head -1 | sed 's/.*std=\([0-9.]*\).*/\1/')
        SRC_ENT=$(echo "$BLOCK" | grep "avg_source_entropy:" | head -1 | awk -F'avg_source_entropy: *' '{print $2}' | awk '{print $1}')

        echo -e "${EMB}\t${DIST_METRIC}\t${DIST_MODE}\t${LINKAGE}\t${K}\t${SIL:-NA}\t${SZ_MIN:-NA}\t${SZ_MAX:-NA}\t${SZ_MED:-NA}\t${SZ_STD:-NA}\t${SRC_ENT:-NA}" >> "$OUTPUT_TSV"

        echo "  [$COUNT] $EMB / $DIST_METRIC / $DIST_MODE / $LINKAGE / k=$K  sil=${SIL:-NA}"
    done

    rm -f "$LOGFILE"

done; done; done

echo ""
echo "================================================================"
echo "  SWEEP v3 COMPLETE: $COUNT runs"
echo "  Results: $OUTPUT_TSV"
echo "================================================================"

echo ""
echo "=== TOP 15 BY SILHOUETTE ==="
column -t -s$'\t' "$OUTPUT_TSV" | head -1
tail -n +2 "$OUTPUT_TSV" | grep -v ERROR | sort -t$'\t' -k6 -rn | head -15 | column -t -s$'\t'
