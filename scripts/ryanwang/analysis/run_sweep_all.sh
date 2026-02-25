#!/bin/bash
# Full sweep over embedding × transform × cluster × k.
#
# Runs transform_and_cluster.py for every combination and collects all metrics
# into a single TSV file for easy comparison.
#
# GMM is skipped for transforms that don't reduce dimensionality (identity, l2)
# since full-covariance GMM on 2032 dims will OOM/segfault.
#
# Usage:
#   bash scripts/ryanwang/analysis/run_sweep_all.sh <DATA_DIR>
#
# Example:
#   bash scripts/ryanwang/analysis/run_sweep_all.sh \
#       claude_outputs/analysis/router_clustering_pretraining/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <DATA_DIR>"
    echo "  DATA_DIR: per-model directory containing embeddings_*.npy, metadata.jsonl.gz, info.json"
    exit 1
fi

DATA_DIR="$1"
OUTPUT_TSV="${DATA_DIR}/sweep_results.tsv"

EMBEDDINGS="logits probs logits_sparse probs_sparse"
TRANSFORMS="identity l2 mean_pca mean_pca_l2 mean_l2_pca tsvd l2_tsvd tsvd_l2"
CLUSTERS="kmeans gmm"
K_VALUES="8 16 32 64 128"

# Transforms that do NOT reduce dimensionality — skip GMM for these
NO_DIMREDUCE="identity l2"

# Back up existing TSV if present
if [ -f "$OUTPUT_TSV" ]; then
    BACKUP="${OUTPUT_TSV%.tsv}_$(date +%Y%m%d_%H%M%S).tsv"
    cp "$OUTPUT_TSV" "$BACKUP"
    echo "Backed up existing sweep results → $BACKUP"
fi

# Write TSV header
echo -e "embedding\ttransform\tcluster\tk\tsilhouette\tcalinski_harabasz\tdavies_bouldin\tcluster_size_min\tcluster_size_max\tcluster_size_median\tcluster_size_std\tavg_source_entropy" > "$OUTPUT_TSV"

COUNT=0
for EMB in $EMBEDDINGS; do
for TRANS in $TRANSFORMS; do
for CLUST in $CLUSTERS; do

    # Skip GMM when transform doesn't reduce dimensionality
    if [ "$CLUST" = "gmm" ]; then
        SKIP=false
        for nd in $NO_DIMREDUCE; do
            if [ "$TRANS" = "$nd" ]; then
                SKIP=true
                break
            fi
        done
        if [ "$SKIP" = "true" ]; then
            echo ""
            echo "  SKIP: $EMB / $TRANS / $CLUST (GMM needs dim reduction)"
            for K in $K_VALUES; do
                COUNT=$((COUNT + 1))
                echo -e "${EMB}\t${TRANS}\t${CLUST}\t${K}\tSKIPPED\t\t\t\t\t\t\t" >> "$OUTPUT_TSV"
            done
            continue
        fi
    fi

    echo ""
    echo "================================================================"
    echo "  Embedding: $EMB | Transform: $TRANS | Cluster: $CLUST"
    echo "================================================================"

    # Run sweep for all k values at once, write output to temp file for streaming
    LOGFILE=$(mktemp)
    OPENBLAS_NUM_THREADS=16 python -u -m src.scripts.analysis.transform_and_cluster \
        --data-dir "$DATA_DIR" \
        --embedding "$EMB" \
        --transform "$TRANS" \
        --cluster "$CLUST" \
        --k $K_VALUES 2>&1 | tee "$LOGFILE" || {
        echo "  FAILED: $EMB / $TRANS / $CLUST"
        for K in $K_VALUES; do
            COUNT=$((COUNT + 1))
            echo -e "${EMB}\t${TRANS}\t${CLUST}\t${K}\tERROR\t\t\t\t\t\t\t" >> "$OUTPUT_TSV"
        done
        rm -f "$LOGFILE"
        continue
    }

    # Parse metrics from log lines for each k
    for K in $K_VALUES; do
        COUNT=$((COUNT + 1))

        # Extract the block for this k value
        BLOCK=$(sed -n "/--- k=${K} ---/,/--- k=\|=== SWEEP/p" "$LOGFILE")

        # Parse metrics — field after the metric name (field 6 in log format: timestamp [LEVEL] name: VALUE ...)
        SIL=$(echo "$BLOCK" | grep "silhouette:" | head -1 | awk -F'silhouette: *' '{print $2}' | awk '{print $1}')
        CH=$(echo "$BLOCK" | grep "calinski_harabasz:" | head -1 | awk -F'calinski_harabasz: *' '{print $2}' | awk '{print $1}')
        DB=$(echo "$BLOCK" | grep "davies_bouldin:" | head -1 | awk -F'davies_bouldin: *' '{print $2}' | awk '{print $1}')
        SZ_MIN=$(echo "$BLOCK" | grep "cluster sizes:" | head -1 | sed 's/.*min=\([0-9]*\).*/\1/')
        SZ_MAX=$(echo "$BLOCK" | grep "cluster sizes:" | head -1 | sed 's/.*max=\([0-9]*\).*/\1/')
        SZ_MED=$(echo "$BLOCK" | grep "cluster sizes:" | head -1 | sed 's/.*median=\([0-9]*\).*/\1/')
        SZ_STD=$(echo "$BLOCK" | grep "cluster sizes:" | head -1 | sed 's/.*std=\([0-9.]*\).*/\1/')
        SRC_ENT=$(echo "$BLOCK" | grep "avg_source_entropy:" | head -1 | awk -F'avg_source_entropy: *' '{print $2}' | awk '{print $1}')

        echo -e "${EMB}\t${TRANS}\t${CLUST}\t${K}\t${SIL:-NA}\t${CH:-NA}\t${DB:-NA}\t${SZ_MIN:-NA}\t${SZ_MAX:-NA}\t${SZ_MED:-NA}\t${SZ_STD:-NA}\t${SRC_ENT:-NA}" >> "$OUTPUT_TSV"

        echo "  [$COUNT] $EMB / $TRANS / $CLUST / k=$K  sil=${SIL:-NA}"
    done

    rm -f "$LOGFILE"

done; done; done

echo ""
echo "================================================================"
echo "  SWEEP COMPLETE: $COUNT runs"
echo "  Results: $OUTPUT_TSV"
echo "================================================================"

# Print top-10 by silhouette
echo ""
echo "=== TOP 10 BY SILHOUETTE ==="
head -1 "$OUTPUT_TSV"
tail -n +2 "$OUTPUT_TSV" | grep -v ERROR | grep -v SKIPPED | sort -t'	' -k5 -rn | head -10
