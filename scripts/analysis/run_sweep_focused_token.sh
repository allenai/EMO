#!/bin/bash
# Focused sweep on token-level embeddings:
#   {token_probs, token_topk_binary} × {identity, l2} × kmeans × k∈{32,64,128}
#
# Same grid as run_sweep_focused.sh but adapted for token-level data:
#   - Uses token_probs (per-token softmax) instead of probs (doc-averaged softmax)
#   - Uses token_topk_binary (per-token binary mask) instead of topk_freq (doc-averaged frequency)
#
# Usage:
#   bash scripts/analysis/run_sweep_focused_token.sh <DATA_DIR>
#
# Example:
#   bash scripts/analysis/run_sweep_focused_token.sh \
#       claude_outputs/analysis/router_clustering_pretraining_shuffled_token/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <DATA_DIR>"
    echo "  DATA_DIR: per-model directory containing embeddings_token_*.npy, metadata_tokens.jsonl.gz, info.json"
    exit 1
fi

DATA_DIR="$1"
OUTPUT_TSV="${DATA_DIR}/sweep_results_focused_token.tsv"

EMBEDDINGS="token_probs token_topk_binary"
TRANSFORMS="identity l2"
CLUSTER="kmeans"
K_VALUES="32 64 128"

# Back up existing TSV if present
if [ -f "$OUTPUT_TSV" ]; then
    BACKUP="${OUTPUT_TSV%.tsv}_$(date +%Y%m%d_%H%M%S).tsv"
    cp "$OUTPUT_TSV" "$BACKUP"
    echo "Backed up existing results → $BACKUP"
fi

# Write TSV header
echo -e "embedding\ttransform\tcluster\tk\tsilhouette_euclidean\tsilhouette_cosine\tcalinski_harabasz\tdavies_bouldin\tcluster_size_min\tcluster_size_max\tcluster_size_median\tcluster_size_std\tavg_source_entropy" > "$OUTPUT_TSV"

COUNT=0
for EMB in $EMBEDDINGS; do
for TRANS in $TRANSFORMS; do

    echo ""
    echo "================================================================"
    echo "  Embedding: $EMB | Transform: $TRANS | Cluster: $CLUSTER"
    echo "================================================================"

    LOGFILE=$(mktemp)
    OPENBLAS_NUM_THREADS=16 python -u -m src.scripts.analysis.transform_and_cluster \
        --data-dir "$DATA_DIR" \
        --embedding "$EMB" \
        --transform "$TRANS" \
        --cluster "$CLUSTER" \
        --k $K_VALUES 2>&1 | tee "$LOGFILE" || {
        echo "  FAILED: $EMB / $TRANS / $CLUSTER"
        for K in $K_VALUES; do
            COUNT=$((COUNT + 1))
            echo -e "${EMB}\t${TRANS}\t${CLUSTER}\t${K}\tERROR\t\t\t\t\t\t\t\t" >> "$OUTPUT_TSV"
        done
        rm -f "$LOGFILE"
        continue
    }

    for K in $K_VALUES; do
        COUNT=$((COUNT + 1))

        BLOCK=$(sed -n "/--- k=${K} ---/,/--- k=\|=== SWEEP/p" "$LOGFILE")

        SIL=$(echo "$BLOCK" | grep "silhouette:" | head -1 | awk -F'silhouette: *' '{print $2}' | awk '{print $1}')
        SIL_COS=$(echo "$BLOCK" | grep "silhouette_cosine:" | head -1 | awk -F'silhouette_cosine: *' '{print $2}' | awk '{print $1}')
        CH=$(echo "$BLOCK" | grep "calinski_harabasz:" | head -1 | awk -F'calinski_harabasz: *' '{print $2}' | awk '{print $1}')
        DB=$(echo "$BLOCK" | grep "davies_bouldin:" | head -1 | awk -F'davies_bouldin: *' '{print $2}' | awk '{print $1}')
        SZ_MIN=$(echo "$BLOCK" | grep "cluster sizes:" | head -1 | sed 's/.*min=\([0-9]*\).*/\1/')
        SZ_MAX=$(echo "$BLOCK" | grep "cluster sizes:" | head -1 | sed 's/.*max=\([0-9]*\).*/\1/')
        SZ_MED=$(echo "$BLOCK" | grep "cluster sizes:" | head -1 | sed 's/.*median=\([0-9]*\).*/\1/')
        SZ_STD=$(echo "$BLOCK" | grep "cluster sizes:" | head -1 | sed 's/.*std=\([0-9.]*\).*/\1/')
        SRC_ENT=$(echo "$BLOCK" | grep "avg_source_entropy:" | head -1 | awk -F'avg_source_entropy: *' '{print $2}' | awk '{print $1}')

        echo -e "${EMB}\t${TRANS}\t${CLUSTER}\t${K}\t${SIL:-NA}\t${SIL_COS:-NA}\t${CH:-NA}\t${DB:-NA}\t${SZ_MIN:-NA}\t${SZ_MAX:-NA}\t${SZ_MED:-NA}\t${SZ_STD:-NA}\t${SRC_ENT:-NA}" >> "$OUTPUT_TSV"

        echo "  [$COUNT] $EMB / $TRANS / k=$K  sil_euc=${SIL:-NA}  sil_cos=${SIL_COS:-NA}"
    done

    rm -f "$LOGFILE"

done; done

echo ""
echo "================================================================"
echo "  SWEEP COMPLETE: $COUNT runs"
echo "  Results: $OUTPUT_TSV"
echo "================================================================"

echo ""
echo "=== RESULTS (sorted by silhouette_cosine) ==="
column -t -s$'\t' "$OUTPUT_TSV" | head -1
tail -n +2 "$OUTPUT_TSV" | sort -t$'\t' -k6 -rn | column -t -s$'\t'
