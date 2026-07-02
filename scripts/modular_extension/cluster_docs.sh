#!/usr/bin/env bash
# Merge doc-embedding shards, cluster at k=64 (doc_probs / mean_pca_l2 / spherical_kmeans,
# same recipe as the published pretraining clustering but document-level), and export the
# doc -> cluster partition. CPU-only; run locally after launch_embed_docs.sh jobs finish.
#
#   SHARDS=0-15 bash scripts/modular_extension/cluster_docs.sh          # small run subset
#   SHARDS=0-127 EXPECT_DOCS=9167088 bash scripts/modular_extension/cluster_docs.sh  # full
#   K=32 bash scripts/modular_extension/cluster_docs.sh                # other k on same embeddings
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUN="emo_64exp_50b_wsd_lr2e-3"
STEP=23842
BASE="modular_extension/cluster/emo100b_step${STEP}"
DATA_WINDOW="modular_extension/data/${RUN}_100B-110B"

SHARDS="${SHARDS:-0-15}"
NUM_SHARDS="${NUM_SHARDS:-128}"
K="${K:-64}"
EMBEDDING="${EMBEDDING:-doc_probs}"
PREPROCESS="${PREPROCESS:-mean_pca_l2}"
METHOD="${METHOD:-spherical_kmeans}"
EXPECT_DOCS="${EXPECT_DOCS:-}"

echo "=== 1/3 merge shards ${SHARDS} into ${BASE}"
PYTHONPATH=.:src python -u -m src.scripts.clustering.build_doc_window_datadir \
    --embeddings-dir "${BASE}/embeddings" \
    --data-dir "${BASE}" \
    --shards "${SHARDS}" --num-shards "${NUM_SHARDS}" \
    ${EXPECT_DOCS:+--expect-docs ${EXPECT_DOCS}}

echo "=== 2/3 cluster ${EMBEDDING} ${PREPROCESS} ${METHOD} k=${K}"
PYTHONPATH=.:src python -u -m src.scripts.clustering.cluster \
    --data-dir "${BASE}" \
    --embedding "${EMBEDDING}" \
    --preprocess "${PREPROCESS}" \
    --method "${METHOD}" \
    --k "${K}" \
    --save

RESULT_DIR="${BASE}/${EMBEDDING}_${PREPROCESS}_${METHOD}_k${K}"
echo "=== 3/3 export partition"
PYTHONPATH=.:src python -u -m src.scripts.clustering.export_doc_partition \
    --data-dir "${BASE}" \
    --result-dir "${RESULT_DIR}" \
    --output-prefix "${DATA_WINDOW}/doc_clusters_k${K}"

echo "DONE: partition at ${DATA_WINDOW}/doc_clusters_k${K}.jsonl.gz ; clustering artifacts in ${RESULT_DIR}/"
