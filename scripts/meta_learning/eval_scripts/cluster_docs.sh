#!/usr/bin/env bash
# PARENT: "scripts/modular_extension/cluster_docs_joint_100b_130b.sh"
# DESCRIPTION:
#     Per-arm k=32 partition of the meta_learning 20B-40B doc window: merge the arm's
#     128 embedding shards, fit ONE spherical k-means over all 18.3M docs (doc_probs ->
#     mean_pca_l2, the published clustering recipe), and export the provenance-keyed
#     partition. CPU-only, local. Run after the arm's embed sweep is complete
#     (128/128 info-*.json).
#
#     The partition file is suffixed by arm (doc_clusters_k32_<MODEL>.jsonl.gz) because
#     both arms share the same extraction window but cluster it with their own routers.
#
#   MODEL=sametok_ws_lam05 bash scripts/meta_learning/eval_scripts/cluster_docs.sh
#   MODEL=vanilla          bash scripts/meta_learning/eval_scripts/cluster_docs.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

MODEL="${MODEL:?set MODEL=vanilla|sametok_ws_lam05|...}"
STEP="${STEP:-4768}"
BASE="meta_learning/cluster/${MODEL}_step${STEP}"
DATA_WINDOW="meta_learning/data/meta128_20B-40B"

K="${K:-32}"
EMBEDDING="${EMBEDDING:-doc_probs}"
PREPROCESS="${PREPROCESS:-mean_pca_l2}"
METHOD="${METHOD:-spherical_kmeans}"

# ground-truth doc count from the extraction manifest
EXPECT_DOCS=$(python -c "import json; print(json.load(open('${DATA_WINDOW}/manifest.json'))['stats']['docs'])")
echo "expecting ${EXPECT_DOCS} docs"

echo "=== 1/3 merge 128 shards into ${BASE}"
PYTHONPATH=.:src python -u -m src.scripts.clustering.build_doc_window_datadir \
    --embeddings-dir "${BASE}/embeddings" \
    --data-dir "${BASE}" \
    --shards 0-127 --num-shards 128 \
    --expect-docs "${EXPECT_DOCS}"

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
    --output-prefix "${DATA_WINDOW}/doc_clusters_k${K}_${MODEL}"

echo "DONE: partition at ${DATA_WINDOW}/doc_clusters_k${K}_${MODEL}.jsonl.gz ; clustering artifacts in ${RESULT_DIR}/"
