#!/usr/bin/env bash
# Joint k=64 partition of the FULL 100B-130B doc stream: merge both windows' embedding
# shards (100B-110B `embeddings/` + 110B-130B `embeddings_110B-130B/`, both fingerprinted
# by the same EMO 100B checkpoint) into one data dir, fit ONE clustering over all docs,
# and export the provenance-keyed partition. CPU-only, local; needs ~350GB RAM peak.
#
# Run after BOTH embed sweeps are complete (128/128 info-*.json in each dir).
#
# Note: a long document whose shuffled chunks straddle the 110B boundary legitimately
# appears in both windows; such rows share the (source_path, doc_start_offset) key and
# get (near-)identical embeddings, so the exported partition may contain duplicate keys
# with consistent labels.
#
#   bash scripts/modular_extension/cluster_docs_joint_100b_130b.sh
#   K=32 bash scripts/modular_extension/cluster_docs_joint_100b_130b.sh   # other k, same merge
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUN="emo_64exp_50b_wsd_lr2e-3"
BASE="modular_extension/cluster/emo100b_step23842"
JOINT="${BASE}_100B-130B"
OUT_DATA="modular_extension/data/${RUN}_100B-130B"

K="${K:-64}"
EMBEDDING="${EMBEDDING:-doc_probs}"
PREPROCESS="${PREPROCESS:-mean_pca_l2}"
METHOD="${METHOD:-spherical_kmeans}"

# ground-truth doc count = sum of the two extraction manifests
EXPECT_DOCS=$(python - <<'PY'
import json
tot = 0
for w in ("100B-110B", "110B-130B"):
    m = json.load(open(f"modular_extension/data/emo_64exp_50b_wsd_lr2e-3_{w}/manifest.json"))
    tot += m["stats"]["docs"]
print(tot)
PY
)
echo "expecting ${EXPECT_DOCS} docs across both windows"

echo "=== 1/3 merge both windows (128 shards each) into ${JOINT}"
PYTHONPATH=.:src python -u -m src.scripts.clustering.build_doc_window_datadir \
    --embeddings-dir "${BASE}/embeddings,${BASE}/embeddings_110B-130B" \
    --data-dir "${JOINT}" \
    --shards 0-127 --num-shards 128 \
    --expect-docs "${EXPECT_DOCS}"

echo "=== 2/3 cluster ${EMBEDDING} ${PREPROCESS} ${METHOD} k=${K}"
PYTHONPATH=.:src python -u -m src.scripts.clustering.cluster \
    --data-dir "${JOINT}" \
    --embedding "${EMBEDDING}" \
    --preprocess "${PREPROCESS}" \
    --method "${METHOD}" \
    --k "${K}" \
    --save

RESULT_DIR="${JOINT}/${EMBEDDING}_${PREPROCESS}_${METHOD}_k${K}"
echo "=== 3/3 export partition"
mkdir -p "${OUT_DATA}"
PYTHONPATH=.:src python -u -m src.scripts.clustering.export_doc_partition \
    --data-dir "${JOINT}" \
    --result-dir "${RESULT_DIR}" \
    --output-prefix "${OUT_DATA}/doc_clusters_k${K}"

echo "DONE: partition at ${OUT_DATA}/doc_clusters_k${K}.jsonl.gz ; clustering artifacts in ${RESULT_DIR}/"
