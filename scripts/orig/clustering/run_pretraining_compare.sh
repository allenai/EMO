#!/bin/bash
# End-to-end pretraining clustering comparison: generate_mix -> extract ->
# transform -> cluster -> visualize_compare for two HF Hub MoE checkpoints.
#
# Per model (loops over MODELS):
#   1. Extract token-level router logits from the pretraining mix.
#   2. Transform: derive softmax probs from logits.
#   3. Cluster: PCA + spherical k-means at k=32 on `probs`.
# Then runs visualize_compare.sh to render a side-by-side HTML.
#
# Final cluster config matches the published runs:
#   probs_mean_pca_l2_spherical_kmeans_k32
#
# All artifacts land under cluster_eval_final/ at the repo root:
#   cluster_eval_final/
#   ├── pretraining_mix.json
#   └── pretraining/
#       ├── Emo_1b14b_1T/...
#       └── StdMoE_1b14b_1T/...
#
# Usage:
#   bash scripts/clustering/run_pretraining_compare.sh
#
# Override:
#   CUDA_VISIBLE_DEVICES=0 bash ...     # restrict GPUs
#   TARGET_TOKENS=500000 bash ...       # smaller extraction budget
#   MAX_TOKENS_PER_DOC=200 bash ...     # different doc truncation
#   CLUSTER_ROOT=/tmp/foo bash ...      # custom output root
set -euo pipefail

# Hardcoded final config (matches the probs_mean_pca_l2_spherical_kmeans_k32 dirs).
EMBEDDING="probs"
PREPROCESS="mean_pca_l2"
METHOD="spherical_kmeans"
K=32

TARGET_TOKENS="${TARGET_TOKENS:-1000000}"
MAX_TOKENS_PER_DOC="${MAX_TOKENS_PER_DOC:-100}"

# Output root. Holds pretraining_mix.json plus per-model extraction/cluster dirs.
CLUSTER_ROOT="${CLUSTER_ROOT:-cluster_eval_final}"
COMPOSITION_FILE="${CLUSTER_ROOT}/pretraining_mix.json"
PRETRAINING_DIR="${CLUSTER_ROOT}/pretraining"

# (HF Hub id, output subdir name, display label) — listed in the order they
# appear in the comparison HTML (left, right).
MODELS=(
    "allenai/Emo_1b14b_1T:Emo_1b14b_1T:EMO"
    "allenai/StdMoE_1b14b_1T:StdMoE_1b14b_1T:Standard MoE"
)

CLUSTER_SUBDIR="${EMBEDDING}_${PREPROCESS}_${METHOD}_k${K}"

# 0. Generate the pretraining-mix composition file (one-time; skip if cached).
if [ ! -f "${COMPOSITION_FILE}" ]; then
    echo "=== Generating pretraining mix composition: ${COMPOSITION_FILE}"
    OUTPUT_DIR="${CLUSTER_ROOT}" \
        bash scripts/clustering/pretraining/generate_mix.sh
else
    echo "=== Reusing existing pretraining mix: ${COMPOSITION_FILE}"
fi

# Per-model: extract -> transform -> cluster
for entry in "${MODELS[@]}"; do
    IFS=':' read -r model_id output_name label <<< "$entry"
    data_dir="${PRETRAINING_DIR}/${output_name}"

    echo ""
    echo "=========================================="
    echo "=== Model: ${model_id}  (${label})"
    echo "=== Output: ${data_dir}"
    echo "=========================================="

    # 1. Extract — token logits from the pretraining mix.
    MODEL_NAME="${output_name}" \
    BASE_DIR="${PRETRAINING_DIR}" \
    COMPOSITION_FILE="${COMPOSITION_FILE}" \
        bash scripts/clustering/pretraining/extract.sh \
            "${model_id}" "${TARGET_TOKENS}" "${MAX_TOKENS_PER_DOC}"

    # 2. Transform — derive softmax probs from logits.
    bash scripts/clustering/common/transform.sh "${data_dir}" "${EMBEDDING}"

    # 3. Cluster — PCA + spherical k-means at k=K on probs.
    bash scripts/clustering/common/cluster.sh \
        "${data_dir}" "${EMBEDDING}" "${PREPROCESS}" "${METHOD}" "${K}"
done

# 4. Visualize — side-by-side HTML.
# MODELS[0] becomes the LEFT panel, MODELS[1] the RIGHT panel.
IFS=':' read -r _ name1 label1 <<< "${MODELS[0]}"
IFS=':' read -r _ name2 label2 <<< "${MODELS[1]}"

CLUSTER_DIR_1="${PRETRAINING_DIR}/${name1}/${CLUSTER_SUBDIR}"
CLUSTER_DIR_2="${PRETRAINING_DIR}/${name2}/${CLUSTER_SUBDIR}"
OUTPUT_HTML="${PRETRAINING_DIR}/compare_${name1}_vs_${name2}.html"

bash scripts/clustering/common/visualize_compare.sh \
    "${CLUSTER_DIR_1}" "${CLUSTER_DIR_2}" \
    "${label1}" "${label2}" \
    "${OUTPUT_HTML}"

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Comparison HTML: ${OUTPUT_HTML}"
echo "=========================================="
