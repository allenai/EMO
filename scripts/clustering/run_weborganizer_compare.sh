#!/bin/bash
# End-to-end weborganizer expert-coverage analysis: extract -> plot for two
# HF Hub MoE checkpoints.
#
# Per model (loops over MODELS):
#   1. Extract per-document expert-activation embeddings (top-k frequency
#      and softmax probs) on the cc_all_dressed weborganizer topic mix.
#   2. Plot 5 expert-coverage heatmaps per embedding type (10 PNGs).
#
# Both models share a single topic_order.json so heatmaps are visually
# comparable side-by-side (same row/column ordering).
#
# All artifacts land under cluster_eval_final/ at the repo root:
#   cluster_eval_final/
#   └── weborganizer/
#       ├── mix_composition.json   (auto-generated on first run)
#       ├── topic_order.json       (shared row/column ordering)
#       ├── Emo_1b14b_1T/
#       │   ├── embeddings_doc_topk_freq.npy
#       │   ├── embeddings_doc_probs.npy
#       │   └── *.png              (5 per embedding type)
#       └── StdMoE_1b14b_1T/
#           └── (same structure)
#
# Usage:
#   bash scripts/clustering/run_weborganizer_compare.sh
#
# Override:
#   CUDA_VISIBLE_DEVICES=0 bash ...     # restrict GPUs
#   TARGET_TOKENS=10000000 bash ...     # smaller extraction budget (default 20M)
#   CLUSTER_ROOT=/tmp/foo bash ...      # custom output root
set -euo pipefail

TARGET_TOKENS="${TARGET_TOKENS:-20000000}"

# Output root. Holds shared mix_composition.json + topic_order.json plus
# per-model extraction/plot dirs.
CLUSTER_ROOT="${CLUSTER_ROOT:-cluster_eval_final}"
WEBORG_DIR="${CLUSTER_ROOT}/weborganizer"

# (HF Hub id, output subdir name, display label)
MODELS=(
    "allenai/Emo_1b14b_1T:Emo_1b14b_1T:EMO"
    "allenai/StdMoE_1b14b_1T:StdMoE_1b14b_1T:Standard MoE"
)

# Per-model: extract -> plot
for entry in "${MODELS[@]}"; do
    IFS=':' read -r model_id output_name label <<< "$entry"
    data_dir="${WEBORG_DIR}/${output_name}"

    echo ""
    echo "=========================================="
    echo "=== Model: ${model_id}  (${label})"
    echo "=== Output: ${data_dir}"
    echo "=========================================="

    # 1. Extract — per-doc expert-coverage embeddings.
    MODEL_NAME="${output_name}" \
    BASE_DIR="${WEBORG_DIR}" \
        bash scripts/clustering/weborganizer/extract.sh \
            "${model_id}" "${TARGET_TOKENS}"

    # 2. Plot — heatmaps for both embedding types (topk_freq + probs).
    bash scripts/clustering/weborganizer/plot.sh "${data_dir}" both
done

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Heatmaps under: ${WEBORG_DIR}/{Emo_1b14b_1T,StdMoE_1b14b_1T}/*.png"
echo "=========================================="
