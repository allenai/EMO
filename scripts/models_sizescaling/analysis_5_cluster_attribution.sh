#!/bin/bash
# Analysis 5: are the document clusters driven by individual experts or by the
# broad activation pattern?
#
# For each model: (a) reproduce the published doc_probs clustering
# (mean_pca_l2 + spherical k-means, k=32) if not already saved, then (b) run
# the attribution tests (signature concentration, single-dim AUC, drop/keep
# ablations with matched random controls). Reads the analysis-1 extractions;
# CPU only, ~12 min for all four models in parallel.
#
# Usage:
#   bash scripts/models_sizescaling/analysis_5_cluster_attribution.sh
set -euo pipefail

MODELS=(emo_1b4b_130b emo_1b7b_130b emo_1b11b_130b emo_1b14b_130b)
WB_ROOT="claude_outputs/models_sizescaling/weborganizer"
OUT_ROOT="claude_outputs/models_sizescaling/expert_attribution"
K=32

mkdir -p "${OUT_ROOT}"

for m in "${MODELS[@]}"; do
    (
        data_dir="${WB_ROOT}/${m}"
        out_dir="${OUT_ROOT}/${m}"
        mkdir -p "${out_dir}"
        if [ ! -f "${data_dir}/doc_probs_mean_pca_l2_spherical_kmeans_k${K}/assignments.npy" ]; then
            OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 \
                python -m src.scripts.clustering.cluster \
                --data-dir "${data_dir}" --embedding doc_probs \
                --preprocess mean_pca_l2 --method spherical_kmeans \
                --k "${K}" --save
        fi
        OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 \
            python -m src.scripts.clustering.cluster_expert_attribution \
            --data-dir "${data_dir}" --embedding doc_probs \
            --preprocess mean_pca_l2 --method spherical_kmeans \
            --k "${K}" --output-dir "${out_dir}"
    ) > "${OUT_ROOT}/${m}.log" 2>&1 &
done
wait
echo "Analysis 5 complete: ${OUT_ROOT}/"
