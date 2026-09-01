#!/usr/bin/env bash
# DESCRIPTION:
#     Build one arm's per-cluster train token shards + held-out sets for the meta_learning
#     phase-2 k=32 CPT, by joining the arm's k=32 partition with the shared 20B-40B
#     extraction JSONLs (src.scripts.clustering.build_cluster_token_data, parameterized).
#     Budget 20B train tokens total (phase 2 = the run's second 20B), split across
#     clusters proportionally to cluster size. CPU, local, ~16 workers.
#
#   MODEL=sametok_ws_lam05 bash scripts/meta_learning/eval_scripts/build_cluster_token_data.sh
#   MODEL=vanilla          bash scripts/meta_learning/eval_scripts/build_cluster_token_data.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

MODEL="${MODEL:?set MODEL=vanilla|sametok_ws_lam05|...}"
WINDOW="meta_learning/data/meta128_20B-40B"

PYTHONPATH=.:src python -u -m src.scripts.clustering.build_cluster_token_data \
    --out "${WINDOW}/k32_cpt_tokens_${MODEL}" \
    --part "${WINDOW}/doc_clusters_k32_${MODEL}.jsonl.gz" \
    --window-dirs "${WINDOW}" \
    --budget 20e9 \
    --k 32
