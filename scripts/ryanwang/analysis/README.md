# Router Clustering Analysis Scripts

## Overview

These scripts analyze how MoE routers implicitly cluster documents by examining per-layer expert activation patterns. The pipeline extracts router embeddings from a trained model, clusters them, and produces interactive HTML visualizations.

## Pipeline

```
analyze_data_mix / analyze_weborganizer     (1. characterize data sources)
        ↓
extract_router_embeddings                   (2. GPU inference → embedding .npy files)
        ↓
exclude_layers (optional)                   (3. extract layer subsets for ablations)
layer_pattern_analysis (optional)           (3b. per-layer diversity stats)
        ↓
cluster_embeddings --mode sweep             (4. k-means sweep → pick best k)
        ↓
cluster_embeddings --mode cluster           (5. final clustering at chosen k)
        ↓
generate_cluster_viz                        (6. UMAP + interactive HTML explorer)
        ↓
extend_previews (optional)                  (7. re-stream S3 for longer doc previews)
```

## Shell Scripts (runners)

| Script | Purpose |
|--------|---------|
| `run_cluster_analysis.sh` | End-to-end pipeline for **pretraining** data (OLMoE-mix-0824). Runs steps 1-2. |
| `run_weborganizer_analysis.sh` | End-to-end pipeline for **weborganizer** data (cc_all_dressed). Runs steps 1-2. |
| `run_optB_layer_ablation.sh` | Layer ablation study for optB embeddings. Generates layer subsets (L1-15, L6-10, L15), runs pattern analysis, and sweeps k-means for each. |
| `run_optC_ablation.sh` | Transform ablation study for optC embeddings. Compares different PCA/normalization pipelines. |
| `push_router_clustering.sh` | Sync `claude_outputs/analysis/` to S3 (`s3://ai2-sewonm/ryanwang/`). |
| `pull_router_clustering.sh` | Sync from S3 back to local. |

## Python Scripts (`src/scripts/analysis/`)

### Data Preparation

| Script | Purpose |
|--------|---------|
| `analyze_data_mix.py` | Scans OLMoE-mix-0824 S3 paths to compute per-source token fractions. Outputs `mix_composition.json`. |
| `analyze_weborganizer.py` | Same for cc_all_dressed/weborganizer data with uniform mixing across topics. |

### Embedding Extraction

| Script | Purpose |
|--------|---------|
| `extract_router_embeddings.py` | Runs documents through the MoE model (GPU) and records router activations. Produces 5 embedding types: optA (avg softmax), optB (binary top-k mask), optC (sparse top-32 softmax), optD (sparse top-32 logits), optE (binary top-32). Also saves `metadata.jsonl.gz` with doc previews (up to 3000 chars). |

### Layer Analysis & Subsetting

| Script | Purpose |
|--------|---------|
| `exclude_layers.py` | Extracts a subset of layers from full embeddings. Uses `--keep-layers` (e.g., `6-10`, `15`, `1-15`). Output named by layers kept (e.g., `_L6-10.npy`). |
| `layer_pattern_analysis.py` | Reports per-layer routing diversity: unique patterns, Shannon entropy, coverage stats, active experts per doc. Saves JSON. |

### Clustering

| Script | Purpose |
|--------|---------|
| `cluster_embeddings.py` | PCA + k-means clustering pipeline. Two modes: `sweep` (try multiple k values, produce elbow/silhouette plots) and `cluster` (final clustering at chosen k, produce summary.json with per-cluster stats and representative docs). Supports multiple transform pipelines via `--transform` (pca_l2, raw, l2, log_l2, etc.). |

### Visualization

| Script | Purpose |
|--------|---------|
| `generate_cluster_viz.py` | Generates interactive HTML cluster explorer with UMAP scatter plot, per-cluster source breakdowns, representative docs, and document browser. Loads cluster labels from `cluster_labels.json` if present (auto-generated or hand-labeled). |
| `extend_previews.py` | Re-streams S3 data (no GPU needed) to replace short doc previews with longer ones. Updates metadata, summary.json, and regenerates HTML. |

## Output Directory Layout

```
claude_outputs/analysis/router_clustering_<experiment>/
    mix_composition.json          ← data source fractions
    metadata.jsonl.gz             ← per-doc source, length, preview text
    info.json                     ← model config + extraction stats
    layer_pattern_analysis.json   ← per-layer diversity stats
    embeddings_optA_avgprob.npy   ← (N, num_layers * num_experts) float16
    embeddings_optB_binary.npy    ← (N, num_layers * num_experts) bool
    embeddings_optB_binary_L6-10.npy  ← layer subset variants
    embeddings_optC_*.npy, optD_*.npy, optE_*.npy
    optB_binary/pca_l2/           ← one subdir per embedding+transform combo
        kmeans_sweep.json/png
        pca_variance.png
        clusters_k128/
            assignments.npy
            summary.json
            cluster_labels.json   ← auto-generated descriptive labels
            umap_coords.npy
        cluster_explorer.html     ← interactive visualizer
    optB_binary_L6-10/pca_l2/     ← same structure per variant
        ...
```
