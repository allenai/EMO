# MoE Router Clustering Pipeline

Clean pipeline for clustering MoE router activations. Supports token-level and document-level clustering across multiple data sources.

## Pipeline

```
extract.py    →  raw per-token router logits from any data source
transform.py  →  derive representations (probs, topk, doc-level aggregation)
cluster.py    →  cluster + evaluate
visualize.py  →  interactive HTML explorer
```

**Key design**: Logits are the single primitive. Extraction saves only raw router logits per token. Everything else (softmax probs, top-k masks, document-level embeddings) is derived in the transform step.

## Quick Start

```bash
# 1. Extract logits from pretraining data (1M tokens, 100 tok/doc)
bash scripts/ryanwang/clustering/extract_pretraining.sh

# 2. Derive probs from logits
bash scripts/ryanwang/clustering/transform.sh \
    claude_outputs/clustering/pretraining/<model> probs

# 3. Cluster
bash scripts/ryanwang/clustering/cluster.sh \
    claude_outputs/clustering/pretraining/<model>

# 4. Visualize
bash scripts/ryanwang/clustering/visualize.sh \
    claude_outputs/clustering/pretraining/<model>/probs_mean_pca_l2_spherical_kmeans_k64
```

## Step 1: Extract

Saves per-token router logits from any data source. Always token-level — document-level is recovered by aggregation in step 2.

```bash
# Pretraining (S3-based, shuffled, truncated)
bash scripts/ryanwang/clustering/extract_pretraining.sh [MODEL_PATH] [TARGET_TOKENS] [MAX_TOKENS_PER_DOC]

# MMLU (all 57 subjects, validation split)
bash scripts/ryanwang/clustering/extract_mmlu.sh [MODEL_PATH]

# HellaSwag (all splits)
bash scripts/ryanwang/clustering/extract_hellaswag.sh [MODEL_PATH]
```

**Output format** (identical across sources):
```
embeddings_logits.npy       # (num_tokens, num_layers * num_experts), float16
documents.npy               # flat token IDs for context recovery
doc_boundaries.npy          # (num_docs + 1,) cumulative offsets
metadata_tokens.jsonl.gz    # per-token: {source, doc_index, token_position, token_id}
metadata_docs.jsonl.gz      # per-doc: {source, doc_len, ...}
info.json                   # extraction config
```

### Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--max-tokens-per-doc` | 100 (pretraining), 0 (tasks) | Truncate docs. 0 = no truncation. |
| `--target-tokens` | 1,000,000 | Post-truncation token budget (pretraining only) |
| `--shuffle-seed` | 42 | Random seed for S3 sampling (pretraining only) |
| `--subjects` | all 57 | Comma-separated MMLU subjects |
| `--hellaswag-splits` | train,validation,test | Comma-separated splits |

## Step 2: Transform

Derive embeddings from raw logits.

```bash
bash scripts/ryanwang/clustering/transform.sh <DATA_DIR> <DERIVE>
```

### Available derivations

| Name | Output | Description |
|------|--------|-------------|
| `probs` | `embeddings_probs.npy` | Per-token softmax probabilities |
| `topk_binary` | `embeddings_topk_binary.npy` | Per-token binary top-k mask |
| `doc_probs` | `embeddings_doc_probs.npy` | Mean softmax per document |
| `doc_logits` | `embeddings_doc_logits.npy` | Mean logits per document |
| `doc_topk_freq` | `embeddings_doc_topk_freq.npy` | Top-k selection frequency per document |

Document-level derivations (`doc_*`) automatically aggregate token embeddings using `doc_boundaries.npy`. They will auto-derive their token-level dependency if needed (e.g., `doc_topk_freq` derives `topk_binary` first).

## Step 3: Cluster

```bash
# Single run with save (default: probs / mean_pca_l2 / spherical_kmeans / k=64)
bash scripts/ryanwang/clustering/cluster.sh <DATA_DIR>

# Custom config
bash scripts/ryanwang/clustering/cluster.sh <DATA_DIR> doc_topk_freq mean_pca_l2 spherical_kmeans 32

# Sweep over k values
bash scripts/ryanwang/clustering/sweep.sh <DATA_DIR>
```

### Preprocessing options

| Name | Description |
|------|-------------|
| `identity` | No preprocessing |
| `l2` | L2 normalize |
| `mean_pca` | Mean-center + PCA (95% variance) |
| `mean_pca_l2` | Mean-center + PCA + L2 normalize **(recommended)** |

### Clustering methods

| Name | Description |
|------|-------------|
| `kmeans` | MiniBatchKMeans |
| `spherical_kmeans` | KMeans with centroid normalization **(recommended)** |
| `hierarchical` | Agglomerative with precomputed distances |
| `gmm` | Gaussian Mixture Model |

### Output (saved run)
```
<embedding>_<preprocess>_<method>_k<K>/
    assignments.npy     # cluster labels
    run_info.json       # full config + metrics
    summary.json        # per-cluster breakdown
```

## Step 4: Visualize

```bash
bash scripts/ryanwang/clustering/visualize.sh <CLUSTER_DIR>
```

Auto-detects token-level vs document-level from the embedding name in `run_info.json`. Generates `cluster_explorer.html` with UMAP scatter plot and cluster detail view.

## Directory Structure

```
claude_outputs/clustering/
├── pretraining/<model>/              # token-level extraction
│   ├── embeddings_logits.npy         # raw (from extract)
│   ├── embeddings_probs.npy          # derived (from transform)
│   ├── embeddings_doc_topk_freq.npy  # derived doc-level
│   ├── documents.npy, doc_boundaries.npy
│   ├── metadata_tokens.jsonl.gz, metadata_docs.jsonl.gz
│   ├── info.json
│   └── probs_mean_pca_l2_spherical_kmeans_k64/
│       ├── assignments.npy, run_info.json, summary.json
│       └── cluster_explorer.html
├── mmlu/<model>/                     # task extraction
└── hellaswag/<model>/                # task extraction
```

## Python Modules

All modules run via `python -m src.scripts.clustering.<module>`.

| Module | Description |
|--------|-------------|
| `extract` | Unified extraction for all data sources |
| `transform` | Derive embeddings + preprocessing |
| `cluster` | Clustering + evaluation |
| `visualize` | HTML explorer |
| `utils` | Shared S3, tokenization, model loading |
