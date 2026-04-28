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

## Script layout

```
scripts/ryanwang/clustering/
├── README.md
├── benchmark_batch_size.py        # one-off GPU throughput benchmark
├── common/                        # source-agnostic primitives
│   ├── transform.sh               # derive <embedding> from logits
│   ├── cluster.sh                 # single cluster run (with save)
│   ├── visualize.sh               # simple HTML explorer (single view)
│   ├── visualize_token.sh         # rich 4-view token explorer (Clusters/
│   │                              # Documents/Tokens/UMAP + doc reader modal)
│   ├── visualize_compare.sh       # two-model side-by-side comparison HTML
│   │                              # (Clusters tab + Documents tab)
│   └── sample_clusters.sh         # per-cluster token contexts for labeling
├── pretraining/                   # pretraining-source specific
│   ├── extract.sh
│   ├── generate_mix.sh            # one-time S3 mix-composition
│   ├── sweep.sh                   # generic k-sweep (2 methods × 4 k)
│   └── sweep_two_models.sh        # two-1T-model harness over probs/topk
├── mmlu/
│   ├── extract.sh                 # 57 mmlu_merged_<subject> tasks
│   └── sweep.sh                   # MMLU-specific sweep: k=16 fixed,
│                                  # 3 embeddings × 2 balance × 2 methods
├── hellaswag/
│   └── extract.sh
└── weborganizer/                   # standalone doc-level expert coverage
    ├── extract.sh                  # → embeddings_doc_topk_freq.npy
    │                               #   embeddings_doc_probs.npy
    └── plot.sh                     # → 5 heatmaps per embedding type
```

## Quick Start

```bash
# 0. Generate pretraining mix composition (one-time)
bash scripts/ryanwang/clustering/pretraining/generate_mix.sh

# 1. Extract logits from pretraining data (1M tokens, 100 tok/doc)
bash scripts/ryanwang/clustering/pretraining/extract.sh

# 2. Derive probs from logits
bash scripts/ryanwang/clustering/common/transform.sh \
    claude_outputs/clustering/pretraining/<model> probs

# 3. Cluster
bash scripts/ryanwang/clustering/common/cluster.sh \
    claude_outputs/clustering/pretraining/<model>

# 4. Visualize
bash scripts/ryanwang/clustering/common/visualize.sh \
    claude_outputs/clustering/pretraining/<model>/probs_mean_pca_l2_spherical_kmeans_k64
```

## Step 1: Extract

Saves per-token router logits from any data source. Always token-level — document-level is recovered by aggregation in step 2.

```bash
# Pretraining (S3-based, shuffled, truncated)
bash scripts/ryanwang/clustering/pretraining/extract.sh [MODEL_PATH] [TARGET_TOKENS] [MAX_TOKENS_PER_DOC]

# MMLU (57 per-subject mmlu_merged_<subject>:rc_validation::olmes tasks.
#       The "merged validation" pool is test[:60%]+validation shuffled
#       per subject. All prompts used — no subsampling. 5-shot
#       subject-matched demos built by OLMES. Source label in metadata
#       is "mmlu_merged_<subject>".)
bash scripts/ryanwang/clustering/mmlu/extract.sh [MODEL_PATH]

# HellaSwag (hellaswag_merged:rc_validation::olmes — merged train+val
#           set. Subsampled to --num-calibration (default 100) via the
#           same seeded torch.randperm as easy_ep_prune. Source label
#           "hellaswag_merged".)
bash scripts/ryanwang/clustering/hellaswag/extract.sh [MODEL_PATH]
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
| `--max-tokens-per-doc` | 250 (pretraining), 0 (tasks) | Truncate docs. 0 = no truncation. |
| `--target-tokens` | 1,000,000 | Post-truncation token budget (pretraining only) |
| `--shuffle-seed` | 42 | Random seed for S3 sampling (pretraining only) |
| `--num-calibration` | 100 | Per-task prompt cap for mmlu / hellaswag. Seeded torch.randperm matches easy_ep_prune / greedy_prune_layerwise. Set `<=0` to disable. |
| `--calibration-seed` | 0 | Seed for the calibration subsample. Matches the pruning pipeline default. |

## Step 2: Transform

Derive embeddings from raw logits.

```bash
bash scripts/ryanwang/clustering/common/transform.sh <DATA_DIR> <DERIVE>
```

### Available derivations

| Name | Output | Description |
|------|--------|-------------|
| `probs` | `embeddings_probs.npy` | Per-token softmax probabilities (all layers, num_layers×num_experts dims) |
| `topk_binary` | `embeddings_topk_binary.npy` | Per-token binary top-k mask (all layers) |
| `layer0_probs` | `embeddings_layer0_probs.npy` | Per-token softmax for **layer 0 only** (num_experts dims) |
| `doc_probs` | `embeddings_doc_probs.npy` | Mean softmax per document (all layers) |
| `doc_logits` | `embeddings_doc_logits.npy` | Mean logits per document (all layers) |
| `doc_topk_freq` | `embeddings_doc_topk_freq.npy` | Top-k selection frequency per document (all layers) |
| `doc_layer0_probs` | `embeddings_doc_layer0_probs.npy` | Mean layer-0 softmax per document (num_experts dims) |

Document-level derivations (`doc_*`) automatically aggregate token embeddings using `doc_boundaries.npy`. They will auto-derive their token-level dependency if needed (e.g., `doc_topk_freq` derives `topk_binary` first).

## Step 3: Cluster

```bash
# Single run with save (default: probs / mean_pca_l2 / spherical_kmeans / k=64)
bash scripts/ryanwang/clustering/common/cluster.sh <DATA_DIR>

# Custom config
bash scripts/ryanwang/clustering/common/cluster.sh <DATA_DIR> doc_topk_freq mean_pca_l2 spherical_kmeans 32

# Generic k-sweep (pretraining-shaped harness)
bash scripts/ryanwang/clustering/pretraining/sweep.sh <DATA_DIR>

# MMLU-specific sweep (k fixed to 16, iterates embeddings × balance × method)
bash scripts/ryanwang/clustering/mmlu/sweep.sh <MMLU_DATA_DIR>

# Class-balanced clustering (stratified subsample before PCA)
bash scripts/ryanwang/clustering/common/cluster.sh <DATA_DIR> doc_probs mean_pca_l2 spherical_kmeans 32 source
bash scripts/ryanwang/clustering/common/cluster.sh <DATA_DIR> doc_probs mean_pca_l2 spherical_kmeans 32 source 100
```

### Class balancing

Raw MMLU / pretraining sources have heavily imbalanced class sizes (e.g. `professional_law` is 1090 docs vs `college_chemistry` at 68). Balancing stratified-subsamples the rows by a metadata field before preprocessing, so PCA and k-means see equal-weight classes.

| Flag | Default | Description |
|------|---------|-------------|
| `--balance-by` | unset | Metadata key to stratify on (typical: `source`). Unset = no balancing. |
| `--balance-n` | min class count | Per-class cap. Classes with fewer rows are kept in full. |
| `--balance-seed` | 42 | Seed for the stratified draw. |

`cluster.sh` and `sweep.sh` take `[BALANCE_BY]` and `[BALANCE_N]` as 6th and 7th positional args. Cache filenames include the balance config, so balanced and unbalanced runs don't overwrite each other (`preprocessed_<emb>_<prep>_bal<key>N<n>seed<s>.npy` + `.meta.json`).

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

Two explorers:

```bash
# Simple single-view (Clusters + UMAP)
bash scripts/ryanwang/clustering/common/visualize.sh <CLUSTER_DIR>

# Rich 4-view token explorer (token-level only)
#   Clusters  -- source breakdown + doc previews with per-token highlighting
#   Documents -- per-doc cluster-spread histogram + full-doc reader modal
#   Tokens    -- per-unique-token cluster distribution & entropy
#   UMAP      -- cluster/source/doc-spread colorings
bash scripts/ryanwang/clustering/common/visualize_token.sh <CLUSTER_DIR>

# Two-model side-by-side comparison (ModMoE vs Standard MoE style)
#   Clusters tab  -- both cluster lists side-by-side; click a cluster to see
#                    docs with its tokens highlighted; click a doc to jump.
#   Documents tab -- same doc rendered twice, each token colored by its
#                    respective model's cluster; click a cluster legend chip
#                    to filter-highlight that cluster.
# Both cluster dirs must come from extraction runs sharing the same
# shuffle seed (identical documents.npy).
bash scripts/ryanwang/clustering/common/visualize_compare.sh \
    <CLUSTER_DIR_1> <CLUSTER_DIR_2> "ModMoE" "Standard MoE" out.html
```

Both write `cluster_explorer.html` into `<CLUSTER_DIR>`. Reads optional
`cluster_labels.json` (`{cluster_id: {label, category}}`) if present to use
human labels instead of `Cluster N`.

## Step 5: Label clusters (optional)

To human-label clusters, sample token contexts per cluster:

```bash
bash scripts/ryanwang/clustering/common/sample_clusters.sh <CLUSTER_DIR> [N_SAMPLES]
```

Writes `<CLUSTER_DIR>/cluster_samples/cluster_NN.txt` — one file per cluster
with source breakdown, top-frequent and top-enriched tokens, and N sampled
context snippets. Use the contents to hand-write `<CLUSTER_DIR>/cluster_labels.json`;
then re-run `visualize_token.sh` to embed the labels.

## Directory Structure

```
claude_outputs/clustering/
├── pretraining/<model>/
├── mmlu/<model>/
│   ├── embeddings_logits.npy              # raw (from extract)
│   ├── embeddings_{probs,topk_binary,layer0_probs}.npy  # token-level derived
│   ├── embeddings_doc_{probs,logits,topk_freq,layer0_probs}.npy  # doc-level derived
│   ├── documents.npy, doc_boundaries.npy
│   ├── metadata_tokens.jsonl.gz, metadata_docs.jsonl.gz
│   ├── info.json, extraction.log
│   ├── preprocessed_<emb>_<prep>.npy                    # preprocess cache (unbalanced)
│   ├── preprocessed_<emb>_<prep>_bal<key>N<n>seed<s>.{npy,meta.json}
│   │                                                    # preprocess cache (balanced)
│   └── <emb>_<prep>_<method>_k<K>/
│       ├── assignments.npy, run_info.json, summary.json
│       └── cluster_explorer.html
└── hellaswag/<model>/
```

## Weborganizer doc-level expert coverage

A **standalone** sub-pipeline for measuring how MoE experts cover the 24
cc_all_dressed weborganizer topics. Does not share extract.py / transform.py
with the rest of this folder — extraction goes straight from raw S3 docs to
per-document aggregates in a single forward pass, so no per-token data is
ever persisted.

```bash
# 1. Extract (~20M tokens, uniformly sampled across topics, shuffled seed=42)
bash scripts/ryanwang/clustering/weborganizer/extract.sh [MODEL_PATH] [TARGET_TOKENS]

# 2. Plot heatmaps (defaults to both embedding types, 10 PNGs)
bash scripts/ryanwang/clustering/weborganizer/plot.sh <DATA_DIR> [topk_freq|probs|both]
```

### Two per-doc embeddings (computed in the same forward pass)

| File | Description | Layer slice sums to |
|------|-------------|---------------------|
| `embeddings_doc_topk_freq.npy` | Top-k expert selection frequency. For each (doc, layer, expert): (# tokens whose top-`routed_top_k` includes this expert) / doc_len. Mirrors `analyze_expert_coverage.py`. | `routed_top_k` |
| `embeddings_doc_probs.npy` | Mean softmax probability per expert across the doc's tokens. Softmax is taken over standard experts only (shared experts excluded), matching the slicing convention used by the analysis pipeline. | `1.0` |

Both shape `(num_docs, num_layers * num_standard_experts)`, float32.

### Output layout

```
claude_outputs/clustering/weborganizer/
├── mix_composition.json                  # shared across model runs
└── <model_name>/
    ├── embeddings_doc_topk_freq.npy
    ├── embeddings_doc_probs.npy
    ├── metadata_docs.jsonl.gz            # per-doc {doc_index, source, doc_len, preview}
    ├── info.json                         # model + extraction config
    ├── extraction.log
    └── PNG plots (5 per embedding type, prefixed by emb name)
```

### Plots

For each `<prefix>` ∈ {`doc_topk_freq`, `doc_probs`}:

| File | Content |
|------|---------|
| `<prefix>_coverage_above_uniform_heatmap.png` | topics × layers, # experts with weight `> 1/num_experts`. Universal threshold — directly comparable across the two embedding types. |
| `<prefix>_coverage_nonzero_heatmap.png` | topics × layers, # experts with weight `> 0`. Meaningful for `topk_freq`; degenerate (= num_experts) for `probs`. Emitted for symmetry. |
| `<prefix>_entropy_heatmap.png` | topics × layers, entropy in bits of the per-doc, per-layer expert distribution (renormalized to sum to 1 first), averaged over docs. Max = `log2(num_experts)`. |
| `<prefix>_similarity_heatmap.png` | 2×2 panel of topic-topic cosine similarities at four evenly-spaced layers (`np.linspace(0, num_layers-1, 4)`). |
| `<prefix>_l2_distance_heatmap.png` | 2×2 panel of topic-topic L2 distances at the same layers. |

Topic ordering is **shared** across all plots and all model runs via
`claude_outputs/clustering/weborganizer/topic_order.json`. The file is
created on the first plot invocation (using that embedding's ascending
mean-entropy order) and reused by every subsequent run, so model-vs-model
and topk_freq-vs-probs comparisons are visually aligned. To re-derive the
order from a different reference, delete `topic_order.json` and rerun
`plot.sh` on the new reference first.

## Python Modules

All modules run via `python -m src.scripts.clustering.<module>`.

| Module | Description |
|--------|-------------|
| `generate_pretraining_mix` | Generate pretraining data composition |
| `extract` | Unified extraction for all data sources (token-level logits) |
| `extract_document` | Standalone weborganizer doc-level extractor (topk_freq + probs in one pass) |
| `transform` | Derive embeddings + preprocessing |
| `cluster` | Clustering + evaluation |
| `visualize` | Simple HTML explorer (single view) |
| `visualize_token` | Rich 4-view token explorer (Clusters/Documents/Tokens/UMAP + doc reader) |
| `visualize_compare` | Two-model side-by-side comparison (Clusters + Documents tabs) |
| `sample_clusters` | Per-cluster token sampling for manual labeling |
| `plot_doc_expert_coverage` | Topic-vs-layer + topic-topic heatmaps from extract_document output |
| `utils` | Shared S3, tokenization, model loading |
