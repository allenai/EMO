# MoE Analysis Pipeline

Analyze MoE router activations: clustering into implicit domains and expert coverage across topics.

## Directory Structure

All analysis outputs live under `claude_outputs/analysis/`. Each analysis type uses a shared
top-level directory with per-model subdirectories:

```
claude_outputs/analysis/
├── router_clustering_pretraining/
│   ├── mix_composition.json                    # shared: data source fractions (one-time)
│   └── <model_name>/                           # per-model outputs
│       ├── embeddings_logits.npy
│       ├── embeddings_probs.npy
│       ├── embeddings_logits_sparse.npy
│       ├── embeddings_probs_sparse.npy
│       ├── metadata.jsonl.gz
│       ├── info.json
│       ├── sweep_results.tsv
│       └── probs_mean_pca_l2_gmm_k64/         # per-config clustering results
│           ├── assignments.npy
│           ├── summary.json
│           ├── run_info.json
│           ├── cluster_labels.json
│           ├── cluster_explorer.html
│           └── umap_coords.npy
│
├── expert_coverage_weborganizer/
│   ├── mix_composition.json                    # shared: uniform topic fractions
│   └── <model_name>/                           # per-model outputs
│       ├── expert_freq.npy
│       ├── topic_stats.json
│       └── *.png (heatmaps)
```

The `<model_name>` is the parent directory of the HF checkpoint path
(e.g. `models/twolevelbatchlbreducedp512sharedexp1-32_.../step30995-hf` → `twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211`).

## Pipeline Overview

```
utils.py                      → shared S3 streaming, token parsing, document loading, model loading
analyze_data_mix.py           → mix_composition.json (data source fractions)
analyze_weborganizer.py       → mix_composition.json (uniform fractions across weborganizer topics)
extract_router_embeddings.py  → embeddings_*.npy + metadata.jsonl.gz + info.json
sparsify_embeddings.py        → derived embeddings (sparse variants) from existing files
transform_and_cluster.py      → apply transforms + cluster + evaluate
generate_cluster_viz.py       → UMAP + interactive HTML visualizer
analyze_expert_coverage.py    → expert coverage analysis across weborganizer topics
plot_expert_coverage.py       → heatmap visualization of expert coverage per topic/layer
```

## Step 1: Analyze Data Composition

One-time step. Queries S3 for file sizes and computes per-source token fractions.

```bash
bash scripts/ryanwang/analysis/run_analyze_data_mix.sh
```

Output: `claude_outputs/analysis/router_clustering_pretraining/mix_composition.json`

## Step 2: Extract Router Embeddings

GPU inference. Runs documents through the model and captures router activations.

```bash
# Default model
bash scripts/ryanwang/analysis/run_extract_embeddings.sh

# Specific model
bash scripts/ryanwang/analysis/run_extract_embeddings.sh models/<model_name>/step<N>-hf
```

Output goes to `claude_outputs/analysis/router_clustering_pretraining/<model_name>/`.

### Embedding Types

All embeddings have shape `(num_docs, num_layers * num_experts)` = `(N, 2032)` for a 16-layer, 127-expert model.

| Name | File | Description |
|------|------|-------------|
| `logits` | `embeddings_logits.npy` | Average pre-softmax router logits per expert per layer |
| `probs` | `embeddings_probs.npy` | Per-token softmax probabilities, averaged per expert per layer |
| `logits_sparse` | `embeddings_logits_sparse.npy` | Sparse logits: top-32 experts per layer, rest zeroed (~25% density) |
| `probs_sparse` | `embeddings_probs_sparse.npy` | Sparse probs: top-32 experts per layer, rest zeroed (~25% density) |

## Step 3: Transform, Cluster, and Sweep

Transform embeddings, cluster, and evaluate. Pass the per-model data directory.

```bash
DATA_DIR="claude_outputs/analysis/router_clustering_pretraining/<model_name>"

# Sweep all combinations
bash scripts/ryanwang/analysis/run_sweep_all.sh "$DATA_DIR"

# Single run
bash scripts/ryanwang/analysis/run_transform_and_cluster.sh "$DATA_DIR" probs mean_pca_l2 gmm 64

# Single run with saving
bash scripts/ryanwang/analysis/run_transform_and_cluster.sh "$DATA_DIR" probs mean_pca_l2 gmm 64 --save
```

## Step 4: HTML Visualizer

UMAP projection + interactive browser. Reads from a saved clustering directory.

```bash
python -u -m src.scripts.analysis.generate_cluster_viz \
    --cluster-dir "$DATA_DIR/probs_mean_pca_l2_gmm_k64"
```

The `--data-dir` defaults to the parent of `--cluster-dir` (i.e. the per-model directory).

## Expert Coverage Analysis

Analyzes how MoE experts cover different weborganizer topic domains.

```bash
# Default model
bash scripts/ryanwang/analysis/run_expert_coverage.sh

# Specific model
bash scripts/ryanwang/analysis/run_expert_coverage.sh models/<model_name>/step<N>-hf
```

## Shell Scripts Summary

| Script | Args | Description |
|--------|------|-------------|
| `run_analyze_data_mix.sh` | (none) | One-time: compute data source fractions |
| `run_extract_embeddings.sh` | `[MODEL_PATH]` | Extract embeddings (GPU) + sparsify |
| `run_sweep_all.sh` | `<DATA_DIR>` | Full sweep over all combinations |
| `run_transform_and_cluster.sh` | `<DATA_DIR> <emb> <transform> <cluster> <k> [--save]` | Single transform+cluster run |
| `run_expert_coverage.sh` | `[MODEL_PATH]` | Expert coverage analysis |
| `push_router_clustering.sh` | (none) | Sync outputs to S3 |
| `pull_router_clustering.sh` | (none) | Pull outputs from S3 |

## Shared Utilities (`utils.py`)

Common functions used across all analysis scripts:

| Function | Description |
|----------|-------------|
| `s3_ls(prefix)` | List S3 directory children |
| `list_npy_files(topic, vigintile)` | List .npy files for a topic/vigintile |
| `stream_bytes_from_s3(path, n)` | Range-GET first N bytes from S3 |
| `tokens_from_bytes(raw)` | Parse uint32 binary to token IDs |
| `iter_documents(tokens)` | Split tokens on EOS into documents |
| `load_source_documents(files, target)` | Stream docs from S3 up to target tokens |
| `load_model_and_tokenizer(path)` | Load HF MoE model |
| `get_moe_config(model)` | Extract num_layers, num_experts, etc. |
