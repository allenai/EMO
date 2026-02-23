# Router Clustering Analysis Pipeline

Analyze how MoE router activations cluster documents into implicit domains.

## Pipeline Overview

```
analyze_data_mix.py          → mix_composition.json (data source fractions)
extract_router_embeddings.py → embeddings_*.npy + metadata.jsonl.gz + info.json
compute_embeddings.py        → derived embeddings (sparse variants) from existing files
cluster_embeddings.py        → k-means sweep/clustering + reports
generate_cluster_viz.py      → UMAP + interactive HTML visualizer
```

## Step 1: Analyze Data Composition

One-time step. Queries S3 for file sizes and computes per-source token fractions.

```bash
python -u -m src.scripts.analysis.analyze_data_mix \
    --mix-file src/olmo_core/data/mixes/OLMoE-mix-0824.txt \
    --output-dir claude_outputs/analysis/router_clustering_pretraining \
    --num-preview-docs 2 \
    --stream-bytes 3000000
```

Output: `mix_composition.json`

## Step 2: Extract Router Embeddings

GPU inference. Runs documents through the model and captures router activations.
Proportionally samples documents from each data source.

```bash
python -u -m src.scripts.analysis.extract_router_embeddings \
    --model-path models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf \
    --composition-file claude_outputs/analysis/router_clustering_pretraining/mix_composition.json \
    --output-dir claude_outputs/analysis/router_clustering_pretraining \
    --target-tokens 20_000_000 \
    --batch-size 32
```

### Embedding Types

All embeddings have shape `(num_docs, num_layers * num_experts)` = `(N, 2032)` for a 16-layer, 127-expert model.

| Name | File | Description |
|------|------|-------------|
| `logits` | `embeddings_logits.npy` | Average pre-softmax router logits per expert per layer |
| `probs` | `embeddings_probs.npy` | Per-token softmax probabilities, averaged per expert per layer |
| `logits_sparse` | `embeddings_logits_sparse.npy` | Sparse logits: top-32 experts per layer, rest zeroed (~25% density) |
| `probs_sparse` | `embeddings_probs_sparse.npy` | Sparse probs: top-32 experts per layer, rest zeroed (~25% density) |

Use `--embeddings logits,probs` to select specific types (default: `all`).

### Adding New Embedding Types

In `extract_router_embeddings.py`:
1. Define a function with signature: `(per_layer_logits: List[Tensor], attention_mask: Tensor, num_layers: int, num_experts: int) -> np.ndarray`
2. Add an `EmbeddingType` entry to `EMBEDDING_REGISTRY`

## Step 2b: Derive Sparse Embeddings (from existing files)

CPU-only post-processing. Avoids re-running GPU extraction when dense embeddings already exist.

```bash
python -u -m src.scripts.analysis.compute_embeddings \
    --data-dir claude_outputs/analysis/router_clustering_pretraining
```

## Step 3: Cluster Embeddings

K-means sweep to find optimal k, then final clustering.

```bash
# Sweep
OPENBLAS_NUM_THREADS=16 python -u -m src.scripts.analysis.cluster_embeddings \
    --output-dir claude_outputs/analysis/router_clustering_pretraining/probs \
    --emb-file claude_outputs/analysis/router_clustering_pretraining/embeddings_probs.npy \
    --data-dir claude_outputs/analysis/router_clustering_pretraining \
    --mode sweep --k-values 8 16 32 64 128

# Final cluster (after reviewing sweep)
OPENBLAS_NUM_THREADS=16 python -u -m src.scripts.analysis.cluster_embeddings \
    --output-dir claude_outputs/analysis/router_clustering_pretraining/probs \
    --emb-file claude_outputs/analysis/router_clustering_pretraining/embeddings_probs.npy \
    --data-dir claude_outputs/analysis/router_clustering_pretraining \
    --mode cluster --k 128
```

## Step 4: HTML Visualizer

UMAP projection + interactive browser.

```bash
OPENBLAS_NUM_THREADS=16 python -u -m src.scripts.analysis.generate_cluster_viz \
    --output-dir claude_outputs/analysis/router_clustering_pretraining/probs \
    --emb-file claude_outputs/analysis/router_clustering_pretraining/embeddings_probs.npy \
    --data-dir claude_outputs/analysis/router_clustering_pretraining \
    --k 128
```

## Output Directory Structure

```
claude_outputs/analysis/router_clustering_pretraining/
    mix_composition.json          # data source fractions
    embeddings_logits.npy         # dense logits
    embeddings_probs.npy          # dense probs
    embeddings_logits_sparse.npy  # sparse logits (top-32)
    embeddings_probs_sparse.npy   # sparse probs (top-32)
    metadata.jsonl.gz             # per-doc metadata (source, length, preview)
    info.json                     # model info + extraction params
    probs/                        # clustering results for probs embedding
        kmeans_sweep.json/png
        pca_variance.png
        clusters_k128/
            assignments.npy
            summary.json
            report.txt
            cluster_explorer.html
```
