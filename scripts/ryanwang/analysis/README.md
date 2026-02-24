# MoE Analysis Pipeline

Analyze MoE router activations: clustering into implicit domains and expert coverage across topics.

## Pipeline Overview

```
utils.py                      → shared S3 streaming, token parsing, document loading, model loading
analyze_data_mix.py           → mix_composition.json (data source fractions)
analyze_weborganizer.py       → mix_composition.json (uniform fractions across weborganizer topics)
extract_router_embeddings.py  → embeddings_*.npy + metadata.jsonl.gz + info.json
sparsify_embeddings.py        → derived embeddings (sparse variants) from existing files
cluster_embeddings.py         → k-means sweep/clustering + reports
generate_cluster_viz.py       → UMAP + interactive HTML visualizer
analyze_expert_coverage.py    → expert coverage analysis across weborganizer topics
plot_expert_coverage.py       → heatmap visualization of expert coverage per topic/layer
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

## Expert Coverage Analysis

Analyzes how MoE experts cover different weborganizer topic domains. Samples 20M tokens
uniformly across topics and runs them through the model.

```bash
# Using existing mix_composition.json from the weborganizer clustering analysis
python -u -m src.scripts.analysis.analyze_expert_coverage \
    --model-path models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf \
    --composition-file claude_outputs/analysis/router_clustering_weborganizer/mix_composition.json \
    --output-dir claude_outputs/analysis/expert_coverage_weborganizer \
    --target-tokens 20_000_000 \
    --batch-size 32

# Or without a composition file (auto-discovers topics on S3)
python -u -m src.scripts.analysis.analyze_expert_coverage \
    --model-path models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf \
    --output-dir claude_outputs/analysis/expert_coverage_weborganizer \
    --target-tokens 20_000_000

# Or use the shell wrapper
bash scripts/ryanwang/analysis/run_expert_coverage.sh

# Plot heatmap from results
python -u -m src.scripts.analysis.plot_expert_coverage \
    --stats-file claude_outputs/analysis/expert_coverage_weborganizer/topic_stats.json \
    --output-dir claude_outputs/analysis/expert_coverage_weborganizer
```

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

claude_outputs/analysis/expert_coverage_weborganizer/
    <model_name>/                 # one subdir per model (auto-derived from model path)
        mix_composition.json          # uniform topic fractions (auto-generated or copied)
        metadata.jsonl.gz             # per-doc metadata
        info.json                     # model + extraction params
        expert_freq.npy               # per-doc normalized frequencies (num_docs, num_layers*num_experts)
        topic_stats.json              # per-topic avg experts/layer and entropy/layer
        expert_coverage_heatmap.png   # heatmap: topics x layers x avg experts used
        expert_entropy_heatmap.png    # heatmap: topics x layers x entropy
        topic_similarity_heatmap.png  # 2x2 cosine similarity at L0/L5/L10/L15
        topic_l2_distance_heatmap.png # 2x2 L2 distance at L0/L5/L10/L15
```
