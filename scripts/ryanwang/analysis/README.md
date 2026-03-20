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
extract_router_embeddings.py       → embeddings from pretraining data (S3-based)
extract_router_embeddings_mmlu.py  → embeddings from MMLU validation data (HF-based)
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

#### Document-level (default, `--granularity document`)

All embeddings have shape `(num_docs, num_layers * num_experts)` = `(N, 2032)` for a 16-layer, 127-expert model.

| Name | File | Description |
|------|------|-------------|
| `logits` | `embeddings_logits.npy` | Average pre-softmax router logits per expert per layer |
| `probs` | `embeddings_probs.npy` | Per-token softmax probabilities, averaged per expert per layer |
| `logits_sparse` | `embeddings_logits_sparse.npy` | Sparse logits: top-32 experts per layer, rest zeroed (~25% density) |
| `probs_sparse` | `embeddings_probs_sparse.npy` | Sparse probs: top-32 experts per layer, rest zeroed (~25% density) |
| `topk_freq` | `embeddings_topk_freq.npy` | Top-k expert selection frequency per expert per layer |

#### Token-level (`--granularity token`)

Per-token activations without document-level aggregation. Shape `(num_tokens, num_layers * num_experts)`.
Also saves source documents for context recovery.

| Name | File | Description |
|------|------|-------------|
| `logits` | `embeddings_token_logits.npy` | Raw router logits per token (float16) |
| `probs` | `embeddings_token_probs.npy` | Softmax probabilities per token (float16) |
| `topk_binary` | `embeddings_token_topk_binary.npy` | Binary top-k mask per token (uint8) |
| — | `documents.npy` | All document tokens concatenated (int32) |
| — | `doc_boundaries.npy` | Document start indices; doc `i` = `documents[bounds[i]:bounds[i+1]]` |
| — | `metadata_tokens.jsonl.gz` | Per-token: source, doc_index, token_position, token_id |
| — | `metadata_docs.jsonl.gz` | Per-document: source, doc_len |

```bash
# Token-level extraction (shuffled, ~100K tokens by default)
bash scripts/ryanwang/analysis/run_extract_embeddings_shuffled_token.sh models/<model_name>/step<N>-hf

# Token-level extraction with custom token count (e.g. 1M tokens)
bash scripts/ryanwang/analysis/run_extract_embeddings_shuffled_token.sh models/<model_name>/step<N>-hf 1000000

# Token-level extraction with per-doc truncation (1M tokens, 100 tokens/doc → ~10K docs)
# Increases document diversity by using only the first N tokens per document.
bash scripts/ryanwang/analysis/run_extract_embeddings_shuffled_token_truncated.sh models/<model_name>/step<N>-hf 1000000 100
```

### MMLU Validation Embeddings

Extract router embeddings from all 57 MMLU subjects' validation data. Each subject
is treated as a separate "source" so clustering can discover groupings that may
differ from the 17 human-defined MMLU categories.

```bash
# Default model
bash scripts/ryanwang/analysis/run_extract_embeddings_mmlu.sh

# Specific model
bash scripts/ryanwang/analysis/run_extract_embeddings_mmlu.sh models/<model_name>/step<N>-hf
```

Output goes to `claude_outputs/analysis/router_clustering_mmlu_val/<model_name>/`.
Metadata includes both subject (as `source`) and category for downstream comparison.

**Script**: `src/scripts/analysis/extract_router_embeddings_mmlu.py`

## Step 3: Transform, Cluster, and Sweep

Transform embeddings, cluster, and evaluate. Pass the per-model data directory.

### Focused sweeps (recommended)

The `run_sweep_focused*.sh` scripts run targeted sweep grids based on prior results.
Each script documents its grid and rationale in comments at the top. Edit the
`DATA_DIR` variable inside the script to point at your model's data directory.

```bash
# v1: baseline grid — {probs, topk_freq} × {identity, l2} × kmeans × k={32,64,128}
bash scripts/ryanwang/analysis/run_sweep_focused.sh

# v2: based on v1 findings — topk_freq × mean_pca_l2 × {kmeans, spherical_kmeans} × k={16,32}
bash scripts/ryanwang/analysis/run_sweep_focused_v2.sh

# v3: hierarchical clustering — {topk_freq, probs} × {cosine, euclidean, jensenshannon} × {flat, per_layer} × average × k={16,32,64,128}
bash scripts/ryanwang/analysis/run_sweep_focused_v3.sh
```

Results are saved to `sweep_results_focused*.tsv` in the model's data directory.
See `sweep_notes.md` (per-model) for full results tables and takeaways.

### Legacy sweep (deprecated)

`run_sweep_all.sh` runs an exhaustive grid over all embedding types, transforms,
and cluster methods. It was used for the original `router_clustering_pretraining/`
and `router_clustering_weborganizer/` analyses. Superseded by the focused sweeps above.

### Single run

```bash
DATA_DIR="claude_outputs/analysis/router_clustering_pretraining_shuffled/<model_name>"

# Single run
bash scripts/ryanwang/analysis/run_transform_and_cluster.sh "$DATA_DIR" probs mean_pca_l2 spherical_kmeans 32

# Single run with saving (writes assignments.npy, summary.json, run_info.json)
bash scripts/ryanwang/analysis/run_transform_and_cluster.sh "$DATA_DIR" probs mean_pca_l2 spherical_kmeans 32 --save
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
| `run_extract_embeddings.sh` | `[MODEL_PATH]` | Extract embeddings (GPU, sequential sampling) + sparsify |
| `run_extract_embeddings_shuffled.sh` | (none) | Extract document-level embeddings (GPU, shuffled sampling) for randpool + baseline |
| `run_extract_embeddings_shuffled_token.sh` | `[MODEL_PATH] [TARGET_TOKENS]` | Extract token-level embeddings (GPU, shuffled, default 100K tokens) |
| `run_extract_embeddings_shuffled_token_truncated.sh` | `[MODEL_PATH] [TARGET_TOKENS] [MAX_TOKENS_PER_DOC]` | Token-level embeddings with per-doc truncation (default 1M tokens, 100 tokens/doc) |
| `run_sweep_focused.sh` | (none) | Sweep v1: baseline grid (edit DATA_DIR inside) |
| `run_sweep_focused_v2.sh` | (none) | Sweep v2: PCA + spherical k-means (edit DATA_DIR inside) |
| `run_sweep_focused_v3.sh` | (none) | Sweep v3: hierarchical clustering (edit DATA_DIR inside) |
| `run_sweep_all.sh` | `<DATA_DIR>` | **(deprecated)** Full sweep over all combinations |
| `run_transform_and_cluster.sh` | `<DATA_DIR> <emb> <transform> <cluster> <k> [--save]` | Single transform+cluster run |
| `run_extract_embeddings_mmlu.sh` | `[MODEL_PATH]` | Extract embeddings from MMLU validation data (GPU) + sparsify |
| `run_expert_coverage.sh` | `[MODEL_PATH]` | Expert coverage analysis |
| `push_router_clustering.sh` | (none) | Sync outputs to S3 |
| `pull_router_clustering.sh` | (none) | Pull outputs from S3 |

## Cumulative Expert Probability Mass Analysis

**Motivation**: During pre-training with top-p routing, the effective k (number of experts selected) tends to be relatively small. This analysis investigates why by examining how probability mass is distributed across experts for each document and layer.

**Method**: For each document and each layer, experts are sorted by their average token-level probability (descending). The cumulative probability mass is then computed over the sorted experts, yielding a curve from k=1 to num_experts (where the final value is 1.0). This reveals how concentrated vs. diffuse the router's probability distribution is.

**Script**: `src/scripts/analysis/plot_cumulative_expert_mass.py`

```bash
python -u -m src.scripts.analysis.plot_cumulative_expert_mass \
    --emb-file "$DATA_DIR/embeddings_probs.npy" \
    --info-file "$DATA_DIR/info.json" \
    --output-dir "$DATA_DIR/cumulative_mass" \
    --model-label "descriptive label"
```

**Output**: Per-layer scatterplots (4x4 grid, one per layer) with p10/p50/p90 percentile lines and mean overlay. Also produces a zoomed version focusing on the top-32 experts. Saved to `<model_dir>/cumulative_mass/`.

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
