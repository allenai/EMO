<div align="center">
  <img src="assets/Emo_Logo.png" alt="Emo Logo" style="width: 300px; margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
</div>

<p align="center">
  <a href="LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/badge/license-TODO-lightgrey">
  </a>
  <a href="#">
    <img alt="Blog Post" src="https://img.shields.io/badge/Emo-blog-F0529C">
  </a>
  <a href="#">
    <img alt="Paper URL" src="https://img.shields.io/badge/Emo-arxiv-blue">
  </a>
  <a href="https://huggingface.co/collections/allenai/emo">
    <img alt="Model Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow">
  </a>
</p>

EMO is a new Mixture-of-Experts model trained so that modular structure emerges during pretraining without requiring human-defined priors. EMO enables selective expert use, down to 12.5% of total experts, with minimal performance degradation. We find that its expert groups specialize to higher-level topics and capabilities rather than low-level lexical patterns.

## Table of Contents

- [Installation](#installation)
- [Released Models](#released-models)
    - [Main Release (1T tokens)](#main-release-1t-tokens)
    - [Ablation Models (130B tokens)](#ablation-models-130b-tokens)
    - [Midtraining Ablation Models](#midtraining-ablation-models)
- [Inference](#inference)
    - [With Hugging Face Transformers](#with-hugging-face-transformers)
    - [With vLLM](#with-vllm)
- [Training scripts](#training-scripts)
- [Evaluation scripts](#evaluation-scripts)
    - [Selective Expert Usage](#selective-expert-usage)
    - [Clustering Pretraining Document Tokens](#clustering-pretraining-document-tokens)
    - [Weborganizer Expert Coverage](#weborganizer-expert-coverage)
- [Contact and Contributing](#contact-and-contributing)
- [Citing](#citing)

## Installation

```bash
git clone https://github.com/allenai/EMO.git
cd EMO
conda create -n emo python==3.12
uv pip install -e .[all]
```

## Released Models

All checkpoints are available in the [EMO collection](https://huggingface.co/collections/allenai/emo) on the Hugging Face Hub.

### Main Release (1T tokens)

| Model | Active | Total | Pretraining (1T) | Annealing (50B) | Description |
|---|---|---|---|---|---|
| [`allenai/Emo_1b14b_1T`](https://huggingface.co/allenai/Emo_1b14b_1T) | 1B | 14B | EMO [\[train_script\]](scripts/models/emo_1b14b_1t.sh) | EMO [\[train_script\]](scripts/models/emo_1b14b_1t_emoanneal.sh) | Main EMO release |
| [`allenai/StdMoE_1b14b_1T`](https://huggingface.co/allenai/StdMoE_1b14b_1T) | 1B | 14B | standard [\[train_script\]](scripts/models/stdmoe_1b14b_1t.sh) | standard [\[train_script\]](scripts/models/stdmoe_1b14b_1t_stdanneal.sh) | Architecture-matched standard MoE baseline |

### Ablation Models (130B tokens)

Smaller-scale checkpoints used for memory-matched comparisons. These models were not midtrained. 

| Model | Active | Total | Pretraining (130B) | Description |
|---|---|---|---|---|
| [`allenai/Emo_1b14b_130B`](https://huggingface.co/allenai/Emo_1b14b_130B) | 1B | 14B | EMO [\[train_script\]](scripts/models/emo_1b14b_130b.sh) | EMO at the 130B-token ablation scale |
| [`allenai/StdMoE_1b14b_130B`](https://huggingface.co/allenai/StdMoE_1b14b_130B) | 1B | 14B | standard [\[train_script\]](scripts/models/stdmoe_1b14b_130b.sh) | Standard MoE baseline at the 130B-token scale |
| [`allenai/StdMoE_1b4b_130B`](https://huggingface.co/allenai/StdMoE_1b4b_130B) | 1B | 4B | standard [\[train_script\]](scripts/models/stdmoe_1b4b_130b.sh) | Memory-matched standard MoE with 32 experts ("Reg. MoE @ 32" in Figure 4), used as a memory-matched baseline for EMO's 32-expert subsets |
| [`allenai/Dense_1b_130B`](https://huggingface.co/allenai/Dense_1b_130B) | 1B | 1B | dense LM [\[train_script\]](scripts/models/dense_1b_130b.sh) | Dense baseline matched to active parameters ("Dense @ 8" in Figure 4), used as a memory-matched baseline for EMO's 8-expert subsets |

### Midtraining Ablation Models

Checkpoints used in Appendix B.3 to test whether modularity can be induced after pretraining via annealing alone, rather than during pretraining.

| Model | Active | Total | Pretraining (1T) | Annealing (50B) | Description                                                                                                             |
|---|---|---|---|---|-------------------------------------------------------------------------------------------------------------------------|
| [`allenai/StdMoE_1b14b_1T_Preanneal`](https://huggingface.co/allenai/StdMoE_1b14b_1T_Preanneal) | 1B | 14B | standard [\[train_script\]](scripts/models/stdmoe_1b14b_1t.sh) | — | Standard MoE checkpoint after 1T-token pretraining, before any annealing. Starting point for the EMO-anneal experiment |
| [`allenai/StdMoE_1b14b_1T_EmoAnnealed`](https://huggingface.co/allenai/StdMoE_1b14b_1T_EmoAnnealed) | 1B | 14B | standard [\[train_script\]](scripts/models/stdmoe_1b14b_1t.sh) | EMO [\[train_script\]](scripts/models/stdmoe_1b14b_1t_emoanneal.sh) | EMO-anneal: a standard MoE annealed under the document-level expert pool constraint for 50B tokens |


## Inference

See [Released Models](#released-models) for the available checkpoints. All inference snippets below require `trust_remote_code=True` since the models use custom modeling code from the [ryanyxw/transformers](https://github.com/ryanyxw/transformers/tree/flexmoe_v4_57_1) fork (Note: you do not need to clone this fork yourself, the Hugging Face Hub will pull the necessary code when you load the model with `trust_remote_code=True`).

### With Hugging Face Transformers

You can use our Hugging Face [transformers](https://github.com/huggingface/transformers) integration to run inference on the released checkpoints:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "allenai/Emo_1b14b_1T"
olmo = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# inputs = {k: v.to('cuda') for k,v in inputs.items()} # optional verifying cuda
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=1.0, top_p=0.7)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

Alternatively, with the Hugging Face pipeline abstraction:

```python
from transformers import pipeline
olmo_pipe = pipeline("text-generation", model="allenai/Emo_1b14b_1T", trust_remote_code=True)
print(olmo_pipe("Language modeling is"))
```

### With vLLM

[vLLM](https://docs.vllm.ai/en/latest/) provides high-throughput inference. We ship a small out-of-tree plugin at [`src/vllm_plugin/`](src/vllm_plugin/) that registers `EmoForCausalLM` with vLLM's native model registry

```bash
pip install vllm>=0.11.0
pip install -e src/vllm_plugin  # optional; only needed for the native path
```

You can run offline batched inference:

```python
from vllm import LLM, SamplingParams
llm = LLM(model="allenai/Emo_1b14b_1T", trust_remote_code=True)
sampling_params = SamplingParams(temperature=1.0, top_p=0.7)
prompts = ["Language modeling is"]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

For more details, see the [vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart/#offline-batched-inference).

## Training scripts

Project-specific pretraining recipes live in [`scripts`](scripts/). Please refer to [Released Models](#released-models) for the training scripts corresponding to each released checkpoint.

Run a script locally:

```bash
bash scripts/models/dense_1b_lr-4e-3_0213.sh
```

Submit it as a Beaker job:

```bash
MODE=beaker bash scripts/models/dense_1b_lr-4e-3_0213.sh
```

Override paths via env vars before launching:

- `PREFIX` — output root
- `MODELS_DIR` — derived from `PREFIX` (`${PREFIX}/models`)
- `DATASET_CACHE` — tokenizer-mapped dataset cache

Override Beaker cluster sizing per script with `BEAKER_GPUS=8 BEAKER_NODES=4 ...`.

After training, OLMo-core checkpoints can be converted to the HuggingFace format (suitable for inference and the evaluation pipelines below) with [`scripts/convert_olmo_to_hf.sh`](scripts/convert_olmo_to_hf.sh).

## Evaluation scripts

### Selective Expert Usage

The launch scripts in [`scripts/selective_hf/`](scripts/selective_hf/) exercise the full router-activation → expert selection → finetuning → eval pipeline on the released checkpoints. Each (model × keep-k × task × method) combination lands in its own subdirectory under `selective_evals_final/<model>/...`, with the pruned-expert model, finetuned checkpoint, and per-checkpoint metrics all colocated. Three scripts target different questions:

| Script | Investigates | Sweep |
|---|---|---|
| [`launch_selective_hf.sh`](scripts/selective_hf/launch_selective_hf.sh) | Main selective-expert evaluation — how each released model performs when only a subset of experts is retained for a given task (Figures 3 and 4 of the paper). | All released models × keep-k ∈ {8, 16, 32, 64, 128} × MC9 / Gen5 / MMLU / MMLU-Pro / GSM8K task groups. |
| [`launch_selective_method_hf.sh`](scripts/selective_hf/launch_selective_method_hf.sh) | Robustness to the choice of expert-selection method (Figure 5 of the paper). | {`layerwise`, `easy_ep`, `random`} selection methods × main 1T models × keep-k × tasks. |
| [`launch_selective_validation_hf.sh`](scripts/selective_hf/launch_selective_validation_hf.sh) | Calibration-data ablation — how much validation data and how many few-shot examples are needed to identify the right experts (Appendix B.1 of the paper). | Validation-set sizes ∈ {1, 5, 10, 100, All} × 3 shot-count configurations × `Emo_1b14b_1T` × keep-k ∈ {8, 16, 32, 128} × tasks. |

#### Output layout

Every (model × keep-k × task × method) combination produces one self-contained subdirectory under `selective_evals_final/`:

```
selective_evals_final/
└── <sanitized_model>/                            # e.g. allenaiEmo_1b14b_1T
    └── <task>_keepk_<K>_bs-<B>_lr-<LR>_epoch-<E>_selectivemode-{layerwise,easy_ep,random}[_nselective-<N>][_pseed-<S>][_pshots-<X>][_eshots-<Y>]/
        ├── selected_model/                       # pruned-expert HF checkpoint + pruning_metadata.json
        ├── finetuned_model/
        │   └── checkpoint-<N>/                   # HF Trainer-format finetuned weights
        └── results/
            └── checkpoint-<N>/
                ├── task-<name>-metrics.json      # aggregate metrics for the task
                ├── task-<name>-predictions.jsonl # per-instance predictions
                └── per_subject/                  # only for MMLU category tasks
                    └── <subject>/
                        └── task-<name>-metrics.json
```

The optional `_nselective-`, `_pseed-`, `_pshots-`, `_eshots-` suffixes only appear when the corresponding override is set (e.g. you'll only see `_nselective-100` when running with a sub-sampled calibration set).

#### Customization

Each script writes its config (`MODELS`, `SELECTIVE_KEEP_K_VALUES`, `TASK_GROUPS_LIST`, etc.) at the top — comment lines out to skip combinations. Override the output root with `OUTPUT_DIR=…` and the per-worker GPU count with `NUM_GPUS=…`.

We recommend running these on a slurm or other scheduling system, since each script launches many sequential worker invocations.

#### Aggregating results into tables

Once one of the launchers above has populated `selective_evals_final/`, two scripts in [`scripts/plotting/`](scripts/plotting/) walk the per-run subdirectories and produce flat CSV/TSV/markdown tables suitable for downstream analysis:

| Script | Source launcher | What it produces |
|---|---|---|
| [`get_table_scores_selective_evals_final.py`](scripts/plotting/get_table_scores_selective_evals_final.py) | `launch_selective_hf.sh` and `launch_selective_method_hf.sh` | Per-metric tables with rows = (model × keep-k variant) and paired columns "task (lw) / task (ep) / task (rd)" — one column per selection method. Group averages (`mc9_avg`, `gen5_avg`, `mmlu_merged_avg_no_other`, `mmlu_pro_merged_avg_no_other`) are prepended automatically. Both `last`-checkpoint (post-finetune) and `first`-checkpoint (pre-finetune) variants are emitted by default. |
| [`get_table_scores_nselective_ablation.py`](scripts/plotting/get_table_scores_nselective_ablation.py) | `launch_selective_validation_hf.sh` | Validation-data-ablation tables: rows = (model, selection-method, task group), columns = `keepk_K (1) / keepk_K (5) / keepk_K (10) / keepk_K (100) / keepk_K (All) / keepk_K (Random)`. Includes optional 0-shot variants when `_pshots-0`/`_eshots-0` runs are present. |

Both scripts default to reading from `<repo>/selective_evals_final/` and writing to `<repo>/plots/`. Overrides:

```bash
# Main + method-comparison tables
python -m scripts.plotting.get_table_scores_selective_evals_final \
    --selective-evals-root selective_evals_final \
    --output-dir plots

# Validation-size ablation tables
python -m scripts.plotting.get_table_scores_nselective_ablation \
    --selective-evals-root selective_evals_final \
    --output-dir plots
```

The model registries (`MODEL_SPECS` at the top of each file) currently list the released HF Hub checkpoints — add new entries there if you point either script at a directory built from a different model.

### Clustering Pretraining Document Tokens

[`scripts/clustering/run_pretraining_compare.sh`](scripts/clustering/run_pretraining_compare.sh) reproduces the side-by-side router-activation clustering used to compare EMO and the standard MoE baseline (Section 5.3 / Figure 6 of the paper). For each of `allenai/Emo_1b14b_1T` and `allenai/StdMoE_1b14b_1T` it:

1. Streams ~1M tokens of the OLMoE pretraining mix from S3
2. Runs a forward pass and saves token-level router logits
3. Derives softmax probs, runs PCA + spherical k-means at `k=32`
4. Renders an interactive side-by-side HTML explorer of both models' clusters

```bash
bash scripts/clustering/run_pretraining_compare.sh
# → cluster_eval_final/pretraining/compare_Emo_1b14b_1T_vs_StdMoE_1b14b_1T.html
```

#### Output layout

```
cluster_eval_final/
├── pretraining_mix.json                   # generated once, then reused
└── pretraining/
    ├── Emo_1b14b_1T/
    │   ├── embeddings_logits.npy + ...    # extract outputs (tokens, doc boundaries, metadata)
    │   ├── embeddings_probs.npy           # transform output
    │   └── probs_mean_pca_l2_spherical_kmeans_k32/
    │       ├── assignments.npy, run_info.json, summary.json
    │       └── cluster_explorer.html
    ├── StdMoE_1b14b_1T/
    │   └── (same structure)
    └── compare_Emo_1b14b_1T_vs_StdMoE_1b14b_1T.html
```

The underlying primitives (extract / transform / cluster / visualize) live in [`scripts/clustering/`](scripts/clustering/) — see its [README](scripts/clustering/README.md) for the modular pipeline.

#### Customization

- `CLUSTER_ROOT=…` overrides the output root (default `cluster_eval_final/`).
- `TARGET_TOKENS=…` and `MAX_TOKENS_PER_DOC=…` change the extraction budget and per-doc truncation.
- `CUDA_VISIBLE_DEVICES=…` restricts which GPUs the model is sharded across.

Pretraining shards stream directly from `s3://ai2-llm/…` (no weka mount required), so the pipeline needs AWS credentials with read access to the `ai2-llm` bucket.

### Weborganizer Expert Coverage

[`scripts/clustering/run_weborganizer_compare.sh`](scripts/clustering/run_weborganizer_compare.sh) reproduces the per-domain expert-activation heatmaps used to compare EMO and the standard MoE baseline (Section 5.3 / Figure 7 of the paper). For each of `allenai/Emo_1b14b_1T` and `allenai/StdMoE_1b14b_1T` it:

1. Streams ~20M tokens of the cc_all_dressed weborganizer mix from S3, sampled uniformly across the 24 topics
2. Runs a single forward pass and aggregates router activations into per-document expert vectors (top-k frequency + softmax probs)
3. Renders 5 expert-coverage heatmaps per embedding type (10 PNGs total per model)

Both models share a single `topic_order.json` (stratified row/column ordering) so the resulting heatmaps are directly comparable side-by-side.

```bash
bash scripts/clustering/run_weborganizer_compare.sh
# → cluster_eval_final/weborganizer/{Emo_1b14b_1T,StdMoE_1b14b_1T}/*.png
```

#### Output layout

```
cluster_eval_final/
└── weborganizer/
    ├── mix_composition.json      # auto-generated on first run by extract_document.py
    ├── topic_order.json          # shared row/column ordering for cross-model comparison
    ├── Emo_1b14b_1T/
    │   ├── embeddings_doc_topk_freq.npy
    │   ├── embeddings_doc_probs.npy
    │   └── *.png                 # 5 heatmaps × 2 embedding types = 10 PNGs
    └── StdMoE_1b14b_1T/
        └── (same structure)
```

The underlying primitives (extract_document / plot_doc_expert_coverage) live in [`scripts/clustering/weborganizer/`](scripts/clustering/weborganizer/).

#### Customization

- `CLUSTER_ROOT=…` overrides the output root (default `cluster_eval_final/`).
- `TARGET_TOKENS=…` changes the extraction budget (default 20M).
- `CUDA_VISIBLE_DEVICES=…` restricts which GPUs the model is sharded across.

Same data dependency as the pretraining-clustering flow: cc_all_dressed shards stream directly from `s3://ai2-llm/…`, so AWS credentials with read access to the `ai2-llm` bucket are required.

<!--

### Run templates

[`scripts/RUN_TEMPLATES.md`](scripts/RUN_TEMPLATES.md) has the full code for each recipe. Summary:

| Template | Description | Entry point |
|---|---|---|
| Dense 1B pretrain | Dense 1B on `OLMoE-mix-0824` | `src/scripts/train/olmo2-1B.py` |
| MoE 1B/14B single-level | 128-expert single-level MoE | `src/scripts/train/olmoe-1B-7B_fsl.py` |
| MoE 1B/14B two-level (shared experts) | Two-level router with `--document-expert-pool`, `--num_shared_experts*` | `src/scripts/train/olmoe-1B-7B_fsl.py` |
| Continual pretrain / extension | Resume from a checkpoint on a domain mix (e.g. `mj_finemath4plus`) | `src/scripts/train/olmoe-1B-7B_fsl_extension.py` |
| Eval | OLMES eval on HF checkpoints, results uploaded to S3 | via [`scripts/extensions/launch_eval.sh`](scripts/extensions/launch_eval.sh) |
| Pruning + task finetune | Compute router activations → prune to top-k experts → finetune | [`scripts/pruning_hf/hf_finetune_with_pruning.sh`](scripts/pruning_hf/hf_finetune_with_pruning.sh) |

Two-level model variants (`--model-type`):

- `moe` — single-level
- `two-level_lb-batch_reduce-dp` — two-level, no shared experts
- `two-level_lb-batch_reduce-dp_sharedexppool` — two-level with fixed shared-expert pool
- `two-level_lb-batch_reduce-dp_sharedexp_randpool` — two-level with random shared-expert pool sampling (used in anneal runs)

Pruning modes (`PRUNING_MODE`):

- `global` — single-pass activation collection + top-k prune across the whole model
- `layerwise` — greedy layer-by-layer pruning (each layer conditioned on earlier pruned layers)
- `layerwise_variable` — greedy layerwise with a per-layer keep-k schedule
- `easy_ep` — EASY-EP ([arXiv 2504.06792](https://arxiv.org/abs/2504.06792)): domain-specific one-shot prune on calibration data

The `runname` naming convention (size · router · LR · LB · date · phase) is documented in the cheatsheet at the bottom of `scripts/RUN_TEMPLATES.md`.

WebOrganizer Domain Similarity Analysis

-->

## Contact and Contributing

If you have a fix, improvement, or extension you'd like to share, **please open a pull request** — direct contributions are the best way to help the project, and we're happy to review them.

For other interactions:

- **Public questions, bug reports, or feature suggestions**: please file a [GitHub issue](https://github.com/allenai/EMO/issues). This keeps the conversation visible to everyone and lets others benefit from the answer.
- **Private or sensitive inquiries** (e.g. anything you'd rather not discuss in public): email [ryanyxw@berkeley.edu](mailto:ryanyxw@berkeley.edu).

## Citing

TODO
