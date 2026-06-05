# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EMO** is a Mixture-of-Experts model where modular structure emerges during pretraining without human-defined priors. EMO enables selective expert use (down to 12.5% of total experts) with minimal performance degradation. The repository is a research extension of **OLMo-core**; the underlying package is still named `ai2-olmo-core` (v2.3.0) and lives under `src/olmo_core/`. See `README.md` for the public-facing description, released checkpoints (under the `allenai/emo` HF collection), and the inference snippets for HF Transformers and vLLM.

## Important Caveats

This codebase was adapted from a well-maintained upstream (OLMo-core). As a result, **many tests, docs, and scripts outside `scripts/` and `src/scripts/` may be outdated or unused**. Treat them as reference rather than ground truth.

## Environment

The conda env per the README is `emo` (Python 3.12, `uv pip install -e .[all]`). Assume it is already set up; activate and use it directly.

## Commands

### Code Quality
```bash
make style-check    # Validate formatting (isort + black)
make lint-check     # Ruff linting
make type-check     # Mypy type checking
make style          # Auto-format code
```

### Pretraining

Pretraining recipes live in `scripts/models/` — one shell script per released checkpoint (e.g. `emo_1b14b_1t.sh`, `stdmoe_1b14b_1t.sh`, `dense_1b_130b.sh`, `*_emoanneal.sh`). Each script sources `scripts/launch_common.sh`, which defines a `launch` helper that dispatches between local `torchrun` and Beaker submission based on the `MODE` env var:

```bash
bash scripts/models/emo_1b14b_1t.sh              # local torchrun (default MODE=local)
MODE=beaker bash scripts/models/emo_1b14b_1t.sh  # submit to Beaker
```

Override I/O roots via env vars before invoking: `PREFIX` (output root), `MODELS_DIR` (defaults to `${PREFIX}/models`), `DATASET_CACHE`, `DATA_ROOT`. For local runs, also `NPROC` (GPUs per node). For Beaker runs, `BEAKER_GPUS`, `BEAKER_NODES`, etc.

Some older training scripts in `scripts/models/` still carry a commented-out `torchrun` block above the `launch` call — these are leftover debugging snippets, not the way to run locally. The `launch` helper is the supported path.

### Selective-Expert Evaluation

The full pipeline (router activations → expert selection → finetune → eval) is driven by three launchers in `scripts/selective_hf/`:

| Launcher | Purpose |
|---|---|
| `launch_selective_hf.sh` | Main selective-expert sweep (Figure 3 of the paper) — all released models × keep-k ∈ {8,16,32,64,128} × MC9/Gen5/MMLU/MMLU-Pro/GSM8K |
| `launch_selective_method_hf.sh` | Selection-method robustness (Figure 4) — layerwise vs easy_ep vs random |
| `launch_selective_validation_hf.sh` | Calibration-data ablation (Appendix B.2) — validation-set sizes × shot counts |

Each launcher fans out into many per-config invocations of one of the worker scripts:

- `hf_finetune_with_selective_layerwise.sh` — greedy layer-by-layer expert selection
- `hf_finetune_with_selective_easy_ep.sh` — EASY-EP one-shot selection ([arXiv 2504.06792](https://arxiv.org/abs/2504.06792))
- `hf_finetune_with_selective_random.sh` — random expert selection (calibration-free baseline)

Workers export `PYTHONPATH="$(pwd)/src"` so bare imports like `offline_evals` and `scripts.eval.tasks` resolve. Override the output root with `OUTPUT_DIR=...` and per-worker GPU count with `NUM_GPUS=...`. Output layout (`selective_evals_final/<model>/<run_subdir>/{selected_model,finetuned_model,results}/`) is documented in `README.md`.

Aggregate per-run results into tables with `scripts/plotting/get_table_scores_selective_evals_final.py` (main + method sweeps) or `scripts/plotting/get_table_scores_nselective_ablation.py` (validation-size ablation).

### Router-Activation Clustering

`scripts/clustering/run_pretraining_compare.sh` and `scripts/clustering/run_weborganizer_compare.sh` reproduce the clustering / expert-coverage analyses (Figures 5 and 6). Outputs land under `cluster_eval_final/`. See `scripts/clustering/README.md` for the modular extract/transform/cluster/visualize primitives.

## Architecture

### Core Library (`src/olmo_core/`)

Config-driven design: nearly every component (model, trainer, optimizer, data) has a corresponding `*Config` dataclass that instantiates the component. Training scripts define `build_model_config()`, `build_train_module_config()`, and `build_trainer_config()`.

Key subsystems:
- **`nn/moe/`** — Primary research area. 23 router implementations; `router.py` is the base class. EMO's published router is `twolevel_batchlb_reducedp_sharedexp_randpool_router.py` (model-type `two-level_lb-batch_reduce-dp_sharedexp_randpool`). `moe.py` is the core MoE layer.
- **`nn/transformer/`** — Transformer blocks (attention + MoE FFN).
- **`train/trainer.py`** — Main training loop with FSDP support.
- **`train/train_module/`** — Per-architecture training modules that wrap model + optimizer.
- **`train/callbacks/`** — Checkpointing, WandB, Comet, downstream eval.
- **`distributed/`** — FSDP, tensor parallelism, distributed checkpointing.

### Training Entry Points (`src/scripts/train/`)

`olmoe-1B-7B_fsl.py` is the entry point used by every script in `scripts/models/`. It builds the config via `olmo_core.internal.experiment.build_config()` / `main()`; config params can be overridden with `--key=value` CLI syntax. Annealing variants use `olmoe-1B-7B_fsl_anneal.py`.

### Evaluation Internals (`src/scripts/eval/`)

- `tasks.py` / `task_suites.py` — Eval task definitions (MMLU, ARC, HellaSwag, etc.)
- `launch_eval.py` — Orchestrates eval runs
- `prune_moe_checkpoint.py` — Builds the pruned-expert HF checkpoint consumed by the selective-eval pipeline
- `router_analysis.py`, `examine_expert_overlap.py` — Diagnostic tooling

### Default Output Roots

- `selective_evals_final/` — selective-expert eval outputs (pruned models, finetuned checkpoints, metrics)
- `cluster_eval_final/` — clustering and weborganizer outputs
- `claude_outputs/` — scratch root for ad-hoc runs (synced to S3, see below)
- `plots/` — table aggregations from `scripts/plotting/`

## Custom Transformers Fork

The released HF checkpoints require a custom transformers fork (`ryanyxw/transformers#flexmoe_v4_57_1`), referenced in `pyproject.toml` under the `transformers` extra. End users load checkpoints with `trust_remote_code=True`; the Hub pulls the necessary modeling code automatically.

## S3 Sync for Scratch Outputs

The `claude_outputs/` tree round-trips to `s3://ai2-sewonm/ryanwang/claude_outputs/` via:

```bash
bash scripts/pull_claude_outputs.sh   # pull from S3 (no --delete)
bash scripts/push_claude_outputs.sh   # push to S3 (--delete enabled)
```

`push_claude_outputs.sh` excludes regeneratable large files (`*.npy`, `*.safetensors`, `*.bin`, `*.parquet`, archives, etc.) but includes `cluster_explorer.html`. Pass `--dryrun` first when in doubt — push uses `--delete` and will remove S3 objects that don't exist locally.
