# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlexMoE is a research extension of **OLMo-core** (Allen AI's open LLM training framework) focused on flexible Mixture-of-Experts (MoE) architectures. The package is named `ai2-olmo-core` (v2.3.0) and lives under `src/olmo_core/`. Custom router implementations are the primary research contribution.

## Important Caveats

This codebase was adapted from a well-maintained upstream (OLMo-core). As a result, **many tests, docs, and scripts outside `scripts/` may be outdated or unused**. Treat them as reference rather than ground truth.

## Environment

The active conda environment is `flexmoe` (already set up via `pip install -e '.[all]'`). Use it directly.

## Commands

### Code Quality
```bash
make style-check    # Validate formatting (isort + black)
make lint-check     # Ruff linting
make type-check     # Mypy type checking
make style          # Auto-format code
```

### Pretraining

Pretraining launch scripts live in `scripts/models_0116/`. Each script contains:
- A **commented-out `torchrun` block** for local/single-node iteration and debugging
- An active `python -m olmo_core.launch.beaker` block for cluster runs

**Always use the commented-out `torchrun` block when running locally** (GPUs are available in the current session). Example from `twolevelbatchlbreducedpsharedexp4c2-32_1b14b_lr-4e-3_lb-1e-2_0215.sh`:
```bash
torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl.py \
  $runname \
  --save-folder="./claude_outputs/models/$runname" \
  --dataset.mix=arc-easy-train \
  --work-dir="./claude_outputs/dataset-cache" \
  --trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
  --trainer.callbacks.wandb="{enabled: false, ...}" \
  --global_batch_size=2 \
  ...
```
The beaker paths (`/weka/...`) must be changed to local paths when running locally.

### Finetuning / Evaluation

Finetuning/evaluation is launched via `scripts/pruning_hf/test_arc_challenge.sh`. Despite the name, it runs many tasks beyond ARC (MMLU, HellaSwag, BoolQ, GSM8K, etc.) by iterating over `TASK_GROUPS_LIST`.

The same beaker-vs-bash pattern applies: the script has commented-out `bash scripts/pruning_hf/hf_finetune_with_pruning.sh ...` blocks — **use those directly** instead of the `python -m olmo_core.launch.beaker` blocks when running locally.

The underlying worker script is `scripts/pruning_hf/hf_finetune_with_pruning.sh`.

## Architecture

### Core Library (`src/olmo_core/`)

The library uses a **config-driven** design: almost every component (model, trainer, optimizer, data) has a corresponding `*Config` dataclass that instantiates the component. Training scripts define `build_model_config()`, `build_train_module_config()`, and `build_trainer_config()` functions.

Key subsystems:
- **`nn/moe/`** — The primary research area. Contains 23+ router implementations. `router.py` is the base class; all custom routers extend it (e.g., `twolevel_batchlb_reducedp_sharedexppool_router.py`). `moe.py` is the core MoE layer.
- **`nn/transformer/`** — Transformer blocks using attention + MoE FFN layers.
- **`train/trainer.py`** — Main training loop with FSDP support.
- **`train/train_module/`** — Per-architecture training modules that wrap the model and optimizer.
- **`train/callbacks/`** — Extensible callback system (checkpointing, WandB, Comet).
- **`distributed/`** — FSDP, tensor parallelism, distributed checkpointing.

### Training Scripts (`src/scripts/train/`)

Scripts like `OLMoE-1B-7B.py` use `olmo_core.internal.experiment.build_config()` and `main()`. The `main()` function handles argument parsing (config params can be overridden with `--key=value` CLI syntax).

### Evaluation (`src/scripts/eval/`)

- `tasks.py` — Defines all evaluation tasks (MMLU, ARC, HellaSwag, etc.)
- `launch_eval.py` — Orchestrates evaluation runs
- `prune_moe_checkpoint.py` — Creates pruned checkpoint variants

### Data Layout

- `models/` — Trained checkpoints (dense 1B, MoE 1B-7B/14B/35B variants)
- `prune/` — Pruned checkpoint variants and tokenized eval datasets
- `prune_evals/` — Evaluation results for pruned models
- `evals/` — Standard evaluation results

### S3 Data Access

Eval results are stored on S3 at `s3://ai2-sewonm/ryanwang/prune_evals_0313/`. Use `aws s3 ls`, `aws s3 cp` to browse and read files. Example:
```bash
# List models
aws s3 ls s3://ai2-sewonm/ryanwang/prune_evals_0313/

# List checkpoints for a task
aws s3 ls s3://ai2-sewonm/ryanwang/prune_evals_0313/<model>/<task>_keepk_<k>_.../results/

# Read a metrics file
aws s3 cp s3://.../<checkpoint>/task-<name>-metrics.json -
```

Each checkpoint directory contains `task-*-metrics.json` (aggregate metrics), `task-*-predictions.jsonl` (per-instance), and optionally `per_subject/<subject>/` subdirectories with per-MMLU-subject results.

### Custom Transformers Fork

The project uses a custom transformers fork (`ryanyxw/transformers#flexmoe_v4_57_1`) for inference/HF integration, referenced in `pyproject.toml` under the `transformers` extra.

## Current Branch Context

The active branch `ryanyxw/prune_evals` is focused on evaluating and comparing pruned MoE model checkpoints against baseline models.
