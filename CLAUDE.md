# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EMO** is a Mixture-of-Experts model where modular structure emerges during pretraining without human-defined priors. EMO enables selective expert use (down to 12.5% of total experts) with minimal performance degradation. The repository is a research extension of **OLMo-core**; the underlying package is still named `ai2-olmo-core` (v2.3.0) and lives under `src/olmo_core/`. See `README.md` for the public-facing description, released checkpoints (under the `allenai/emo` HF collection), and the inference snippets for HF Transformers and vLLM.

## Session Type

At the start of every session, confirm with the user which of these two modes applies:

1. **GPU-attached session** — direct access to GPU resources. Used for debugging and testing code pipelines: run training/eval scripts locally, iterate on routers and configs, reproduce failures end-to-end.
2. **Launch-and-monitor session** — no direct GPU access. Used for launching jobs (Beaker submissions, S3-staged eval sweeps) and monitoring their progress, results, and logs.

The two modes call for different defaults: a GPU-attached session can just `torchrun`/`bash` a script and inspect outputs; a launch-and-monitor session should prefer `MODE=beaker`, queue up sweeps, and rely on `aws s3` / `beaker` CLIs to check state. Ask up front rather than guessing.

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

**Always launch big pretraining jobs from a checked-in bash file, not from an ad-hoc one-liner.** When the user asks for a pretraining (or other large) job, create a dedicated script under `scripts/models/` (or a logically-equivalent subdir) that hardcodes every env-var override and CLI arg, then run `bash <that script>`. This keeps the exact config — runname, save path, data root, node count, dataset overrides, env-var tweaks — versioned alongside the launch so the job can be reconstructed later. Don't paste `MODELS_DIR=... DATASET_CACHE=... BEAKER_NODES=... bash scripts/models/foo.sh` into the terminal; instead write `scripts/models/foo_<variant>.sh` that sets those values and invokes the launcher.

**Before every Beaker launch, commit AND push your changes.** Gantry clones the source for each replica from the GitHub remote — it does not bundle your local working tree. The launcher's `--allow-dirty` flag only tolerates uncommitted diffs locally; on the workers, any commit you haven't pushed will fail with `upload-pack: not our ref` and crash all replicas within ~20s. Workflow for every launch: `git add ... && git commit -m ... && git push origin <branch>`, then `MODE=beaker bash <script>`. Pushing is non-optional — skipping it wastes a scheduling slot.

**Experiment conventions for new pretraining jobs.** Each new pretraining experiment (e.g. a size-scaling sweep, an architecture ablation) lives in a dedicated subfolder of `scripts/` (e.g. `scripts/models_sizescaling/`). Within an experiment subfolder, every script must:

- **WandB project**: log to `emo-extension`. All new pretraining runs go to this project, regardless of which subfolder they live in.
- **WandB tags**: include the subfolder name as a tag (e.g. `[pretraining, models_sizescaling]`) so runs can be filtered by experiment.
- **Save path**: `MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/{experiment_name}"`, where `{experiment_name}` matches the subfolder name. Checkpoints land under `${MODELS_DIR}/${runname}/`.
- **Data root**: `DATA_ROOT="s3://ai2-llm"`. The weka mirror at `/weka/oe-training-default/ai2-llm/` is incomplete; S3 is the source of truth for tokenized data.

Each script should set these as bash variables after sourcing `scripts/launch_common.sh` so they override the launcher's defaults.

**Beaker operational notes.** Hard-won during the `models_sizescaling` sweep:

- **Diagnosing failures: read the `failed` replica's logs, not the cancelled ones.** When one replica crashes, Beaker cancels all siblings, so the experiment summary typically reads `15 canceled, 1 failed`. Only the `failed` replica's traceback is informative — the cancelled ones just show the cancellation signal. Find the failing task from the launcher output (it lists `failed with exit code N - see https://beaker.org/job/<jobid>`), then pull just that one: `gantry logs <experiment_id> --tail=500 --task=main-replica-<N>`.
- **`--allow-dirty` is a launcher-only flag.** It only suppresses the launcher's own "uncommitted changes" guard locally. Gantry on the workers always `git fetch`es from origin, so unpushed commits crash all replicas in ~20s with `upload-pack: not our ref` regardless of `--allow-dirty`. The flag is convenient for ad-hoc local launches; it does not substitute for `git push`.
- **`MODE=beaker bash <script>` blocks by default until the experiment finalizes**, streaming logs the whole time (the launcher passes `--follow` to `olmo_core.launch.beaker` implicitly). Pass `--no-follow` to make it fire-and-forget (verified — confirmed in `python -m olmo_core.launch.beaker --help`). Implications of the default: (a) for any real run, launch in background or you lose the shell; (b) `Ctrl+C` only kills your local watcher — the Beaker job keeps running; (c) the launcher's exit code is itself a coarse "did the experiment succeed" monitor.
- **Cancel a live Beaker experiment with `beaker experiment stop <id>`**, not Ctrl+C on the launcher. The experiment ID is in the launcher output (Beaker URL `https://beaker.org/ex/<id>`) or via `beaker workspace experiments ai2/flex2 --format=json | jq '.[].id'`.
- **`X canceled` paired with `running` or `succeeded` is preemption noise, not failure.** Preemptible replicas on jupiter get cancelled and rescheduled mid-run; the cancellation count grows over time without it being a problem. Only treat `canceled` as a failure signal when it's paired with `failed`.
- **Walltime calibration**: a 130B-token EMO run on 128 GPUs (16 nodes × 8 H100s) takes roughly **15–16 hours** end-to-end. Useful for deciding when to come back and check.

### OLMo-core → HF Conversion

`scripts/convert_emo_to_hf.py` converts an EMO OLMo-core checkpoint to a HuggingFace checkpoint and stages the `trust_remote_code` files into the output dir, so the result is directly loadable with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` on stock transformers. Validation (logit comparison against the OLMo-core model) is on by default; run with `--validation-device cuda` and leave conversion on CPU.

`scripts/models_sizescaling/convert_to_hf.sh` is the checked-in sweep wrapper (idempotent — skips outputs that already exist). It matches the released-checkpoint settings: `--dtype float32 --max-sequence-length 4096`.

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

## HF Modeling Code (`src/hf_trust_remote_code/`)

This project uses stock transformers — no custom fork. The `EmoForCausalLM` modeling code lives in `src/hf_trust_remote_code/` and ships inside each checkpoint; everything (conversion, evals, end users) loads checkpoints with `trust_remote_code=True`. The fork referenced in `pyproject.toml` (`ryanyxw/transformers#flexmoe_v4_57_1`) is a leftover from the pre-release workflow and is not needed.

## S3 Sync for Scratch Outputs

The `claude_outputs/` tree round-trips to `s3://ai2-sewonm/ryanwang/claude_outputs/` via:

```bash
bash scripts/pull_claude_outputs.sh   # pull from S3 (no --delete)
bash scripts/push_claude_outputs.sh   # push to S3 (--delete enabled)
```

`push_claude_outputs.sh` excludes regeneratable large files (`*.npy`, `*.safetensors`, `*.bin`, `*.parquet`, archives, etc.) but includes `cluster_explorer.html`. Pass `--dryrun` first when in doubt — push uses `--delete` and will remove S3 objects that don't exist locally.
