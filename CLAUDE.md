# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OLMo-Core is a production-grade framework for training large language models (LLMs) from Allen Institute for AI. It supports both **dense transformer models** (Llama-like, OLMo-2) and **Mixture of Experts (MoE) models** (OLMoE).

- Package: `ai2-olmo-core` (version 2.3.0)
- Documentation: https://olmo-core.readthedocs.io/

## Common Commands

```bash
# Install for development
pip install -e .[all]

# Run tests
pytest -v src/test                    # All tests
pytest -m gpu src/test                # GPU-only tests
pytest -m "not gpu" src/test          # CPU-only tests
pytest src/test/path/to/test_file.py  # Single test file

# Code quality
make style-check   # Check formatting (isort, black)
make style         # Auto-format code
make lint-check    # Ruff linting
make type-check    # Mypy type checking
make checks        # All checks combined

# Build docs
make docs
```

## Environment

- Always use `uv pip` instead of `pip` for package installation
- Always run `source .venv/bin/activate` from FlexMoE dir before running anything
- Development happens over SSH on remote nodes. To serve files (e.g., HTML reports), use `python -m http.server` on the node and SSH tunnel from local (`ssh -L <port>:localhost:<port>`).

## Running Training

### Dense Model Training (e.g., Llama-3 8B)
```bash
torchrun --nproc-per-node=8 src/scripts/train/Llama3-8B.py \
  --save-folder=/path/to/checkpoints
```

### MoE Model Training (e.g., OLMoE-1B-7B)
```bash
torchrun --nproc-per-node=8 src/scripts/train/OLMoE-1B-7B.py \
  --save-folder=/path/to/checkpoints
```

### Small Development Training (public example)
```bash
torchrun --nproc-per-node=2 src/examples/moe/train.py run_name \
  --model.n_layers=4 --train_module.rank_microbatch_size=256
```

### Configuration Overrides
All scripts support dot-notation overrides:
```bash
--model.n_layers=24 --train_module.optim.lr=5e-4 --trainer.save_interval=5000
```

## Key Architecture Concepts

### Configuration System
All configs inherit from the base `Config` dataclass in `src/olmo_core/config.py`. Configs support:
- YAML/JSON serialization
- Dot-notation command-line overrides
- Validation and merging

### Model Presets
Located in `src/olmo_core/nn/transformer/config.py`:
- Dense: `TransformerConfig.llama3_8B()`, `olmo2_32B()`, etc.
- MoE: `TransformerConfig.olmoe_1B_7B()`, `llama_like_moe()`

### Training Scripts Structure
- **Internal scripts** (`src/scripts/train/`): Use `olmo_core.internal.experiment` module with `CommonComponents` pattern
- **Public examples** (`src/examples/`): Self-contained, use explicit config building

### MoE-Specific Components
- Router: `src/olmo_core/nn/moe/router.py` - Top-K selection with softmax/sigmoid gating
- MoE layers: `src/olmo_core/nn/moe/moe.py` - `MoE` (default) and `DroplessMoE` (requires grouped_gemm)
- Load balancing loss: Prevents expert collapse, controlled by `lb_loss_weight`
- Z-loss: Encourages router entropy, controlled by `z_loss_multiplier`

### Block Types for MoE
- `moe`: All layers are sparse expert layers
- `moe_hybrid`: Mix of dense FFN and sparse expert layers
- `moe_reordered_norm` / `moe_hybrid_reordered_norm`: Variants with different layer norm placement

### Distributed Training
- **Data parallel**: FSDP (`DataParallelType.fsdp`) or HSDP (`DataParallelType.hsdp`)
- **Pipeline parallel**: `TransformerPipelineParallelConfig` with 1F1B scheduling
- **Expert parallel**: Distributes experts across devices for large MoE models

## Key File Locations

| Component | Location |
|-----------|----------|
| Config base class | `src/olmo_core/config.py` |
| Model configs | `src/olmo_core/nn/transformer/config.py` |
| Transformer blocks | `src/olmo_core/nn/transformer/block.py` |
| MoE implementation | `src/olmo_core/nn/moe/` |
| Trainer | `src/olmo_core/train/trainer.py` |
| Train module | `src/olmo_core/train/train_module/transformer/` |
| Callbacks | `src/olmo_core/train/callbacks/` |
| Data loading | `src/olmo_core/data/` |
| Optimizers | `src/olmo_core/optim/` |
| Tests | `src/test/` |

## Optional Dependencies

For MoE with dropless routing:
- `grouped_gemm` (compile from https://github.com/tgale96/grouped_gemm)

For attention backends:
- `flash-attn`, `ring-flash-attn`, `TransformerEngine`

For experiment tracking:
- `comet`, `wandb`

## Data Pipeline: HuggingFace → JSONL → Tokenized

### Overview

Three-step process to prepare HuggingFace datasets for training:
1. **Download** - Fast local download using `snapshot_download` (123x faster than streaming)
2. **Convert** - Arrow files → JSONL (parallelized, ~550k docs/s with 100 workers)
3. **Tokenize** - JSONL → tokenized .npy files using Dolma

### Why Download First?

| Method | Speed | Time (13M docs) |
|--------|-------|-----------------|
| Streaming from HF | 1,366 docs/s | ~2.6 hours |
| Download + local convert | 167,851 docs/s | ~78 seconds |

### Scripts Location

All data scripts in `src/scripts/kevinf/data/`:

| Script | Purpose |
|--------|---------|
| `download_hf_dataset.py` | Download HF datasets using snapshot_download |
| `convert_arrow_to_jsonl.py` | Convert Arrow → JSONL (parallelized) |
| `download_and_convert.sh` | Combined download + convert pipeline |
| `count_tokens.sh` | Count tokens in tokenized .npy files |

### Quick Start

```bash
# Full pipeline (download + convert):
./src/scripts/kevinf/data/download_and_convert.sh \
    croissantllm/croissant_dataset \
    /data/output/croissant \
    french_303b_1 french_303b_2 code_140b

# Then tokenize with Dolma (sequential, 200 workers per config):
for config in french_303b_1 french_303b_2 code_140b; do
    dolma tokens \
        --documents "/data/output/croissant/jsonl/${config}/**/*.jsonl.gz" \
        --destination "/data/output/croissant/tokenized/${config}" \
        --tokenizer.name_or_path allenai/dolma2-tokenizer \
        --tokenizer.eos_token_id 100257 \
        --tokenizer.pad_token_id 100277 \
        --dtype uint32 \
        --processes 200
done
```

### Performance Tips

- **CPU vs I/O bound**: Use `iostat -x 1` to check. If `%util` < 20%, you're CPU-bound → add more workers
- **Conversion**: 100 workers is good for most systems
- **Tokenization**: Run configs sequentially with max workers (e.g., 200) rather than parallel with fewer workers per config. This ensures full worker utilization when small configs finish.
- **Corrupted files**: If download/conversion was interrupted, some .jsonl.gz files may be corrupted. Dolma skips them automatically. Re-convert affected configs if needed.

### Output Structure

```
/data/output/croissant/
├── raw/                    # Downloaded Arrow files
│   ├── french_303b_1/train/*.arrow
│   └── ...
├── jsonl/                  # Converted JSONL
│   ├── french_303b_1/chunk*.jsonl.gz
│   └── ...
└── tokenized/              # Tokenized for training
    ├── french_303b_1/part-*.npy
    └── ...
```

## Running Evaluations (Offline Evals)

### Architecture

The eval system extends `oe_eval` (installed in site-packages) with custom tasks in the project source:

| Component | Location |
|-----------|----------|
| Entry point | `src/scripts/eval/launch_eval.py` |
| Task suite configs | `src/scripts/eval/task_suites.py` |
| Task config overrides | `src/scripts/eval/tasks.py` |
| Custom task implementations | `src/offline_evals/tasks/` (legalbench, chembench, frenchbench, etc.) |
| Task registration | `src/offline_evals/tasks/__init__.py` → merged into `TASK_REGISTRY` |
| Launch script (Beaker) | `scripts/kevinf/eval/launch.sh` |
| Print results | `scripts/kevinf/eval/print_evals.sh` / `src/scripts/eval/print_evals.py` |

### Quick Local Test (Inspect Mode)

Uses pythia-160m with 5 instances for fast sanity checks:

```bash
cd src/scripts/eval && PYTHONPATH=src python -u launch_eval.py \
    --task chembench:mc \
    --inspect \
    --output-dir /tmp/eval_test \
    --gpus 1 \
    --batch-size 1
```

Or use the Claude Code command: `/test-eval chembench:mc`

### Available Task Suites

```bash
# List all task suites
cd src/scripts/eval && PYTHONPATH=src python -u launch_eval.py --list-task-suites

# List all individual tasks
cd src/scripts/eval && PYTHONPATH=src python -u launch_eval.py --list-tasks
```

Key custom suites: `chembench:mc`, `chembench:rc`, `chembench:gen`, `legalbench:rc`, `frenchbench:rc`, `sciriff5`

### Adding a New Eval Benchmark

1. Create task file in `src/offline_evals/tasks/<benchmark>.py`
   - Inherit from `MultipleChoiceTask` (for MC/RC) or `Task` (for generative)
   - Define `TASK_CONFIG_DEFAULTS` with `dataset_path`, `primary_metric`, `split`, etc.
   - Implement: `_process_doc()`, `doc_to_text()`, `doc_to_target()`, `has_*_docs()`, `*_docs()`
   - Use a factory function (e.g., `create_<benchmark>_tasks()`) to generate subtask classes from a data dict
2. Register in `src/offline_evals/tasks/__init__.py` via `**create_<benchmark>_tasks()` spread into `new_task_registry`
3. Add task suite in `src/scripts/eval/task_suites.py` — use dynamic generation from source lists, not hardcoded task names
4. Test locally: `/test-eval <benchmark>:<eval_type>`

### Running on Beaker (Production Evals)

Use `scripts/kevinf/eval/launch.sh` to launch evals on Beaker via Gantry. This runs real models on GPU with full datasets.

**Setup:**
1. Edit `scripts/kevinf/eval/launch.sh`:
   - Set `MODELS=()` array with checkpoint paths (local `/data/input/...` paths or HF model names)
   - Set `TASKS=()` array with task specs (e.g., `arc_easy:mc::olmes`, `chembench:mc`, `legalbench:rc`)
   - Set `BASE_OUTPUT_DIR` for where results are stored
   - Adjust `CLUSTER`, `LIMIT`, `BATCH_SIZE` as needed

2. Launch:
```bash
bash scripts/kevinf/eval/launch.sh
```

**Key Gantry flags used:**
- `--weka oe-training-default:/data/input` — mounts the shared Weka filesystem
- `--install "pip install uv && UV_CACHE_DIR=/tmp/uv-cache uv pip install -e '.[eval]'"` — installs the project in the Beaker job
- `--budget ai2/oceo` — billing budget
- `--workspace ai2/flex2` — Beaker workspace
- `--cluster ai2/saturn` — GPU cluster (default)
- `--priority urgent` — job priority
- `--env-secret HF_TOKEN=KEVINF_HF_TOKEN` — HF token from Beaker secrets
- `--env-secret AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID` — AWS credentials from Beaker secrets
- `--allow-dirty` — allows launching from a dirty git tree

**What each job runs:**
```bash
PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
    --model <MODEL_PATH> \
    --model-type hf \
    --task <TASK> \
    --limit 1000 \
    --output-dir <OUTPUT_DIR> \
    --batch-size <BATCH_SIZE> \
    --gpus 1 \
    --fewshot-seed 1234 \
    --random-subsample-seed 1234
```

**Viewing results:**
```bash
# Print eval results table
bash scripts/kevinf/eval/print_evals.sh

# Or directly:
python src/scripts/eval/print_evals.py \
    -r -t \
    -b /data/input/kevinf/flexmoe/eval/results/ \
    --show-models "<model_substr>" \
    --show-tasks "<task_substr>" \
    --avg-all
```

**Batch size rules** (auto-applied in launch.sh):
- Default: 4
- CoT, minerva_math, mbpp, bigcodebench, ruler, sciriff tasks: 1

### Adding Shot Variants to an Existing Benchmark

To add N-shot variants (e.g., 0-shot and 5-shot) to an already-registered benchmark. Requires 4 files to be modified.

The eval system has 3 layers:
1. **TASK_REGISTRY** (`src/offline_evals/__init__.py`) — maps task names to Python classes (e.g., `"frenchbench_boolq:rc" → FrenchBenchBoolQRC`)
2. **TASK_CONFIGS** (`src/scripts/eval/tasks.py`) — maps config aliases to parameter dicts with `task_name` (pointing to TASK_REGISTRY) + overrides like `num_shots`
3. **TASK_SUITE_CONFIGS** (`src/scripts/eval/task_suites.py`) — groups multiple TASK_CONFIGS entries under one shorthand name

**Step 1: Add stub subclasses and register them** in the benchmark's task file (e.g., `src/offline_evals/tasks/frenchbench.py`):

Each shot variant needs its own empty subclass so it gets a unique `task_name` in the TASK_REGISTRY. This is required because `run_eval.py` uses `task_name` for output filenames — without unique names, different shot variants would overwrite each other's results.

```python
# Empty stubs — inherit all behavior from parent, exist only for unique TASK_REGISTRY keys
class FrenchBenchBoolQRC_0shot(FrenchBenchBoolQRC):
    pass

class FrenchBenchBoolQRC_5shot(FrenchBenchBoolQRC):
    pass
```

Also add a `BASE_TASKS` constant (to avoid the suite picking up shot variants):
```python
FRENCHBENCH_BASE_TASKS = ["frenchbench_boolq", "frenchbench_arc_challenge", ...]
```

Register all variants in the `create_*_tasks()` function:
```python
"frenchbench_boolq:rc": FrenchBenchBoolQRC,
"frenchbench_boolq:rc:0shot": FrenchBenchBoolQRC_0shot,
"frenchbench_boolq:rc:5shot": FrenchBenchBoolQRC_5shot,
```

**Step 2: Add TASK_CONFIGS entries** in `src/scripts/eval/tasks.py`:
```python
# task_name must match the new TASK_REGISTRY key (with shot count)
# num_shots is what actually controls the shot count at runtime
TASK_CONFIGS["frenchbench_boolq:rc:5shot::olmes"] = {
    "task_name": "frenchbench_boolq:rc:5shot",  # matches TASK_REGISTRY key
    "num_shots": 5,
    "primary_metric": "acc_per_char",
    "metadata": {"regimes": []},
}
```

**Step 3: Add task suite** in `src/scripts/eval/task_suites.py` (only if benchmark has multiple subtasks):
```python
"frenchbench:rc:5shot": {
    "tasks": [f"{t}:rc:5shot::olmes" for t in frenchbench.FRENCHBENCH_BASE_TASKS],
    "primary_metric": "macro",
},
```

**Step 4: Add to launch script** (`scripts/kevinf/eval/launch.sh`):
```bash
TASKS=(
    frenchbench:rc:5shot
)
```

**Key rules:**
- Each shot variant needs a unique stub subclass + TASK_REGISTRY entry so output files don't collide
- The stub classes are empty (`pass`) — they inherit everything from the parent class
- `num_shots` in TASK_CONFIGS is what actually controls the shot count at runtime
- The `::olmes` suffix is just a namespace convention for the config key, has no runtime effect
- Use a `BASE_TASKS` constant in suites to avoid picking up shot variants in the base suite
- Result files are named from `task_name` (colons → underscores), so `frenchbench_boolq:rc:5shot` → `task-frenchbench_boolq_rc_5shot-metrics.json`

### Design Notes

- **`unconditioned_prompt`**: Return `"Answer:"` for RC tasks (enables PMI-DC normalization for length-biased choices). Return `None` for MC tasks or non-English benchmarks.
- **Subfield lists**: If a benchmark has subfields where not all support every eval mode, use separate lists (e.g., `CHEMBENCH_SUBFIELDS` vs `CHEMBENCH_GEN_SUBFIELDS`) to avoid registering empty tasks.
- **Splits**: Use `test` or `validation` when available. If only `train` exists (e.g., ChemBench), document this limitation.

