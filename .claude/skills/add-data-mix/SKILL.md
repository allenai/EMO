---
name: add-data-mix
description: Add a new data mix for training. Use when the user wants to create a new dataset mix, register it, and create a launch script.
---

# Add a New Data Mix

Add a new data mix for training: $ARGUMENTS

## Overview

Adding a new data mix requires 4 steps:
1. List `.npy` files from S3
2. Create the mix `.txt` file
3. Register in `__init__.py`
4. Create a Beaker launch script

## Step 1: List .npy Files from S3

Find all `.npy` files in the S3 bucket:
```bash
aws s3 ls s3://ai2-llm/preprocessed/<dataset>/<subset>/dolma2-tokenizer/ --recursive | grep '\.npy$' | awk '{print $NF}'
```

Check all subdirectories under the dataset path — the actual folder names may differ from what you expect (e.g., `oa_noncomm` not `non_comm`):
```bash
aws s3 ls s3://ai2-llm/preprocessed/<dataset>/
```

## Step 2: Create the Mix .txt File

**Location**: `src/olmo_core/data/mixes/<mix_name>.txt`

**Format**: CSV with `label,path` per line. No header, no comments needed.

```
<label>,preprocessed/<dataset>/<subset>/dolma2-tokenizer/part-00-00000.npy
<label>,preprocessed/<dataset>/<subset>/dolma2-tokenizer/part-01-00000.npy
...
```

**Rules**:
- Label convention: `<dataset>-<subset>` (e.g., `pmc-oa_comm`, `chempile-paper`)
- Paths are relative to the `mix_base_dir` (which defaults to `/weka/oe-training-default/ai2-llm` or `s3://ai2-llm`)
- If the data is tokenized with `dolma2-tokenizer` only, hardcode it in the path. Use `{TOKENIZER}` placeholder only if multiple tokenizers are supported.
- Reference: `src/olmo_core/data/mixes/croissant.txt`, `src/olmo_core/data/mixes/chempile.txt`

## Step 3: Register in `__init__.py`

**File**: `src/olmo_core/data/mixes/__init__.py`

Add the new mix as an enum member in the `DataMix` class:
```python
class DataMix(DataMixBase):
    ...
    chempile = "chempile"
    pmc = "pmc"          # <-- add here
```

The enum value must match the `.txt` filename (without extension). The `build()` method reads from `data/mixes/{value}.txt`.

## Step 4: Create a Beaker Launch Script

**Location**: `scripts/kevinf/train/launch_<mix_name>.sh`

**Template** (based on `scripts/kevinf/train/launch_croissant.sh`):

```bash
#!/bin/bash
dataset="<mix_name>"
warmup_fraction=0.1
train_tokens_B=10
train_tokens_raw=$((train_tokens_B * 1000000000))
load_path="/weka/oe-training-default/kevinf/checkpoints-new/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995"

for lr in 5e-5; do
  runname="olmo3-1b-${dataset}-${train_tokens_B}B-lr${lr}-warmup${warmup_fraction}"
  if [ -n "$load_path" ]; then
    runname="${runname}-ctd"
  fi

  python -m olmo_core.launch.beaker \
    --name $runname \
    --gpus 8 \
    --nodes 1 \
    --weka=oe-training-default \
    --is_private_repo \
    --priority urgent \
    --shared-filesystem \
    --workspace ai2/flex2 \
    --cluster ai2/jupiter \
    --preemptible \
    --allow-dirty \
    --env-secret "GITHUB_TOKEN=KEVINF_GITHUB_TOKEN" "WANDB_API_KEY=KEVINF_WANDB_API_KEY" "BEAKER_TOKEN=KEVINF_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY" \
    --env "S3_PROFILE=" \
    -- src/scripts/kevinf/train/OLMo3-1B.py \
    $runname \
    --save-folder="/weka/oe-training-default/kevinf/checkpoints/${runname}/" \
    --work-dir="/weka/oe-training-default/kevinf/dataset-cache" \
    --trainer.max_duration="{value: ${train_tokens_raw}, unit: tokens}" \
    --trainer.hard_stop="{value: ${train_tokens_raw}, unit: tokens}" \
    --trainer.callbacks.downstream_evaluator.eval_interval=100 \
    --dataset.mix=$dataset \
    --dataset.mix_base_dir=s3://ai2-llm \
    --trainer.callbacks.lm_evaluator.eval_dataset.mix=$dataset \
    --trainer.callbacks.lm_evaluator.eval_dataset.mix_base_dir=s3://ai2-llm \
    --trainer.callbacks.lm_evaluator.enabled=true \
    --train_module.optim.lr=$lr \
    ${load_path:+--load_path=$load_path} \
    --train_module.scheduler.warmup_fraction=$warmup_fraction

  sleep 5
done

echo "All jobs submitted! Check Beaker for status."
```

## Important: Data Location (mix_base_dir)

The `mix_base_dir` determines where the data is read from. It prefixes every path in the `.txt` file.

| Data location | `mix_base_dir` value | Notes |
|---|---|---|
| Weka (local) | `/weka/oe-training-default/ai2-llm` | Default in `OLMo3-1B.py`. Fast. |
| S3 (remote) | `s3://ai2-llm` | Needed if data isn't synced to weka. Slower. |

- **Override via launch script**: `--dataset.mix_base_dir=s3://ai2-llm`
- **Training and eval datasets are separate configs** — you must override `mix_base_dir` on BOTH if using S3:
  - `--dataset.mix_base_dir=s3://ai2-llm` (training)
  - `--trainer.callbacks.lm_evaluator.eval_dataset.mix_base_dir=s3://ai2-llm` (eval)

## Important: S3 Access on Beaker

When using `s3://` paths, Beaker jobs need AWS credentials and the S3 profile cleared:

1. **AWS credentials**: Add to `--env-secret` (BEFORE the `--` separator):
   ```
   --env-secret "AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY"
   ```

2. **Clear S3 profile**: `olmo_core.launch.beaker` auto-sets `S3_PROFILE=S3` (in `beaker.py:339`), which expects a named AWS profile that doesn't exist. Override it:
   ```
   --env "S3_PROFILE="
   ```
   This makes boto fall back to env var credentials. The code in `io.py:816` explicitly treats empty string as None for this purpose.

3. **CRITICAL**: Both `--env-secret` and `--env` flags must go BEFORE the `--` separator. Anything after `--` is passed to the Python training script as config overrides, and `--env` would cause `Key 'env' not in 'ExperimentConfig'`.

## LM Evaluator (Perplexity Eval on Training Data)

`OLMo3-1B.py` has an `LMEvaluatorCallbackConfig` callback that computes perplexity on a data mix during training. It's disabled by default.

**Two ways to enable and configure it**:

1. **Config overrides in launch script** (preferred):
   ```
   --trainer.callbacks.lm_evaluator.eval_dataset.mix=$dataset
   --trainer.callbacks.lm_evaluator.eval_dataset.mix_base_dir=s3://ai2-llm
   --trainer.callbacks.lm_evaluator.enabled=true
   ```

2. **CLI arg**: `--eval-mix <mix_name>` (sets both the mix and enables the callback in Python code)

OmegaConf correctly preserves concrete subclass types through merge/reconstruct cycles, so config overrides work even though `eval_dataset` is typed as `NumpyDatasetConfig` (ABC) but holds a `NumpyPaddedFSLDatasetConfig` instance.

## Verification

Before launching, dry-run to verify the config resolves correctly:
```bash
.venv/bin/python src/scripts/kevinf/train/OLMo3-1B.py test-run --dry-run \
  --dataset.mix=<mix_name> \
  --dataset.mix_base_dir=s3://ai2-llm \
  --trainer.callbacks.lm_evaluator.eval_dataset.mix=<mix_name> \
  --trainer.callbacks.lm_evaluator.eval_dataset.mix_base_dir=s3://ai2-llm \
  --trainer.callbacks.lm_evaluator.enabled=true
```

Check that:
- `dataset.mix` shows the correct mix name
- `dataset.mix_base_dir` shows the correct base dir
- `lm_evaluator.eval_dataset.mix` matches
- `lm_evaluator.enabled` is `True`

## Key Files Reference

| Purpose | Path |
|---------|------|
| Mix definitions | `src/olmo_core/data/mixes/*.txt` |
| Mix enum registry | `src/olmo_core/data/mixes/__init__.py` |
| Training script | `src/scripts/kevinf/train/OLMo3-1B.py` |
| Launch scripts | `scripts/kevinf/train/launch_*.sh` |
| S3 profile logic | `src/olmo_core/io.py` (`_get_s3_profile_name`) |
| Beaker env defaults | `src/olmo_core/launch/beaker.py` (line ~339) |
| Task groups (in-loop eval) | `src/olmo_core/eval/task_groups.py` |
