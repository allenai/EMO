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

`OLMo3-1B.py` has an `LMEvaluatorCallbackConfig` callback that computes perplexity on a data mix during training.

**Use config overrides in the launch script** (preferred over `--eval-mix` CLI arg):
```
--trainer.callbacks.lm_evaluator.eval_dataset.mix=$eval_dataset
--trainer.callbacks.lm_evaluator.eval_dataset.mix_base_dir=s3://ai2-llm
--trainer.callbacks.lm_evaluator.enabled=true
```

OmegaConf correctly preserves concrete subclass types through merge/reconstruct cycles, so config overrides work even though `eval_dataset` is typed as `NumpyDatasetConfig` (ABC) but holds a `NumpyPaddedFSLDatasetConfig` instance.

**Eval dataset should be the full parent mix, not the training subset.** For example, if training on `dolma2-code-python`, eval on `dolma2-code` (all 15 languages). This gives cross-domain perplexity signal — you can see how training on one language affects perplexity across all languages. Use a separate `eval_dataset` variable in the launch script:
```bash
dataset="dolma2-code-python"
eval_dataset="dolma2-code"
```

## Beaker Launch Checklist

Before launching jobs on Beaker:

1. **Push your branch first.** Beaker clones the repo at the specific commit hash. If the commit doesn't exist on the remote, the job fails with `fatal: reference is not a tree`.

2. **Use the correct venv.** Always `source /Users/kevinfarhat/repos/FlexMoE/.venv/bin/activate` before running launch scripts. The shell may have another project's venv active.

3. **Verify eval task groups exist.** The `TASK_GROUPS["fast"]` in `task_groups.py` is used by the downstream evaluator. If it references tasks not available in the `olmo_eval` package installed on Beaker (e.g., `legalbench:rc`), the job crashes at config build time with `KeyError: 'Downstream evaluation config not found'`.

4. **The LM evaluator callback must exist in OLMo3-1B.py.** Config overrides like `--trainer.callbacks.lm_evaluator.eval_dataset.mix=X` will fail with `OLMoConfigurationError: Key 'eval_dataset' not in 'Callback'` if the `lm_evaluator` callback isn't defined in the training script. The callback with default values must be present for overrides to work.

5. **Monitor by experiment ID, not name.** Use `beaker experiment get <ID>` to check status. Names can vary with random suffixes. To find recent experiments: `beaker workspace experiments ai2/flex2 --format json`.

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

## Weighted / Multi-Source Mixes

When combining multiple datasets with a target token budget (e.g., "all of dataset A + fill the rest with dataset B to reach 5B tokens"), you need to control the ratio by **subsampling the larger dataset at the file level**.

### Why ratios matter

The data loader consumes files proportionally to their byte sizes. Simply concatenating all files from both datasets would make the smaller dataset negligibly represented. For example, 1.7B tokens of mimic + 63.8B tokens of PMC would make mimic only ~2.6% of training.

### Step 1: Count tokens per dataset

Use the token counting script on an existing mix file:
```bash
bash scripts/kevinf/data/count_tokens_s3.sh s3://ai2-llm src/olmo_core/data/mixes/<mix>.txt
```

Or count tokens directly from S3 for a dataset without a mix file:
```bash
aws s3 ls --recursive s3://ai2-llm/preprocessed/<dataset>/ | grep '\.npy$' | \
  awk 'BEGIN{OFMT="%.0f"} {total += $3} END {printf "%.2fB tokens\n", total/4/1e9}'
```

**Important**: Use `OFMT="%.0f"` or floating-point math in awk — default integer arithmetic overflows at 2^31 bytes (~0.5B tokens).

### Step 2: Calculate the ratio

Given:
- Dataset A: X tokens (include all)
- Dataset B: Y tokens available (sample from this)
- Target: T total tokens

Then you need `T - X` tokens from dataset B. Find the right number of files:
```bash
# List per-file token counts with cumulative totals
aws s3 ls s3://ai2-llm/preprocessed/<dataset>/<subset>/dolma2-tokenizer/ | grep '\.npy$' | \
  awk '{sizes[NR]=$3; names[NR]=$NF; count++} END {
    cumul=0; for(i=1;i<=count;i++){
      cumul+=sizes[i];
      printf "%s: %.3fB (cumul: %.3fB)\n", names[i], sizes[i]/4/1e9, cumul/4/1e9;
      if(cumul/4/1e9 > <TARGET_B>) break
    }
  }'
```

### Step 3: Using create_weighted_mix.py (automated)

For automated ratio-based selection with S3 file size lookups:
```bash
python src/scripts/kevinf/data/create_weighted_mix.py \
  --sources <mix_a>.txt:<pct_a> <mix_b>.txt:<pct_b> \
  --total-tokens 5B \
  --output src/olmo_core/data/mixes/<output>.txt \
  --dry-run
```

**Note**: This script queries S3 for file sizes, which can be slow (~5 min for large datasets). If you already know the file sizes, it's faster to calculate manually and write the mix file directly.

### Step 3 (alternative): Manual mix file creation

Write the combined mix file directly with comments documenting the ratio:
```
# Weighted mix: ~5B tokens
# dataset-a: 34% (~1.70B tokens, all files)
# dataset-b: 66% (~3.34B tokens, subset)

# dataset-a
dataset-a-subset,preprocessed/dataset-a/subset/dolma2-tokenizer/part-00-00000.npy
...

# dataset-b (sampled)
dataset-b-subset,preprocessed/dataset-b/subset/dolma2-tokenizer/part-00-00000.npy
...
```

### Granularity note

Ratio precision is limited by file sizes. If each `.npy` file is ~0.5B tokens, you can only adjust in ~0.5B increments. The trainer's `max_duration` controls the exact training budget — having slightly more data in the mix than the token budget is fine.

### Reference: Existing multi-source mixes

- `mimic-pmc-5B.txt`: mimic-iv-note (all, ~1.70B) + PMC oa_comm (7 files, ~3.34B) = ~5.04B total
- `tpol-70-dclm-30.txt`: the-pile-of-law 70% + DCLM 30%

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
| Token counting (S3) | `scripts/kevinf/data/count_tokens_s3.sh` |
| Weighted mix creator | `src/scripts/kevinf/data/create_weighted_mix.py` |
| S3 path validator | `src/scripts/kevinf/data/check_s3_paths_exist.py` |
