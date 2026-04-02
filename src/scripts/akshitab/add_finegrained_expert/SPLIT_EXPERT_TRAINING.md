# Split Expert Training

Memory-efficient selective expert training by splitting MoE expert parameters into separate frozen and trainable `nn.Parameter` objects.

## Motivation

When training 4 out of 128 experts, the default approach (`SelectedExpertGradientMaskCallback`) zeros gradients for frozen experts but the Adam optimizer still allocates states (2 × fp32 moment tensors) for all expert parameters:

| | Params in optimizer | Adam state memory |
|---|---|---|
| Default (gradient masking) | ~12.9B | ~103 GB |
| Split expert params | ~403M (4 experts) | ~3.2 GB |

## How it works

### `SplitExpertDroplessMoEMLP`

Defined in `src/olmo_core/nn/moe/mlp.py`. Extends `DroplessMoEMLP`.

Each of `w1`, `w2`, `w3` (shape `num_experts * hidden_size, d_model`) is split into two `nn.Parameter` objects by expert index:

- `w1_frozen` (shape `num_frozen * hidden_size, d_model`, `requires_grad=False`)
- `w1_trainable` (shape `num_trainable * hidden_size, d_model`, `requires_grad=True`)

The optimizer's `build_groups()` (in `src/olmo_core/optim/config.py`) already skips `requires_grad=False` params — no optimizer state allocated for frozen experts.

In `forward()`, the full stacked tensor is reconstructed from both parts using precomputed row indices, then passed to `gmm()` as usual.

### Checkpoint conversion

The model's state dict has split keys (`w1_frozen`, `w1_trainable`, ...) which differ from the standard format (`w1`, `w2`, `w3`). Conversion is handled offline:

```
src/scripts/akshitab/add_finegrained_expert/convert_split_checkpoint.py
```

```bash
# Regular → Split (before training)
python src/scripts/akshitab/add_finegrained_expert/convert_split_checkpoint.py \
    --checkpoint-path /path/to/regular/checkpoint \
    --save-path /path/to/split/checkpoint \
    --experts-to-train 127,128,129,130 \
    --to-split

# Split → Regular (after training)
python src/scripts/akshitab/add_finegrained_expert/convert_split_checkpoint.py \
    --checkpoint-path /path/to/split/checkpoint \
    --save-path /path/to/regular/checkpoint \
    --experts-to-train 127,128,129,130 \
    --to-regular
```

### Post-training callback

`SplitExpertConverterCallback` (in `src/olmo_core/train/callbacks/split_expert_converter.py`) automatically converts the final checkpoint back to regular format in `post_train()`. It:

1. Gathers the full model state dict from FSDP (collective op, all ranks participate)
2. Merges `w1_frozen`/`w1_trainable` → `w1` on rank 0
3. Saves the regular checkpoint to `{checkpoint_path}_regular`
4. Passes the merged state dict to `HFConverterCallback` so HF conversion works on regular keys

## Full pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Add new experts (optional, for extension experiments)    │
│    python .../add_new_expert.py \                           │
│        --num_shared_experts 1 --exclude_experts 127         │
│    Output: regular checkpoint with added experts            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 2. Convert to split format (CPU, on weka)                   │
│    python .../convert_split_checkpoint.py \                  │
│        --to-split --experts-to-train 127,128,129,130        │
│    Output: split checkpoint                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 3. Train (distributed, on cluster)                          │
│    python .../train_selected_experts.py \                    │
│        --split-expert-params \                               │
│        --base-model-config="${BASE_MODEL_PATH}" \            │
│        --experts-to-train=127,128,129,130                   │
│    - Loads split checkpoint (keys match natively)           │
│    - Saves checkpoints in split format                      │
│    - SplitExpertConverterCallback merges final checkpoint   │
│    - HFConverterCallback converts merged to HF format       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 4. (Optional) Convert intermediate checkpoints back         │
│    python .../convert_split_checkpoint.py \                  │
│        --to-regular --experts-to-train 127,128,129,130      │
└─────────────────────────────────────────────────────────────┘
```

## Usage in shell scripts

Example from `add_4math_expert_init_top2_average_noise_split.sh`:

```bash
# Compute expert indices (insert before shared expert)
NUM_SHARED_EXPERTS=1
INSERT_POS=$((128 - NUM_SHARED_EXPERTS))
EXPERTS_TO_TRAIN=$(seq -s, $INSERT_POS $((INSERT_POS + NUM_NEW_EXPERTS - 1)) | sed 's/,$//')

# Part 2: Convert to split (run once on weka)
# python .../convert_split_checkpoint.py \
#     --checkpoint-path ${NEW_BASE_MODEL_PATH} \
#     --save-path ${SPLIT_MODEL_PATH} \
#     --experts-to-train ${EXPERTS_TO_TRAIN} --to-split

# Part 3: Train
python -m olmo_core.launch.beaker \
    -- src/scripts/akshitab/add_finegrained_expert/train_selected_experts.py \
    ${RUN_NAME} \
        --trainer.load_path="${SPLIT_MODEL_PATH}/model_and_optim" \
        --base-model-config="${NEW_BASE_MODEL_PATH}" \
        --experts-to-train=${EXPERTS_TO_TRAIN} \
        --split-expert-params
```

## Compatibility

| Feature | Compatible? | Notes |
|---|---|---|
| FSDP | ✅ | Both frozen and trainable params sharded. `num_trainable * hidden_size` must be ≥ world_size. |
| torch.compile | ✅ | Reconstruction is basic tensor ops. `gmm()` already has `@torch._dynamo.disable()`. |
| Expert parallelism | ❌ | `_shard_experts()` raises `OLMoConfigurationError`. Use gradient masking instead. |
| Checkpoint format | ✅ | Offline conversion handles regular ↔ split. Callback auto-converts post-training. |
| Weight decay | ✅ for experts | Frozen params excluded from optimizer. Router still needs `weight_decay=0.0` (uses gradient zeroing). |

## Implementation details

### `SplitExpertDroplessMoEMLP` overrides

| Method | Why |
|---|---|
| `__init__` | Creates split params instead of full w1/w2/w3. Calls `MoEMLPBase.__init__` directly (skips parent). |
| `forward` | Reconstructs full tensors via `_reconstruct()`, then calls `gmm()`. |
| `reset_parameters` | Inits split params. `fan_in = d_model` is same for both splits. |
| `_shard_experts` | Raises error (EP incompatible). |
| `w1`/`w2`/`w3` properties | Returns reconstructed tensors for backward compat with `init_feed_forward_moe()`. In-place init on these is a no-op (weights come from checkpoint). |

### Router handling

The router weight (`num_experts, d_model`) is NOT split — too small (~4.2M params total). `SelectedExpertGradientMaskCallback` with `layer_patterns=["router"]` handles router gradient masking.

## Tests

`src/test/nn/moe/split_expert_mlp_test.py` — 9 tests:

1. Correct param shapes and requires_grad
2. Reconstruction preserves expert ordering
3. State dict has split keys (no hooks)
4. Copy regular → split roundtrip
5. Copy split → regular roundtrip
6. Split state dict save/load roundtrip
7. Forward output matches original
8. Gradients only flow to trainable params
9. `split_expert_mlps()` works on full Transformer model (2-layer, 8-expert)
