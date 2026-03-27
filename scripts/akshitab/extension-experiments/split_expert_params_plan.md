# Plan: Split MoE Expert Parameters into Frozen/Trainable Groups

## Context

When selectively training 4 out of 128 experts, the current `SelectedExpertGradientMaskCallback` zeros gradients for frozen experts but the optimizer still allocates Adam states (2 × fp32 moments) for ALL ~12.9B expert params. This wastes ~100 GB of optimizer memory. By splitting expert parameters into separate `nn.Parameter` objects with different `requires_grad`, the optimizer naturally skips frozen params (already implemented in `OptimConfig.build_groups()`).

## Approach: Post-build model transformation

A conversion function replaces each `DroplessMoEMLP` in the model with a `SplitExpertDroplessMoEMLP` that has separate frozen/trainable parameters. This happens after `model.build()` but before FSDP wrapping — no core MoE config changes needed.

## Files to modify

1. **`src/olmo_core/nn/moe/mlp.py`** — Add `SplitExpertDroplessMoEMLP` subclass
2. **`src/scripts/akshitab/add_finegrained_expert/train_selected_experts.py`** — Add conversion step, optionally remove gradient mask callback

## Implementation

### Step 1: `SplitExpertDroplessMoEMLP` in `mlp.py`

Extends `DroplessMoEMLP`. In `__init__`:

```python
class SplitExpertDroplessMoEMLP(DroplessMoEMLP):
    def __init__(self, *, d_model, hidden_size, num_experts, experts_to_train, dtype, init_device):
        # Call MoEMLPBase.__init__ directly (skip DroplessMoEMLP to avoid creating full w1/w2/w3)
        MoEMLPBase.__init__(self, d_model=d_model, hidden_size=hidden_size, num_experts=num_experts)

        self.experts_to_train = sorted(experts_to_train)
        self.experts_frozen = sorted(set(range(num_experts)) - set(experts_to_train))
        num_trainable = len(self.experts_to_train)
        num_frozen = len(self.experts_frozen)

        # Frozen params — no optimizer state allocated
        self.w1_frozen = nn.Parameter(torch.empty(num_frozen * hidden_size, d_model, ...), requires_grad=False)
        self.w1_trainable = nn.Parameter(torch.empty(num_trainable * hidden_size, d_model, ...))
        # Same for w2, w3

        # Precompute row indices for reconstruction (non-persistent buffer)
        # frozen_row_indices: which rows in the full tensor come from w_frozen
        # trainable_row_indices: which rows come from w_trainable
        self.register_buffer("_frozen_row_indices", ..., persistent=False)
        self.register_buffer("_trainable_row_indices", ..., persistent=False)

        # State dict hooks for checkpoint compatibility
        self._register_load_state_dict_pre_hook(self._split_checkpoint_weights)
        self._register_state_dict_hook(self._merge_checkpoint_weights)

        self._gmm = gmm
```

**Forward** — reconstruct full tensors then call gmm as normal:

```python
def _reconstruct(self, w_frozen, w_trainable):
    full = torch.empty(self.num_experts * self.hidden_size, self.d_model, ...)
    full[self._frozen_row_indices] = w_frozen
    full[self._trainable_row_indices] = w_trainable  # grad flows through indexing
    return full

def forward(self, x, batch_size_per_expert):
    w1 = self._reconstruct(self.w1_frozen, self.w1_trainable)
    w2 = self._reconstruct(self.w2_frozen, self.w2_trainable)
    w3 = self._reconstruct(self.w3_frozen, self.w3_trainable)
    # Then same as DroplessMoEMLP.forward: view, get_local_tensor, gmm
    w1, w2, w3 = (
        get_local_tensor(w1.view(self.num_experts, self.hidden_size, self.d_model)),
        get_local_tensor(w2.view(...)),
        get_local_tensor(w3.view(...)),
    )
    x1 = self.gmm(x, w1, batch_size_per_expert, trans_b=True)
    x2 = self.gmm(x, w3, batch_size_per_expert, trans_b=True)
    x1 = F.silu(x1) * x2
    return self.gmm(x1, w2, batch_size_per_expert)
```

**State dict hooks** for checkpoint compatibility:

- `_split_checkpoint_weights`: On load, if state dict has `w1`/`w2`/`w3` (original format), split into `w1_frozen`/`w1_trainable` by expert index. Remove original keys, insert split keys.
- `_merge_checkpoint_weights`: On save, reconstruct `w1` from `w1_frozen` + `w1_trainable`. Remove split keys, insert original key.

**Expert parallelism**: Not supported with split params (see Risks section). The conversion function should skip the split if EP is enabled.

### Step 2: Conversion function in `train_selected_experts.py`

```python
def split_expert_mlps(model, experts_to_train):
    """Replace DroplessMoEMLP with SplitExpertDroplessMoEMLP in all MoE layers."""
    for block in model.blocks:
        moe = block.feed_forward_moe
        old_mlp = moe.experts.mlp  # ParallelDroplessMLP.mlp is DroplessMoEMLP
        new_mlp = SplitExpertDroplessMoEMLP(
            d_model=old_mlp.d_model,
            hidden_size=old_mlp.hidden_size,
            num_experts=old_mlp.num_experts,
            experts_to_train=experts_to_train,
            dtype=...,
            init_device="meta",
        )
        moe.experts.mlp = new_mlp
```

Called in `train()` between model build and train_module build:

```python
model = config.model.build(init_device="meta")
if experts_to_train is not None:
    split_expert_mlps(model, experts_to_train)  # <-- new
train_module = config.train_module.build(model)
```

### Step 3: Remove `SelectedExpertGradientMaskCallback` (optional)

With split params, the expert gradients are naturally zero for frozen experts (requires_grad=False). The callback is no longer needed for expert MLP params. However, the **router** gradient masking is still needed — the router weight is a single `(num_experts, d_model)` tensor and we still want to mask non-selected expert rows. Two options:

- **Keep the callback** but only for router params (change `layer_patterns=["router"]`)
- **Split router too** — but the router is small (~4.2M params), so the memory savings are negligible. Not worth the complexity.

Recommendation: keep the callback for router only.

## Memory savings

| | Current | Proposed |
|---|---|---|
| Expert params in optimizer | ~12.9B | ~403M (4 experts) |
| Optimizer state (Adam, fp32×2) | ~103 GB | ~3.2 GB |
| **Savings** | | **~100 GB** |

Parameter memory (bf16) is unchanged — frozen params still need to be in memory for forward pass.

## Risks / considerations

- **torch.compile**: Reconstruction uses basic tensor ops (empty + indexing). The `gmm` call is already `@torch._dynamo.disable()`. Should be fine, but needs testing.
- **FSDP (current setup)**: The original code flattens expert dims so dim 0 = `num_experts * hidden_size` (128 * 1024 = 131,072) to ensure enough rows for FSDP sharding. With split params, `w1_trainable` has dim 0 = `num_trainable * hidden_size`. For 4 trainable experts with 32 GPUs: `4 * 1024 = 4,096` rows → `128` per GPU — works fine. Even 1 expert gives `1024 / 32 = 32` rows per GPU. Constraint: `num_trainable * hidden_size >= world_size` and divisible. Holds for all realistic setups (hidden_size=1024 is a power of 2, world_size is also power of 2). Mixed requires_grad in one module is supported by FSDP2.
- **Expert parallelism (NOT currently used)**: EP requires `num_experts % ep_world_size == 0`. With split tensors, `num_trainable` (e.g., 4) may not be divisible by the EP world size. **This optimization is incompatible with EP.** The conversion function should check for EP and skip the split (fall back to gradient masking) when EP is enabled. This is fine since the current selective training scripts only use FSDP.
- **Checkpoint format**: State dict hooks maintain backward compatibility — saves in original format.
- **Reconstruction overhead**: One `torch.empty` + 2 index assignments per weight per layer per forward step. Negligible vs gmm compute.

## Verification

1. Dry-run with `--dry-run` to verify model config prints correctly
2. Run locally with `torchrun --nproc-per-node=1` on a small dataset to verify:
   - Checkpoint loads correctly (state dict hooks work)
   - Forward pass produces same output as original
   - Only trainable expert params appear in optimizer groups
   - Gradient is zero for frozen expert params, nonzero for trainable
3. Save checkpoint and verify it loads back into the original (non-split) model
