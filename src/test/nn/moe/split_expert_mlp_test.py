"""Tests for SplitExpertDroplessMoEMLP and checkpoint conversion."""

import torch
import torch.nn.functional as F

from olmo_core.nn.moe.mlp import DroplessMoEMLP, SplitExpertDroplessMoEMLP


D_MODEL = 64
HIDDEN_SIZE = 32
NUM_EXPERTS = 8
EXPERTS_TO_TRAIN = [2, 5]


def _make_original_mlp():
    return DroplessMoEMLP(
        d_model=D_MODEL, hidden_size=HIDDEN_SIZE, num_experts=NUM_EXPERTS,
        dtype=torch.float32, init_device="cpu",
    )


def _make_split_mlp():
    return SplitExpertDroplessMoEMLP(
        d_model=D_MODEL, hidden_size=HIDDEN_SIZE, num_experts=NUM_EXPERTS,
        experts_to_train=EXPERTS_TO_TRAIN, dtype=torch.float32, init_device="cpu",
    )


def _copy_regular_to_split(original, split):
    """Manually copy weights from regular MLP to split MLP (simulates convert_to_split)."""
    with torch.no_grad():
        for name in ("w1", "w2", "w3"):
            full_weight = getattr(original, name).data
            getattr(split, f"{name}_frozen").data.copy_(full_weight[split._frozen_row_indices])
            getattr(split, f"{name}_trainable").data.copy_(full_weight[split._trainable_row_indices])


def _copy_split_to_regular(split, regular):
    """Manually copy weights from split MLP to regular MLP (simulates convert_to_regular)."""
    with torch.no_grad():
        for name in ("w1", "w2", "w3"):
            full_weight = getattr(regular, name).data
            full_weight[split._frozen_row_indices] = getattr(split, f"{name}_frozen").data
            full_weight[split._trainable_row_indices] = getattr(split, f"{name}_trainable").data


def test_split_mlp_has_correct_param_shapes():
    mlp = _make_split_mlp()
    num_frozen = NUM_EXPERTS - len(EXPERTS_TO_TRAIN)

    assert mlp.w1_trainable.shape == (len(EXPERTS_TO_TRAIN) * HIDDEN_SIZE, D_MODEL)
    assert mlp.w1_frozen.shape == (num_frozen * HIDDEN_SIZE, D_MODEL)
    assert mlp.w1_trainable.requires_grad is True
    assert mlp.w1_frozen.requires_grad is False


def test_split_mlp_reconstruct_roundtrip():
    """Verify _reconstruct produces a tensor with the correct expert ordering."""
    mlp = _make_split_mlp()

    # Fill with identifiable values: expert i's rows are all i+1
    with torch.no_grad():
        for i, idx in enumerate(mlp.experts_frozen):
            start = i * HIDDEN_SIZE
            end = start + HIDDEN_SIZE
            mlp.w1_frozen[start:end, :] = float(idx + 1)
        for i, idx in enumerate(mlp.experts_to_train):
            start = i * HIDDEN_SIZE
            end = start + HIDDEN_SIZE
            mlp.w1_trainable[start:end, :] = float(idx + 1)

    full = mlp._reconstruct(mlp.w1_frozen, mlp.w1_trainable)
    assert full.shape == (NUM_EXPERTS * HIDDEN_SIZE, D_MODEL)

    # Check each expert's rows
    for expert_idx in range(NUM_EXPERTS):
        start = expert_idx * HIDDEN_SIZE
        end = start + HIDDEN_SIZE
        expected_val = float(expert_idx + 1)
        assert torch.allclose(full[start:end], torch.full((HIDDEN_SIZE, D_MODEL), expected_val)), \
            f"Expert {expert_idx} rows mismatch"


def test_split_state_dict_has_split_keys():
    """State dict from SplitExpertDroplessMoEMLP should have split keys (no hooks)."""
    mlp = _make_split_mlp()
    sd = mlp.state_dict()

    # Should have split keys
    assert "w1_frozen" in sd
    assert "w1_trainable" in sd
    assert "w2_frozen" in sd
    assert "w2_trainable" in sd
    # Should NOT have original keys
    assert "w1" not in sd
    assert "w2" not in sd


def test_copy_regular_to_split_roundtrip():
    """Copy regular → split → regular and verify weights match."""
    original = _make_original_mlp()

    split = _make_split_mlp()
    _copy_regular_to_split(original, split)

    # Verify reconstruction matches original
    full_w1 = split._reconstruct(split.w1_frozen, split.w1_trainable)
    assert torch.allclose(full_w1, original.w1.data)

    full_w2 = split._reconstruct(split.w2_frozen, split.w2_trainable)
    assert torch.allclose(full_w2, original.w2.data)

    full_w3 = split._reconstruct(split.w3_frozen, split.w3_trainable)
    assert torch.allclose(full_w3, original.w3.data)


def test_copy_split_to_regular_roundtrip():
    """Copy regular → split → regular and verify round-trip."""
    original = _make_original_mlp()
    split = _make_split_mlp()
    _copy_regular_to_split(original, split)

    # Copy back to a new regular
    original2 = _make_original_mlp()
    _copy_split_to_regular(split, original2)

    assert torch.allclose(original.w1.data, original2.w1.data)
    assert torch.allclose(original.w2.data, original2.w2.data)
    assert torch.allclose(original.w3.data, original2.w3.data)


def test_split_state_dict_save_load_roundtrip():
    """Save split state dict, load into another split MLP."""
    original = _make_original_mlp()
    split1 = _make_split_mlp()
    _copy_regular_to_split(original, split1)

    sd = split1.state_dict()

    split2 = _make_split_mlp()
    split2.load_state_dict(sd)

    assert torch.allclose(split1.w1_trainable.data, split2.w1_trainable.data)
    assert torch.allclose(split1.w1_frozen.data, split2.w1_frozen.data)


def test_forward_matches_original():
    """SplitExpertDroplessMoEMLP forward should match DroplessMoEMLP given same weights."""
    original = _make_original_mlp()

    split = _make_split_mlp()
    _copy_regular_to_split(original, split)

    # Create input: 4 tokens, 2 per expert for experts 0 and 1
    x = torch.randn(4, D_MODEL)
    batch_size_per_expert = torch.zeros(NUM_EXPERTS, dtype=torch.long)
    batch_size_per_expert[0] = 2
    batch_size_per_expert[1] = 2

    out_original = original(x, batch_size_per_expert)
    out_split = split(x, batch_size_per_expert)

    assert torch.allclose(out_original, out_split, atol=1e-5), \
        f"Max diff: {(out_original - out_split).abs().max()}"


def test_gradient_only_flows_to_trainable():
    """Gradients should only flow to trainable expert params, not frozen."""
    original = _make_original_mlp()
    split = _make_split_mlp()
    _copy_regular_to_split(original, split)

    x = torch.randn(4, D_MODEL)
    batch_size_per_expert = torch.zeros(NUM_EXPERTS, dtype=torch.long)
    # Route tokens to a trainable expert (index 2)
    batch_size_per_expert[EXPERTS_TO_TRAIN[0]] = 4

    out = split(x, batch_size_per_expert)
    out.sum().backward()

    assert split.w1_trainable.grad is not None
    assert split.w1_trainable.grad.abs().sum() > 0
    assert split.w1_frozen.grad is None  # requires_grad=False, so no grad


if __name__ == "__main__":
    test_split_mlp_has_correct_param_shapes()
    print("PASSED: test_split_mlp_has_correct_param_shapes")

    test_split_mlp_reconstruct_roundtrip()
    print("PASSED: test_split_mlp_reconstruct_roundtrip")

    test_split_state_dict_has_split_keys()
    print("PASSED: test_split_state_dict_has_split_keys")

    test_copy_regular_to_split_roundtrip()
    print("PASSED: test_copy_regular_to_split_roundtrip")

    test_copy_split_to_regular_roundtrip()
    print("PASSED: test_copy_split_to_regular_roundtrip")

    test_split_state_dict_save_load_roundtrip()
    print("PASSED: test_split_state_dict_save_load_roundtrip")

    test_forward_matches_original()
    print("PASSED: test_forward_matches_original")

    test_gradient_only_flows_to_trainable()
    print("PASSED: test_gradient_only_flows_to_trainable")

    print("\nAll tests passed!")
