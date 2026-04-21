"""
Test that the static-shape replacements in router_lbreducedp_sharedexp.py
produce identical results to the original masked_select-based logic.
"""

import pytest
import torch

from olmo_core.ops.moe import batched_histc


def _original_masked_select_logic(expert_indices, scores, num_standard, padding_mask):
    """
    Original implementation using masked_select (causes torch.compile graph breaks).
    """
    if padding_mask is not None:
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(expert_indices)
        valid_expert_indices = expert_indices.masked_select(padding_mask_expanded).view(
            -1, expert_indices.size(-1)
        )
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(scores)
        valid_scores = scores.masked_select(padding_mask_expanded).view(-1, num_standard)

        batched_bspe = batched_histc(valid_expert_indices, num_standard)
        batch_size_per_expert = batched_bspe.sum(dim=0)
    else:
        valid_expert_indices = expert_indices.view(-1, expert_indices.size(-1))
        valid_scores = scores.view(-1, num_standard)
        batch_size_per_expert = batched_histc(
            expert_indices.view(-1, expert_indices.size(-1)), num_standard
        ).sum(dim=0)

    # unique experts
    unique_experts = torch.unique(valid_expert_indices.view(-1))
    num_unique_experts = unique_experts.numel()

    # entropy
    token_entropies = -torch.sum(valid_scores * torch.log(valid_scores + 1e-10), dim=-1)
    avg_entropy = token_entropies.mean().item()

    return batch_size_per_expert, num_unique_experts, avg_entropy


def _new_static_shape_logic(expert_indices, scores, num_standard, padding_mask):
    """
    New implementation using static-shape ops (compile-friendly).
    """
    if padding_mask is not None:
        weights = padding_mask.unsqueeze(-1).expand_as(expert_indices).to(expert_indices.dtype)
        hist = torch.zeros(
            (*expert_indices.shape[:-1], num_standard),
            dtype=expert_indices.dtype,
            device=expert_indices.device,
        )
        hist.scatter_add_(-1, expert_indices, weights)
        batched_bspe = hist.sum(dim=1)
        batch_size_per_expert = batched_bspe.sum(dim=0)
    else:
        batch_size_per_expert = batched_histc(
            expert_indices.view(-1, expert_indices.size(-1)), num_standard
        ).sum(dim=0)

    # unique experts via histogram
    num_unique_experts = (batch_size_per_expert > 0).sum().item()

    # entropy with masked mean
    token_entropies = -torch.sum(scores * torch.log(scores + 1e-10), dim=-1)  # (B, S)
    if padding_mask is not None:
        token_entropies = token_entropies * padding_mask.float()
        num_valid = padding_mask.sum().clamp(min=1)
        avg_entropy = (token_entropies.sum() / num_valid).item()
    else:
        avg_entropy = token_entropies.mean().item()

    return batch_size_per_expert, num_unique_experts, avg_entropy


def _make_test_data(batch_size, seq_len, num_standard, top_k, padding_frac, seed):
    torch.manual_seed(seed)
    expert_indices = torch.randint(0, num_standard, (batch_size, seq_len, top_k))
    # softmax scores so they sum to 1 per token
    raw = torch.randn(batch_size, seq_len, num_standard).float()
    scores = raw.softmax(dim=-1)

    if padding_frac > 0:
        padding_mask = torch.rand(batch_size, seq_len) > padding_frac
        # Ensure at least one valid token
        padding_mask[0, 0] = True
    else:
        padding_mask = None

    return expert_indices, scores, num_standard, padding_mask


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [8, 32])
@pytest.mark.parametrize("num_standard", [8, 64])
@pytest.mark.parametrize("top_k", [1, 4])
@pytest.mark.parametrize("padding_frac", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("seed", [42, 123])
def test_batch_size_per_expert_equivalence(
    batch_size, seq_len, num_standard, top_k, padding_frac, seed
):
    expert_indices, scores, num_standard, padding_mask = _make_test_data(
        batch_size, seq_len, num_standard, top_k, padding_frac, seed
    )

    old_bspe, _, _ = _original_masked_select_logic(
        expert_indices, scores, num_standard, padding_mask
    )
    new_bspe, _, _ = _new_static_shape_logic(expert_indices, scores, num_standard, padding_mask)

    assert torch.equal(
        old_bspe, new_bspe
    ), f"batch_size_per_expert mismatch:\nold={old_bspe}\nnew={new_bspe}"


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [8, 32])
@pytest.mark.parametrize("num_standard", [8, 64])
@pytest.mark.parametrize("top_k", [1, 4])
@pytest.mark.parametrize("padding_frac", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("seed", [42, 123])
def test_num_unique_experts_equivalence(
    batch_size, seq_len, num_standard, top_k, padding_frac, seed
):
    expert_indices, scores, num_standard, padding_mask = _make_test_data(
        batch_size, seq_len, num_standard, top_k, padding_frac, seed
    )

    _, old_unique, _ = _original_masked_select_logic(
        expert_indices, scores, num_standard, padding_mask
    )
    _, new_unique, _ = _new_static_shape_logic(expert_indices, scores, num_standard, padding_mask)

    assert (
        old_unique == new_unique
    ), f"num_unique_experts mismatch: old={old_unique}, new={new_unique}"


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [8, 32])
@pytest.mark.parametrize("num_standard", [8, 64])
@pytest.mark.parametrize("top_k", [1, 4])
@pytest.mark.parametrize("padding_frac", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("seed", [42, 123])
def test_entropy_equivalence(batch_size, seq_len, num_standard, top_k, padding_frac, seed):
    expert_indices, scores, num_standard, padding_mask = _make_test_data(
        batch_size, seq_len, num_standard, top_k, padding_frac, seed
    )

    _, _, old_entropy = _original_masked_select_logic(
        expert_indices, scores, num_standard, padding_mask
    )
    _, _, new_entropy = _new_static_shape_logic(expert_indices, scores, num_standard, padding_mask)

    assert (
        abs(old_entropy - new_entropy) < 1e-5
    ), f"avg_entropy mismatch: old={old_entropy}, new={new_entropy}"
