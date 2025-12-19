from pathlib import Path
from typing import Optional

import pytest
import torch

from scripts.akshitab.add_finegrained_expert.add_new_expert import (
    add_experts,
    get_model_config,
    load_checkpoint,
)


@pytest.mark.parametrize("init_method", ["random", "average", "zero", "similar"])
@pytest.mark.parametrize("num_new_experts", [1, 3, 8])
@pytest.mark.parametrize("top_k_expert_indices", [None, [4], [1, 2]])
def test_add_expert(tmp_path: Path, init_method: str, num_new_experts: int, top_k_expert_indices: Optional[list[int]]):
    checkpoint_path = "src/test_fixtures/smallmoe"
    output_path = tmp_path / "smallmoe_with_new_expert"

    old_config = get_model_config(checkpoint_path)

    if init_method == "similar" and top_k_expert_indices is None:
        pytest.skip("top_k_expert_indices must be provided for 'similar' init_method")

    add_experts(
        checkpoint_path="src/test_fixtures/smallmoe",
        save_path=str(output_path),
        init_method=init_method,
        num_new_experts=num_new_experts,
        top_k_expert_indices=top_k_expert_indices,
    )

    assert output_path.exists()
    new_config = get_model_config(str(output_path))
    assert (
        new_config.block.feed_forward_moe.num_experts
        == old_config.block.feed_forward_moe.num_experts + num_new_experts
    )

    num_experts = old_config.block.feed_forward_moe.num_experts

    old_model = load_checkpoint(old_config, checkpoint_path)
    new_model = load_checkpoint(new_config, str(output_path))
    for (old_name, old_param), (new_name, new_param) in zip(
        old_model.named_parameters(), new_model.named_parameters()
    ):
        if "experts.mlp" in old_name:
            if init_method == "zero":
                assert torch.all(new_param.data[-1] == 0)
            elif init_method == "random":
                assert not torch.all(new_param.data[-1] == 0)
            elif init_method == "average":
                source_param = old_model.state_dict()[old_name]
                source_rows, source_columns = source_param.shape
                expected_value = source_param.view(
                    num_experts, source_rows // num_experts, source_columns
                ).data.mean(dim=0)
                actual_value = new_param.data.view(
                    num_experts + num_new_experts, source_rows // num_experts, source_columns
                )[-1]
                assert torch.allclose(
                    actual_value, expected_value, atol=1e-6
                ), f"Discrepancy in {old_name} and {new_name}: expected {expected_value}, got {new_param.data[-1]}"
            elif init_method == "similar" and top_k_expert_indices is not None:
                source_param = old_model.state_dict()[old_name]
                source_rows, source_columns = source_param.shape
                expected_value = torch.zeros(
                    source_rows // num_experts, source_columns
                )
                for idx in top_k_expert_indices:
                    expected_value += source_param.view(
                        num_experts, source_rows // num_experts, source_columns
                    )[idx]
                expected_value /= len(top_k_expert_indices)
                actual_value = new_param.data.view(
                    num_experts + num_new_experts, source_rows // num_experts, source_columns
                )[-1]
                assert torch.allclose(
                    actual_value, expected_value, atol=1e-6
                ), f"Discrepancy in {old_name} and {new_name}: expected {expected_value}, got {new_param.data[-1]}"
        elif "router.weight" in old_name:
            if init_method == "zero":
                assert torch.all(new_param.data[-1] == 0)
            elif init_method == "random":
                assert not torch.all(new_param.data[-1] == 0)
            elif init_method == "average":
                source_param = old_model.state_dict()[old_name].view(num_experts, -1)
                source_rows, source_columns = source_param.shape
                expected_value = source_param.data.mean(dim=0)
                actual_value = new_param.data.view(num_experts + num_new_experts, source_columns)[-1]
                assert torch.allclose(actual_value, expected_value, atol=1e-6)
            elif init_method == "similar" and top_k_expert_indices is not None:
                source_param = old_model.state_dict()[old_name].view(num_experts, -1)
                source_rows, source_columns = source_param.shape
                expected_value = torch.zeros(source_columns)
                for idx in top_k_expert_indices:
                    expected_value += source_param.data[idx]
                expected_value /= len(top_k_expert_indices)
                actual_value = new_param.data.view(num_experts + num_new_experts, source_columns)[-1]
                assert torch.allclose(actual_value, expected_value, atol=1e-6)
        else:
            assert torch.all(
                old_param.data == new_param.data
            ), f"Discrepancy between {old_name} and {new_name}"
