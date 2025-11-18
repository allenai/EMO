import pytest
import torch
from pathlib import Path
from scripts.akshitab.add_finegrained_expert.add_new_expert import add_expert, get_model_config, load_checkpoint

@pytest.mark.parametrize("init_method", ["random"]) #, "zero", "average"])
def test_add_expert(
    tmp_path: Path,
    init_method: str
):
    checkpoint_path = "src/test_fixtures/smallmoe"
    output_path = tmp_path / "smallmoe_with_new_expert"
    
    old_config = get_model_config(checkpoint_path)

    add_expert(
        checkpoint_path="src/test_fixtures/smallmoe",
        save_path=str(output_path),
        init_method=init_method,
    )

    assert output_path.exists()
    new_config = get_model_config(str(output_path))
    assert new_config.block.feed_forward_moe.num_experts == old_config.block.feed_forward_moe.num_experts + 1

    old_model = load_checkpoint(old_config, checkpoint_path)
    new_model = load_checkpoint(new_config, str(output_path))
    for (old_name, old_param), (new_name, new_param) in zip(old_model.named_parameters(), new_model.named_parameters()):
        if "experts.mlp" in old_name:
            if init_method == "zero":
                assert torch.all(new_param.data[-1] == 0)
            elif init_method == "random":
                assert not torch.all(new_param.data[-1] == 0)
            elif init_method == "average":
                expert_idx = int(old_name.split(".")[-3])
                expected_value = old_model.state_dict()[old_name].data.mean(dim=0)
                assert torch.allclose(new_param.data[-1], expected_value, atol=1e-6)
        elif "router.weight" in old_name:
            if init_method == "zero":
                assert torch.all(new_param.data[-1] == 0)
            elif init_method == "random":
                assert not torch.all(new_param.data[-1] == 0)
            elif init_method == "average":
                expected_value = old_model.state_dict()[old_name].data.mean(dim=0)
                assert torch.allclose(new_param.data[-1], expected_value, atol=1e-6)
        else:
            assert torch.all(old_param.data == new_param.data), f"Discrepancy between {old_name} and {new_name}"