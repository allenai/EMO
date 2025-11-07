# Part 1: Add a new expert to an existing Mixture of Experts (MoE) model checkpoint
# Load existing MoE checkpoint
# Create new MoE with same config + new expert
# Copy weights from loaded MoE to new MoE.
# Save new MoE checkpoint

# Part 2:
# Train the new expert on some data, keeping other experts frozen

import argparse
import json
import logging
import os
from typing import Optional

import torch

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.transformer import TransformerConfig

logger = logging.getLogger(__name__)


def get_model_config(checkpoint_path: str):
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    model_config = TransformerConfig.from_dict(config["model"])
    return model_config


def load_checkpoint(model_config: TransformerConfig, checkpoint_path: str):
    model = model_config.build(init_device="cpu")

    # Load model weights
    model_weights_path = os.path.join(checkpoint_path, "model_and_optim")
    load_model_and_optim_state(dir=model_weights_path, model=model, optim=None)
    return model


def save_checkpoint(config: dict, model: torch.nn.Module, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving new model checkpoint to {save_path}")
    new_config_path = os.path.join(save_path, "config.json")

    with open(new_config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved new model config to {new_config_path}")

    model_weights_path = os.path.join(save_path, "model_and_optim")
    save_model_and_optim_state(dir=model_weights_path, model=model, optim=None)
    logger.info(f"Saved new model weights to {model_weights_path}")


def copy_param_with_resize(source_param, target_param, num_experts: int):
    """
    Copy parameters from source to target, handling two cases:

    Case 1: Multi-block tensors
        source_param: (num_experts * b, c)
        target_param: ((num_experts + 1) * b, c)

    Case 2: Flat tensors that should be viewed as matrices
        source_param: (num_experts * c,)
        target_param: ((num_experts + 1) * c,)
    """
    # Check if we're dealing with flat tensors (1D) - Case 2
    if len(source_param.shape) == 1:
        source_len = source_param.shape[0]
        target_len = target_param.shape[0]

        # Calculate c by dividing the length by num_experts
        if source_len % num_experts != 0:
            raise ValueError(
                f"Source length {source_len} is not divisible by num_experts {num_experts}"
            )

        c = source_len // num_experts

        # Verify target length
        expected_target_len = (num_experts + 1) * c
        if target_len != expected_target_len:
            raise ValueError(f"Expected target length {expected_target_len}, got {target_len}")

        # Reshape flat tensors to matrices
        source_matrix = source_param.view(num_experts, c)

        # Create a new tensor for target
        target_new = torch.zeros(
            (num_experts + 1) * c, device=target_param.device, dtype=target_param.dtype
        )
        target_matrix = target_new.view(num_experts + 1, c)

        # Copy data
        target_matrix[:num_experts, :] = source_matrix

        # Update target parameter
        with torch.no_grad():
            target_param.copy_(target_new)

    # Case 1: 2D tensors (multi-block)
    else:
        source_rows, columns = source_param.shape
        target_rows, target_columns = target_param.shape

        # Check column dimensions match
        if columns != target_columns:
            raise ValueError(f"Column dimensions don't match: {columns} vs {target_columns}")

        # Calculate b
        if source_rows % num_experts != 0:
            raise ValueError(
                f"Source rows {source_rows} is not divisible by num_experts {num_experts}"
            )

        b = source_rows // num_experts
        expected_target_rows = (num_experts + 1) * b

        # Verify target shape
        if target_rows != expected_target_rows:
            raise ValueError(
                f"Expected target rows to be {expected_target_rows}, got {target_rows}"
            )

        # Reshape for manipulation
        source_reshaped = source_param.view(num_experts, b, columns)

        # Create new tensor
        a_new = num_experts + 1
        target_new = torch.zeros(
            a_new, b, columns, device=target_param.device, dtype=target_param.dtype
        )

        # Copy data
        target_new[:num_experts, :, :] = source_reshaped

        # Reshape back
        target_flat = target_new.view(a_new * b, columns)

        # Update target
        with torch.no_grad():
            target_param.copy_(target_flat)

    return target_param


def add_expert(checkpoint_path: str, save_path: Optional[str] = None):
    # Load model config
    config_path = os.path.join(checkpoint_path, "config.json")
    logger.info(f"Loading model config from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    model_config = TransformerConfig.from_dict(config["model"])
    logger.info(f"Model config {model_config}")

    # Load model weights
    logger.info(f"Loading model weights from {checkpoint_path}")
    model = load_checkpoint(model_config=model_config, checkpoint_path=checkpoint_path)
    logger.info("Model loaded successfully")

    # Update config
    logger.info("Adding new expert to the model")
    new_config = model_config.copy()
    assert new_config.block.feed_forward_moe is not None, "Model is not MoE"
    new_config.block.feed_forward_moe.num_experts += 1
    new_model = new_config.build(init_device="cpu")

    num_experts = model_config.block.feed_forward_moe.num_experts

    # Copy weights from old model to new model
    for name, param in model.named_parameters():
        if name in new_model.state_dict():
            new_param = new_model.state_dict()[name]
            if param.shape == new_param.shape:
                new_param.data.copy_(param.data)
            else:
                # assert "router" in name, f"Shape mismatch for parameter {name}"
                if "router" in name or "experts" in name:
                    copy_param_with_resize(param, new_param, num_experts)
                else:
                    logger.warning(
                        f"Shape mismatch for parameter {name}, but not a router or expert parameter. Skipping."
                    )
        else:
            logger.debug(f"Parameter {name} not found in new model, not updating weights")
            # TODO: how should expert weights for new expert be initialized?

    logger.info("Weights copied to new model successfully")

    # Save new model checkpoint
    if save_path is not None:
        config["model"] = new_config.as_config_dict()
        save_checkpoint(config, new_model, save_path)

    return new_model


def parse_args():
    parser = argparse.ArgumentParser(description="Add new expert to MoE model")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to existing MoE checkpoint"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save new MoE checkpoint"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    new_model = add_expert(args.checkpoint_path, args.save_path)
