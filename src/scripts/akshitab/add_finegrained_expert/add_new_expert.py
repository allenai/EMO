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


class AddExpertInitMethod:
    RANDOM = "random"
    """
    Initialize new expert with random weights.
    """

    AVERAGE = "average"
    """
    Initialize new expert with average of existing experts.
    """

    ZERO = "zero"
    """
    Initialize new expert with zeros.
    """

    SIMILAR = "similar"
    """
    Initialize new expert with weights similar to existing experts.
    """


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


def add_expert(
    checkpoint_path: str, save_path: Optional[str] = None, init_method: Optional[str] = None
):
    # Load model config
    old_config_path = os.path.join(checkpoint_path, "config.json")
    logger.info(f"Loading model config from {old_config_path}")
    with open(old_config_path, "r") as f:
        config = json.load(f)

    old_model_config = TransformerConfig.from_dict(config["model"])
    logger.info(f"Model config {old_model_config}")

    # Load model weights
    logger.info(f"Loading model weights from {checkpoint_path}")
    old_model = load_checkpoint(model_config=old_model_config, checkpoint_path=checkpoint_path)
    logger.info("Model loaded successfully")

    # Update config
    logger.info("Adding new expert to the model")
    new_config = old_model_config.copy()
    assert new_config.block.feed_forward_moe is not None, "Model is not MoE"
    new_config.block.feed_forward_moe.num_experts += 1
    new_model = new_config.build(init_device="cpu")

    new_model.init_weights()  # Initialized with random init

    assert old_model_config.block.feed_forward_moe is not None
    num_experts = old_model_config.block.feed_forward_moe.num_experts

    init_method = init_method or AddExpertInitMethod.RANDOM

    # Copy weights from old model to new model
    for name, old_param in old_model.named_parameters():
        if name in new_model.state_dict():
            new_param = new_model.state_dict()[name]
            if old_param.shape == new_param.shape:
                new_param.data.copy_(old_param.data)

            elif "router.weight" in name:
                source_param = old_param.view(num_experts, -1)
                _, source_columns = source_param.shape

                target_param = new_param.view(num_experts + 1, source_columns).clone()

                if init_method == AddExpertInitMethod.ZERO:
                    target_param = torch.zeros(
                        (num_experts + 1), source_columns, device=new_param.device, dtype=new_param.dtype
                    )
                elif init_method == AddExpertInitMethod.RANDOM:
                    # Do nothing, the new_model was initialized randomly already
                    pass
                elif init_method == AddExpertInitMethod.AVERAGE:
                    target_param = torch.empty(
                        (num_experts + 1), source_columns, device=new_param.device, dtype=new_param.dtype
                    )
                    # Compute average of existing experts
                    avg_expert = source_param.data.mean(dim=0)
                    # Copy average to new expert position
                    with torch.no_grad():
                        target_param[-1, :].copy_(avg_expert)
                elif init_method == AddExpertInitMethod.SIMILAR:
                    print("Similar initialization not implemented yet.")
                    raise NotImplementedError("Similar initialization not implemented yet.")

                target_param[:num_experts, :] = source_param
                with torch.no_grad():
                    new_param.data.copy_(target_param.view(-1))

            elif "experts.mlp" in name:
                source_param = old_param.clone()
                source_rows, source_columns = source_param.shape

                target_param = new_param.view(num_experts + 1, source_rows // num_experts, source_columns).clone()

                if init_method == AddExpertInitMethod.ZERO:
                    target_param = torch.zeros(
                        (num_experts + 1), (source_rows // num_experts), source_columns,
                        device=new_param.device,
                        dtype=new_param.dtype,
                    )
                elif init_method == AddExpertInitMethod.RANDOM:
                    # Do nothing, the new_model was initialized randomly already
                    pass
                elif init_method == AddExpertInitMethod.AVERAGE:
                    target_param = torch.empty(
                        (num_experts + 1), (source_rows // num_experts), source_columns,
                        device=new_param.device,
                        dtype=new_param.dtype,
                    )
                    # Compute average of existing experts
                    source_param = source_param.view(num_experts, source_rows // num_experts, source_columns)
                    avg_expert = source_param.data.mean(dim=0)
                    # Copy average to new expert position
                    with torch.no_grad():
                        target_param[-1, :, :].copy_(avg_expert)
                elif init_method == AddExpertInitMethod.SIMILAR:
                    print("Similar initialization not implemented yet.")
                    raise NotImplementedError("Similar initialization not implemented yet.")

                target_param[:num_experts, :, :] = source_param.view(num_experts, source_rows // num_experts, source_columns)
                with torch.no_grad():
                    new_param.data.copy_(target_param.view(-1, source_columns))

        else:
            logger.debug(f"Parameter {name} not found in new model, not updating weights")


    logger.info("Weights copied to new model successfully")

    # Save new model checkpoint
    if save_path is not None:
        config["model"] = new_config.as_config_dict()
        save_checkpoint(config, new_model, save_path)

    return new_model



def parse_args():
    parser = argparse.ArgumentParser(description="Add new expert to MoE model")
    parser.add_argument(
        "-c", "--checkpoint_path", type=str, required=True, help="Path to existing MoE checkpoint"
    )
    parser.add_argument(
        "-o", "--save_path", type=str, required=True, help="Path to save new MoE checkpoint"
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default=AddExpertInitMethod.RANDOM,
        help="Initialization method for new expert",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    new_model = add_expert(args.checkpoint_path, args.save_path)
