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
from olmo_core.nn.transformer.init import InitMethod

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

    # Copy .metadata from model_and_optim to root so the checkpoint is recognized
    import shutil
    metadata_source = os.path.join(model_weights_path, ".metadata")
    metadata_dest = os.path.join(save_path, ".metadata")
    if os.path.exists(metadata_source):
        shutil.copy2(metadata_source, metadata_dest)
        logger.info(f"Copied .metadata to root: {metadata_dest}")
    else:
        logger.warning(f".metadata file not found at {metadata_source}")


def copy_param_with_prune(source_param, target_param, num_experts, prune_keep_k, model_config, experts_to_keep):
    """
    Copy parameters from source to target, handling two cases:
    NOTE: c means hidden dim of model, NOT context length
    NOTE: b means hidden dim of expert (usually hidden dim of model / 2)

    Case 1: Multi-block tensors (expert weights)
        source_param: (num_experts * b, c)
        target_param: (prune_keep_k * b, c)

    Case 2: Flat tensors that should be viewed as matrices (router weights)
        source_param: (num_experts * c,)
        target_param: (prune_keep_k * c,)

    "source_param": parameter tensor from the source model
    "target_param": parameter tensor from the target model
    "num_experts": number of experts in the source model
    "prune_keep_k": number of experts to keep in the target model
    "model_config": config of the model to check dimensions
    "experts_to_keep": list of experts to keep for the layer being processed. Tensor of shape (prune_keepk,) with range [0, num_experts-1]
    """

    # Router weights
    if len(source_param.shape) == 1:
        source_len = source_param.shape[0]
        target_len = target_param.shape[0]

        # Calculate c by dividing the length by num_experts
        if source_len % num_experts != 0:
            raise ValueError(
                f"Source length {source_len} is not divisible by num_experts {num_experts}"
            )

        c = source_len // num_experts
        assert c == model_config.d_model, f"Expected hidden dim {model_config.d_model}, got {c}"

        # Verify target length
        expected_target_len = prune_keep_k * c
        if target_len != expected_target_len:
            raise ValueError(f"Expected target length {expected_target_len}, got {target_len}")

        # Reshape flat tensors to matrices
        source_matrix = source_param.view(num_experts, c)

        # initialize new tensor for target
        target_new = torch.zeros(
            prune_keep_k * c, device=target_param.device, dtype=target_param.dtype
        )

        target_matrix = target_new.view(prune_keep_k, c)


        # Copy data for selected experts
        for i, expert_idx in enumerate(experts_to_keep):
            target_matrix[i, :] = source_matrix[expert_idx, :]

        # Update target parameter
        with torch.no_grad():
            target_param.copy_(target_new)

    # Expert weights
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
        expected_target_rows = prune_keep_k * b

        # Verify target shape
        if target_rows != expected_target_rows:
            raise ValueError(
                f"Expected target rows to be {expected_target_rows}, got {target_rows}"
            )

        # Reshape for manipulation
        source_reshaped = source_param.view(num_experts, b, columns)

        # Create new tensor
        target_new = torch.zeros(
            prune_keep_k, b, columns, device=target_param.device, dtype=target_param.dtype
        )

        # Copy data
        for i, expert_idx in enumerate(experts_to_keep):
            target_new[i, :, :] = source_reshaped[expert_idx, :, :]

        # Reshape back
        target_flat = target_new.view(prune_keep_k * b, columns)

        # Update target
        with torch.no_grad():
            target_param.copy_(target_flat)

    return target_param


def prune_experts(args):
    # Load model config
    config_path = os.path.join(args.checkpoint_path, "config.json")
    logger.info(f"Loading model config from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    model_config = TransformerConfig.from_dict(config["model"])
    logger.info(f"Model config {model_config}")

    # Load model weights
    logger.info(f"Loading model weights from {args.checkpoint_path}")
    model = load_checkpoint(model_config=model_config, checkpoint_path=args.checkpoint_path)
    logger.info("Model loaded successfully")

    # Update config
    logger.info("Adding new expert to the model")
    new_config = model_config.copy()
    assert new_config.block.feed_forward_moe is not None, "Model is not MoE"
    # Set the total number of experts to prune_keep_k
    new_config.block.feed_forward_moe.num_experts = args.prune_keep_k
    new_model = new_config.build(init_device="cpu")

    new_model.init_weights() # Initialized with random init

    num_experts = model_config.block.feed_forward_moe.num_experts
    assert args.prune_keep_k < num_experts, f"prune_keep_k {args.prune_keep_k} must be less than original number of experts {num_experts}"

    # we now load in the activation file to determine which experts to keep
    with open(args.activation_file, 'r') as f:
        line = f.readline()
        activations = json.loads(line)["avg_router_probabilities"]
    assert len(activations) == model_config.n_layers, f"Number of layers in activation file {len(activations)} does not match orig model {model_config.n_layers}"

    # Copy weights from old model to new model
    for name, param in model.named_parameters():
        assert name in new_model.state_dict() , f"Parameter {name} not found in new model, expected to be same since we're pruning router weights only"
        if name in new_model.state_dict():
            new_param = new_model.state_dict()[name]
            if param.shape == new_param.shape:
                new_param.data.copy_(param.data)
            else:
                # assert "router" in name, f"Shape mismatch for parameter {name}"
                if "router" in name or "experts" in name:
                    # we extract the layer number
                    layer_idx = int(name.split(".")[1]) # Assumes naming convention like 'blocks.15.feed_forward_moe.router.weight', 'blocks.15.feed_forward_moe.experts.mlp.w1'
                    layer_activation = activations[layer_idx]
                    experts_to_keep = torch.topk(
                        torch.tensor(layer_activation),
                        min(args.prune_keep_k, len(layer_activation))
                    ).indices.tolist()

                    copy_param_with_prune(param, new_param, num_experts, args.prune_keep_k, model_config, experts_to_keep)
                else:
                    raise ValueError(f"Shape mismatch for parameter {name}, cannot prune non-router/expert weights")

        else:
            logger.debug(f"Parameter {name} not found in new model, not updating weights")

    logger.info("Weights copied to new model successfully")

    # Save new model checkpoint
    if args.save_path is not None:
        config["model"] = new_config.as_config_dict()
        save_checkpoint(config, new_model, args.save_path)

    return new_model


def parse_args():
    parser = argparse.ArgumentParser(description="Add new expert to MoE model")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to existing MoE checkpoint"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save new MoE checkpoint"
    )
    parser.add_argument(
        "--prune_keep_k", type=int, required=True, help="Number of experts to keep after pruning"
    )
    parser.add_argument(
        "--activation_file", type=str, required=True, help="Path to activation file that records expert activations"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    new_model = prune_experts(args)