"""
Remove expert(s) from an existing Mixture of Experts (MoE) model checkpoint.
"""
import argparse
import json
import logging
import os
from typing import Optional

import smart_open
import torch

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import setup_logging

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
    if os.path.exists(save_path):
        logger.warning(f"Save path {save_path} already exists. Not overwriting.")
    else:
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving new model checkpoint to {save_path}")
        new_config_path = os.path.join(save_path, "config.json")

        with open(new_config_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved new model config to {new_config_path}")

        model_weights_path = os.path.join(save_path, "model_and_optim")
        save_model_and_optim_state(dir=model_weights_path, model=model, optim=None)
        logger.info(f"Saved new model weights to {model_weights_path}")


def remove_experts(
    checkpoint_path: str,
    save_path: Optional[str] = None,
    experts_to_remove: list[int] = [],
):
    # Load model config
    old_config_path = os.path.join(checkpoint_path, "config.json")
    logger.info(f"Loading model config from {old_config_path}")
    with open(old_config_path, "r") as f:
        config = json.load(f)

    old_model_config = TransformerConfig.from_dict(config["model"])
    backend = old_model_config.block.attention.backend
    old_model_config.block.attention.backend = 'torch'
    logger.info(f"Model config {old_model_config}")

    assert old_model_config.block.feed_forward_moe is not None
    num_experts = old_model_config.block.feed_forward_moe.num_experts

    experts_to_remove = [expert_id + num_experts if expert_id < 0 else expert_id for expert_id in experts_to_remove]

    for expert_id in experts_to_remove:
        assert 0 <= expert_id < num_experts, f"Expert id {expert_id} out of range"

    experts_to_keep = [i for i in range(num_experts) if i not in experts_to_remove]

    # Load model weights
    logger.info(f"Loading model weights from {checkpoint_path}")
    old_model = load_checkpoint(model_config=old_model_config, checkpoint_path=checkpoint_path)
    logger.info("Model loaded successfully")

    # Update config
    logger.info("Removing experts from the model")
    new_config = old_model_config.copy()
    assert new_config.block.feed_forward_moe is not None, "Model is not MoE"
    new_config.block.feed_forward_moe.num_experts -= len(experts_to_remove)
    new_model = new_config.build(init_device="cpu")

    new_model.init_weights()  # Initialized with random init

    # Copy weights from old model to new model
    for name, old_param in old_model.named_parameters():
        if name in new_model.state_dict():
            new_param = new_model.state_dict()[name]
            if old_param.shape == new_param.shape:
                logger.info(f"Copying parameter {name} without changes")
                new_param.data.copy_(old_param.data)

            elif "router.weight" in name:
                logger.info(f"Copying parameter {name} with expert addition")
                source_param = old_param.view(num_experts, -1)
                _, source_columns = source_param.shape

                target_param = new_param.view(num_experts - len(experts_to_remove), source_columns).clone()

                target_param[experts_to_keep, :] = source_param
                with torch.no_grad():
                    new_param.data.copy_(target_param.view(-1))

            elif "experts.mlp" in name:
                source_param = old_param.clone()
                source_rows, source_columns = source_param.shape

                target_param = new_param.view(
                    num_experts - len(experts_to_remove), source_rows // num_experts, source_columns
                ).clone()

                target_param[experts_to_keep, :, :] = source_param.view(
                    num_experts, source_rows // num_experts, source_columns
                )
                with torch.no_grad():
                    new_param.data.copy_(target_param.view(-1, source_columns))

        else:
            logger.debug(f"Parameter {name} not found in new model, not updating weights")

    logger.info("Weights copied to new model successfully")

    # Save new model checkpoint
    if save_path is not None:
        new_config.block.attention.backend = backend
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
        "-n",
        "--num-experts-to-remove",
        type=int,
        default=1,
        help="Number of experts to remove from the model (from the end)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    
    new_model = remove_experts(
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        experts_to_remove=list(range(-args.num_experts_to_remove, 0))
    )
