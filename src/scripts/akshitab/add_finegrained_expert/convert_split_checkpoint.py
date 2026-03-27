"""
Convert MoE checkpoints between regular (DroplessMoEMLP) and split-expert
(SplitExpertDroplessMoEMLP) formats.

Usage:

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
"""

import argparse
import json
import logging
import os
from typing import List

import torch

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.moe.mlp import DroplessMoEMLP, SplitExpertDroplessMoEMLP
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import setup_logging

logger = logging.getLogger(__name__)


def load_config(checkpoint_path: str) -> dict:
    config_path = os.path.join(checkpoint_path, "config.json")
    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def build_and_load_model(model_config: TransformerConfig, checkpoint_path: str):
    model = model_config.build(init_device="cpu")
    model_weights_path = os.path.join(checkpoint_path, "model_and_optim")
    load_model_and_optim_state(dir=model_weights_path, model=model, optim=None)
    return model


def save_model(config: dict, model: torch.nn.Module, save_path: str):
    if os.path.exists(save_path):
        logger.warning(f"Save path {save_path} already exists. Not overwriting.")
        return
    os.makedirs(save_path, exist_ok=True)

    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved config to {config_path}")

    model_weights_path = os.path.join(save_path, "model_and_optim")
    save_model_and_optim_state(dir=model_weights_path, model=model, optim=None)
    logger.info(f"Saved model weights to {model_weights_path}")


def replace_mlps_with_split(model, experts_to_train: List[int]):
    """Replace DroplessMoEMLP with SplitExpertDroplessMoEMLP in all MoE layers."""
    for block in model.blocks:
        if block.feed_forward_moe is None:
            continue
        old_mlp = block.feed_forward_moe.experts.mlp
        if not isinstance(old_mlp, DroplessMoEMLP):
            logger.warning(f"Skipping non-DroplessMoEMLP: {type(old_mlp)}")
            continue

        new_mlp = SplitExpertDroplessMoEMLP(
            d_model=old_mlp.d_model,
            hidden_size=old_mlp.hidden_size,
            num_experts=old_mlp.num_experts,
            experts_to_train=experts_to_train,
            dtype=old_mlp.w1.dtype,
            init_device="cpu",
        )

        # Split weights from old_mlp into new_mlp
        with torch.no_grad():
            for name in ("w1", "w2", "w3"):
                full_weight = getattr(old_mlp, name).data
                getattr(new_mlp, f"{name}_frozen").data.copy_(
                    full_weight[new_mlp._frozen_row_indices]
                )
                getattr(new_mlp, f"{name}_trainable").data.copy_(
                    full_weight[new_mlp._trainable_row_indices]
                )

        block.feed_forward_moe.experts.mlp = new_mlp

    logger.info(
        f"Converted to split format: {len(experts_to_train)} trainable, "
        f"{old_mlp.num_experts - len(experts_to_train)} frozen"
    )


def replace_mlps_with_regular(model, experts_to_train: List[int]):
    """Replace SplitExpertDroplessMoEMLP with DroplessMoEMLP in all MoE layers."""
    for block in model.blocks:
        if block.feed_forward_moe is None:
            continue
        old_mlp = block.feed_forward_moe.experts.mlp
        if not isinstance(old_mlp, SplitExpertDroplessMoEMLP):
            logger.warning(f"Skipping non-SplitExpertDroplessMoEMLP: {type(old_mlp)}")
            continue

        new_mlp = DroplessMoEMLP(
            d_model=old_mlp.d_model,
            hidden_size=old_mlp.hidden_size,
            num_experts=old_mlp.num_experts,
            dtype=old_mlp.w1_trainable.dtype,
            init_device="cpu",
        )

        # Merge split weights back into full tensors
        with torch.no_grad():
            for name in ("w1", "w2", "w3"):
                full_weight = getattr(new_mlp, name).data
                full_weight[old_mlp._frozen_row_indices] = getattr(old_mlp, f"{name}_frozen").data
                full_weight[old_mlp._trainable_row_indices] = getattr(old_mlp, f"{name}_trainable").data

        block.feed_forward_moe.experts.mlp = new_mlp

    logger.info("Converted back to regular format")


def convert_to_split(checkpoint_path: str, save_path: str, experts_to_train: List[int]):
    config = load_config(checkpoint_path)
    model_config = TransformerConfig.from_dict(config["model"])

    logger.info(f"Loading regular checkpoint from {checkpoint_path}")
    model = build_and_load_model(model_config, checkpoint_path)

    replace_mlps_with_split(model, experts_to_train)

    # Store experts_to_train in config for reference
    config["split_expert_params"] = {
        "experts_to_train": experts_to_train,
    }

    save_model(config, model, save_path)
    logger.info(f"Split checkpoint saved to {save_path}")


def convert_to_regular(checkpoint_path: str, save_path: str, experts_to_train: List[int]):
    config = load_config(checkpoint_path)
    model_config = TransformerConfig.from_dict(config["model"])

    # Build model with split MLPs, then load checkpoint
    logger.info(f"Loading split checkpoint from {checkpoint_path}")
    model = model_config.build(init_device="cpu")

    # Replace with split MLPs before loading (so keys match)
    replace_mlps_with_split_empty(model, experts_to_train)
    model_weights_path = os.path.join(checkpoint_path, "model_and_optim")
    load_model_and_optim_state(dir=model_weights_path, model=model, optim=None)

    # Now convert back to regular
    replace_mlps_with_regular(model, experts_to_train)

    # Remove split metadata from config
    config.pop("split_expert_params", None)

    save_model(config, model, save_path)
    logger.info(f"Regular checkpoint saved to {save_path}")


def replace_mlps_with_split_empty(model, experts_to_train: List[int]):
    """Replace MLPs with SplitExpertDroplessMoEMLP (empty, for loading split checkpoints)."""
    for block in model.blocks:
        if block.feed_forward_moe is None:
            continue
        old_mlp = block.feed_forward_moe.experts.mlp
        if not isinstance(old_mlp, DroplessMoEMLP):
            continue

        new_mlp = SplitExpertDroplessMoEMLP(
            d_model=old_mlp.d_model,
            hidden_size=old_mlp.hidden_size,
            num_experts=old_mlp.num_experts,
            experts_to_train=experts_to_train,
            dtype=torch.float32,
            init_device="cpu",
        )
        block.feed_forward_moe.experts.mlp = new_mlp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MoE checkpoints between regular and split-expert formats."
    )
    parser.add_argument(
        "--checkpoint-path", type=str, required=True,
        help="Path to the input checkpoint directory.",
    )
    parser.add_argument(
        "--save-path", type=str, required=True,
        help="Path to save the converted checkpoint.",
    )
    parser.add_argument(
        "--experts-to-train", type=str, required=True,
        help="Comma-separated list of expert indices to keep trainable (e.g. '127,128,129,130').",
    )

    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument(
        "--to-split", action="store_true",
        help="Convert from regular to split-expert format.",
    )
    direction.add_argument(
        "--to-regular", action="store_true",
        help="Convert from split-expert to regular format.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()

    experts_to_train = [int(x) for x in args.experts_to_train.split(",")]
    logger.info(f"Experts to train: {experts_to_train}")

    if args.to_split:
        convert_to_split(args.checkpoint_path, args.save_path, experts_to_train)
    else:
        convert_to_regular(args.checkpoint_path, args.save_path, experts_to_train)
