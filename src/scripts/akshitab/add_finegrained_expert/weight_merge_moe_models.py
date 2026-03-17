"""
Script to weight-merge n MoE model checkpoints.

Given n checkpoints of the same architecture, produces a single checkpoint
whose parameters are a weighted average of the inputs. All models must share
the same TransformerConfig (same num_experts, layers, hidden size, etc.).

Usage:
    python weight_merge_moe_models.py \
        -c /path/to/ckpt1 /path/to/ckpt2 /path/to/ckpt3 \
        -o /path/to/merged \
        --weights 0.5 0.25 0.25
"""
import argparse
import json
import logging
import os
from typing import List, Optional

import torch

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import setup_logging

logger = logging.getLogger(__name__)


def get_model_config(checkpoint_path: str) -> TransformerConfig:
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return TransformerConfig.from_dict(config["model"])


def load_checkpoint(model_config: TransformerConfig, checkpoint_path: str):
    model = model_config.build(init_device="cpu")
    model_weights_path = os.path.join(checkpoint_path, "model_and_optim")
    load_model_and_optim_state(dir=model_weights_path, model=model, optim=None)
    return model


def save_checkpoint(config: dict, model: torch.nn.Module, save_path: str):
    if os.path.exists(save_path):
        logger.warning(f"Save path {save_path} already exists. Not overwriting.")
        return
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving merged model checkpoint to {save_path}")

    new_config_path = os.path.join(save_path, "config.json")
    with open(new_config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved config to {new_config_path}")

    model_weights_path = os.path.join(save_path, "model_and_optim")
    save_model_and_optim_state(dir=model_weights_path, model=model, optim=None)
    logger.info(f"Saved model weights to {model_weights_path}")


def weight_merge(
    checkpoint_paths: List[str],
    weights: Optional[List[float]] = None,
    save_path: Optional[str] = None,
):
    n = len(checkpoint_paths)
    assert n >= 2, "Need at least 2 checkpoints to merge"

    if weights is None:
        weights = [1.0 / n] * n
    else:
        assert len(weights) == n, f"Got {len(weights)} weights for {n} checkpoints"
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            weights = [w / weight_sum for w in weights]

    logger.info(f"Merging {n} checkpoints with weights: {weights}")

    # Load and validate configs — all must match
    configs_raw = []
    model_configs = []
    for path in checkpoint_paths:
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            configs_raw.append(json.load(f))
        model_configs.append(get_model_config(path))

    base_config = model_configs[0]
    backend = base_config.block.attention.backend
    base_config.block.attention.backend = AttentionBackendName.torch

    assert base_config.block.feed_forward_moe is not None, "Model is not MoE"
    base_num_experts = base_config.block.feed_forward_moe.num_experts
    for i, mc in enumerate(model_configs[1:], start=1):
        assert mc.block.feed_forward_moe is not None, f"Checkpoint {i} is not MoE"
        assert mc.block.feed_forward_moe.num_experts == base_num_experts, (
            f"Checkpoint {i} has {mc.block.feed_forward_moe.num_experts} experts, "
            f"expected {base_num_experts}"
        )

    # Build the output model
    merged_model = base_config.build(init_device="cpu")

    # Load first model and scale its weights
    logger.info(f"Loading checkpoint 0: {checkpoint_paths[0]}")
    model_0 = load_checkpoint(base_config, checkpoint_paths[0])
    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            if name in model_0.state_dict():
                param.data.copy_(model_0.state_dict()[name] * weights[0])
    del model_0

    # Accumulate remaining models
    for i in range(1, n):
        logger.info(f"Loading checkpoint {i}: {checkpoint_paths[i]}")
        cfg_i = model_configs[i]
        cfg_i.block.attention.backend = AttentionBackendName.torch
        model_i = load_checkpoint(cfg_i, checkpoint_paths[i])
        with torch.no_grad():
            for name, param in merged_model.named_parameters():
                if name in model_i.state_dict():
                    param.data.add_(model_i.state_dict()[name] * weights[i])
        del model_i
        logger.info(f"Accumulated checkpoint {i}")

    logger.info("Weight merging complete")

    if save_path is not None:
        base_config.block.attention.backend = backend
        config_out = configs_raw[0].copy()
        config_out["model"] = base_config.as_config_dict()
        save_checkpoint(config_out, merged_model, save_path)

    return merged_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Weight-merge n MoE model checkpoints (model soup / weighted average)"
    )
    parser.add_argument(
        "-c",
        "--checkpoint_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to MoE checkpoints to merge (must share the same architecture)",
    )
    parser.add_argument(
        "-o",
        "--save_path",
        type=str,
        required=True,
        help="Path to save the merged checkpoint",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Per-checkpoint weights (must match number of checkpoints). "
        "Defaults to uniform averaging. Normalized to sum to 1.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    logger.info(f"Args: {args}")

    merged = weight_merge(
        checkpoint_paths=args.checkpoint_paths,
        weights=args.weights,
        save_path=args.save_path,
    )
