"""
Script to merge various MoE checkpoints by selecting and combining experts from them,
with router row norm matching to prevent new experts from acting as probability sinks.

After merging, each new expert's router row is rescaled so its L2 norm matches the
average L2 norm of the base experts' router rows (computed per layer). This prevents
new experts from dominating the softmax on out-of-domain inputs.
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


def normalize_router_rows(
    router_param: torch.Tensor,
    base_num_experts: int,
    total_num_experts: int,
):
    """
    Rescale each new expert's router row so its L2 norm matches the average
    L2 norm of the base experts' router rows.

    Args:
        router_param: Flat router weight tensor for one layer.
        base_num_experts: Number of original base experts.
        total_num_experts: Total number of experts after merging.

    Returns:
        The modified router parameter (in-place).
    """
    rows = router_param.view(total_num_experts, -1)

    base_rows = rows[:base_num_experts]
    new_rows = rows[base_num_experts:]

    target_norm = base_rows.norm(dim=1).mean()

    for i in range(new_rows.shape[0]):
        current_norm = new_rows[i].norm()
        if current_norm > 0:
            scale = target_norm / current_norm
            new_rows[i].mul_(scale)
            logger.info(
                f"  New expert {base_num_experts + i}: norm {current_norm:.4f} -> {target_norm:.4f} (scale={scale:.4f})"
            )
        else:
            logger.warning(f"  New expert {base_num_experts + i}: zero norm, skipping")

    return router_param


def merge_experts(
    base_checkpoint_path: str,
    merge_checkpoint_paths: List[str],
    expert_indices: List[List[int]],
    save_path: Optional[str] = None,
):
    assert len(merge_checkpoint_paths) == len(
        expert_indices
    ), "Mismatch in checkpoint paths and expert indices"

    # Load model config
    old_config_path = os.path.join(base_checkpoint_path, "config.json")
    logger.info(f"Loading model config from {old_config_path}")
    with open(old_config_path, "r") as f:
        config = json.load(f)

    old_model_config = TransformerConfig.from_dict(config["model"])
    backend = old_model_config.block.attention.backend
    old_model_config.block.attention.backend = AttentionBackendName.torch
    logger.info(f"Model config {old_model_config}")

    merge_model_config = old_model_config.copy()
    merge_models = []
    assert old_model_config.block.feed_forward_moe is not None
    base_num_experts = old_model_config.block.feed_forward_moe.num_experts
    for idx, ckpt_path in enumerate(merge_checkpoint_paths):
        assert merge_model_config.block.feed_forward_moe is not None
        merge_model_config.block.feed_forward_moe.num_experts = base_num_experts + len(
            expert_indices[idx]
        )
        merge_model = load_checkpoint(model_config=merge_model_config, checkpoint_path=ckpt_path)
        merge_models.append(merge_model)
        logger.info(f"Merge model loaded successfully from {ckpt_path}")

    new_num_experts = base_num_experts + sum(len(indices) for indices in expert_indices)

    new_config = old_model_config.copy()
    assert new_config.block.feed_forward_moe is not None, "Model is not MoE"
    new_config.block.feed_forward_moe.num_experts = new_num_experts

    new_model = new_config.build(init_device="cpu")
    new_model.init_weights()  # Initialized with random init

    old_model = load_checkpoint(model_config=old_model_config, checkpoint_path=base_checkpoint_path)
    logger.info("Base model loaded successfully")

    # Copy weights from old model to new model
    for name, old_param in old_model.named_parameters():
        if name in new_model.state_dict():
            new_param = new_model.state_dict()[name]
            if old_param.shape == new_param.shape:
                logger.info(f"Copying parameter {name} without changes")
                new_param.data.copy_(old_param.data)

            elif "router.weight" in name:
                logger.info(f"Copying parameter {name} with expert merging")
                source_param = old_param.view(base_num_experts, -1)
                _, source_columns = source_param.shape

                target_param = new_param.view(new_num_experts, source_columns).clone()

                # Copy base model's router rows first
                target_param[:base_num_experts, :].copy_(source_param)
                logger.info(f"Copied {base_num_experts} base router rows for {name}")

                current_expert = base_num_experts
                for ckpt_path, indices in zip(merge_checkpoint_paths, expert_indices):
                    merge_model = merge_models[merge_checkpoint_paths.index(ckpt_path)]
                    merge_num_experts = base_num_experts + len(indices)
                    merge_param = merge_model.state_dict()[name].view(merge_num_experts, -1)
                    for idx in indices:
                        logger.info(
                            f"Copying expert {idx} from {ckpt_path} to new model at position {current_expert}"
                        )
                        target_param[current_expert, :].copy_(merge_param[idx, :])
                        current_expert += 1

                with torch.no_grad():
                    new_param.data.copy_(target_param.view(-1))

                # Apply router row norm matching for this layer
                logger.info(f"Applying router row norm matching for {name}")
                with torch.no_grad():
                    normalize_router_rows(new_param.data, base_num_experts, new_num_experts)

            elif "experts.mlp" in name:
                source_param = old_param.clone()
                source_rows, source_columns = source_param.shape

                expert_dim = source_rows // base_num_experts

                target_param = new_param.view(
                    new_num_experts,
                    expert_dim,
                    source_columns,
                ).clone()

                # Copy base model's expert weights first
                target_param[:base_num_experts, :, :].copy_(
                    source_param.view(base_num_experts, expert_dim, source_columns)
                )
                logger.info(f"Copied {base_num_experts} base expert weights for {name}")

                current_expert = base_num_experts
                for ckpt_path, indices in zip(merge_checkpoint_paths, expert_indices):
                    merge_model = merge_models[merge_checkpoint_paths.index(ckpt_path)]
                    merge_num_experts = base_num_experts + len(indices)
                    merge_param = merge_model.state_dict()[name].view(
                        merge_num_experts, expert_dim, source_columns
                    )
                    for idx in indices:
                        logger.info(
                            f"Copying expert {idx} from {ckpt_path} to new model at position {current_expert}"
                        )
                        target_param[current_expert, :, :].copy_(merge_param[idx, :, :])
                        current_expert += 1

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
    parser = argparse.ArgumentParser(
        description="Merge experts from MoE models with router row norm matching"
    )
    parser.add_argument(
        "-b", "--base_checkpoint_path", type=str, required=True, help="Path to base MoE checkpoint"
    )
    parser.add_argument(
        "-m",
        "--merge_checkpoint_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to MoE checkpoints to merge experts from",
    )
    parser.add_argument(
        "-e",
        "--expert_indices",
        type=int,
        nargs="+",
        action="append",
        required=True,
        help="List of expert indices to merge from each checkpoint, provide multiple times for multiple checkpoints",
    )
    parser.add_argument(
        "-o", "--save_path", type=str, required=True, help="Path to save new MoE checkpoint"
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    print(args)

    new_model = merge_experts(
        base_checkpoint_path=args.base_checkpoint_path,
        merge_checkpoint_paths=args.merge_checkpoint_paths,
        expert_indices=args.expert_indices,
        save_path=args.save_path,
    )
