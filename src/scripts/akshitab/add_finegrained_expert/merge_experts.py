"""
Script to merge various MoE checkpoints by selecting and combining experts from them.
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
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig
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


def merge_experts(
    base_checkpoint_path: str,
    merge_checkpoint_paths: List[str],
    expert_indices: List[List[int]],
    save_path: Optional[str] = None,
    num_shared_experts: int = 0,
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
    assert isinstance(old_model_config.block, TransformerBlockConfig)
    assert isinstance(old_model_config.block.sequence_mixer, AttentionConfig)
    backend = old_model_config.block.sequence_mixer.backend
    old_model_config.block.sequence_mixer.backend = AttentionBackendName.torch
    logger.info(f"Model config {old_model_config}")

    merge_models = []
    assert old_model_config.block.feed_forward_moe is not None
    base_num_experts = old_model_config.block.feed_forward_moe.num_experts
    # Position where new experts will be inserted (before shared experts)
    insert_pos = base_num_experts - num_shared_experts
    logger.info(
        f"Base model has {base_num_experts} experts ({num_shared_experts} shared). "
        f"New experts will be inserted at position {insert_pos}."
    )
    for idx, ckpt_path in enumerate(merge_checkpoint_paths):
        merge_model_config = get_model_config(ckpt_path)
        assert isinstance(merge_model_config.block, TransformerBlockConfig)
        assert isinstance(merge_model_config.block.sequence_mixer, AttentionConfig)
        merge_model_config.block.sequence_mixer.backend = AttentionBackendName.torch
        merge_model = load_checkpoint(model_config=merge_model_config, checkpoint_path=ckpt_path)
        merge_models.append(merge_model)
        logger.info(f"Merge model loaded successfully from {ckpt_path}")

    total_new_experts = sum(len(indices) for indices in expert_indices)
    new_config = old_model_config.copy()
    assert isinstance(new_config.block, TransformerBlockConfig)
    assert new_config.block.feed_forward_moe is not None, "Model is not MoE"
    new_config.block.feed_forward_moe.num_experts = base_num_experts + total_new_experts
    new_total_experts = new_config.block.feed_forward_moe.num_experts

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

                target_param = new_param.view(new_total_experts, source_columns).clone()

                # Copy non-shared experts from base model (positions 0..insert_pos-1)
                target_param[:insert_pos, :].copy_(source_param[:insert_pos, :])
                logger.info(f"Copied {insert_pos} non-shared base router rows for {name}")

                # Copy merged experts into positions insert_pos..insert_pos+total_new_experts-1
                current_expert = insert_pos
                for ckpt_idx, (ckpt_path, indices) in enumerate(
                    zip(merge_checkpoint_paths, expert_indices)
                ):
                    merge_model = merge_models[ckpt_idx]
                    merge_state = merge_model.state_dict()
                    merge_num_experts = merge_state[name].shape[0] // source_columns
                    merge_param = merge_state[name].view(merge_num_experts, -1)
                    for idx in indices:
                        logger.info(
                            f"Copying expert {idx} from {ckpt_path} to new model at position {current_expert}"
                        )
                        target_param[current_expert, :].copy_(merge_param[idx, :])
                        current_expert += 1

                # Copy shared experts from base model to the end
                if num_shared_experts > 0:
                    target_param[insert_pos + total_new_experts :, :].copy_(
                        source_param[insert_pos:, :]
                    )
                    logger.info(
                        f"Copied {num_shared_experts} shared base router rows to positions "
                        f"{insert_pos + total_new_experts}..{new_total_experts - 1} for {name}"
                    )

                with torch.no_grad():
                    new_param.data.copy_(target_param.view(-1))

            elif "experts.mlp" in name:
                source_param = old_param.clone()
                source_rows, source_columns = source_param.shape
                expert_dim = source_rows // base_num_experts

                source_param_3d = source_param.view(base_num_experts, expert_dim, source_columns)
                target_param = new_param.view(new_total_experts, expert_dim, source_columns).clone()

                # Copy non-shared experts from base model
                target_param[:insert_pos, :, :].copy_(source_param_3d[:insert_pos, :, :])
                logger.info(f"Copied {insert_pos} non-shared base expert weights for {name}")

                # Copy merged experts
                current_expert = insert_pos
                for ckpt_idx, (ckpt_path, indices) in enumerate(
                    zip(merge_checkpoint_paths, expert_indices)
                ):
                    merge_model = merge_models[ckpt_idx]
                    merge_state = merge_model.state_dict()
                    merge_num_experts = merge_state[name].shape[0] // expert_dim
                    merge_param = merge_state[name].view(
                        merge_num_experts, expert_dim, source_columns
                    )
                    for idx in indices:
                        logger.info(
                            f"Copying expert {idx} from {ckpt_path} to new model at position {current_expert}"
                        )
                        target_param[current_expert, :, :].copy_(merge_param[idx, :, :])
                        current_expert += 1

                # Copy shared experts from base model to the end
                if num_shared_experts > 0:
                    target_param[insert_pos + total_new_experts :, :, :].copy_(
                        source_param_3d[insert_pos:, :, :]
                    )
                    logger.info(
                        f"Copied {num_shared_experts} shared base expert weights to positions "
                        f"{insert_pos + total_new_experts}..{new_total_experts - 1} for {name}"
                    )

                with torch.no_grad():
                    new_param.data.copy_(target_param.view(-1, source_columns))

        else:
            logger.debug(f"Parameter {name} not found in new model, not updating weights")

    logger.info("Weights copied to new model successfully")

    # Save new model checkpoint
    if save_path is not None:
        assert isinstance(new_config.block.sequence_mixer, AttentionConfig)
        new_config.block.sequence_mixer.backend = backend
        config["model"] = new_config.as_config_dict()
        save_checkpoint(config, new_model, save_path)

    return new_model


def parse_args():
    parser = argparse.ArgumentParser(description="Merge experts from MoE models")
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
    parser.add_argument(
        "--num_shared_experts",
        type=int,
        default=0,
        help="Number of shared experts (last N) in the base model. New experts will be inserted before them.",
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
        num_shared_experts=args.num_shared_experts,
    )
