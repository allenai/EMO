"""
Verify that a split checkpoint preserves expert weights exactly compared to the
original regular checkpoint.

For each MoE layer, checks that:
  - w1_frozen rows match the corresponding expert rows in the original w1
  - w1_trainable rows match the corresponding expert rows in the original w1
  - Same for w2 and w3

Usage:
    python src/scripts/akshitab/add_finegrained_expert/verify_split_conversion.py \
        --original-path /path/to/regular/checkpoint \
        --split-path /path/to/split/checkpoint \
        --experts-to-train 127,128,129,130
"""

import argparse
import logging
import os
import sys

import torch

from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.moe.mlp import SplitExpertDroplessMoEMLP
from olmo_core.utils import setup_logging

from convert_split_checkpoint import (
    build_and_load_model,
    load_config,
    replace_mlps_with_split_empty,
)

logger = logging.getLogger(__name__)


def verify(original_path: str, split_path: str, experts_to_train: list[int]):
    from olmo_core.nn.transformer import TransformerConfig

    # Load original regular checkpoint
    logger.info(f"Loading original checkpoint from {original_path}...")
    original_config = load_config(original_path)
    original_model = build_and_load_model(
        TransformerConfig.from_dict(original_config["model"]), original_path
    )

    # Load split checkpoint
    logger.info(f"Loading split checkpoint from {split_path}...")
    split_config = load_config(split_path)
    split_model = TransformerConfig.from_dict(split_config["model"]).build(init_device="cpu")
    replace_mlps_with_split_empty(split_model, experts_to_train)
    load_model_and_optim_state(
        dir=os.path.join(split_path, "model_and_optim"), model=split_model, optim=None
    )

    # Check expert weights per layer
    num_checked = 0
    errors = []

    for block_name, block in split_model.blocks.items():
        if block.feed_forward_moe is None:
            continue
        split_mlp = block.feed_forward_moe.experts.mlp
        if not isinstance(split_mlp, SplitExpertDroplessMoEMLP):
            continue

        orig_mlp = original_model.blocks[block_name].feed_forward_moe.experts.mlp
        hidden_size = split_mlp.hidden_size
        layer_prefix = f"blocks.{block_name}.feed_forward_moe.experts.mlp"

        for wname in ("w1", "w2", "w3"):
            original_w = getattr(orig_mlp, wname).data
            frozen_w = getattr(split_mlp, f"{wname}_frozen").data
            trainable_w = getattr(split_mlp, f"{wname}_trainable").data

            # Check frozen experts
            for i, expert_idx in enumerate(split_mlp.experts_frozen):
                orig_rows = original_w[expert_idx * hidden_size : (expert_idx + 1) * hidden_size]
                split_rows = frozen_w[i * hidden_size : (i + 1) * hidden_size]
                if not torch.equal(orig_rows, split_rows):
                    max_diff = (orig_rows.float() - split_rows.float()).abs().max().item()
                    errors.append(
                        f"{layer_prefix}.{wname}_frozen expert {expert_idx}: "
                        f"max abs diff = {max_diff:.6e}"
                    )

            # Check trainable experts
            for i, expert_idx in enumerate(split_mlp.experts_to_train):
                orig_rows = original_w[expert_idx * hidden_size : (expert_idx + 1) * hidden_size]
                split_rows = trainable_w[i * hidden_size : (i + 1) * hidden_size]
                if not torch.equal(orig_rows, split_rows):
                    max_diff = (orig_rows.float() - split_rows.float()).abs().max().item()
                    errors.append(
                        f"{layer_prefix}.{wname}_trainable expert {expert_idx}: "
                        f"max abs diff = {max_diff:.6e}"
                    )

            num_checked += 1

    # Also check all non-MLP params match
    orig_state = original_model.state_dict()
    split_state = split_model.state_dict()
    non_mlp_keys = [k for k in orig_state if k in split_state]
    for key in non_mlp_keys:
        if not torch.equal(orig_state[key], split_state[key]):
            max_diff = (orig_state[key].float() - split_state[key].float()).abs().max().item()
            errors.append(f"{key}: max abs diff = {max_diff:.6e}")

    if errors:
        logger.error(f"{len(errors)} mismatch(es) found:")
        for err in errors:
            logger.error(f"  {err}")
        sys.exit(1)

    logger.info(
        f"Verified {num_checked} weight matrices across all MoE layers. "
        f"All frozen and trainable expert rows match the original exactly."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify split checkpoint conversion preserves expert weights exactly."
    )
    parser.add_argument(
        "--original-path", type=str, required=True,
        help="Path to the original regular checkpoint directory.",
    )
    parser.add_argument(
        "--split-path", type=str, required=True,
        help="Path to the split checkpoint directory.",
    )
    parser.add_argument(
        "--experts-to-train", type=str, required=True,
        help="Comma-separated list of expert indices (e.g. '127,128,129,130').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    experts_to_train = [int(x) for x in args.experts_to_train.split(",")]
    logger.info(f"Experts to train: {experts_to_train}")
    verify(args.original_path, args.split_path, experts_to_train)
