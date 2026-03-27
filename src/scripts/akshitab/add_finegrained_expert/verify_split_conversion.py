"""
Verify that a split checkpoint preserves weights exactly compared to the
original regular checkpoint.

Loads both checkpoints (regular and split), reconstructs full tensors from the
split params, and checks that all weights match.

Usage:
    python src/scripts/akshitab/add_finegrained_expert/verify_split_conversion.py \
        --original-path /path/to/regular/checkpoint \
        --split-path /path/to/split/checkpoint \
        --experts-to-train 127,128,129,130
"""

import argparse
import logging
import sys

import torch

from olmo_core.utils import setup_logging

from convert_split_checkpoint import (
    build_and_load_model,
    load_config,
    replace_mlps_with_regular,
    replace_mlps_with_split_empty,
)

logger = logging.getLogger(__name__)


def verify(original_path: str, split_path: str, experts_to_train: list[int]):
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.transformer import TransformerConfig
    import os

    # Load original regular checkpoint
    logger.info(f"Loading original checkpoint from {original_path}...")
    original_config = load_config(original_path)
    original_model_config = TransformerConfig.from_dict(original_config["model"])
    original = build_and_load_model(original_model_config, original_path)
    original_state = {k: v.clone() for k, v in original.state_dict().items()}

    # Load split checkpoint: build regular model, replace with split MLPs, then load
    logger.info(f"Loading split checkpoint from {split_path}...")
    split_config = load_config(split_path)
    split_model_config = TransformerConfig.from_dict(split_config["model"])
    split_model = split_model_config.build(init_device="cpu")
    replace_mlps_with_split_empty(split_model, experts_to_train)
    load_model_and_optim_state(
        dir=os.path.join(split_path, "model_and_optim"), model=split_model, optim=None
    )

    # Convert split back to regular so state dict keys match
    logger.info("Converting split → regular for comparison...")
    replace_mlps_with_regular(split_model, experts_to_train)
    roundtrip_state = split_model.state_dict()

    # Compare keys
    if set(original_state.keys()) != set(roundtrip_state.keys()):
        only_original = set(original_state.keys()) - set(roundtrip_state.keys())
        only_roundtrip = set(roundtrip_state.keys()) - set(original_state.keys())
        if only_original:
            logger.error(f"Keys only in original: {only_original}")
        if only_roundtrip:
            logger.error(f"Keys only in split: {only_roundtrip}")
        sys.exit(1)

    # Compare values
    mismatches = []
    for key in sorted(original_state.keys()):
        if not torch.equal(original_state[key], roundtrip_state[key]):
            max_diff = (original_state[key].float() - roundtrip_state[key].float()).abs().max().item()
            mismatches.append((key, max_diff))

    if mismatches:
        logger.error(f"{len(mismatches)} parameter(s) differ:")
        for key, max_diff in mismatches:
            logger.error(f"  {key}: max abs diff = {max_diff:.6e}")
        sys.exit(1)

    logger.info(
        f"All {len(original_state)} parameters match exactly between "
        f"original and split checkpoints."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify split checkpoint conversion preserves weights exactly."
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
