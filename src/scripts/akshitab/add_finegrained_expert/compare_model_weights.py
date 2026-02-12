"""
Check if two existing model checkpoints have the same weights.
"""
import argparse
import json
import logging
import os
from typing import Optional

import torch

from olmo_core.distributed.checkpoint import load_model_and_optim_state
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


def load_model(checkpoint_path):
    # Load model config
    config_path = os.path.join(checkpoint_path, "config.json")
    logger.info(f"Loading model config from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    config = TransformerConfig.from_dict(config["model"])
    config.block.attention.backend = "torch"
    logger.info(f"Model config {config}")

    # Load model weights
    logger.info(f"Loading model weights from {checkpoint_path}")
    old_model = load_checkpoint(model_config=config, checkpoint_path=checkpoint_path)
    logger.info("Model loaded successfully")
    return old_model


def check_model_weights(
    checkpoint_path1: str,
    checkpoint_path2: Optional[str] = None,
):
    model1 = load_model(checkpoint_path1)
    model2 = load_model(checkpoint_path2)

    # Compare weights
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2:
            logger.error(f"Parameter names do not match: {name1} vs {name2}")
            return False
        if not torch.equal(param1.data, param2.data):
            logger.error(f"Weights do not match for parameter: {name1}")
            return False

    logger.info("All model weights match successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare model weights between two checkpoints.")
    parser.add_argument(
        "-c1", "--checkpoint_path1", type=str, required=True, help="Path to existing checkpoint 1"
    )
    parser.add_argument(
        "-c2", "--checkpoint_path2", type=str, required=True, help="Path to existing checkpoint 2"
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()

    new_model = check_model_weights(
        checkpoint_path1=args.checkpoint_path1,
        checkpoint_path2=args.checkpoint_path2,
    )
