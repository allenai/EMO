"""
Add a shared expert to an existing Mixture of Experts (MoE) model checkpoint.

The shared expert processes all tokens (no routing) and its output is combined
with the routed expert output using weighted averaging:
    1/(top_k+1) * shared + top_k/(top_k+1) * routed

The shared expert weights are initialized from a specified base expert.

Example:

```bash
python src/scripts/akshitab/add_finegrained_expert/add_shared_expert.py \
    -c ${BASE_MODEL_PATH} \
    -o ${NEW_MODEL_PATH} \
    --shared-expert-init-idx 0
```
"""

import argparse
import json
import logging
import os


from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import setup_logging

logger = logging.getLogger(__name__)


def add_shared_expert(
    checkpoint_path: str,
    save_path: str,
    shared_expert_init_idx: int,
):
    # Load model config
    old_config_path = os.path.join(checkpoint_path, "config.json")
    logger.info(f"Loading model config from {old_config_path}")
    with open(old_config_path, "r") as f:
        config = json.load(f)

    old_model_config = TransformerConfig.from_dict(config["model"])
    backend = old_model_config.block.attention.backend
    old_model_config.block.attention.backend = AttentionBackendName.torch

    assert old_model_config.block.feed_forward_moe is not None, "Model is not MoE"
    moe_config = old_model_config.block.feed_forward_moe
    assert moe_config.shared_mlp is None, "Model already has a shared_mlp"

    num_experts = moe_config.num_experts
    hidden_size = moe_config.hidden_size
    assert (
        0 <= shared_expert_init_idx < num_experts
    ), f"shared_expert_init_idx {shared_expert_init_idx} out of range [0, {num_experts})"

    logger.info(f"Model config: {old_model_config}")
    logger.info(f"Initializing shared expert from expert {shared_expert_init_idx}")

    # Load old model weights
    logger.info(f"Loading model weights from {checkpoint_path}")
    old_model = old_model_config.build(init_device="cpu")
    model_weights_path = os.path.join(checkpoint_path, "model_and_optim")
    load_model_and_optim_state(dir=model_weights_path, model=old_model, optim=None)
    logger.info("Old model loaded successfully")

    # Build new config with shared_mlp
    new_model_config = old_model_config.copy()
    assert new_model_config.block.feed_forward_moe is not None
    new_model_config.block.feed_forward_moe.shared_mlp = FeedForwardConfig(
        hidden_size=hidden_size, bias=False
    )
    logger.info(f"New model config: {new_model_config}")

    # Build new model
    new_model = new_model_config.build(init_device="cpu")
    new_model.init_weights()

    # Copy all weights from old model to new model
    for name, old_param in old_model.named_parameters():
        if name in new_model.state_dict():
            new_param = new_model.state_dict()[name]
            if old_param.shape == new_param.shape:
                logger.info(f"Copying parameter {name}")
                new_param.data.copy_(old_param.data)
            else:
                logger.warning(
                    f"Shape mismatch for {name}: old {old_param.shape} vs new {new_param.shape}, skipping"
                )
        else:
            logger.warning(f"Parameter {name} not found in new model")

    # Initialize shared_mlp weights from the specified expert
    # DroplessMoEMLP stores weights as (num_experts * hidden_size, d_model)
    # viewed as (num_experts, hidden_size, d_model).
    # FeedForward uses nn.Linear:
    #   w1.weight: (hidden_size, d_model) - same layout as expert, direct copy
    #   w2.weight: (d_model, hidden_size) - transposed vs expert, needs .T
    #   w3.weight: (hidden_size, d_model) - same layout as expert, direct copy
    for name, param in new_model.named_parameters():
        if "shared_mlp" not in name:
            continue

        # Find the corresponding expert parameter
        # e.g. "blocks.0.feed_forward_moe.shared_mlp.w1.weight"
        #   -> "blocks.0.feed_forward_moe.experts.mlp.w1"
        expert_name = name.replace("shared_mlp.", "experts.mlp.").replace(".weight", "")

        if expert_name not in dict(old_model.named_parameters()):
            logger.warning(f"Could not find expert parameter {expert_name} for {name}")
            continue

        expert_param = dict(old_model.named_parameters())[expert_name]

        # Extract the single expert's weights
        # Expert param shape: (num_experts * hidden_size, d_model)
        # -> view as (num_experts, hidden_size, d_model) -> select [init_idx]
        expert_weight = expert_param.view(num_experts, hidden_size, -1)[shared_expert_init_idx]

        # Check if transpose is needed (w2: expert is (hidden_size, d_model), linear expects (d_model, hidden_size))
        if expert_weight.shape == param.shape:
            logger.info(f"Copying {expert_name}[{shared_expert_init_idx}] -> {name} (direct)")
            param.data.copy_(expert_weight)
        elif expert_weight.T.shape == param.shape:
            logger.info(f"Copying {expert_name}[{shared_expert_init_idx}] -> {name} (transposed)")
            param.data.copy_(expert_weight.T)
        else:
            raise ValueError(
                f"Cannot map expert weight {expert_name} shape {expert_weight.shape} "
                f"to shared_mlp param {name} shape {param.shape}"
            )

    logger.info("Shared expert weights initialized successfully")

    # Save new checkpoint
    if os.path.exists(save_path):
        logger.warning(f"Save path {save_path} already exists. Not overwriting.")
    else:
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving new model checkpoint to {save_path}")

        new_model_config.block.attention.backend = backend
        config["model"] = new_model_config.as_config_dict()

        new_config_path = os.path.join(save_path, "config.json")
        with open(new_config_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved new model config to {new_config_path}")

        model_weights_path = os.path.join(save_path, "model_and_optim")
        save_model_and_optim_state(dir=model_weights_path, model=new_model, optim=None)
        logger.info(f"Saved new model weights to {model_weights_path}")

    return new_model


def parse_args():
    parser = argparse.ArgumentParser(description="Add shared expert to MoE model")
    parser.add_argument(
        "-c", "--checkpoint_path", type=str, required=True, help="Path to existing MoE checkpoint"
    )
    parser.add_argument(
        "-o", "--save_path", type=str, required=True, help="Path to save new MoE checkpoint"
    )
    parser.add_argument(
        "--shared-expert-init-idx",
        type=int,
        required=True,
        help="Index of the base expert to copy weights from for the shared expert",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    add_shared_expert(
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        shared_expert_init_idx=args.shared_expert_init_idx,
    )
