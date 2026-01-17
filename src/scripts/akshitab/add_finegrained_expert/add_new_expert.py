"""
Add new expert(s) to an existing Mixture of Experts (MoE) model checkpoint.

Examples:

Initialize new expert with randomly selected expert:

```bash
python src/scripts/akshitab/add_finegrained_expert/add_new_expert.py \
	-c ${BASE_MODEL_PATH}\
	-o ${NEW_BASE_MODEL_PATH} \
	--num_new_experts 1 \
	--init_method random_expert
```

Initialize new expert with average of top 2 most similar experts by domain

```bash
python src/scripts/akshitab/add_finegrained_expert/add_new_expert.py \
	-c ${BASE_MODEL_PATH}\
	-o ${NEW_BASE_MODEL_PATH} \
	--num_new_experts 1 \
	--init_method similar \
        --activation_file ${ACTIVATION_FILE} \
	--top_k 2
```

Initialize 8 new experts with average of previous experts:
```bash
python src/scripts/akshitab/add_finegrained_expert/add_new_expert.py \
	-c ${BASE_MODEL_PATH}\
	-o ${NEW_BASE_MODEL_PATH} \
	--num_new_experts 8 \
	--init_method average
```

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


class AddExpertInitMethod:
    RANDOM = "random"
    """
    Initialize new expert with random weights.
    """

    RANDOM_EXPERT = "random_expert"
    """
    Initialize new expert with weights of randomly selected existing experts.
    """

    AVERAGE = "average"
    """
    Initialize new expert with average of existing experts.
    """

    ZERO = "zero"
    """
    Initialize new expert with zeros.
    """

    SIMILAR = "similar"
    """
    Initialize new expert with weights from similar existing experts. Takes the average of top_k experts if top_k > 1.
    """

    SIMILAR_NO_AVERAGE = "similar_no_average"
    """
    Initialize new expert with weights from similar existing experts without averaging. When adding multiple experts, uses top_k experts in order.
    """


def get_similar_experts(activation_file: str, top_k: int = 1):
    with smart_open.open(activation_file, "r") as f:
        line = f.readline()
        activations = json.loads(line)["avg_router_probabilities"]  # (layers, num_experts)

    all_activations = torch.tensor(activations)

    # Expert with highest activation across all layers
    expert_sums = all_activations.sum(dim=0)
    top_k_similar_experts = torch.topk(expert_sums, top_k).indices.tolist()
    return top_k_similar_experts


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


def add_experts(
    checkpoint_path: str,
    save_path: Optional[str] = None,
    init_method: Optional[str] = None,
    num_new_experts: int = 1,
    top_k_expert_indices: Optional[list[int]] = None,
    noise_scale: float = 0.0,
    noise_std_fraction: Optional[float] = None,
):
    # Load model config
    old_config_path = os.path.join(checkpoint_path, "config.json")
    logger.info(f"Loading model config from {old_config_path}")
    with open(old_config_path, "r") as f:
        config = json.load(f)

    old_model_config = TransformerConfig.from_dict(config["model"])
    logger.info(f"Model config {old_model_config}")

    # Load model weights
    logger.info(f"Loading model weights from {checkpoint_path}")
    old_model = load_checkpoint(model_config=old_model_config, checkpoint_path=checkpoint_path)
    logger.info("Model loaded successfully")

    # Update config
    logger.info("Adding new expert to the model")
    new_config = old_model_config.copy()
    assert new_config.block.feed_forward_moe is not None, "Model is not MoE"
    new_config.block.feed_forward_moe.num_experts += num_new_experts
    new_model = new_config.build(init_device="cpu")

    new_model.init_weights()  # Initialized with random init

    assert old_model_config.block.feed_forward_moe is not None
    num_experts = old_model_config.block.feed_forward_moe.num_experts

    if top_k_expert_indices is not None:
        assert (
            len(top_k_expert_indices) <= num_experts
        ), "top_k_expert_indices cannot be more than existing experts"
        for idx in top_k_expert_indices:
            assert 0 <= idx < num_experts, f"Expert index {idx} out of range"

    init_method = init_method or AddExpertInitMethod.RANDOM

    if init_method == AddExpertInitMethod.RANDOM_EXPERT:
        random_expert_ids = torch.randint(0, num_experts, (num_new_experts,)).tolist()

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

                target_param = new_param.view(num_experts + num_new_experts, source_columns).clone()

                if init_method == AddExpertInitMethod.ZERO:
                    logger.info(f"Initializing new expert weights to zero for {name}")
                    target_param.fill_(0)
                elif init_method == AddExpertInitMethod.RANDOM:
                    # Do nothing, the new_model was initialized randomly already
                    logger.info(f"Keeping random initialization for new expert weights for {name}")
                    pass
                elif init_method == AddExpertInitMethod.RANDOM_EXPERT:
                    logger.info(
                        f"Initializing new expert weights from random existing experts {random_expert_ids} for {name}"
                    )
                    with torch.no_grad():
                        target_param[-num_new_experts:, :].copy_(source_param[random_expert_ids, :])
                elif init_method == AddExpertInitMethod.AVERAGE:
                    logger.info(
                        f"Initializing new expert weights with average of all experts for {name}"
                    )
                    # Compute average of existing experts
                    avg_expert = source_param.data.mean(dim=0)
                    # Compute noise scale
                    actual_noise_scale = (
                        noise_std_fraction * avg_expert.std()
                        if noise_std_fraction is not None
                        else noise_scale
                    )
                    # Copy average to new expert position with noise for each expert
                    with torch.no_grad():
                        for i in range(num_new_experts):
                            noise = torch.randn_like(avg_expert) * actual_noise_scale
                            target_param[num_experts + i, :].copy_(avg_expert + noise)
                elif init_method == AddExpertInitMethod.SIMILAR:
                    logger.info(
                        f"Initializing new expert weights with {top_k_expert_indices} experts for {name}"
                    )
                    assert (
                        top_k_expert_indices is not None
                    ), "top_k_expert_indices must be provided for SIMILAR initialization"
                    avg_expert = source_param[top_k_expert_indices, :].data.mean(dim=0)
                    # Compute noise scale
                    actual_noise_scale = (
                        noise_std_fraction * avg_expert.std()
                        if noise_std_fraction is not None
                        else noise_scale
                    )
                    with torch.no_grad():
                        for i in range(num_new_experts):
                            noise = torch.randn_like(avg_expert) * actual_noise_scale
                            target_param[num_experts + i, :].copy_(avg_expert + noise)
                elif init_method == AddExpertInitMethod.SIMILAR_NO_AVERAGE:
                    logger.info(
                        f"Initializing new expert weights with {top_k_expert_indices} experts (no averaging) for {name}"
                    )
                    assert (
                        top_k_expert_indices is not None
                    ), "top_k_expert_indices must be provided for SIMILAR_NO_AVERAGE initialization"
                    assert (
                        len(top_k_expert_indices) >= num_new_experts
                    ), f"Need at least {num_new_experts} expert indices for SIMILAR_NO_AVERAGE, got {len(top_k_expert_indices)}"
                    with torch.no_grad():
                        for i in range(num_new_experts):
                            target_param[num_experts + i, :].copy_(
                                source_param[top_k_expert_indices[i], :]
                            )

                target_param[:num_experts, :] = source_param
                with torch.no_grad():
                    new_param.data.copy_(target_param.view(-1))

            elif "experts.mlp" in name:
                source_param = old_param.clone()
                source_rows, source_columns = source_param.shape

                target_param = new_param.view(
                    num_experts + num_new_experts, source_rows // num_experts, source_columns
                ).clone()

                if init_method == AddExpertInitMethod.ZERO:
                    logger.info(f"Initializing new expert weights to zero for {name}")
                    target_param.fill_(0)
                elif init_method == AddExpertInitMethod.RANDOM:
                    # Do nothing, the new_model was initialized randomly already
                    logger.info(f"Keeping random initialization for new expert weights for {name}")
                    pass
                elif init_method == AddExpertInitMethod.RANDOM_EXPERT:
                    logger.info(
                        f"Initializing new expert weights from random existing experts {random_expert_ids} for {name}"
                    )
                    source_param = source_param.view(
                        num_experts, source_rows // num_experts, source_columns
                    )
                    with torch.no_grad():
                        target_param[-num_new_experts:, :, :].copy_(
                            source_param[random_expert_ids, :, :]
                        )
                elif init_method == AddExpertInitMethod.AVERAGE:
                    logger.info(
                        f"Initializing new expert weights with average for all experts for {name}"
                    )
                    # Compute average of existing experts
                    source_param = source_param.view(
                        num_experts, source_rows // num_experts, source_columns
                    )
                    avg_expert = source_param.data.mean(dim=0)
                    # Compute noise scale
                    actual_noise_scale = (
                        noise_std_fraction * avg_expert.std()
                        if noise_std_fraction is not None
                        else noise_scale
                    )
                    # Copy average to new expert position with noise for each expert
                    with torch.no_grad():
                        for i in range(num_new_experts):
                            noise = torch.randn_like(avg_expert) * actual_noise_scale
                            target_param[num_experts + i, :, :].copy_(avg_expert + noise)
                elif init_method == AddExpertInitMethod.SIMILAR:
                    logger.info(
                        f"Initializing new expert weights with {top_k_expert_indices} experts for {name}"
                    )
                    assert (
                        top_k_expert_indices is not None
                    ), "top_k_expert_indices must be provided for SIMILAR initialization"
                    source_param = source_param.view(
                        num_experts, source_rows // num_experts, source_columns
                    )
                    avg_expert = source_param[top_k_expert_indices, :, :].data.mean(dim=0)
                    # Compute noise scale
                    actual_noise_scale = (
                        noise_std_fraction * avg_expert.std()
                        if noise_std_fraction is not None
                        else noise_scale
                    )
                    with torch.no_grad():
                        for i in range(num_new_experts):
                            noise = torch.randn_like(avg_expert) * actual_noise_scale
                            target_param[num_experts + i, :, :].copy_(avg_expert + noise)
                elif init_method == AddExpertInitMethod.SIMILAR_NO_AVERAGE:
                    logger.info(
                        f"Initializing new expert weights with {top_k_expert_indices} experts (no averaging) for {name}"
                    )
                    assert (
                        top_k_expert_indices is not None
                    ), "top_k_expert_indices must be provided for SIMILAR_NO_AVERAGE initialization"
                    assert (
                        len(top_k_expert_indices) >= num_new_experts
                    ), f"Need at least {num_new_experts} expert indices for SIMILAR_NO_AVERAGE, got {len(top_k_expert_indices)}"
                    source_param = source_param.view(
                        num_experts, source_rows // num_experts, source_columns
                    )
                    with torch.no_grad():
                        for i in range(num_new_experts):
                            target_param[num_experts + i, :, :].copy_(
                                source_param[top_k_expert_indices[i], :, :]
                            )

                target_param[:num_experts, :, :] = source_param.view(
                    num_experts, source_rows // num_experts, source_columns
                )
                with torch.no_grad():
                    new_param.data.copy_(target_param.view(-1, source_columns))

        else:
            logger.debug(f"Parameter {name} not found in new model, not updating weights")

    logger.info("Weights copied to new model successfully")

    # Save new model checkpoint
    if save_path is not None:
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
        "--init_method",
        type=str,
        default=AddExpertInitMethod.RANDOM_EXPERT,
        help="Initialization method for new expert, options: RANDOM, RANDOM_EXPERT, AVERAGE, ZERO, SIMILAR. SIMILAR requires --activation_file.",
    )
    parser.add_argument(
        "-n",
        "--num_new_experts",
        type=int,
        default=1,
        help="Number of new experts to add",
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=1,
        help="Top k similar experts to use for SIMILAR init method",
    )
    parser.add_argument(
        "--activation_file",
        type=str,
        default=None,
        help="Path to activation file for SIMILAR init method",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.0,
        help="Scale of noise to add to each new expert when averaging (default: 0.0, no noise)",
    )
    parser.add_argument(
        "--noise_std_fraction",
        type=float,
        default=None,
        help="If set, compute noise scale as this fraction of the weight std (e.g., 0.01 for 1%%). Overrides --noise_scale.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    if args.activation_file is not None:
        top_k_expert_indices = get_similar_experts(
            activation_file=args.activation_file, top_k=args.top_k
        )
        print(top_k_expert_indices)
    else:
        top_k_expert_indices = None
    new_model = add_experts(
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        init_method=args.init_method,
        num_new_experts=args.num_new_experts,
        top_k_expert_indices=top_k_expert_indices,
        noise_scale=args.noise_scale,
        noise_std_fraction=args.noise_std_fraction,
    )
