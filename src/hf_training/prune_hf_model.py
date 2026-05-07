"""
Prune HuggingFace MoE model based on router activations.

This script loads an HF MoE model, reads activation probabilities from file,
and creates a pruned model keeping only the top-k experts per layer.

Usage:
    python -m src.hf_training.prune_hf_model \
        --model allenai/OLMoE-1B-7B-0924 \
        --activation-file activations.jsonl \
        --prune-keep-k 4 \
        --save-path ./pruned_model

Supports OLMoE-style architectures where each layer has:
- model.layers[i].mlp.experts (nn.ModuleList of expert MLPs)
- model.layers[i].mlp.gate (router linear layer)
"""

import argparse
import json
import logging
import os
from typing import List

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def get_experts_to_keep(activations: List[List[float]], prune_keep_k: int) -> List[List[int]]:
    """
    Determine which experts to keep for each layer based on activation probabilities.

    Args:
        activations: List of per-layer activation probabilities, shape (num_layers, num_experts)
        prune_keep_k: Number of experts to keep per layer

    Returns:
        List of expert indices to keep for each layer
    """
    experts_to_keep = []
    for layer_idx, layer_activations in enumerate(activations):
        # Get top-k experts by activation probability
        activation_tensor = torch.tensor(layer_activations)
        top_k_indices = torch.topk(
            activation_tensor, min(prune_keep_k, len(layer_activations))
        ).indices.tolist()
        experts_to_keep.append(sorted(top_k_indices))  # Sort for consistency
        logger.debug(f"Layer {layer_idx}: keeping experts {top_k_indices}")
    return experts_to_keep


def prune_olmoe_model(
    model: torch.nn.Module,
    config,
    experts_to_keep: List[List[int]],
    prune_keep_k: int,
) -> torch.nn.Module:
    """
    Prune an OLMoE-style model by removing experts.

    OLMoE structure:
    - model.model.layers[i].mlp.experts (nn.ModuleList)
    - model.model.layers[i].mlp.gate.weight (router, shape: num_experts x hidden_size)

    Args:
        model: The HuggingFace model
        config: Model config
        experts_to_keep: List of expert indices to keep per layer
        prune_keep_k: Number of experts to keep

    Returns:
        Pruned model
    """
    num_layers = config.num_hidden_layers

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]

        # Check if this is an MoE layer
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
            logger.debug(f"Layer {layer_idx} is not an MoE layer, skipping")
            continue

        moe = layer.mlp
        keep_indices = experts_to_keep[layer_idx]

        logger.info(f"Layer {layer_idx}: pruning to keep experts {keep_indices}")

        # Prune experts - create new ModuleList with only kept experts
        old_experts = moe.experts
        new_experts = torch.nn.ModuleList([old_experts[i] for i in keep_indices])
        moe.experts = new_experts

        # Prune router/gate weights
        # Gate weight shape: (num_experts, hidden_size)
        if hasattr(moe, "gate"):
            old_gate_weight = moe.gate.weight.data
            new_gate_weight = old_gate_weight[keep_indices, :]
            moe.gate.weight = torch.nn.Parameter(new_gate_weight)

            # Handle bias if present
            if moe.gate.bias is not None:
                old_gate_bias = moe.gate.bias.data
                new_gate_bias = old_gate_bias[keep_indices]
                moe.gate.bias = torch.nn.Parameter(new_gate_bias)

            # Update gate output features
            moe.gate.out_features = prune_keep_k

    # Update config
    if hasattr(config, "num_local_experts"):
        config.num_local_experts = prune_keep_k
    if hasattr(config, "num_experts"):
        config.num_experts = prune_keep_k

    return model


def prune_mixtral_model(
    model: torch.nn.Module,
    config,
    experts_to_keep: List[List[int]],
    prune_keep_k: int,
) -> torch.nn.Module:
    """
    Prune a Mixtral-style model by removing experts.

    Mixtral structure:
    - model.model.layers[i].block_sparse_moe.experts (nn.ModuleList)
    - model.model.layers[i].block_sparse_moe.gate.weight (router)

    Args:
        model: The HuggingFace model
        config: Model config
        experts_to_keep: List of expert indices to keep per layer
        prune_keep_k: Number of experts to keep

    Returns:
        Pruned model
    """
    num_layers = config.num_hidden_layers

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]

        # Check if this is an MoE layer (Mixtral-style)
        if not hasattr(layer, "block_sparse_moe"):
            logger.debug(f"Layer {layer_idx} is not an MoE layer, skipping")
            continue

        moe = layer.block_sparse_moe
        keep_indices = experts_to_keep[layer_idx]

        logger.info(f"Layer {layer_idx}: pruning to keep experts {keep_indices}")

        # Prune experts
        old_experts = moe.experts
        new_experts = torch.nn.ModuleList([old_experts[i] for i in keep_indices])
        moe.experts = new_experts

        # Prune gate weights
        if hasattr(moe, "gate"):
            old_gate_weight = moe.gate.weight.data
            new_gate_weight = old_gate_weight[keep_indices, :]
            moe.gate.weight = torch.nn.Parameter(new_gate_weight)

            if moe.gate.bias is not None:
                old_gate_bias = moe.gate.bias.data
                new_gate_bias = old_gate_bias[keep_indices]
                moe.gate.bias = torch.nn.Parameter(new_gate_bias)

            moe.gate.out_features = prune_keep_k

    # Update config
    if hasattr(config, "num_local_experts"):
        config.num_local_experts = prune_keep_k
    if hasattr(config, "num_experts"):
        config.num_experts = prune_keep_k

    return model


def detect_model_type(model: torch.nn.Module) -> str:
    """Detect the MoE architecture type."""
    # Check first layer structure
    first_layer = model.model.layers[0]

    if hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "experts"):
        return "olmoe"
    elif hasattr(first_layer, "block_sparse_moe"):
        return "mixtral"
    else:
        raise ValueError("Unknown MoE architecture. Expected OLMoE or Mixtral-style model.")


def prune_hf_model(
    model_name: str,
    activation_file: str,
    prune_keep_k: int,
    num_shared_experts: int,
    save_path: str,
    device: str = "cpu",
) -> None:
    """
    Prune a HuggingFace MoE model based on router activations.

    Args:
        model_name: HuggingFace model name or path
        activation_file: Path to activation file (JSON with avg_router_probabilities)
        prune_keep_k: Number of experts to keep per layer
        num_shared_experts: Number of shared experts to keep (if applicable). NOTE: model.config.num_shared_experts is total shared experts, this is the number of shared experts to keep
        save_path: Path to save the pruned model
    """
    # Load activation file
    logger.info(f"Loading activations from {activation_file}")
    with open(activation_file, "r") as f:
        line = f.readline()
        activations = json.loads(line)["avg_router_probabilities"]

    # Load model
    logger.info(f"Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Verify activation file matches model
    num_layers = config.num_hidden_layers
    assert (
        len(activations) == num_layers
    ), f"Activation file has {len(activations)} layers but model has {num_layers} layers"

    # Determine experts to keep
    activations = torch.tensor(activations)
    # check if we have shared experts
    if model.config.num_shared_experts > 0:
        activations_standard = activations[
            :, : model.config.num_experts - model.config.num_shared_experts
        ]
        activations_shared = activations[
            :, model.config.num_experts - model.config.num_shared_experts :
        ]

        experts_to_keep_standard = get_experts_to_keep(
            activations_standard, prune_keep_k - num_shared_experts
        )
        experts_to_keep_shared = get_experts_to_keep(activations_shared, num_shared_experts)

        experts_to_keep = []
        for layer_idx in range(num_layers):
            layer_experts_to_keep = sorted(experts_to_keep_standard[layer_idx]) + sorted(
                [
                    idx + model.config.num_experts - model.config.num_shared_experts
                    for idx in experts_to_keep_shared[layer_idx]
                ]
            )
            experts_to_keep.append(layer_experts_to_keep)
    else:
        experts_to_keep = get_experts_to_keep(activations, prune_keep_k)

    # Detect model type and prune
    model_type = detect_model_type(model)
    logger.info(f"Detected model type: {model_type}")

    if model_type == "olmoe":
        model = prune_olmoe_model(
            model, config, experts_to_keep, prune_keep_k
        )  # we pass prune_keep_k that includes both shared and non-shared experts
    elif model_type == "mixtral":
        model = prune_mixtral_model(model, config, experts_to_keep, prune_keep_k)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # update the model configs accordingly
    if model.config.num_experts_per_tok > prune_keep_k:
        model.config.num_experts_per_tok = prune_keep_k
    model.config.num_experts = prune_keep_k

    # update the number of shared experts in the config
    model.config.num_shared_experts = num_shared_experts

    # set the "output_router_logits" to "true" to enable load balancing during finetuning
    model.config.output_router_logits = True

    # Save pruned model
    logger.info(f"Saving pruned model to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Save pruning metadata
    metadata = {
        "original_model": model_name,
        "prune_keep_k": prune_keep_k,
        "num_shared_experts": num_shared_experts,
        "activation_file": activation_file,
        "experts_kept_per_layer": experts_to_keep,
    }
    with open(os.path.join(save_path, "pruning_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Pruned model saved to {save_path}")
    logger.info(f"Original experts: {len(activations[0])}, Kept: {prune_keep_k}")


def main():
    parser = argparse.ArgumentParser(description="Prune HuggingFace MoE model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--activation-file",
        type=str,
        required=True,
        help="Path to activation file",
    )
    parser.add_argument(
        "--prune-keep-k",
        type=int,
        required=True,
        help="Number of experts to keep per layer",
    )
    parser.add_argument(
        "--num-shared-experts",
        type=int,
        default=0,
        help="Number of shared experts to keep (default: 0, meaning all experts are standard)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to save pruned model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading model (default: cpu)",
    )

    args = parser.parse_args()

    prune_hf_model(
        model_name=args.model,
        activation_file=args.activation_file,
        prune_keep_k=args.prune_keep_k,
        num_shared_experts=args.num_shared_experts,
        save_path=args.save_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
