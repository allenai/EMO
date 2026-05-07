"""
Random expert pruning for HuggingFace MoE models.

Baseline for calibration-based pruners: for each MoE layer, randomly select
`prune_keep_k` experts (with a fixed seed for reproducibility) instead of
scoring them. No calibration data is used.

Shared-expert handling mirrors EASY-EP/greedy_layerwise:
    - standard pool:  randomly pick (prune_keep_k - num_shared_experts)
    - shared pool:    randomly pick num_shared_experts

Usage:
    python -m src.hf_training.random_prune \\
        --model /path/to/model \\
        --prune-keep-k 32 --num-shared-experts 1 \\
        --save-path /tmp/pruned
"""

import argparse
import json
import logging
import os
from typing import List, Optional

import torch

from src.hf_training.greedy_prune_layerwise import prune_moe_layer_inplace
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def random_prune(
    model_name: str,
    prune_keep_k: int,
    num_shared_experts: int,
    save_path: str,
    seed: int = 0,
    device: Optional[str] = None,
) -> None:
    logger.info(f"Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device is None else device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = config.num_hidden_layers
    logger.info(f"Model has {num_layers} layers")

    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = False

    g = torch.Generator().manual_seed(seed)

    experts_kept_per_layer: List[Optional[List[int]]] = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        if not (hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")):
            experts_kept_per_layer.append(None)
            continue

        num_experts = layer.mlp.num_experts
        num_shared_current = layer.mlp.num_shared_experts
        num_standard = num_experts - num_shared_current

        if num_shared_current > 0:
            n_keep_standard = min(prune_keep_k - num_shared_experts, num_standard)
            n_keep_shared = min(num_shared_experts, num_shared_current)
            perm_standard = torch.randperm(num_standard, generator=g).tolist()
            perm_shared = torch.randperm(num_shared_current, generator=g).tolist()
            keep_standard = sorted(perm_standard[:n_keep_standard])
            keep_shared = [num_standard + i for i in sorted(perm_shared[:n_keep_shared])]
            experts_to_keep = keep_standard + keep_shared
        else:
            n_keep = min(prune_keep_k, num_experts)
            perm = torch.randperm(num_experts, generator=g).tolist()
            experts_to_keep = sorted(perm[:n_keep])

        logger.info(
            f"Layer {layer_idx}: keeping {len(experts_to_keep)} experts (random): {experts_to_keep}"
        )
        experts_kept_per_layer.append(experts_to_keep)

    for layer_idx in range(num_layers):
        if experts_kept_per_layer[layer_idx] is None:
            continue
        layer = model.model.layers[layer_idx]
        target_shared = num_shared_experts if layer.mlp.num_shared_experts > 0 else 0
        prune_moe_layer_inplace(
            layer,
            experts_kept_per_layer[layer_idx],
            prune_keep_k,
            target_shared,
        )

    if (
        hasattr(model.config, "num_experts_per_tok")
        and model.config.num_experts_per_tok > prune_keep_k
    ):
        model.config.num_experts_per_tok = prune_keep_k
    model.config.num_experts = prune_keep_k
    if hasattr(model.config, "num_local_experts"):
        model.config.num_local_experts = prune_keep_k
    model.config.num_shared_experts = num_shared_experts

    if (
        hasattr(model.config, "num_experts_per_layer")
        and model.config.num_experts_per_layer is not None
    ):
        model.config.num_experts_per_layer = [
            prune_keep_k if n > 0 else 0 for n in model.config.num_experts_per_layer
        ]
    if (
        hasattr(model.config, "num_shared_experts_per_layer")
        and model.config.num_shared_experts_per_layer is not None
    ):
        model.config.num_shared_experts_per_layer = [
            num_shared_experts if n > 0 else 0 for n in model.config.num_shared_experts_per_layer
        ]

    model.config.output_router_logits = True

    logger.info(f"Saving pruned model to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    metadata = {
        "original_model": model_name,
        "prune_keep_k": prune_keep_k,
        "num_shared_experts": num_shared_experts,
        "pruning_method": "random",
        "seed": seed,
        "experts_kept_per_layer": experts_kept_per_layer,
    }
    with open(os.path.join(save_path, "pruning_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done. Pruned model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Random expert pruning for HuggingFace MoE models")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prune-keep-k", type=int, required=True)
    parser.add_argument("--num-shared-experts", type=int, default=0)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    random_prune(
        model_name=args.model,
        prune_keep_k=args.prune_keep_k,
        num_shared_experts=args.num_shared_experts,
        save_path=args.save_path,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
