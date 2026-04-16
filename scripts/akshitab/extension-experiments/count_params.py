"""
Count trainable parameters for different freeze configurations.

Usage:
    python scripts/akshitab/extension-experiments/count_params.py --config-path <checkpoint_dir>
"""

import argparse
import json
import os
import re

from olmo_core.nn.transformer import TransformerConfig


def count_params_by_pattern(model_config, freeze_patterns, expert_mask_indices=None):
    """Count params that would be trained given freeze_params patterns and optional expert gradient mask."""
    model = model_config.build(init_device="meta")

    total_params = 0
    frozen_params = 0
    trained_params = 0
    masked_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n

        # Check if frozen by freeze_params patterns
        is_frozen = any(
            re.fullmatch(pattern.replace("*", ".*"), name) for pattern in freeze_patterns
        )
        if is_frozen:
            frozen_params += n
            continue

        # Check if gradient-masked (expert/router params with selective training)
        if expert_mask_indices is not None:
            is_expert_or_router = "experts" in name or "router" in name
            if is_expert_or_router:
                trained_params += n  # optimizer still tracks these
                # But only some get actual gradients
                num_experts = model_config.block.feed_forward_moe.num_experts
                expert_size = n // num_experts
                active = len(expert_mask_indices) * expert_size
                masked_params += n - active
                continue

        trained_params += n

    return {
        "total": total_params,
        "frozen_by_freeze_params": frozen_params,
        "tracked_by_optimizer": trained_params,
        "gradient_masked": masked_params,
        "actually_updated": trained_params - masked_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to checkpoint dir with config.json"
    )
    args = parser.parse_args()

    config_path = os.path.join(args.config_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    model_config = TransformerConfig.from_dict(config["model"])

    num_experts = model_config.block.feed_forward_moe.num_experts
    print(
        f"Model: {num_experts} experts, num_params={model_config.num_params:,}, "
        f"num_non_embedding_params={model_config.num_non_embedding_params:,}"
    )
    print()

    configs = {
        "selective 4 experts (freeze attn+emb+norms+lmhead)": {
            "freeze_patterns": [
                "embeddings.*",
                "blocks.*.attention.*",
                "blocks.*.feed_forward_norm.*",
                "lm_head.*",
            ],
            "expert_mask_indices": [69, 30, 3, 6],  # example 4 experts
        },
        "all experts (freeze attn+emb+norms+lmhead)": {
            "freeze_patterns": [
                "embeddings.*",
                "blocks.*.attention.*",
                "blocks.*.feed_forward_norm.*",
                "lm_head.*",
            ],
            "expert_mask_indices": None,
        },
        "freeze embeddings+lmhead only": {
            "freeze_patterns": [
                "embeddings.*",
                "lm_head.*",
            ],
            "expert_mask_indices": None,
        },
        "full finetune": {
            "freeze_patterns": [],
            "expert_mask_indices": None,
        },
    }

    for name, cfg in configs.items():
        counts = count_params_by_pattern(
            model_config,
            cfg["freeze_patterns"],
            cfg.get("expert_mask_indices"),
        )
        print(f"=== {name} ===")
        print(f"  Total params:           {counts['total']:>14,}")
        print(f"  Frozen (freeze_params): {counts['frozen_by_freeze_params']:>14,}")
        print(f"  Optimizer tracks:       {counts['tracked_by_optimizer']:>14,}")
        if counts["gradient_masked"] > 0:
            print(f"  Gradient-masked:        {counts['gradient_masked']:>14,}")
        print(f"  Actually updated:       {counts['actually_updated']:>14,}")
        print()


if __name__ == "__main__":
    main()
