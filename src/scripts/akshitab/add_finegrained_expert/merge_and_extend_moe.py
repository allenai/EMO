"""
Merge MoE checkpoints with expert duplication (extension).

Like weight_merge_moe_models.py, this averages parameters across checkpoints.
But for specified experts, instead of averaging, it keeps multiple versions from
different checkpoints, increasing the total expert count.

Example: Two 128-expert checkpoints both trained expert 76 (one for math, one
for code). We want to average everything, but keep both versions of expert 76.
The result is a 129-expert model.

Usage:
    python merge_and_extend_moe.py \
        -c /path/to/math_ckpt /path/to/code_ckpt \
        -o /path/to/merged \
        --weights 0.5 0.5 \
        --duplicate-experts 76:0 76:1

    This means: for expert 76, keep the version from checkpoint 0 AND checkpoint 1
    as separate experts. All other experts are weight-averaged as usual.

    The final model has 129 experts: 0-127 are averaged (expert 76 averaged is
    replaced by the copy from ckpt 0), and expert 128 is expert 76 from ckpt 1.

    More generally, for each unique expert ID in --duplicate-experts, the first
    listed checkpoint's version replaces the averaged slot, and subsequent
    checkpoint versions are appended as new experts.
"""
import argparse
import copy
import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import setup_logging

logger = logging.getLogger(__name__)


def get_model_config(checkpoint_path: str) -> TransformerConfig:
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return TransformerConfig.from_dict(config["model"])


def load_checkpoint(model_config: TransformerConfig, checkpoint_path: str):
    model = model_config.build(init_device="cpu")
    model_weights_path = os.path.join(checkpoint_path, "model_and_optim")
    load_model_and_optim_state(dir=model_weights_path, model=model, optim=None)
    return model


def save_checkpoint(config: dict, model: torch.nn.Module, save_path: str):
    if os.path.exists(save_path):
        logger.warning(f"Save path {save_path} already exists. Not overwriting.")
        return
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving merged model checkpoint to {save_path}")

    new_config_path = os.path.join(save_path, "config.json")
    with open(new_config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved config to {new_config_path}")

    model_weights_path = os.path.join(save_path, "model_and_optim")
    save_model_and_optim_state(dir=model_weights_path, model=model, optim=None)
    logger.info(f"Saved model weights to {model_weights_path}")


def parse_duplicate_experts(specs: List[str]) -> Dict[int, List[int]]:
    """
    Parse --duplicate-experts specs like ["76:0", "76:1", "3:1"].

    Returns: {expert_id: [ckpt_idx, ckpt_idx, ...]}
        e.g. {76: [0, 1], 3: [1]}
    """
    result: Dict[int, List[int]] = defaultdict(list)
    for spec in specs:
        parts = spec.split(":")
        assert len(parts) == 2, f"Expected 'expert_id:ckpt_idx', got '{spec}'"
        expert_id, ckpt_idx = int(parts[0]), int(parts[1])
        result[expert_id].append(ckpt_idx)
    # Validate: each expert must have at least 2 entries to be duplicated
    for eid, ckpt_indices in result.items():
        assert len(ckpt_indices) >= 2, (
            f"Expert {eid} only listed once (ckpt {ckpt_indices[0]}). "
            f"Need at least 2 entries to duplicate (e.g., {eid}:0 {eid}:1)."
        )
        assert len(set(ckpt_indices)) == len(
            ckpt_indices
        ), f"Expert {eid} has duplicate checkpoint indices: {ckpt_indices}"
    return dict(result)


def is_moe_expert_param(name: str) -> bool:
    """Check if a parameter belongs to MoE expert MLPs (w1, w2, w3)."""
    return bool(re.search(r"\.feed_forward_moe\.experts\.mlp\.w[123]$", name))


def is_moe_router_param(name: str) -> bool:
    """Check if a parameter is a MoE router weight."""
    return bool(re.search(r"\.feed_forward_moe\.router\.weight$", name))


def is_moe_param(name: str) -> bool:
    return is_moe_expert_param(name) or is_moe_router_param(name)


def get_expert_slice(param: torch.Tensor, expert_idx: int, num_experts: int) -> torch.Tensor:
    """
    Extract a single expert's slice from a flattened parameter.

    Expert params are stored as (num_experts * dim0, dim1), where dim0 is
    d_model for w1/w3 and hidden_size for w2.
    """
    rows_per_expert = param.shape[0] // num_experts
    start = expert_idx * rows_per_expert
    end = start + rows_per_expert
    return param[start:end].clone()


def set_expert_slice(param: torch.Tensor, expert_idx: int, num_experts: int, value: torch.Tensor):
    """Write a single expert's slice into a flattened parameter."""
    rows_per_expert = param.shape[0] // num_experts
    start = expert_idx * rows_per_expert
    end = start + rows_per_expert
    param[start:end].copy_(value)


def merge_and_extend(
    checkpoint_paths: List[str],
    weights: Optional[List[float]] = None,
    duplicate_experts: Optional[Dict[int, List[int]]] = None,
    save_path: Optional[str] = None,
):
    n = len(checkpoint_paths)
    assert n >= 2, "Need at least 2 checkpoints to merge"

    if weights is None:
        weights = [1.0 / n] * n
    else:
        assert len(weights) == n, f"Got {len(weights)} weights for {n} checkpoints"
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            weights = [w / weight_sum for w in weights]

    if duplicate_experts is None:
        duplicate_experts = {}

    # Validate checkpoint indices in duplicate_experts
    for eid, ckpt_indices in duplicate_experts.items():
        for ci in ckpt_indices:
            assert 0 <= ci < n, f"Checkpoint index {ci} out of range for expert {eid}"

    # Count how many new experts we're adding.
    # For each duplicated expert: first entry replaces the averaged slot,
    # remaining entries become new experts appended at the end.
    num_new_experts = sum(len(cis) - 1 for cis in duplicate_experts.values())

    logger.info(f"Merging {n} checkpoints with weights: {weights}")
    logger.info(
        f"Duplicating {len(duplicate_experts)} experts, adding {num_new_experts} new experts"
    )

    # Load and validate configs
    configs_raw = []
    model_configs = []
    for path in checkpoint_paths:
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            configs_raw.append(json.load(f))
        model_configs.append(get_model_config(path))

    base_config = model_configs[0]
    backend = base_config.block.attention.backend
    base_config.block.attention.backend = AttentionBackendName.torch

    assert base_config.block.feed_forward_moe is not None, "Model is not MoE"
    base_num_experts = base_config.block.feed_forward_moe.num_experts
    for i, mc in enumerate(model_configs[1:], start=1):
        assert mc.block.feed_forward_moe is not None, f"Checkpoint {i} is not MoE"
        assert mc.block.feed_forward_moe.num_experts == base_num_experts, (
            f"Checkpoint {i} has {mc.block.feed_forward_moe.num_experts} experts, "
            f"expected {base_num_experts}"
        )

    for eid in duplicate_experts:
        assert 0 <= eid < base_num_experts, f"Expert {eid} out of range [0, {base_num_experts})"

    new_num_experts = base_num_experts + num_new_experts
    logger.info(f"Original experts: {base_num_experts}, new total: {new_num_experts}")

    # Build the output model with the extended expert count
    extended_config = copy.deepcopy(base_config)
    assert extended_config.block.feed_forward_moe is not None
    extended_config.block.feed_forward_moe.num_experts = new_num_experts
    merged_model = extended_config.build(init_device="cpu")

    # ---- Step 1: Weight-average all checkpoints into the base expert slots ----
    # We first average into the first base_num_experts slots, treating the
    # output model's expert params as if they only have base_num_experts.

    logger.info(f"Loading checkpoint 0: {checkpoint_paths[0]}")
    cfg_0 = model_configs[0]
    cfg_0.block.attention.backend = AttentionBackendName.torch
    model_0 = load_checkpoint(cfg_0, checkpoint_paths[0])
    sd_0 = model_0.state_dict()

    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            if is_moe_expert_param(name):
                # Copy expert-by-expert into the first base_num_experts slots
                src = sd_0[name]
                for e in range(base_num_experts):
                    expert_data = get_expert_slice(src, e, base_num_experts)
                    set_expert_slice(param.data, e, new_num_experts, expert_data * weights[0])
            elif is_moe_router_param(name):
                # Copy router weights for the first base_num_experts slots
                src = sd_0[name]
                # Router: flattened (num_experts * d_model,)
                d_model = src.shape[0] // base_num_experts
                for e in range(base_num_experts):
                    s = e * d_model
                    new_s = e * d_model  # same position in new (larger) param
                    # But new param has new_num_experts * d_model rows
                    new_s_actual = e * d_model
                    param.data[new_s_actual : new_s_actual + d_model].copy_(
                        src[s : s + d_model] * weights[0]
                    )
            else:
                # Non-MoE param: straightforward copy
                if name in sd_0:
                    param.data.copy_(sd_0[name] * weights[0])
    del model_0, sd_0

    for i in range(1, n):
        logger.info(f"Loading checkpoint {i}: {checkpoint_paths[i]}")
        cfg_i = model_configs[i]
        cfg_i.block.attention.backend = AttentionBackendName.torch
        model_i = load_checkpoint(cfg_i, checkpoint_paths[i])
        sd_i = model_i.state_dict()

        with torch.no_grad():
            for name, param in merged_model.named_parameters():
                if is_moe_expert_param(name):
                    src = sd_i[name]
                    for e in range(base_num_experts):
                        expert_data = get_expert_slice(src, e, base_num_experts)
                        # Accumulate weighted average into the base slots
                        rows_per_expert = param.data.shape[0] // new_num_experts
                        start = e * rows_per_expert
                        end = start + rows_per_expert
                        param.data[start:end].add_(expert_data * weights[i])
                elif is_moe_router_param(name):
                    src = sd_i[name]
                    d_model = src.shape[0] // base_num_experts
                    for e in range(base_num_experts):
                        s = e * d_model
                        new_s = e * d_model
                        param.data[new_s : new_s + d_model].add_(src[s : s + d_model] * weights[i])
                else:
                    if name in sd_i:
                        param.data.add_(sd_i[name] * weights[i])
        del model_i, sd_i
        logger.info(f"Accumulated checkpoint {i}")

    logger.info("Base weight averaging complete")

    # ---- Step 2: Overwrite duplicated expert slots and fill new expert slots ----
    # For each duplicated expert, the first listed checkpoint replaces the
    # averaged slot, and subsequent checkpoints fill new appended slots.

    # Build mapping: new_expert_idx -> (source_expert_id, source_ckpt_idx)
    next_new_idx = base_num_experts
    overwrite_map: List[Tuple[int, int, int]] = []  # (target_idx, src_expert, src_ckpt)
    for expert_id, ckpt_indices in duplicate_experts.items():
        # First entry overwrites the original averaged slot
        overwrite_map.append((expert_id, expert_id, ckpt_indices[0]))
        # Remaining entries get new slots
        for ci in ckpt_indices[1:]:
            overwrite_map.append((next_new_idx, expert_id, ci))
            next_new_idx += 1

    logger.info(f"Expert placement: {overwrite_map}")

    # Group by source checkpoint to minimize loads
    by_ckpt: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for target_idx, src_expert, src_ckpt in overwrite_map:
        by_ckpt[src_ckpt].append((target_idx, src_expert))

    for ckpt_idx, placements in by_ckpt.items():
        logger.info(
            f"Loading checkpoint {ckpt_idx} for expert duplication: "
            f"{[(t, s) for t, s in placements]}"
        )
        cfg = model_configs[ckpt_idx]
        cfg.block.attention.backend = AttentionBackendName.torch
        model = load_checkpoint(cfg, checkpoint_paths[ckpt_idx])
        sd = model.state_dict()

        with torch.no_grad():
            for name, param in merged_model.named_parameters():
                if is_moe_expert_param(name):
                    src = sd[name]
                    for target_idx, src_expert in placements:
                        expert_data = get_expert_slice(src, src_expert, base_num_experts)
                        set_expert_slice(param.data, target_idx, new_num_experts, expert_data)
                elif is_moe_router_param(name):
                    src = sd[name]
                    d_model = src.shape[0] // base_num_experts
                    for target_idx, src_expert in placements:
                        src_s = src_expert * d_model
                        tgt_s = target_idx * d_model
                        param.data[tgt_s : tgt_s + d_model].copy_(src[src_s : src_s + d_model])

        del model, sd

    logger.info(f"Merge and extend complete: {new_num_experts} experts")

    if save_path is not None:
        extended_config.block.attention.backend = backend
        config_out = configs_raw[0].copy()
        config_out["model"] = extended_config.as_config_dict()
        save_checkpoint(config_out, merged_model, save_path)

    return merged_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge MoE checkpoints with expert duplication (extension)"
    )
    parser.add_argument(
        "-c",
        "--checkpoint_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to MoE checkpoints to merge (must share the same architecture)",
    )
    parser.add_argument(
        "-o",
        "--save_path",
        type=str,
        required=True,
        help="Path to save the merged checkpoint",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Per-checkpoint weights (must match number of checkpoints). "
        "Defaults to uniform averaging. Normalized to sum to 1.",
    )
    parser.add_argument(
        "--duplicate-experts",
        type=str,
        nargs="+",
        default=None,
        help="Experts to duplicate instead of averaging. Format: expert_id:ckpt_idx. "
        "E.g., '76:0 76:1' keeps expert 76 from both checkpoints. "
        "The first listed checkpoint's version replaces the averaged slot; "
        "subsequent versions are appended as new experts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    logger.info(f"Args: {args}")

    dup_experts = None
    if args.duplicate_experts:
        dup_experts = parse_duplicate_experts(args.duplicate_experts)
        logger.info(f"Duplicate experts: {dup_experts}")

    merged = merge_and_extend(
        checkpoint_paths=args.checkpoint_paths,
        weights=args.weights,
        duplicate_experts=dup_experts,
        save_path=args.save_path,
    )
