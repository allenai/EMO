"""
Merge a finetuned pruned MoE model's experts back into the original full model.

Pipeline context: the pruning_hf flow takes a full model, prunes it (greedy
layerwise) to ``prune_keep_k`` experts per layer, finetunes the small model on
some task, and evals it. Extensions adds a final step: copy the trained expert
MLPs from the small model back into the original full model at their original
indices, then eval the full model.

Defaults (v0): only the routable expert MLPs are copied. The router rows, the
shared-expert MLP, and any non-MoE params (attention, norms, embeddings, lm_head)
in the parent are left untouched. This isolates the hypothesis that domain-trained
expert MLPs improve the parent model when re-injected at their original positions.

Usage:
    python -m src.hf_training.merge_pruned_experts_back \\
        --parent-model /path/to/full_model \\
        --pruned-trained-model /path/to/finetuned/checkpoint-N \\
        --pruning-metadata /path/to/pruned_model/pruning_metadata.json \\
        --output-dir /path/to/merged_model
"""

import argparse
import json
import logging
import os
from typing import List, Optional

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_moe_layer(layer) -> bool:
    return hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")


def _layer_param_norm(layer) -> float:
    return sum(p.detach().float().norm().item() ** 2 for p in layer.parameters()) ** 0.5


def _blend(parent_t: torch.Tensor, small_t: torch.Tensor, average: bool, alpha: float) -> None:
    """In-place merge of small into parent. average=False ⇒ replace; True ⇒ (1-α)*parent + α*small."""
    if not average:
        parent_t.copy_(small_t.to(parent_t.dtype))
    else:
        parent_t.mul_(1.0 - alpha).add_(small_t.to(parent_t.dtype), alpha=alpha)


def merge_pruned_experts_back(
    parent_model_path: str,
    pruned_trained_model_path: str,
    pruning_metadata_path: str,
    output_dir: str,
    *,
    also_copy_router_rows: bool = False,
    also_copy_shared: bool = False,
    also_copy_non_moe: bool = False,
    average: bool = False,
    average_weight: float = 0.5,
) -> None:
    if average:
        assert 0.0 < average_weight < 1.0, f"average_weight must be in (0, 1), got {average_weight}"
        logger.info(
            f"Mode: AVERAGE (parent gets (1-α)*parent + α*small for copied params; α={average_weight})"
        )
    else:
        logger.info("Mode: REPLACE (parent's copied params are overwritten with small's values)")

    logger.info(f"Loading parent (full) model: {parent_model_path}")
    parent = AutoModelForCausalLM.from_pretrained(parent_model_path, torch_dtype=torch.bfloat16)

    logger.info(f"Loading pruned-and-trained model: {pruned_trained_model_path}")
    small = AutoModelForCausalLM.from_pretrained(
        pruned_trained_model_path, torch_dtype=torch.bfloat16
    )

    logger.info(f"Loading pruning metadata: {pruning_metadata_path}")
    with open(pruning_metadata_path, "r") as f:
        metadata = json.load(f)
    experts_kept_per_layer: List[Optional[List[int]]] = metadata["experts_kept_per_layer"]

    # ------------------------------------------------------------------
    # Sanity checks: parent and small must share architecture except for expert count
    # ------------------------------------------------------------------
    assert parent.config.num_hidden_layers == small.config.num_hidden_layers, (
        f"Layer count mismatch: parent={parent.config.num_hidden_layers}, "
        f"small={small.config.num_hidden_layers}"
    )
    assert parent.config.hidden_size == small.config.hidden_size, "hidden_size mismatch"
    assert len(experts_kept_per_layer) == parent.config.num_hidden_layers, (
        f"experts_kept_per_layer length {len(experts_kept_per_layer)} != "
        f"num_hidden_layers {parent.config.num_hidden_layers}"
    )

    # ------------------------------------------------------------------
    # Per-layer expert copy
    # ------------------------------------------------------------------
    n_routable_copied = 0
    n_shared_copied = 0
    n_router_rows_copied = 0

    for layer_idx, kept in enumerate(experts_kept_per_layer):
        parent_layer = parent.model.layers[layer_idx]
        small_layer = small.model.layers[layer_idx]

        if kept is None:
            assert not _is_moe_layer(
                parent_layer
            ), f"Layer {layer_idx}: metadata says non-MoE but parent has MoE"
            continue

        assert _is_moe_layer(parent_layer), f"Layer {layer_idx}: parent missing MoE structure"
        assert _is_moe_layer(small_layer), f"Layer {layer_idx}: small missing MoE structure"

        parent_moe = parent_layer.mlp
        small_moe = small_layer.mlp

        parent_num_experts = parent_moe.num_experts
        parent_num_shared = parent_moe.num_shared_experts
        parent_num_standard = parent_num_experts - parent_num_shared
        small_num_experts = small_moe.num_experts

        assert (
            len(kept) == small_num_experts
        ), f"Layer {layer_idx}: kept indices {len(kept)} != small num_experts {small_num_experts}"

        norm_before = _layer_param_norm(parent_layer)

        for new_pos, orig_idx in enumerate(kept):
            assert (
                0 <= orig_idx < parent_num_experts
            ), f"Layer {layer_idx}: orig_idx {orig_idx} out of range [0, {parent_num_experts})"
            is_shared = orig_idx >= parent_num_standard
            if is_shared and not also_copy_shared:
                continue

            parent_expert = parent_moe.experts[orig_idx]
            small_expert = small_moe.experts[new_pos]

            # The two Expert modules must have matching submodule structure / shapes.
            small_sd = small_expert.state_dict()
            for pname, p in parent_expert.named_parameters():
                if pname not in small_sd:
                    raise KeyError(
                        f"Layer {layer_idx} expert {orig_idx}: param {pname} missing from small expert"
                    )
                _blend(p.data, small_sd[pname], average, average_weight)

            if is_shared:
                n_shared_copied += 1
            else:
                n_routable_copied += 1

        if also_copy_router_rows:
            for new_pos, orig_idx in enumerate(kept):
                _blend(
                    parent_moe.gate.weight.data[orig_idx, :],
                    small_moe.gate.weight.data[new_pos, :],
                    average,
                    average_weight,
                )
                if parent_moe.gate.bias is not None and small_moe.gate.bias is not None:
                    _blend(
                        parent_moe.gate.bias.data[orig_idx : orig_idx + 1],
                        small_moe.gate.bias.data[new_pos : new_pos + 1],
                        average,
                        average_weight,
                    )
                n_router_rows_copied += 1

        norm_after = _layer_param_norm(parent_layer)
        delta = abs(norm_after - norm_before)
        logger.info(
            f"Layer {layer_idx}: kept={kept[:3]}{'...' if len(kept) > 3 else ''} "
            f"({len(kept)} experts), parent param-norm change |Δ|={delta:.4f}"
        )
        if delta == 0.0:
            logger.warning(
                f"Layer {layer_idx}: parent param-norm unchanged after copy — possible silent no-op"
            )

    # ------------------------------------------------------------------
    # Optional: copy non-MoE params (attention / norms / embeddings / lm_head)
    # ------------------------------------------------------------------
    n_non_moe_copied = 0
    if also_copy_non_moe:
        parent_state = parent.state_dict()
        small_state = small.state_dict()
        for k in list(parent_state.keys()):
            if ".mlp." in k:
                continue
            if k not in small_state:
                continue
            if small_state[k].shape != parent_state[k].shape:
                logger.warning(
                    f"Skipping non-MoE param {k}: shape mismatch "
                    f"parent={parent_state[k].shape} small={small_state[k].shape}"
                )
                continue
            _blend(parent_state[k], small_state[k], average, average_weight)
            n_non_moe_copied += 1
        logger.info(f"Copied {n_non_moe_copied} non-MoE params from small → parent")

    logger.info(
        f"Copy summary: {n_routable_copied} routable expert MLPs, "
        f"{n_shared_copied} shared expert MLPs, "
        f"{n_router_rows_copied} router rows, "
        f"{n_non_moe_copied} non-MoE params."
    )

    # ------------------------------------------------------------------
    # Save merged model
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving merged full-sized model to {output_dir}")
    parent.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(parent_model_path).save_pretrained(output_dir)

    merge_metadata = {
        "parent_model": parent_model_path,
        "pruned_trained_model": pruned_trained_model_path,
        "pruning_metadata_source": pruning_metadata_path,
        "experts_copied_per_layer": experts_kept_per_layer,
        "also_copy_router_rows": also_copy_router_rows,
        "also_copy_shared": also_copy_shared,
        "also_copy_non_moe": also_copy_non_moe,
        "n_routable_copied": n_routable_copied,
        "n_shared_copied": n_shared_copied,
        "n_router_rows_copied": n_router_rows_copied,
        "n_non_moe_copied": n_non_moe_copied,
        "merge_mode": "average" if average else "replace",
        "average_weight": average_weight if average else None,
    }
    with open(os.path.join(output_dir, "merge_metadata.json"), "w") as f:
        json.dump(merge_metadata, f, indent=2)
    logger.info(f"Done. Merged model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge a finetuned pruned MoE model's experts back into the original full model."
    )
    parser.add_argument(
        "--parent-model",
        type=str,
        required=True,
        help="Path to the original (unpruned, untrained) HF MoE checkpoint.",
    )
    parser.add_argument(
        "--pruned-trained-model",
        type=str,
        required=True,
        help="Path to the finetuned pruned HF checkpoint (e.g. .../finetuned_model/checkpoint-N).",
    )
    parser.add_argument(
        "--pruning-metadata",
        type=str,
        required=True,
        help="Path to pruning_metadata.json written by greedy_prune_layerwise (lives in the pruned model dir).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Where to write the merged full-sized model."
    )
    parser.add_argument(
        "--also-copy-router-rows",
        action="store_true",
        help="Also copy the trained router rows (for the kept indices) back into the parent's router.",
    )
    parser.add_argument(
        "--also-copy-shared",
        action="store_true",
        help="Also copy the shared-expert MLP back into the parent (default off).",
    )
    parser.add_argument(
        "--also-copy-non-moe",
        action="store_true",
        help="Also copy non-MoE params (attention, norms, embeddings, lm_head) back into the parent.",
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help="Instead of replacing the parent's params with the small model's, blend "
        "(1-α)*parent + α*small for every selected param. Affects experts, router rows, "
        "and non-MoE copies — whichever flags are on.",
    )
    parser.add_argument(
        "--average-weight",
        type=float,
        default=0.5,
        help="Weight α on the small model when --average is set (default 0.5 = arithmetic mean).",
    )
    args = parser.parse_args()

    merge_pruned_experts_back(
        parent_model_path=args.parent_model,
        pruned_trained_model_path=args.pruned_trained_model,
        pruning_metadata_path=args.pruning_metadata,
        output_dir=args.output_dir,
        also_copy_router_rows=args.also_copy_router_rows,
        also_copy_shared=args.also_copy_shared,
        also_copy_non_moe=args.also_copy_non_moe,
        average=args.average,
        average_weight=args.average_weight,
    )


if __name__ == "__main__":
    main()
