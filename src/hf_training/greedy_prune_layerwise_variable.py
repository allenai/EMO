"""
Greedy layer-by-layer expert pruning with variable per-layer keep-k.

Same algorithm as greedy_prune_layerwise.py, but each layer can retain a
different number of experts.  The caller passes a comma-separated list of
keep-k values (one per transformer layer).  Layers whose keep-k equals or
exceeds the current expert count are left untouched.

Usage:
    python -m src.hf_training.greedy_prune_layerwise_variable \
        --model /path/to/model \
        --task arc_challenge \
        --keep-k-per-layer "128,128,32,32,32,32,32,32,32,32,32,32,32,32,32,32" \
        --num-shared-experts 1 \
        --save-path /path/to/pruned_model
"""

import argparse
import json
import logging
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.hf_training.data_utils import get_formatted_prompts
from src.hf_training.greedy_prune_layerwise import (
    EarlyExit,
    _capture_layer_output,
    prune_moe_layer_inplace,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def greedy_prune_layerwise_variable(
    model_name: str,
    task_name: str,
    split: str,
    keep_k_per_layer: List[int],
    num_shared_experts: int,
    save_path: str,
    batch_size: int = 32,
    num_calibration: Optional[int] = None,
    device: Optional[str] = None,
) -> None:
    """
    Greedily prune MoE experts one layer at a time, with a per-layer keep-k schedule.

    Args:
        model_name: Path or HF hub name of the full (unpruned) model
        task_name: Task name used to load the validation set for activation collection
        split: Dataset split to use (default: validation)
        keep_k_per_layer: List of ints, one per transformer layer.  Each entry is the
            number of experts to keep in that layer.  If >= the layer's current expert
            count the layer is left untouched.
        num_shared_experts: Number of shared experts to keep (applied only to pruned layers)
        save_path: Directory to save the pruned model
        batch_size: Batch size for forward passes during activation collection
        device: Device override; defaults to "auto" (multi-GPU if available)
    """
    logger.info(f"Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device is None else device,
    )
    # model = FlexOlmoNoQKNormPrenormForCausalLMDebug.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto" if device is None else device,
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = config.num_hidden_layers
    assert (
        len(keep_k_per_layer) == num_layers
    ), f"keep_k_per_layer has length {len(keep_k_per_layer)} but model has {num_layers} layers"
    logger.info(f"Model has {num_layers} layers")
    logger.info(f"Per-layer keep-k schedule: {keep_k_per_layer}")

    # -------------------------------------------------------------------------
    # Load and tokenize validation data once up front
    # -------------------------------------------------------------------------
    logger.info(f"Loading dataset: {task_name} ({split})")
    prompts, _ = get_formatted_prompts(task_name, split)
    if num_calibration is None:
        logger.info(f"Loaded {len(prompts)} prompts, using all (no subsampling)")
    else:
        n_keep = min(num_calibration, len(prompts))
        logger.info(f"Loaded {len(prompts)} prompts, subsampling to {n_keep}")
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(prompts), generator=g).tolist()
        prompts = [prompts[i] for i in perm[:n_keep]]

    logger.info("Tokenizing prompts into batches...")
    all_batches = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        all_batches.append(inputs)
    logger.info(f"Created {len(all_batches)} batches")

    # -------------------------------------------------------------------------
    # Layer-by-layer greedy pruning
    # -------------------------------------------------------------------------
    experts_kept_per_layer: List[Optional[List[int]]] = []

    for layer_idx in tqdm(range(num_layers), desc="Pruning layers"):
        layer = model.model.layers[layer_idx]
        layer_keep_k = keep_k_per_layer[layer_idx]

        if not (hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")):
            logger.debug(f"Layer {layer_idx}: not an MoE layer, skipping")
            experts_kept_per_layer.append(None)
            continue

        current_num_experts = layer.mlp.num_experts
        current_num_shared = layer.mlp.num_shared_experts

        # Skip layers where keep-k >= current expert count (no pruning needed)
        if layer_keep_k >= current_num_experts:
            logger.info(
                f"Layer {layer_idx}: keep_k={layer_keep_k} >= num_experts={current_num_experts}, "
                "skipping (no pruning)"
            )
            experts_kept_per_layer.append(None)
            continue

        current_num_standard = current_num_experts - current_num_shared

        logger.info(
            f"Layer {layer_idx}: collecting activations "
            f"({current_num_experts} total experts, {current_num_shared} shared), "
            f"will prune to {layer_keep_k}"
        )

        # --- Register hooks -------------------------------------------
        captured_logits: List[torch.Tensor] = []

        def make_capture_hook():
            def hook(module, input, output):
                captured_logits.append(output[1].detach().cpu())

            return hook

        def make_early_exit_hook():
            def hook(module, input):
                raise EarlyExit()

            return hook

        h_capture = layer.mlp.register_forward_hook(make_capture_hook())
        h_exit = (
            model.model.layers[layer_idx + 1].register_forward_pre_hook(make_early_exit_hook())
            if layer_idx + 1 < num_layers
            else None
        )

        # --- Collect gate logits across all batches -----------------------
        tot_probs = torch.zeros(current_num_experts)
        tot_tokens = 0

        for batch_inputs in all_batches:
            batch_on_device = {k: v.to(model.device) for k, v in batch_inputs.items()}
            captured_logits.clear()

            with torch.no_grad():
                try:
                    model(**batch_on_device)
                except EarlyExit:
                    pass

            assert (
                len(captured_logits) == 1
            ), f"Expected exactly 1 captured logit tensor per batch, got {len(captured_logits)}"

            logits = captured_logits[0]
            B, T = batch_inputs["attention_mask"].shape
            logits_reshaped = logits.view(B, T, current_num_experts)

            if current_num_shared > 0:
                logits_standard = logits_reshaped[:, :, :current_num_standard]
                logits_shared = logits_reshaped[:, :, current_num_standard:]
                probs_standard = F.softmax(logits_standard, dim=-1)
                probs_shared = F.softmax(logits_shared, dim=-1)

                mask = batch_inputs["attention_mask"].unsqueeze(-1)
                tot_probs[:current_num_standard] += (probs_standard * mask).sum(dim=(0, 1))
                tot_probs[current_num_standard:] += (probs_shared * mask).sum(dim=(0, 1))
            else:
                probs = F.softmax(logits_reshaped, dim=-1)
                mask = batch_inputs["attention_mask"].unsqueeze(-1)
                tot_probs += (probs * mask).sum(dim=(0, 1))

            tot_tokens += batch_inputs["attention_mask"].sum().item()

        h_capture.remove()
        if h_exit is not None:
            h_exit.remove()

        # --- Determine which experts to keep ------------------------------
        avg_probs = tot_probs / tot_tokens

        if current_num_shared > 0:
            avg_probs_standard = avg_probs[:current_num_standard]
            avg_probs_shared = avg_probs[current_num_standard:]

            keep_standard = sorted(
                torch.topk(
                    avg_probs_standard,
                    min(layer_keep_k - num_shared_experts, current_num_standard),
                ).indices.tolist()
            )
            keep_shared_local = sorted(
                torch.topk(
                    avg_probs_shared,
                    min(num_shared_experts, current_num_shared),
                ).indices.tolist()
            )
            keep_shared = [current_num_standard + i for i in keep_shared_local]
            experts_to_keep = keep_standard + keep_shared
        else:
            experts_to_keep = sorted(
                torch.topk(avg_probs, min(layer_keep_k, current_num_experts)).indices.tolist()
            )

        logger.info(f"Layer {layer_idx}: keeping experts {experts_to_keep}")
        experts_kept_per_layer.append(experts_to_keep)

        # --- Sanity check (first pruned MoE layer only) --------------------
        is_first_pruned_moe = all(e is None for e in experts_kept_per_layer[:-1])
        if is_first_pruned_moe:
            hidden_before = _capture_layer_output(model, layer_idx, all_batches[0])

        # --- Prune layer in-place -----------------------------------------
        target_shared = num_shared_experts if current_num_shared > 0 else 0
        prune_moe_layer_inplace(layer, experts_to_keep, layer_keep_k, target_shared)

        if is_first_pruned_moe:
            hidden_after = _capture_layer_output(model, layer_idx, all_batches[0])
            max_diff = (hidden_before - hidden_after).abs().max().item()
            if torch.allclose(hidden_before, hidden_after):
                raise RuntimeError(
                    f"Sanity check FAILED at layer {layer_idx}: hidden states are "
                    "identical before and after pruning. The forward pass is not "
                    "reflecting the pruned layer weights."
                )
            logger.info(
                f"Sanity check PASSED at layer {layer_idx}: "
                f"max hidden-state difference before/after pruning = {max_diff:.6f}"
            )

    # -------------------------------------------------------------------------
    # Update global model config to reflect pruned expert counts
    # -------------------------------------------------------------------------
    # Compute actual per-layer expert counts after pruning
    # (layers that were skipped retain their original count)
    actual_num_experts_per_layer = []
    actual_num_shared_experts_per_layer = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            actual_num_experts_per_layer.append(layer.mlp.num_experts)
            actual_num_shared_experts_per_layer.append(layer.mlp.num_shared_experts)
        else:
            # Non-MoE layer, use 0 as placeholder
            actual_num_experts_per_layer.append(0)
            actual_num_shared_experts_per_layer.append(0)

    min_keep_k = min(keep_k_per_layer)
    if (
        hasattr(model.config, "num_experts_per_tok")
        and model.config.num_experts_per_tok > min_keep_k
    ):
        model.config.num_experts_per_tok = min_keep_k
    model.config.num_experts = min_keep_k
    if hasattr(model.config, "num_local_experts"):
        model.config.num_local_experts = min_keep_k
    model.config.num_shared_experts = num_shared_experts
    model.config.output_router_logits = True

    # Store per-layer expert counts for FlexOlmoNoQKNormPrenormForCausalLMDebug
    model.config.num_experts_per_layer = actual_num_experts_per_layer
    model.config.num_shared_experts_per_layer = actual_num_shared_experts_per_layer

    # -------------------------------------------------------------------------
    # Save pruned model
    # -------------------------------------------------------------------------
    logger.info(f"Saving pruned model to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    metadata = {
        "original_model": model_name,
        "keep_k_per_layer": keep_k_per_layer,
        "num_shared_experts": num_shared_experts,
        "pruning_method": "greedy_layerwise_variable",
        "task": task_name,
        "split": split,
        "experts_kept_per_layer": experts_kept_per_layer,
    }
    with open(os.path.join(save_path, "pruning_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done. Pruned model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Greedy layer-by-layer expert pruning with variable per-layer keep-k"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path or HF name of the full (unpruned) model"
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Task name for activation data collection"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split for activation collection (default: validation)",
    )
    parser.add_argument(
        "--keep-k-per-layer",
        type=str,
        required=True,
        help="Comma-separated list of per-layer keep-k values (length must equal num_hidden_layers)",
    )
    parser.add_argument(
        "--num-shared-experts",
        type=int,
        default=0,
        help="Number of shared experts to keep in pruned layers (default: 0)",
    )
    parser.add_argument(
        "--save-path", type=str, required=True, help="Output directory for the pruned model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for activation collection forward passes (default: 32)",
    )
    parser.add_argument(
        "--num-calibration",
        type=int,
        default=None,
        help="Subsample calibration set to this many prompts (default: use all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (default: auto)",
    )

    args = parser.parse_args()

    keep_k_per_layer = [int(x.strip()) for x in args.keep_k_per_layer.split(",")]

    greedy_prune_layerwise_variable(
        model_name=args.model,
        task_name=args.task,
        split=args.split,
        keep_k_per_layer=keep_k_per_layer,
        num_shared_experts=args.num_shared_experts,
        save_path=args.save_path,
        batch_size=args.batch_size,
        num_calibration=args.num_calibration,
        device=args.device,
    )


if __name__ == "__main__":
    main()
