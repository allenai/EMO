"""
Greedy layer-by-layer expert pruning for HuggingFace MoE models.

Unlike the global approach (compute_router_activations.py + prune_hf_model.py),
which collects all layer activations from a single full-model forward pass, this
script prunes greedily one layer at a time:

  For layer l = 0 .. num_layers-1:
    1. Run all validation batches through the model with layers 0..l-1 already
       pruned in-place. Capture the raw gate logits at layer l via a forward hook.
    2. Compute average softmax router probabilities over all non-padding tokens.
    3. Select top-k experts, prune layer l in-place.
    4. Proceed to layer l+1, whose activations are now conditioned on the pruned
       representation produced by layers 0..l.

An EarlyExit pre-forward hook on layer l+1 aborts the forward pass right after
layer l finishes, avoiding unnecessary computation through the remaining layers.

Usage:
    python -m src.hf_training.greedy_prune_layerwise \\
        --model /path/to/model \\
        --task arc_challenge \\
        --prune-keep-k 32 \\
        --num-shared-experts 1 \\
        --save-path /path/to/pruned_model
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.hf_training.data_utils import get_formatted_prompts

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyExit(Exception):
    """Raised by a pre-forward hook to abort the model forward pass after a target layer."""

    pass


def prune_moe_layer_inplace(
    layer,
    experts_to_keep: List[int],
    prune_keep_k: int,
    target_num_shared_experts: int,
) -> None:
    """
    Prune a single MoE decoder layer in-place.

    Args:
        layer: FlexOlmoNoQKNormPrenormDecoderLayer (or compatible OLMoE-style layer)
        experts_to_keep: Sorted list of expert indices to retain (length == prune_keep_k)
        prune_keep_k: Total number of experts to keep (standard + shared)
        target_num_shared_experts: Number of shared experts in the pruned layer
    """
    moe = layer.mlp

    # Prune expert MLPs
    moe.experts = torch.nn.ModuleList([moe.experts[i] for i in experts_to_keep])

    # Prune router/gate weights  (shape: num_experts x hidden_size -> prune_keep_k x hidden_size)
    moe.gate.weight = torch.nn.Parameter(moe.gate.weight.data[experts_to_keep, :])
    if moe.gate.bias is not None:
        moe.gate.bias = torch.nn.Parameter(moe.gate.bias.data[experts_to_keep])
    moe.gate.out_features = prune_keep_k

    # Update module-level expert counts so subsequent forward passes are correct
    moe.num_experts = prune_keep_k
    moe.num_shared_experts = target_num_shared_experts
    moe.top_k = min(moe.top_k, prune_keep_k)


def _is_moe_layer(layer) -> bool:
    return hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")


def snapshot_model_state(model) -> Dict[str, Any]:
    """
    Capture a snapshot of all state mutated by ``prune_moe_layer_inplace`` plus the
    ``output_router_logits`` config flag, so that a later call to
    ``restore_model_state`` returns the model to exactly its pre-pruning state.

    The individual expert ``nn.Module`` instances are never mutated by pruning — only
    the ``ModuleList`` wrapping them is replaced — so storing the list of module
    references (not clones) is sufficient.
    """
    layer_snaps: List[Optional[Dict[str, Any]]] = []
    for layer in model.model.layers:
        if not _is_moe_layer(layer):
            layer_snaps.append(None)
            continue
        moe = layer.mlp
        layer_snaps.append(
            {
                "gate_weight": moe.gate.weight.data.clone(),
                "gate_bias": (
                    moe.gate.bias.data.clone() if moe.gate.bias is not None else None
                ),
                "gate_out_features": moe.gate.out_features,
                "experts": list(moe.experts),
                "num_experts": moe.num_experts,
                "num_shared_experts": moe.num_shared_experts,
                "top_k": moe.top_k,
            }
        )
    return {
        "layers": layer_snaps,
        "output_router_logits": getattr(model.config, "output_router_logits", None),
    }


def restore_model_state(model, snapshot: Dict[str, Any]) -> None:
    """Restore model state captured by :func:`snapshot_model_state`."""
    for layer, snap in zip(model.model.layers, snapshot["layers"]):
        if snap is None:
            continue
        moe = layer.mlp
        moe.gate.weight = torch.nn.Parameter(snap["gate_weight"].clone())
        if snap["gate_bias"] is not None:
            moe.gate.bias = torch.nn.Parameter(snap["gate_bias"].clone())
        moe.gate.out_features = snap["gate_out_features"]
        moe.experts = torch.nn.ModuleList(snap["experts"])
        moe.num_experts = snap["num_experts"]
        moe.num_shared_experts = snap["num_shared_experts"]
        moe.top_k = snap["top_k"]
    if snapshot["output_router_logits"] is not None and hasattr(
        model.config, "output_router_logits"
    ):
        model.config.output_router_logits = snapshot["output_router_logits"]


def _capture_layer_output(
    model,
    layer_idx: int,
    batch_inputs: dict,
) -> torch.Tensor:
    """
    Run one batch forward through the model up to and including layer_idx,
    and return that decoder layer's output hidden states (shape: B x T x hidden_dim).

    Uses EarlyExit to avoid computing layers beyond layer_idx.
    """
    num_layers = model.config.num_hidden_layers
    batch_on_device = {k: v.to(model.device) for k, v in batch_inputs.items()}
    captured = []

    def capture_hook(module, input, output):
        captured.append(output.detach().cpu())

    def early_exit_hook(module, input):
        raise EarlyExit()

    layer = model.model.layers[layer_idx]
    h_capture = layer.register_forward_hook(capture_hook)
    h_exit = (
        model.model.layers[layer_idx + 1].register_forward_pre_hook(early_exit_hook)
        if layer_idx + 1 < num_layers
        else None
    )

    with torch.no_grad():
        try:
            model(**batch_on_device)
        except EarlyExit:
            pass

    h_capture.remove()
    if h_exit is not None:
        h_exit.remove()

    assert len(captured) == 1, f"Expected 1 captured tensor, got {len(captured)}"
    return captured[0]


def compute_layerwise_keep_sets(
    model,
    tokenizer,
    task_name: str,
    split: str,
    prune_keep_k: int,
    num_shared_experts: int,
    batch_size: int = 32,
    num_calibration: Optional[int] = None,
    prompts: Optional[List[str]] = None,
    num_shots_override: Optional[int] = None,
    prune_seed: int = 0,
) -> Tuple[List[Optional[List[int]]], List[Optional[List[float]]]]:
    """
    Run greedy layer-by-layer pruning on a preloaded model (in-place) and return the
    experts kept per layer plus the per-expert average router probabilities used to
    make the top-k decision.

    The model is left in its fully-pruned state on return. Callers that want to reuse
    the model should snapshot + restore it around this call via
    :func:`snapshot_model_state` / :func:`restore_model_state`.

    Args:
        model: Preloaded HF causal-LM with MoE layers.
        tokenizer: Corresponding tokenizer.
        task_name, split: Forwarded to ``get_formatted_prompts`` when ``prompts`` is None.
        prune_keep_k: Total experts to keep per MoE layer after pruning.
        num_shared_experts: Of those, how many are shared experts.
        batch_size: Batch size for forward passes during activation collection.
        num_calibration: If not None, subsample the loaded prompts via a seeded
            permutation (seed defaults to 0, matches the historical eval-pipeline
            behaviour). Ignored when ``prompts`` is explicitly provided.
        prompts: Optional pre-loaded / pre-subsampled prompt list. When provided,
            ``get_formatted_prompts`` is not called and ``num_calibration`` is ignored.
        prune_seed: Seed for the calibration-subsample permutation (default 0).

    Returns:
        (experts_kept_per_layer, avg_probs_per_layer) — both length ``num_hidden_layers``.
        Entries are ``None`` for non-MoE layers.
    """
    num_layers = model.config.num_hidden_layers
    logger.info(f"Model has {num_layers} layers")

    # Disable router logit output during pruning to avoid LB loss computation,
    # which crashes when layers have different expert counts mid-pruning.
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = False

    # -------------------------------------------------------------------------
    # Load and tokenize validation data once up front
    # -------------------------------------------------------------------------
    if prompts is None:
        logger.info(
            f"Loading dataset: {task_name} ({split})"
            + (f" [num_shots={num_shots_override}]" if num_shots_override is not None else "")
        )
        prompts, _ = get_formatted_prompts(
            task_name, split, num_shots_override=num_shots_override
        )
        if num_calibration is None:
            logger.info(f"Loaded {len(prompts)} prompts, using all (no subsampling)")
        else:
            n_keep = min(num_calibration, len(prompts))
            logger.info(
                f"Loaded {len(prompts)} prompts, subsampling to {n_keep} (seed={prune_seed})"
            )
            g = torch.Generator().manual_seed(prune_seed)
            perm = torch.randperm(len(prompts), generator=g).tolist()
            prompts = [prompts[i] for i in perm[:n_keep]]
    else:
        logger.info(f"Using {len(prompts)} caller-provided prompts (skipping load/subsample)")

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
    avg_probs_per_layer: List[Optional[List[float]]] = []

    for layer_idx in tqdm(range(num_layers), desc="Pruning layers"):
        layer = model.model.layers[layer_idx]

        if not _is_moe_layer(layer):
            logger.debug(f"Layer {layer_idx}: not an MoE layer, skipping")
            experts_kept_per_layer.append(None)
            avg_probs_per_layer.append(None)
            continue

        # Expert counts for this layer before pruning
        current_num_experts = layer.mlp.num_experts
        current_num_shared = layer.mlp.num_shared_experts
        current_num_standard = current_num_experts - current_num_shared

        logger.info(
            f"Layer {layer_idx}: collecting activations "
            f"({current_num_experts} total experts, {current_num_shared} shared)"
        )

        # --- Register hooks -------------------------------------------
        # Capture hook: fires after layer.mlp.forward(), grabs router_logits
        captured_logits: List[torch.Tensor] = []

        def make_capture_hook():
            def hook(module, input, output):
                # SparseMoeBlock returns (final_hidden_states, router_logits)
                # router_logits shape: (batch_size * seq_len, num_experts)
                captured_logits.append(output[1].detach().cpu())

            return hook

        # Early-exit hook: fires before layer l+1.forward(), aborts the pass
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

            # logits shape: (B*T, current_num_experts)
            logits = captured_logits[0]
            B, T = batch_inputs["attention_mask"].shape
            logits_reshaped = logits.view(B, T, current_num_experts)

            # Softmax separately for standard vs shared experts (matching
            # the forward pass in FlexOlmoNoQKNormPrenormSparseMoeBlock)
            if current_num_shared > 0:
                logits_standard = logits_reshaped[:, :, :current_num_standard]
                logits_shared = logits_reshaped[:, :, current_num_standard:]
                probs_standard = F.softmax(logits_standard, dim=-1)
                probs_shared = F.softmax(logits_shared, dim=-1)

                mask = batch_inputs["attention_mask"].unsqueeze(-1)  # (B, T, 1)
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
                    min(prune_keep_k - num_shared_experts, current_num_standard),
                ).indices.tolist()
            )
            keep_shared_local = sorted(
                torch.topk(
                    avg_probs_shared,
                    min(num_shared_experts, current_num_shared),
                ).indices.tolist()
            )
            # Shift shared indices by the number of standard experts
            keep_shared = [current_num_standard + i for i in keep_shared_local]
            experts_to_keep = keep_standard + keep_shared
        else:
            experts_to_keep = sorted(
                torch.topk(avg_probs, min(prune_keep_k, current_num_experts)).indices.tolist()
            )

        logger.info(f"Layer {layer_idx}: keeping experts {experts_to_keep}")
        experts_kept_per_layer.append(experts_to_keep)
        avg_probs_per_layer.append(avg_probs.tolist())

        # --- Sanity check (first MoE layer only) --------------------------
        # Confirm that after pruning, the hidden states produced by this layer
        # are actually different from the pre-pruning hidden states — i.e., that
        # subsequent layers will receive different inputs once this layer is pruned.
        is_first_moe_layer = all(e is None for e in experts_kept_per_layer[:-1])
        actually_pruning = prune_keep_k < current_num_experts
        if is_first_moe_layer and actually_pruning:
            hidden_before = _capture_layer_output(model, layer_idx, all_batches[0])

        # --- Prune layer in-place -----------------------------------------
        target_shared = num_shared_experts if current_num_shared > 0 else 0
        prune_moe_layer_inplace(layer, experts_to_keep, prune_keep_k, target_shared)

        if is_first_moe_layer and actually_pruning:
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

    return experts_kept_per_layer, avg_probs_per_layer


def greedy_prune_layerwise(
    model_name: str,
    task_name: str,
    split: str,
    prune_keep_k: int,
    num_shared_experts: int,
    save_path: str,
    batch_size: int = 32,
    num_calibration: Optional[int] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    num_shots_override: Optional[int] = None,
    prune_seed: int = 0,
) -> None:
    """
    Greedily prune MoE experts one layer at a time, then save the pruned model.

    Thin wrapper over :func:`compute_layerwise_keep_sets` that handles model loading,
    post-pruning config updates, and disk I/O. Behavior is identical to the pre-refactor
    implementation.
    """
    logger.info(f"Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device is None else device,
        trust_remote_code=trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    experts_kept_per_layer, _ = compute_layerwise_keep_sets(
        model=model,
        tokenizer=tokenizer,
        task_name=task_name,
        split=split,
        prune_keep_k=prune_keep_k,
        num_shared_experts=num_shared_experts,
        batch_size=batch_size,
        num_calibration=num_calibration,
        num_shots_override=num_shots_override,
        prune_seed=prune_seed,
    )

    # -------------------------------------------------------------------------
    # Update global model config to reflect pruned expert counts
    # -------------------------------------------------------------------------
    if (
        hasattr(model.config, "num_experts_per_tok")
        and model.config.num_experts_per_tok > prune_keep_k
    ):
        model.config.num_experts_per_tok = prune_keep_k
    model.config.num_experts = prune_keep_k
    if hasattr(model.config, "num_local_experts"):
        model.config.num_local_experts = prune_keep_k
    model.config.num_shared_experts = num_shared_experts

    # Update per-layer expert counts for densefirst (mixed dense/MoE) models
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

    # Enable router logit output so load-balancing loss is active during finetuning
    model.config.output_router_logits = True

    # -------------------------------------------------------------------------
    # Save pruned model
    # -------------------------------------------------------------------------
    logger.info(f"Saving pruned model to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    metadata = {
        "original_model": model_name,
        "prune_keep_k": prune_keep_k,
        "num_shared_experts": num_shared_experts,
        "pruning_method": "greedy_layerwise",
        "task": task_name,
        "split": split,
        "num_shots_override": num_shots_override,
        "num_calibration": num_calibration,
        "prune_seed": prune_seed,
        "experts_kept_per_layer": experts_kept_per_layer,
    }
    with open(os.path.join(save_path, "pruning_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done. Pruned model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Greedy layer-by-layer expert pruning for HuggingFace MoE models"
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
        "--prune-keep-k",
        type=int,
        required=True,
        help="Total number of experts to keep per layer",
    )
    parser.add_argument(
        "--num-shared-experts",
        type=int,
        default=0,
        help="Number of shared experts to keep (default: 0)",
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
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading from HF Hub (default: False)",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=None,
        help="Override task config's num_shots (default: use config value)",
    )
    parser.add_argument(
        "--prune-seed",
        type=int,
        default=0,
        help="Seed for the calibration-subsample permutation (default: 0)",
    )

    args = parser.parse_args()

    greedy_prune_layerwise(
        model_name=args.model,
        task_name=args.task,
        split=args.split,
        prune_keep_k=args.prune_keep_k,
        num_shared_experts=args.num_shared_experts,
        save_path=args.save_path,
        batch_size=args.batch_size,
        num_calibration=args.num_calibration,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        num_shots_override=args.num_shots,
        prune_seed=args.prune_seed,
    )


if __name__ == "__main__":
    main()
