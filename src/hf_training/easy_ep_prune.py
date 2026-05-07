"""
EASY-EP (Expert Assessment with Simple Yet-effective scoring) expert pruning
for HuggingFace MoE models. Implementation of arXiv 2504.06792.

Core idea: score each expert by the sum over calibration tokens of
    c_{i,t} * s_t
where
    c_{i,t} = g_{i,t} * ||E_i(h_t)||       (output-aware expert importance)
    s_t    = 1 - cos_sim(h_t, h_t + bar_h_t)  (token contribution weight)

Differs from the greedy-layerwise path in two ways:
  1. Single forward pass through the unpruned model — all layers scored at
     once, no greedy conditioning.
  2. Score replaces the plain-softmax-avg used in greedy_prune_layerwise.py.

Usage:
    python -m src.hf_training.easy_ep_prune \\
        --model /path/to/model \\
        --task arc_easy --split train \\
        --prune-keep-k 32 --num-shared-experts 1 \\
        --num-calibration 25 \\
        --save-path /tmp/pruned
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.hf_training.data_utils import get_formatted_prompts
from src.hf_training.greedy_prune_layerwise import prune_moe_layer_inplace
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_easy_ep_score_for_batch(
    hidden_in: torch.Tensor,  # (B*T, D)
    hidden_out_bar: torch.Tensor,  # (B*T, D)
    router_logits: torch.Tensor,  # (B*T, N)
    experts: torch.nn.ModuleList,
    num_shared_experts: int,
    top_k: int,
    valid_mask_flat: torch.Tensor,  # (B*T,) bool, True = non-padding
) -> torch.Tensor:
    """
    Compute per-expert EASY-EP score contributions for a single batch / layer.

    Returns a (num_experts,) tensor on CPU in float64.
    """
    device = hidden_in.device
    num_experts = router_logits.shape[1]
    num_standard = num_experts - num_shared_experts

    # --- Replicate the forward-pass routing -----------------------------------
    # Matches EmoSparseMoeBlock.forward exactly for the
    # num_shared_experts > 0 path (two-softmax), and the plain path otherwise.
    if num_shared_experts > 0:
        logits_standard = router_logits[:, :num_standard]
        logits_shared = router_logits[:, num_standard:]
        rw_standard = F.softmax(logits_standard, dim=1, dtype=torch.float)
        rw_shared = F.softmax(logits_shared, dim=1, dtype=torch.float)
        top_k_standard = top_k - num_shared_experts
        rw_standard_top, se_standard = torch.topk(rw_standard, top_k_standard, dim=-1)
        rw_shared_top, se_shared = torch.topk(rw_shared, num_shared_experts, dim=-1)
        routing_weights = torch.cat([rw_standard_top, rw_shared_top], dim=1)
        selected_experts = torch.cat([se_standard, se_shared + num_standard], dim=1)
    else:
        rw = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(rw, top_k, dim=-1)

    # --- Token contribution s_t = 1 - cos_sim(h_t, h_t + bar_h_t) -------------
    h_post_moe = hidden_in.float() + hidden_out_bar.float()
    cos_sim = F.cosine_similarity(hidden_in.float(), h_post_moe, dim=-1)  # (B*T,)
    s_t = (1.0 - cos_sim).clamp(min=0.0)  # guard against tiny negatives from fp
    s_t = s_t * valid_mask_flat.float()

    # --- Per-expert output norms + score accumulation ------------------------
    # expert_mask[i]: (top_k, B*T) bool. True where token routed to expert i.
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)

    scores = torch.zeros(num_experts, dtype=torch.float64, device=device)
    for expert_idx in range(num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        if top_x.numel() == 0:
            continue
        # Skip padding tokens entirely
        valid = valid_mask_flat[top_x]
        if not valid.any():
            continue
        top_x = top_x[valid]
        idx = idx[valid]
        if top_x.numel() == 0:
            continue

        current_state = hidden_in[top_x]  # (n_routed, D)
        with torch.no_grad():
            expert_out = experts[expert_idx](current_state)  # (n_routed, D)
        e_norms = expert_out.float().norm(dim=-1)  # (n_routed,)
        g_vals = routing_weights[top_x, idx].float()  # (n_routed,)
        c_vals = g_vals * e_norms  # (n_routed,)
        s_vals = s_t[top_x]  # (n_routed,)
        scores[expert_idx] += (c_vals * s_vals).double().sum()

    return scores.cpu()


class _EasyEPCollector:
    """
    Registers pre/post hooks on every MoE mlp module. For each forward pass,
    accumulates per-expert scores per layer into self.scores.

    The attention mask for the *current* batch must be set on self.attn_mask
    before calling model(...).
    """

    def __init__(self, model, num_layers: int, num_experts_per_layer: List[int]):
        self.model = model
        self.num_layers = num_layers
        self.scores: List[Optional[torch.Tensor]] = [
            torch.zeros(n, dtype=torch.float64) if n > 0 else None for n in num_experts_per_layer
        ]
        self.attn_mask: Optional[torch.Tensor] = None  # (B, T) set per batch
        self._hook_handles: List = []
        self._pending_inputs: Dict[int, torch.Tensor] = {}

    def _make_pre_hook(self, layer_idx: int):
        def pre_hook(module, args):
            # args[0]: hidden_states (B, T, D) post-norm
            self._pending_inputs[layer_idx] = args[0].detach()

        return pre_hook

    def _make_post_hook(self, layer_idx: int, mlp_module):
        def post_hook(module, args, output):
            # output: (final_hidden_states (B, T, D), router_logits (B*T, N))
            final_hidden, router_logits = output
            h_in = self._pending_inputs.pop(layer_idx)

            B, T, D = h_in.shape
            h_in_flat = h_in.reshape(B * T, D)
            h_out_flat = final_hidden.detach().reshape(B * T, D)
            rl = router_logits.detach()
            if rl.shape[0] != B * T:
                rl = rl.reshape(B * T, -1)

            assert self.attn_mask is not None, "attn_mask must be set before forward"
            valid_flat = self.attn_mask.reshape(B * T).bool().to(h_in.device)

            score_delta = compute_easy_ep_score_for_batch(
                hidden_in=h_in_flat,
                hidden_out_bar=h_out_flat,
                router_logits=rl,
                experts=mlp_module.experts,
                num_shared_experts=mlp_module.num_shared_experts,
                top_k=mlp_module.top_k,
                valid_mask_flat=valid_flat,
            )
            self.scores[layer_idx] += score_delta

        return post_hook

    def attach(self):
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            if not (hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")):
                continue
            aae = getattr(layer.mlp, "always_active_experts", None)
            if aae is not None and len(aae) > 0:
                raise NotImplementedError(
                    f"Layer {layer_idx}: always_active_experts is set (len={len(aae)}). "
                    "EASY-EP scoring only supports the num_shared_experts>=0 path, not the "
                    "masked always-active path. Extend _EasyEPCollector if you need this."
                )
            h_pre = layer.mlp.register_forward_pre_hook(self._make_pre_hook(layer_idx))
            h_post = layer.mlp.register_forward_hook(self._make_post_hook(layer_idx, layer.mlp))
            self._hook_handles.extend([h_pre, h_post])

    def detach(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def easy_ep_prune(
    model_name: str,
    task_name: str,
    split: str,
    prune_keep_k: int,
    num_shared_experts: int,
    save_path: str,
    batch_size: int = 8,
    num_calibration: Optional[int] = None,
    max_length: int = 4096,
    device: Optional[str] = None,
    num_shots_override: Optional[int] = None,
    prune_seed: int = 0,
) -> None:
    """
    One-shot EASY-EP pruning: collect scores with a single forward pass per
    calibration batch, pick top-k per layer, save pruned model.
    """
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

    # Disable router-logit LB loss path (same reason as greedy_prune_layerwise).
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = False

    # --- Calibration data -----------------------------------------------------
    logger.info(
        f"Loading calibration data: {task_name} ({split})"
        + (f" [num_shots={num_shots_override}]" if num_shots_override is not None else "")
    )
    prompts, _ = get_formatted_prompts(task_name, split, num_shots_override=num_shots_override)
    if num_calibration is None:
        logger.info(f"Loaded {len(prompts)} prompts, using all (no subsampling)")
    else:
        n_keep = min(num_calibration, len(prompts))
        logger.info(f"Loaded {len(prompts)} prompts, subsampling to {n_keep} (seed={prune_seed})")
        # Deterministic subsample: take the first num_calibration after a fixed
        # permutation (avoids always hitting the same few examples for different
        # task sizes). Seed defaults to 0 to preserve historical behavior.
        g = torch.Generator().manual_seed(prune_seed)
        perm = torch.randperm(len(prompts), generator=g).tolist()
        prompts = [prompts[i] for i in perm[:n_keep]]

    logger.info(f"Tokenizing {len(prompts)} calibration prompts")
    all_batches = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        all_batches.append(inputs)
    logger.info(f"Created {len(all_batches)} batches")

    # --- Gather per-layer expert counts --------------------------------------
    num_experts_per_layer: List[int] = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            num_experts_per_layer.append(layer.mlp.num_experts)
        else:
            num_experts_per_layer.append(0)

    # --- Attach hooks & collect scores ---------------------------------------
    collector = _EasyEPCollector(model, num_layers, num_experts_per_layer)
    collector.attach()
    try:
        for batch in tqdm(all_batches, desc="EASY-EP score collection"):
            batch_on_device = {k: v.to(model.device) for k, v in batch.items()}
            collector.attn_mask = batch["attention_mask"]
            with torch.no_grad():
                model(**batch_on_device)
    finally:
        collector.detach()

    # --- Select top-k per layer ----------------------------------------------
    experts_kept_per_layer: List[Optional[List[int]]] = []
    for layer_idx in range(num_layers):
        scores = collector.scores[layer_idx]
        if scores is None:
            experts_kept_per_layer.append(None)
            continue

        layer = model.model.layers[layer_idx]
        num_experts = layer.mlp.num_experts
        num_shared_current = layer.mlp.num_shared_experts
        num_standard = num_experts - num_shared_current

        if num_shared_current > 0:
            scores_standard = scores[:num_standard]
            scores_shared = scores[num_standard:]
            keep_standard = sorted(
                torch.topk(
                    scores_standard,
                    min(prune_keep_k - num_shared_experts, num_standard),
                ).indices.tolist()
            )
            keep_shared_local = sorted(
                torch.topk(
                    scores_shared,
                    min(num_shared_experts, num_shared_current),
                ).indices.tolist()
            )
            keep_shared = [num_standard + i for i in keep_shared_local]
            experts_to_keep = keep_standard + keep_shared
        else:
            experts_to_keep = sorted(
                torch.topk(scores, min(prune_keep_k, num_experts)).indices.tolist()
            )

        logger.info(
            f"Layer {layer_idx}: keeping {len(experts_to_keep)} experts "
            f"(top score {scores.max().item():.4g}, min kept "
            f"{scores[experts_to_keep].min().item():.4g})"
        )
        logger.info(f"Layer {layer_idx}: keeping experts {experts_to_keep}")
        experts_kept_per_layer.append(experts_to_keep)

    # --- Prune in-place and save ---------------------------------------------
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

    # Update global config
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
        "pruning_method": "easy_ep",
        "task": task_name,
        "split": split,
        "num_calibration": num_calibration,
        "num_shots_override": num_shots_override,
        "prune_seed": prune_seed,
        "experts_kept_per_layer": experts_kept_per_layer,
    }
    with open(os.path.join(save_path, "pruning_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done. Pruned model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="EASY-EP expert pruning for HuggingFace MoE models"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prune-keep-k", type=int, required=True)
    parser.add_argument("--num-shared-experts", type=int, default=0)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-calibration", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", type=str, default=None)
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

    easy_ep_prune(
        model_name=args.model,
        task_name=args.task,
        split=args.split,
        prune_keep_k=args.prune_keep_k,
        num_shared_experts=args.num_shared_experts,
        save_path=args.save_path,
        batch_size=args.batch_size,
        num_calibration=args.num_calibration,
        max_length=args.max_length,
        device=args.device,
        num_shots_override=args.num_shots,
        prune_seed=args.prune_seed,
    )


if __name__ == "__main__":
    main()
