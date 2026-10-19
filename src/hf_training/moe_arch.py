"""Architecture dispatch for expert selection on HuggingFace MoE checkpoints.

The selection code was written against the EMO / OLMoE module layout:

* ``layer.mlp`` is a sparse-MoE block whose forward returns ``(hidden, router_logits)``,
* ``layer.mlp.gate`` is the router projection, ``layer.mlp.experts`` a ``ModuleList``,
* ``layer.mlp.num_experts`` / ``num_shared_experts`` are plain attributes, and
* the config carries ``num_experts`` (+ optional per-layer lists).

The OLMo 3.5 hero exports (``model_type == "olmo3moe"``, ``Olmo3MoeForCausalLM`` remote code)
differ: the router is ``layer.mlp.router`` (its projection ``router.gate``), the routed
experts are ``layer.mlp.experts`` (a ``ModuleList`` subclass with its own fused forward), a
separate ``layer.mlp.shared_expert`` is always active and is not part of the routed pool, and
the config field is ``n_routed_experts`` plus the EMO document-pool sizes.  The MoE block
returns a plain tensor, so router probabilities are captured from the router module instead.

Every helper below takes the decoder layer (or model) and dispatches on the block's layout, so
the greedy layerwise selector stays byte-for-byte the same for EMO models.
"""

from __future__ import annotations

import os
import shutil
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

OLMO3MOE_CODE_FILES = ("configuration_olmo3moe.py", "modeling_olmo3moe.py")


def is_olmo3moe_block(moe) -> bool:
    return hasattr(moe, "router") and hasattr(getattr(moe, "router"), "gate") and hasattr(moe, "experts")


def is_moe_layer(layer) -> bool:
    return hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")


def expert_counts(layer) -> Tuple[int, int]:
    """``(num_experts, num_shared_experts)`` of the routed pool of ``layer``."""
    moe = layer.mlp
    if is_olmo3moe_block(moe):
        # The shared expert is outside the routed pool and is never pruned.
        return len(moe.experts), 0
    return moe.num_experts, moe.num_shared_experts


def register_router_logit_capture(layer, store: List[torch.Tensor]):
    """Append the router logits of every forward pass through ``layer.mlp`` to ``store``.

    Logits are ``(batch * seq, num_experts)`` on CPU, before the softmax, in both layouts.
    Returns the hook handle.
    """
    moe = layer.mlp
    if is_olmo3moe_block(moe):
        router = moe.router

        def pre_hook(module, inputs):
            x = inputs[0]
            # The router computes its logits in float32 from bf16 weights; do the same.
            logits = F.linear(x.float(), module.gate.weight.float())
            store.append(logits.reshape(-1, logits.shape[-1]).detach().cpu())

        return router.register_forward_pre_hook(pre_hook)

    def hook(module, input, output):
        # SparseMoeBlock returns (final_hidden_states, router_logits)
        store.append(output[1].detach().cpu())

    return moe.register_forward_hook(hook)


def prune_layer_inplace(layer, experts_to_keep: List[int], prune_keep_k: int, target_num_shared_experts: int) -> None:
    """Keep only ``experts_to_keep`` in ``layer.mlp`` (experts and router rows), in place."""
    moe = layer.mlp
    if is_olmo3moe_block(moe):
        kept = type(moe.experts)()  # same ModuleList subclass, so its fused forward is kept
        for i in experts_to_keep:
            kept.append(moe.experts[i])
        moe.experts = kept
        router = moe.router
        router.gate.weight = torch.nn.Parameter(router.gate.weight.data[experts_to_keep, :].clone())
        if getattr(router.gate, "bias", None) is not None:
            router.gate.bias = torch.nn.Parameter(router.gate.bias.data[experts_to_keep].clone())
        router.gate.out_features = len(experts_to_keep)
        router.n_routed_experts = len(experts_to_keep)
        if getattr(router, "allowed_experts", None) is not None:
            router.allowed_experts = None
        if router.num_experts_per_tok > len(experts_to_keep):
            router.num_experts_per_tok = len(experts_to_keep)
        return

    moe.experts = torch.nn.ModuleList([moe.experts[i] for i in experts_to_keep])
    moe.gate.weight = torch.nn.Parameter(moe.gate.weight.data[experts_to_keep, :])
    if moe.gate.bias is not None:
        moe.gate.bias = torch.nn.Parameter(moe.gate.bias.data[experts_to_keep])
    moe.gate.out_features = prune_keep_k
    moe.num_experts = prune_keep_k
    moe.num_shared_experts = target_num_shared_experts
    moe.top_k = min(moe.top_k, prune_keep_k)


def model_is_olmo3moe(model) -> bool:
    return getattr(model.config, "model_type", "") == "olmo3moe" or any(
        is_moe_layer(l) and is_olmo3moe_block(l.mlp) for l in model.model.layers
    )


def finalize_pruned_config(model, prune_keep_k: int, num_shared_experts: int) -> None:
    """Make ``model.config`` describe the pruned expert counts before ``save_pretrained``."""
    config = model.config
    if model_is_olmo3moe(model):
        config.n_routed_experts = prune_keep_k
        if config.num_experts_per_tok > prune_keep_k:
            config.num_experts_per_tok = prune_keep_k
        # EMO document pools must not exceed the expert count (the vLLM plugin requires the
        # eval pool to be None or equal to it).
        for key in ("emo_eval_document_expert_pool", "emo_max_document_expert_pool", "emo_min_document_expert_pool"):
            value = getattr(config, key, None)
            if value is not None:
                setattr(config, key, min(int(value), prune_keep_k))
        return

    if hasattr(config, "num_experts_per_tok") and config.num_experts_per_tok > prune_keep_k:
        config.num_experts_per_tok = prune_keep_k
    config.num_experts = prune_keep_k
    if hasattr(config, "num_local_experts"):
        config.num_local_experts = prune_keep_k
    config.num_shared_experts = num_shared_experts
    if getattr(config, "num_experts_per_layer", None) is not None:
        config.num_experts_per_layer = [prune_keep_k if n > 0 else 0 for n in config.num_experts_per_layer]
    if getattr(config, "num_shared_experts_per_layer", None) is not None:
        config.num_shared_experts_per_layer = [
            num_shared_experts if n > 0 else 0 for n in config.num_shared_experts_per_layer
        ]
    # Enable router logit output so load-balancing loss is active during finetuning
    config.output_router_logits = True


def copy_remote_code(source_model: str, save_path: str) -> List[str]:
    """Copy the checkpoint's ``trust_remote_code`` files next to a saved pruned model.

    ``save_pretrained`` writes the ``auto_map`` but not always the modules it points to.
    Returns the files copied.  No-op when ``source_model`` is not a local directory.
    """
    copied: List[str] = []
    if not os.path.isdir(source_model):
        return copied
    for name in os.listdir(source_model):
        if not name.endswith(".py"):
            continue
        dst = os.path.join(save_path, name)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(source_model, name), dst)
            copied.append(name)
    return copied


def make_unpadded_batches(tokenizer, prompts: List[str], batch_size: int, max_length: int = 4096):
    """Tokenize ``prompts`` into batches that need no padding.

    The OLMo 3.5 exports' KDA layers reject padded inputs, so prompts are grouped by token
    length (preserving nothing about order: selection statistics are order-independent) and
    each group is split into chunks of at most ``batch_size``.  Every batch carries an
    all-ones ``attention_mask`` so the caller's masking arithmetic is unchanged.
    """
    enc = tokenizer(prompts, truncation=True, max_length=max_length, padding=False)
    by_len = {}
    for ids in enc["input_ids"]:
        by_len.setdefault(len(ids), []).append(ids)
    batches = []
    for length in sorted(by_len):
        rows = by_len[length]
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            input_ids = torch.tensor(chunk, dtype=torch.long)
            batches.append({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})
    return batches
