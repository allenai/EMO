import logging
from typing import Any, Dict, List, Optional

from transformers import Olmo2Config, Olmo2NoQKNormPrenormConfig, PretrainedConfig

from olmo_core.doc_utils import beta_feature
from olmo_core.nn.attention import Attention
from olmo_core.nn.attention.recurrent import GatedDeltaNet
from olmo_core.nn.moe.mlp import DroplessMoEMLP, MoEMLP
from olmo_core.nn.moe.router_lbreducedp_sharedexp import (
    MoELinearLBReduceDPSharedExpRouter,
)
from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexp_randpool_router import (
    MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter,
)
from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexp_router import (
    MoETwoLevelBatchLBReduceDPSharedExpRouter,
)
from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexppool_router import (
    MoETwoLevelBatchLBReduceDPSharedExpPoolRouter,
)
from olmo_core.nn.rope import RoPEScalingConfig
from olmo_core.nn.transformer.block import (
    MoEReorderedNormTransformerBlock,
    MoETransformerBlock,
    ReorderedNormTransformerBlock,
    TransformerBlock,
)
from olmo_core.nn.transformer.model import (
    MoETransformer,
    NormalizedTransformer,
    Transformer,
)

log = logging.getLogger(__name__)

try:
    from transformers import EmoConfig, EmoSharedConfig
except ImportError:
    EmoConfig = None
    EmoSharedConfig = None

try:
    from transformers import Olmo3Config  # type: ignore
except ImportError:
    Olmo3Config = None


def _get_emo_config(model: MoETransformer) -> PretrainedConfig:
    blocks = list(model.blocks.values())
    for block in blocks:
        # Dense TransformerBlock (e.g., densefirst layers 0-1): validate attention only
        if isinstance(block, TransformerBlock) and not isinstance(
            block, (MoETransformerBlock, MoEReorderedNormTransformerBlock)
        ):
            if not isinstance(block.attention, Attention):
                raise NotImplementedError(
                    f"Attention is not a {Attention.__name__}, unable to build HF config for {model.__class__.__name__}"
                )
            if block.attention.rope is None:
                raise NotImplementedError(
                    f"Attention does not use rope, unable to build HF config for {model.__class__.__name__}"
                )
            continue

        # MoE block validation: only MoETransformerBlock with q_norm=None and k_norm=None is supported.
        if not isinstance(block, MoETransformerBlock) or isinstance(
            block, MoEReorderedNormTransformerBlock
        ):
            raise NotImplementedError(
                f"Block is not a plain {MoETransformerBlock.__name__}, "
                f"unable to build HF config for {model.__class__.__name__}"
            )
        if block.attention.q_norm is not None or block.attention.k_norm is not None:
            raise NotImplementedError(
                f"Block has q_norm/k_norm; only no-q_norm/no-k_norm is supported for Emo, "
                f"unable to build HF config for {model.__class__.__name__}"
            )

        if not isinstance(block.experts.mlp, (DroplessMoEMLP, MoEMLP)):
            raise NotImplementedError(
                f"MoE mlp is not a {DroplessMoEMLP.__name__} or {MoEMLP.__name__}, unable to build HF config for {model.__class__.__name__}"
            )

        if not isinstance(block.attention, Attention):
            raise NotImplementedError(
                f"Attention is not a {Attention.__name__}, unable to build HF config for {model.__class__.__name__}"
            )
        if block.attention.rope is None:
            raise NotImplementedError(
                f"Attention does not use rope, unable to build HF config for {model.__class__.__name__}"
            )

    # Find the first MoE block (may not be blocks[0] for densefirst models)
    first_moe_block = None
    for block in blocks:
        if isinstance(block, (MoEReorderedNormTransformerBlock, MoETransformerBlock)):
            first_moe_block = block
            break
    if first_moe_block is None:
        raise NotImplementedError(
            f"No MoE block found in model, unable to build HF config for {model.__class__.__name__}"
        )

    block = first_moe_block
    assert isinstance(block, (MoEReorderedNormTransformerBlock, MoETransformerBlock))
    assert isinstance(block.attention, Attention)
    assert block.attention.rope is not None
    assert block.feed_forward_moe is not None

    if EmoConfig is None:
        raise RuntimeError("The installed transformers version does not support Emo")

    # Build per-layer metadata for dense/MoE mixed models
    dense_intermediate_size = None
    num_experts_per_layer = []
    num_shared_experts_per_layer = []
    has_dense_layers = False

    dense_mlp_bias = False

    for b in blocks:
        if isinstance(b, TransformerBlock) and not isinstance(
            b, (MoETransformerBlock, MoEReorderedNormTransformerBlock)
        ):
            # Dense layer
            has_dense_layers = True
            num_experts_per_layer.append(0)
            num_shared_experts_per_layer.append(0)
            if dense_intermediate_size is None:
                dense_intermediate_size = b.feed_forward.hidden_size
                dense_mlp_bias = b.feed_forward.w1.bias is not None
        else:
            # MoE layer
            num_experts_per_layer.append(b.feed_forward_moe.router.num_experts)
            layer_shared = 0
            if isinstance(b.feed_forward_moe.router, MoETwoLevelBatchLBReduceDPSharedExpPoolRouter):
                layer_shared = b.feed_forward_moe.router.num_shared_experts_pool
            elif isinstance(
                b.feed_forward_moe.router,
                (
                    MoETwoLevelBatchLBReduceDPSharedExpRouter,
                    MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter,
                    MoELinearLBReduceDPSharedExpRouter,
                ),
            ):
                layer_shared = b.feed_forward_moe.router.num_shared_experts
            num_shared_experts_per_layer.append(layer_shared)

    shared_mlp = block.feed_forward_moe.shared_mlp
    if shared_mlp is not None:
        if EmoSharedConfig is None:
            raise RuntimeError("The installed transformers version does not support EmoShared")
        return EmoSharedConfig(
            vocab_size=model.vocab_size,
            hidden_size=model.d_model,
            intermediate_size=block.feed_forward_moe.experts.mlp.hidden_size,
            shared_expert_intermediate_size=shared_mlp.hidden_size,
            num_hidden_layers=model.n_layers,
            num_attention_heads=block.attention.n_heads,
            num_key_value_heads=block.attention.n_kv_heads,
            hidden_act="silu",
            max_position_embeddings=-1,
            attention_bias=block.attention.w_out.bias is not None,
            rope_theta=block.attention.rope.theta,
            pad_token_id=None,  # type: ignore
            bos_token_id=None,
            eos_token_id=None,  # type: ignore
            rms_norm_eps=block.feed_forward_norm.eps,
            num_experts_per_tok=block.feed_forward_moe.router.top_k,
            num_experts=block.feed_forward_moe.router.num_experts,
            tie_word_embeddings=False,
        )
    always_active_experts = getattr(block.feed_forward_moe.router, "always_active_experts", None)
    # find the right number of shared experts accordingly
    num_shared_experts = 0
    if isinstance(block.feed_forward_moe.router, MoETwoLevelBatchLBReduceDPSharedExpPoolRouter):
        num_shared_experts = block.feed_forward_moe.router.num_shared_experts_pool
    elif isinstance(
        block.feed_forward_moe.router,
        (
            MoETwoLevelBatchLBReduceDPSharedExpRouter,
            MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter,
            MoELinearLBReduceDPSharedExpRouter,
        ),
    ):
        num_shared_experts = block.feed_forward_moe.router.num_shared_experts
    return EmoConfig(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        intermediate_size=block.feed_forward_moe.experts.mlp.hidden_size,
        num_hidden_layers=model.n_layers,
        num_attention_heads=block.attention.n_heads,
        num_key_value_heads=block.attention.n_kv_heads,
        hidden_act="silu",
        max_position_embeddings=-1,
        attention_bias=block.attention.w_out.bias is not None,
        rope_theta=block.attention.rope.theta,
        pad_token_id=None,  # type: ignore
        bos_token_id=None,
        eos_token_id=None,  # type: ignore
        rms_norm_eps=block.feed_forward_norm.eps,
        num_experts_per_tok=block.feed_forward_moe.router.top_k,
        num_experts=block.feed_forward_moe.router.num_experts,
        tie_word_embeddings=False,
        always_active_experts=always_active_experts,
        num_shared_experts=num_shared_experts,
        num_experts_per_layer=num_experts_per_layer if has_dense_layers else None,
        num_shared_experts_per_layer=num_shared_experts_per_layer if has_dense_layers else None,
        dense_intermediate_size=dense_intermediate_size,
        dense_mlp_bias=dense_mlp_bias,
        output_router_logits=True,
    )


@beta_feature
def get_hf_config(model: Transformer) -> PretrainedConfig:
    if isinstance(model, NormalizedTransformer):
        raise NotImplementedError(
            f"Building HF config not implemented for {model.__class__.__name__}"
        )

    if isinstance(model, MoETransformer):
        return _get_emo_config(model)

    blocks = list(model.blocks.values())
    first_block = blocks[0]
    if not isinstance(first_block, ReorderedNormTransformerBlock):
        # we support case where we use prenorm and no q/k norm
        if (
            isinstance(first_block, TransformerBlock)
            and first_block.attention.q_norm is None
            and first_block.attention.k_norm is None
        ):
            pass
        else:
            raise NotImplementedError(
                f"Block is not a {ReorderedNormTransformerBlock.__name__}, unable to build HF config for {model.__class__.__name__}"
            )

    if not isinstance(first_block.attention, Attention):
        raise NotImplementedError(
            f"Attention is not a {Attention.__name__}, unable to build HF config for {model.__class__.__name__}"
        )
    if first_block.attention.backend is None:
        raise ValueError("Attention backend is not set.")

    has_rope = first_block.attention.rope is not None

    if has_rope:
        rope_scaling = _get_and_validate_rope_scaling_config(blocks)
        rope_theta = first_block.attention.rope.theta
    else:
        rope_scaling = None
        rope_theta = None

    # Extract common configuration parameters
    common_config_args = {
        "vocab_size": model.vocab_size,
        "hidden_size": model.d_model,
        "intermediate_size": first_block.feed_forward.hidden_size,
        "num_hidden_layers": model.n_layers,
        "num_attention_heads": first_block.attention.n_heads,
        "num_key_value_heads": first_block.attention.n_kv_heads,
        "hidden_act": "silu",
        "max_position_embeddings": -1,
        "attention_bias": first_block.attention.w_out.bias is not None,
        "rope_theta": rope_theta,
        "rope_scaling": rope_scaling,
        "pad_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
        "rms_norm_eps": first_block.feed_forward_norm.eps,
        "tie_word_embeddings": False,
    }

    # The OLMo 3 model family is identical to the OLMo 2 model family, except:
    # - Sliding window attention is used for 3 out of 4 layers.
    # - RoPE scaling is not applied to sliding window attention layers.
    # Therefore, if any layer uses sliding window attention, we assume the model is OLMo 3.
    # Identify layers that use sliding window attention.
    sliding_window_blocks = [
        block for block in blocks if block.attention.backend.window_size != (-1, -1)
    ]

    if sliding_window_blocks:
        if Olmo3Config is None:
            raise RuntimeError("The installed transformers version does not support Olmo3")

        found_window_sizes = {
            block.attention.backend.window_size[0] for block in sliding_window_blocks
        }

        if len(found_window_sizes) > 1:
            raise ValueError(
                "All sliding window attention layers must have the same window size for "
                f"OLMo3Config. Found different window sizes: {found_window_sizes}."
            )

        # This sliding window sizes value is configured to be fed to flash_attention -
        # it is one smaller than the actual window size because FA implicitly includes the
        # current position in the window. HF expects a value one larger than this and will
        # manually adjust the window size down by 1 for FA.
        # See https://github.com/huggingface/transformers/pull/40163
        common_window_size_value = found_window_sizes.pop()

        olmo3_specific_args = {
            "sliding_window": common_window_size_value + 1,
            "layer_types": [
                "sliding_attention"
                if block.attention.backend.window_size != (-1, -1)
                else "full_attention"
                for block in blocks
            ],
        }
        return Olmo3Config(**common_config_args, **olmo3_specific_args)
    else:
        if (
            isinstance(first_block, TransformerBlock)
            and first_block.attention.q_norm is None
            and first_block.attention.k_norm is None
        ):
            return Olmo2NoQKNormPrenormConfig(**common_config_args)
        else:
            return Olmo2Config(**common_config_args)


def _get_and_validate_rope_scaling_config(blocks) -> dict | None:
    """
    Validate RoPE scaling configuration across transformer blocks.

    :param blocks: The list of transformer blocks to validate.
    :returns: The validated RoPE scaling config dict for HF, or None if no scaling.
    :raises NotImplementedError: If RoPE scaling is applied to sliding window layers or if
                               full attention layers have different RoPE scaling configs.
    """
    # Separate full attention layers from sliding window layers
    full_attention_layers = [
        (idx, block)
        for idx, block in enumerate(blocks)
        if block.attention.backend.window_size == (-1, -1)
    ]
    sliding_window_layers = [
        (idx, block)
        for idx, block in enumerate(blocks)
        if block.attention.backend.window_size != (-1, -1)
    ]

    # Check for RoPE scaling on sliding window layers (not allowed)
    sliding_with_scaling = [
        (idx, block)
        for idx, block in sliding_window_layers
        if block.attention.rope is not None and block.attention.rope.scaling is not None
    ]
    if sliding_with_scaling:
        sliding_indices = [idx for idx, _ in sliding_with_scaling]
        raise NotImplementedError(
            f"RoPE scaling is configured on sliding window attention layers {sliding_indices}, "
            f"but HuggingFace only supports RoPE scaling on full attention layers. "
            f"Please remove RoPE scaling from sliding window layers or convert them to full attention."
        )

    # Collect RoPE scaling configs from full attention layers only
    full_layers_with_scaling = [
        (idx, block)
        for idx, block in full_attention_layers
        if block.attention.rope is not None and block.attention.rope.scaling is not None
    ]
    if not full_layers_with_scaling:
        return None

    rope_scaling_configs: list[RoPEScalingConfig] = [
        block.attention.rope.scaling for _, block in full_layers_with_scaling
    ]

    # Validate that all full attention layers with RoPE scaling use the same configuration
    first_config = rope_scaling_configs[0]
    first_config_dict = first_config.to_hf_config()

    for i, rope_config in enumerate(rope_scaling_configs[1:], 1):
        config_dict = rope_config.to_hf_config()
        if config_dict != first_config_dict:
            scaling_indices = [idx for idx, _ in full_layers_with_scaling]
            raise NotImplementedError(
                f"Full attention layers have different RoPE scaling configurations but HuggingFace "
                "only supports a single RoPE scaling configuration per model. "
                f"Full attention layers with scaling: {scaling_indices}. "
                f"First config: {first_config_dict}, Different config at layer {i}: {config_dict}"
            )

    return first_config_dict


# ---------------------------------------------------------------------------
# Hybrid model helpers
# ---------------------------------------------------------------------------


@beta_feature
def is_olmo_hybrid_model(model: Transformer) -> bool:
    """Return ``True`` if the model has both :class:`GatedDeltaNet` and :class:`Attention` layers."""
    has_gdn = False
    has_attn = False
    for block in model.blocks.values():
        if isinstance(block.attention, GatedDeltaNet):
            has_gdn = True
        elif isinstance(block.attention, Attention):
            has_attn = True
        if has_gdn and has_attn:
            return True
    return False


@beta_feature
def get_hybrid_layer_types(model: Transformer) -> List[str]:
    """
    Return a per-layer type list for a hybrid model.

    Each entry is ``"linear_attention"`` (GDN) or ``"full_attention"`` (standard attention),
    matching the HF ``olmo_hybrid`` config format.
    """
    layer_types: List[str] = []
    for idx, block in model.blocks.items():
        if isinstance(block.attention, GatedDeltaNet):
            layer_types.append("linear_attention")
        elif isinstance(block.attention, Attention):
            layer_types.append("full_attention")
        else:
            raise ValueError(f"Unknown sequence mixer type at layer {idx}: {type(block.attention)}")
    return layer_types


def _get_hybrid_rope_scaling(model: Transformer, layer_types: List[str]) -> Optional[dict]:
    """
    Extract the RoPE scaling config from attention blocks.  GDN layers are skipped
    because they don't use RoPE.
    """
    attn_blocks = [
        (int(idx), block)
        for idx, block in model.blocks.items()
        if layer_types[int(idx)] == "full_attention"
    ]

    layers_with_scaling = [
        (idx, block)
        for idx, block in attn_blocks
        if block.attention.rope is not None and block.attention.rope.scaling is not None
    ]
    if not layers_with_scaling:
        return None

    first_config = layers_with_scaling[0][1].attention.rope.scaling.to_hf_config()
    for idx, block in layers_with_scaling[1:]:
        cfg = block.attention.rope.scaling.to_hf_config()
        if cfg != first_config:
            raise NotImplementedError(
                f"Inconsistent RoPE scaling configs. First: {first_config}, Layer {idx}: {cfg}"
            )
    return first_config


@beta_feature
def get_hybrid_hf_config(
    model: Transformer,
    layer_types: List[str],
    max_seq_len: int = 65536,
) -> Dict[str, Any]:
    """
    Build the ``config.json`` dict for a HF ``olmo_hybrid`` model.

    Returns a plain dict (not :class:`PretrainedConfig`) to avoid a hard dependency
    on a specific ``transformers`` version.

    :param model: The OLMo-core hybrid transformer model.
    :param layer_types: Per-layer type list from :func:`get_hybrid_layer_types`.
    :param max_seq_len: Maximum sequence length for ``max_position_embeddings``.
    """
    blocks = list(model.blocks.values())

    attn_block: Optional[TransformerBlock] = None
    gdn_block: Optional[TransformerBlock] = None
    for lt, block in zip(layer_types, blocks):
        if lt == "full_attention" and attn_block is None:
            attn_block = block
        elif lt == "linear_attention" and gdn_block is None:
            gdn_block = block

    if attn_block is None:
        raise ValueError("Hybrid model must have at least one attention layer")
    if gdn_block is None:
        raise ValueError("Hybrid model must have at least one GDN layer")

    attn: Attention = attn_block.attention
    gdn: GatedDeltaNet = gdn_block.attention

    # RoPE (from attention blocks only)
    rope_parameters: Optional[dict] = None
    if attn.rope is not None:
        rope_theta = float(attn.rope.theta)
        rope_scaling = _get_hybrid_rope_scaling(model, layer_types)
        rope_parameters = {"rope_theta": rope_theta}
        if rope_scaling:
            rope_parameters.update(rope_scaling)
        else:
            rope_parameters["rope_type"] = "default"
        log.info(f"RoPE: {rope_parameters}")
    else:
        log.info("No RoPE configured")

    # Warn if GDN blocks are post-norm but HF expects pre-norm.
    if isinstance(gdn_block, ReorderedNormTransformerBlock):
        log.warning(
            "GDN block uses post-norm (ReorderedNormTransformerBlock) but HF olmo_hybrid "
            "expects pre-norm for linear_attention layers. The conversion will proceed, but "
            "outputs may not match exactly."
        )

    config: Dict[str, Any] = {
        "model_type": "olmo_hybrid",
        "architectures": ["OlmoHybridForCausalLM"],
        # Standard transformer fields
        "vocab_size": model.vocab_size,
        "hidden_size": model.d_model,
        "intermediate_size": attn_block.feed_forward.hidden_size,
        "num_hidden_layers": len(blocks),
        "num_attention_heads": attn.n_heads,
        "num_key_value_heads": attn.n_kv_heads,
        "hidden_act": "silu",
        "max_position_embeddings": max_seq_len,
        "initializer_range": 0.02,
        "use_cache": True,
        "attention_bias": attn.w_out.bias is not None,
        "attention_dropout": 0.0,
        "rms_norm_eps": attn_block.feed_forward_norm.eps,  # todo: revisit
        "tie_word_embeddings": False,
        # Hybrid layer configuration
        "layer_types": layer_types,
        # GDN (linear attention) parameters
        "linear_num_key_heads": gdn.n_heads,
        "linear_num_value_heads": gdn.n_v_heads,
        "linear_key_head_dim": gdn.head_k_dim,
        "linear_value_head_dim": gdn.head_v_dim,
        "linear_conv_kernel_dim": gdn.conv_size,
        "linear_allow_neg_eigval": gdn.allow_neg_eigval,
        # Token IDs (updated later after tokenizer is saved)
        "pad_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
    }

    if rope_parameters is not None:
        config["rope_parameters"] = rope_parameters
    else:
        config["rope_theta"] = None

    return config
