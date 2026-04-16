from olmo_core.doc_utils import beta_feature
from olmo_core.nn.attention import Attention
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
from transformers import Olmo2Config, Olmo2NoQKNormPrenormConfig, PretrainedConfig

try:
    from transformers import (  # type: ignore
        FlexOlmoConfig,
        FlexOlmoNoQKNormPrenormConfig,
        FlexOlmoNoQKNormPrenormSharedConfig,
        FlexOlmoPrenormConfig,
    )
except ImportError:
    FlexOlmoConfig = None
    FlexOlmoNoQKNormPrenormSharedConfig = None

try:
    from transformers import Olmo3Config  # type: ignore
except ImportError:
    Olmo3Config = None


def _get_flex_olmo_config(model: MoETransformer) -> PretrainedConfig:
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

        # MoE block validation
        if not isinstance(block, MoEReorderedNormTransformerBlock):
            if (
                isinstance(block, MoETransformerBlock)
                and block.attention.q_norm is None
                and block.attention.k_norm is None
            ):
                pass
            elif (
                isinstance(block, MoETransformerBlock)
                and block.attention.q_norm is not None
                and block.attention.k_norm is not None
            ):
                pass
            else:
                raise NotImplementedError(
                    f"Block is not a {MoEReorderedNormTransformerBlock.__name__}, unable to build HF config for {model.__class__.__name__}"
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

    if FlexOlmoConfig is None:
        raise RuntimeError("The installed transformers version does not support FlexOlmo")

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

    if (
        isinstance(block, MoETransformerBlock)
        and block.attention.q_norm is None
        and block.attention.k_norm is None
    ):
        has_shared_mlp = block.feed_forward_moe.shared_mlp is not None
        if has_shared_mlp:
            if FlexOlmoNoQKNormPrenormSharedConfig is None:
                raise RuntimeError(
                    "The installed transformers version does not support FlexOlmoNoQKNormPrenormShared"
                )
            return FlexOlmoNoQKNormPrenormSharedConfig(
                vocab_size=model.vocab_size,
                hidden_size=model.d_model,
                intermediate_size=block.feed_forward_moe.experts.mlp.hidden_size,
                shared_expert_intermediate_size=block.feed_forward_moe.shared_mlp.hidden_size,
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
        always_active_experts = getattr(
            block.feed_forward_moe.router, "always_active_experts", None
        )
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
        return FlexOlmoNoQKNormPrenormConfig(
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
    elif (
        isinstance(block, MoETransformerBlock)
        and block.attention.q_norm is not None
        and block.attention.k_norm is not None
    ):
        return FlexOlmoPrenormConfig(
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
            output_router_logits=True,
        )
    return FlexOlmoConfig(
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
        output_router_logits=True,
    )


@beta_feature
def get_hf_config(model: Transformer) -> PretrainedConfig:
    if isinstance(model, NormalizedTransformer):
        raise NotImplementedError(
            f"Building HF config not implemented for {model.__class__.__name__}"
        )

    if isinstance(model, MoETransformer):
        return _get_flex_olmo_config(model)

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
    if first_block.attention.rope is None:
        raise NotImplementedError(
            f"Attention does not use rope, unable to build HF config for {model.__class__.__name__}"
        )

    if first_block.attention.backend is None:
        raise ValueError("Attention backend is not set.")

    rope_scaling = _get_and_validate_rope_scaling_config(blocks)

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
        "rope_theta": first_block.attention.rope.theta,
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
        if block.attention.rope.scaling is not None
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
        if block.attention.rope.scaling is not None
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
