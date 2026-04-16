from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Cache, FlexOlmoNoQKNormPrenormConfig
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.models.flex_olmo_noqknorm_prenorm.modeling_flex_olmo_noqknorm_prenorm import (
    FlexOlmoNoQKNormPrenormAttention,
    FlexOlmoNoQKNormPrenormDecoderLayer,
    FlexOlmoNoQKNormPrenormMLP,
    FlexOlmoNoQKNormPrenormPreTrainedModel,
    FlexOlmoNoQKNormPrenormRMSNorm,
    FlexOlmoNoQKNormPrenormRotaryEmbedding,
    MoeModelOutputWithPast,
)
from transformers.utils import ModelOutput


class FlexOlmoNoQKNormPrenormConfigDebug(FlexOlmoNoQKNormPrenormConfig):
    """
    Config class that supports per-layer expert counts.

    Adds two new attributes:
    - num_experts_per_layer: List[int] of expert counts per layer
    - num_shared_experts_per_layer: List[int] of shared expert counts per layer

    These are optional. If not set, falls back to num_experts/num_shared_experts for all layers.
    """

    def __init__(
        self,
        num_experts_per_layer: Optional[List[int]] = None,
        num_shared_experts_per_layer: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts_per_layer = num_experts_per_layer
        self.num_shared_experts_per_layer = num_shared_experts_per_layer


@dataclass
class MoeCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) with mixture of experts outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).

        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.

        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.

        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    lb_loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None


class FlexOlmoNoQKNormPrenormSparseMoeBlockDebug(nn.Module):
    """
    MoE block that supports per-layer expert counts.

    This is identical to FlexOlmoNoQKNormPrenormSparseMoeBlock except it accepts
    num_experts and num_shared_experts as constructor arguments instead of reading
    from config, allowing different layers to have different expert counts.
    """

    def __init__(self, config, num_experts: int, num_shared_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FlexOlmoNoQKNormPrenormMLP(config) for _ in range(self.num_experts)]
        )

        self.num_shared_experts = num_shared_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        if self.num_shared_experts > 0:
            # split the router logits into shared and unshared experts
            router_logits_standard = router_logits[
                :, : -self.num_shared_experts
            ]  # (batch * sequence_length, n_experts - num_shared_experts)
            router_logits_shared = router_logits[
                :, -self.num_shared_experts :
            ]  # (batch * sequence_length, num_shared_experts)

            # compute the routing weights for the standard experts and shared experts separately
            routing_weights_standard = F.softmax(router_logits_standard, dim=1, dtype=torch.float)
            routing_weights_shared = F.softmax(router_logits_shared, dim=1, dtype=torch.float)

            # select the routing weights and experts for the standard experts and shared experts separately
            routing_weights_standard, selected_experts_standard = torch.topk(
                routing_weights_standard, self.top_k - self.num_shared_experts, dim=-1
            )
            routing_weights_shared, selected_experts_shared = torch.topk(
                routing_weights_shared, self.num_shared_experts, dim=-1
            )

            # concatenate the routing weights and selected experts for the standard experts and shared experts
            routing_weights = torch.cat([routing_weights_standard, routing_weights_shared], dim=1)
            selected_experts = torch.cat(
                [
                    selected_experts_standard,
                    selected_experts_shared + (self.num_experts - self.num_shared_experts),
                ],
                dim=1,
            )  # we need to add the offset to the selected experts for the shared experts since they are at the end of the router logits

            # make sure there are self.top_k experts selected in total
            assert (
                routing_weights.shape
                == selected_experts.shape
                == (batch_size * sequence_length, self.top_k)
            ), f"routing_weights and selected_experts should have the same shape of (batch_size * sequence_length, self.top_k), but got {routing_weights.shape} and {selected_experts.shape}"
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            if self.num_shared_experts > 0:
                raise NotImplementedError(
                    "norm_topk_prob is not implemented for the case where num_shared_experts > 0, but should be a simple change"
                )
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class FlexOlmoNoQKNormPrenormDecoderLayerDebug(nn.Module):
    """
    Decoder layer that supports per-layer expert counts.

    This is identical to FlexOlmoNoQKNormPrenormDecoderLayer except it accepts
    num_experts and num_shared_experts as constructor arguments.
    """

    def __init__(
        self,
        config: FlexOlmoNoQKNormPrenormConfig,
        layer_idx: int,
        num_experts: int,
        num_shared_experts: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = FlexOlmoNoQKNormPrenormAttention(config=config, layer_idx=layer_idx)

        self.mlp = FlexOlmoNoQKNormPrenormSparseMoeBlockDebug(
            config, num_experts, num_shared_experts
        )

        self.pre_attention_layernorm = FlexOlmoNoQKNormPrenormRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = FlexOlmoNoQKNormPrenormRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        # apply norm before attention
        hidden_states = self.pre_attention_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # apply norm before feedforward
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class FlexOlmoNoQKNormPrenormModelDebug(FlexOlmoNoQKNormPrenormPreTrainedModel):
    """
    Model that supports per-layer expert counts via config.num_experts_per_layer.

    If config.num_experts_per_layer is set (a list of ints), each layer will be
    initialized with that many experts. Otherwise, falls back to config.num_experts.
    Similarly for config.num_shared_experts_per_layer.
    """

    config_class = FlexOlmoNoQKNormPrenormConfigDebug

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Check if per-layer expert counts are specified
        num_experts_per_layer = getattr(config, "num_experts_per_layer", None)
        num_shared_experts_per_layer = getattr(config, "num_shared_experts_per_layer", None)

        if num_experts_per_layer is not None:
            # Use per-layer expert counts
            assert (
                len(num_experts_per_layer) == config.num_hidden_layers
            ), f"num_experts_per_layer has length {len(num_experts_per_layer)} but model has {config.num_hidden_layers} layers"
            if num_shared_experts_per_layer is None:
                # Default: use config.num_shared_experts for all layers, but cap at layer's num_experts
                num_shared_experts_per_layer = [
                    min(config.num_shared_experts, num_experts_per_layer[i])
                    for i in range(config.num_hidden_layers)
                ]
            self.layers = nn.ModuleList(
                [
                    FlexOlmoNoQKNormPrenormDecoderLayerDebug(
                        config,
                        layer_idx,
                        num_experts_per_layer[layer_idx],
                        num_shared_experts_per_layer[layer_idx],
                    )
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
        else:
            # Fall back to original behavior: all layers use config.num_experts
            self.layers = nn.ModuleList(
                [
                    FlexOlmoNoQKNormPrenormDecoderLayer(config, layer_idx)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )

        self.norm = FlexOlmoNoQKNormPrenormRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = FlexOlmoNoQKNormPrenormRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class FlexOlmoNoQKNormPrenormForCausalLMDebug(
    FlexOlmoNoQKNormPrenormPreTrainedModel, GenerationMixin
):
    """
    CausalLM that supports per-layer expert counts.

    Changes from FlexOlmoNoQKNormPrenormForCausalLM:
    1. Uses FlexOlmoNoQKNormPrenormModelDebug which supports num_experts_per_layer
    2. Uses load_balancing_loss_func_olmoe_variable which handles variable-size logits
    """

    _tied_weights_keys = ["lm_head.weight"]
    config_class = FlexOlmoNoQKNormPrenormConfigDebug

    def __init__(self, config):
        super().__init__(config)
        self.model = FlexOlmoNoQKNormPrenormModelDebug(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, MoeCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlexOlmoNoQKNormPrenormForCausalLM

        >>> model = FlexOlmoNoQKNormPrenormForCausalLM.from_pretrained("allenai/FlexOlmoNoQKNormPrenorm-1B-7B-0924")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/FlexOlmoNoQKNormPrenorm-1B-7B-0924")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'Hey, are you conscious? Can you talk to me?\nI'm not sure if you're conscious of this, but I'm'
        ```
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        ce_loss = None
        if labels is not None:
            ce_loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
            loss = ce_loss

        lb_loss = None

        if output_router_logits:
            # Get per-layer expert counts if available
            num_experts_per_layer = getattr(self.config, "num_experts_per_layer", None)
            num_shared_experts_per_layer = getattr(
                self.config, "num_shared_experts_per_layer", None
            )

            lb_loss = load_balancing_loss_func_olmoe_variable(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
                labels,
                num_shared_experts=self.config.num_shared_experts,
                num_experts_per_layer=num_experts_per_layer,
                num_shared_experts_per_layer=num_shared_experts_per_layer,
                **kwargs,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * lb_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (lb_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=lb_loss,
            lb_loss=lb_loss.detach().clone()
            if lb_loss is not None
            else None,  # for logging callback
            ce_loss=ce_loss.detach().clone()
            if ce_loss is not None
            else None,  # for logging callback
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


def load_balancing_loss_func_olmoe_variable(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    num_items_in_batch: Optional[
        torch.Tensor
    ] = None,  # the number of tokens within a global batch (including across dp ranks)
    ignore_index=-100,
    num_shared_experts=0,
    num_experts_per_layer: Optional[List[int]] = None,
    num_shared_experts_per_layer: Optional[List[int]] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    This version supports variable per-layer expert counts by computing the loss
    per-layer individually and averaging across layers.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts]. This has not been softmaxed yet.
            Note: each layer may have a different num_experts if num_experts_per_layer is set.
        num_experts:
            Number of experts (used as fallback if num_experts_per_layer is None)
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts_per_layer:
            List of expert counts per layer. If None, uses num_experts for all layers.
        num_shared_experts_per_layer:
            List of shared expert counts per layer. If None, uses num_shared_experts for all layers.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    compute_device = gate_logits[0].device
    num_hidden_layers = len(gate_logits)

    # Check if we have variable expert counts
    has_variable_experts = num_experts_per_layer is not None and len(set(num_experts_per_layer)) > 1

    if not has_variable_experts:
        # All layers have the same expert count - use the original stacking approach
        concatenated_gate_logits = torch.stack(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )  # shape: (num_hidden_layers, batch_size * sequence_length, num_experts)

        # remove the shared experts from the gate logits since they are not used for routing in the loss function
        if num_shared_experts > 0:
            concatenated_gate_logits = concatenated_gate_logits[:, :, :-num_shared_experts]
            # adjust the num_experts and top_k accordingly for the loss computation
            num_experts = num_experts - num_shared_experts
            top_k = top_k - num_shared_experts

        routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

        _, selected_experts = torch.topk(
            routing_weights, top_k, dim=-1
        )  # shape: (num_hidden_layers, batch_size * sequence_length, top_k)

        expert_counts_onehot = torch.nn.functional.one_hot(
            selected_experts, num_experts
        )  # shape: (num_hidden_layers, batch_size * sequence_length, top_k, num_experts)

        if attention_mask is None and labels is None:
            # Compute the percentage of tokens routed to each experts
            counts_per_expert = torch.mean(
                expert_counts_onehot.float(), dim=(1, 2)
            )  # shape: (num_hidden_layers, num_experts)

            # Compute the average probability of routing to these experts
            prob_per_expert = torch.mean(
                routing_weights, dim=1
            )  # shape: (num_hidden_layers, num_experts)
        else:
            # if there are labels, then we want to ignore the indices that are in the prompt as well (if there is any)
            if labels is not None:
                attention_mask = labels != ignore_index
            batch_size, sequence_length = attention_mask.shape

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = (
                attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape(num_hidden_layers, -1, top_k, num_experts)
                .to(compute_device)
            )

            # Compute the percentage of tokens routed to each experts
            counts_per_expert = torch.sum(
                expert_counts_onehot.float() * expert_attention_mask, dim=(1, 2)
            )

            # Compute the mask that masks all padding tokens as 0 with the same shape of frequency_per_expert
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape(num_hidden_layers, -1, num_experts)
                .to(compute_device)
            )

            # average the probability across valid tokens
            prob_per_expert = torch.sum(
                routing_weights * router_per_expert_attention_mask, dim=1
            ) / torch.sum(
                attention_mask
            )  # shape: (num_hidden_layers, num_experts)

        overall_loss = torch.sum(counts_per_expert * prob_per_expert)

        # Fallback when num_items_in_batch isn't provided (e.g., manual forward calls)
        if num_items_in_batch is None:
            if labels is not None:
                num_items_in_batch = (labels != ignore_index).sum()
            elif attention_mask is not None:
                num_items_in_batch = attention_mask.sum()
            else:
                # fall back to total tokens in batch/seq from gate logits
                num_items_in_batch = gate_logits[0].shape[0]

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(compute_device)

        # we follow olmo-core and use counts for dot product instead of frequency, and divide by total number token across gradient accumulation steps
        overall_loss = overall_loss / (num_items_in_batch * top_k)

        overall_loss = (
            overall_loss * num_experts / num_hidden_layers
        )  # times num_experts according to lb equation, divide by num_hidden_layers to get average over layers

        return overall_loss

    else:
        # Variable expert counts - compute loss per layer and average
        if num_shared_experts_per_layer is None:
            num_shared_experts_per_layer = [num_shared_experts] * num_hidden_layers

        # Compute attention mask once
        if labels is not None:
            attention_mask = labels != ignore_index

        if attention_mask is not None:
            batch_size, sequence_length = attention_mask.shape

        # Fallback when num_items_in_batch isn't provided
        if num_items_in_batch is None:
            if labels is not None:
                num_items_in_batch = (labels != ignore_index).sum()
            elif attention_mask is not None:
                num_items_in_batch = attention_mask.sum()
            else:
                num_items_in_batch = gate_logits[0].shape[0]

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(compute_device)

        layer_losses = []

        for layer_idx, layer_gate in enumerate(gate_logits):
            layer_gate = layer_gate.to(compute_device)
            layer_num_experts = num_experts_per_layer[layer_idx]
            layer_num_shared = num_shared_experts_per_layer[layer_idx]

            # Remove shared experts from logits
            if layer_num_shared > 0:
                layer_gate = layer_gate[:, :-layer_num_shared]
                effective_num_experts = layer_num_experts - layer_num_shared
                effective_top_k = top_k - layer_num_shared
            else:
                effective_num_experts = layer_num_experts
                effective_top_k = top_k

            # Compute routing weights
            routing_weights = torch.nn.functional.softmax(layer_gate, dim=-1)

            _, selected_experts = torch.topk(
                routing_weights, effective_top_k, dim=-1
            )  # shape: (batch_size * sequence_length, top_k)

            expert_counts_onehot = torch.nn.functional.one_hot(
                selected_experts, effective_num_experts
            )  # shape: (batch_size * sequence_length, top_k, num_experts)

            if attention_mask is None:
                counts_per_expert = torch.mean(
                    expert_counts_onehot.float(), dim=(0, 1)
                )  # shape: (num_experts,)
                prob_per_expert = torch.mean(routing_weights, dim=0)  # shape: (num_experts,)
            else:
                # Reshape for masking
                expert_attention_mask = (
                    attention_mask[:, :, None, None]
                    .expand((batch_size, sequence_length, effective_top_k, effective_num_experts))
                    .reshape(-1, effective_top_k, effective_num_experts)
                    .to(compute_device)
                )

                counts_per_expert = torch.sum(
                    expert_counts_onehot.float() * expert_attention_mask, dim=(0, 1)
                )

                router_attention_mask = (
                    attention_mask[:, :, None]
                    .expand((batch_size, sequence_length, effective_num_experts))
                    .reshape(-1, effective_num_experts)
                    .to(compute_device)
                )

                prob_per_expert = torch.sum(
                    routing_weights * router_attention_mask, dim=0
                ) / torch.sum(attention_mask)

            layer_loss = torch.sum(counts_per_expert * prob_per_expert)
            layer_loss = layer_loss / (num_items_in_batch * effective_top_k)
            layer_loss = layer_loss * effective_num_experts

            layer_losses.append(layer_loss)

        # Average across layers
        overall_loss = torch.stack(layer_losses).mean()

        return overall_loss
