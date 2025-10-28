import json
from typing import Optional

from olmo_core.nn.moe.router import MoERouterConfig, MoERouterType
from dataclasses import dataclass

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

import olmo_core.ops.moe as ops
from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.utils import (
    _HiddenTensor,
    distribute_like,
    get_local_tensor,
    hide_from_torch,
    is_distributed,
    unhide_from_torch,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_default_device

from .loss import MoELoadBalancingLossGranularity, load_balancing_loss, router_z_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from olmo_core.nn.moe.router import MoELinearRouter, MoERouter
from olmo_core.nn.moe.router import MoERouterGatingFunction, MoELoadBalancingLossGranularity
from olmo_core.distributed.utils import get_local_tensor



class PruningMoELinearRouter(MoELinearRouter):
    """
    Custom MoE router with modified forward pass and additional class variables.
    """

    def __init__(
            self,
            *,
            dtype: torch.dtype = torch.float32,
            init_device: str = "cpu",
            prune_keep_k: int = 32,  # the number of experts to keep after pruning
            activation_file: str = "",  # path to activation file for pruning
            layer_idx: int, # index of the layer for which this router is being created -> needs to know this to know which experts to keep
            **kwargs,
    ):
        super().__init__(dtype=dtype, init_device=init_device, **kwargs)

        # Read in activation file to know which experts to prune
        self.prune_keep_k = prune_keep_k
        self.layer_idx = layer_idx

        # Load expert activations and determine which experts to keep
        if activation_file:
            # replace all occurences of ":" with "\:" in activation_file path
            activation_file = activation_file.replace(":", "\:")
            with open(activation_file, 'r') as f:
                line = f.readline()
                activations = json.loads(line)["avg_router_probabilities"]
            expert_activations = activations[layer_idx] # choose the layer to activate
            # Get indices of top-k most activated experts
            self.experts_to_keep = torch.topk(
                torch.tensor(expert_activations),
                min(self.prune_keep_k, len(expert_activations))
            ).indices.tolist()
        else:
            # If no activation file, keep all experts
            self.experts_to_keep = list(range(self.num_experts))

        # ensure we can pick top_k among active experts
        assert len(self.experts_to_keep) >= self.top_k, (
            f"prune_keep_k={len(self.experts_to_keep)} < top_k={self.top_k}. "
            "Increase prune_keep_k or reduce top_k."
        )

        # Register mask as a buffer so it moves with the module
        expert_mask = torch.zeros(self.num_experts, dtype=torch.bool, device=init_device)
        expert_mask[self.experts_to_keep] = True
        self.register_buffer("expert_mask", expert_mask, persistent=False)

        # Precompute active indices buffer
        active_idx = torch.nonzero(self.expert_mask, as_tuple=False).view(-1)
        self.register_buffer("active_indices", active_idx, persistent=False)

        print(f"Layer {layer_idx}: Keeping experts {self.experts_to_keep} out of {self.num_experts}")

    def forward(
            self,
            x: torch.Tensor,
            *,
            loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Custom forward pass with modifications to implement pruned experts (i.e never activate certain experts).
        Given the input ``x`` of shape ``(B, S, d_model)``, compute the experts assignment.

        :returns: The expert weights of shape ``(B, S, top_k)``,
            the expert indices of shape ``(B, S, top_k)``,
            the total number of items routed to each expert, with shape ``(num_experts,)``,
            and optionally the auxiliary losses.
        """
        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x).float()

        # Mask out pruned experts by setting their logits to a very large negative value
        logits = logits.masked_fill(~self.expert_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # shape: (batch_size, seq_len, num_experts)
        if self.gating_function == MoERouterGatingFunction.softmax:
            scores = logits.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = F.sigmoid(logits) + 1e-7
        else:
            raise NotImplementedError(self.gating_function)

        # shape: (batch_size, seq_len, top_k)
        expert_weights, expert_indices = self.get_top_k(scores)

        if self.normalize_expert_weights is not None:
            expert_weights = expert_weights.div(
                torch.norm(
                    expert_weights,
                    p=self.normalize_expert_weights,
                    dim=-1,
                    keepdim=True,
                )
            )

        with torch.no_grad():
            # Histogram the expert ids to identify the number of items/tokens routed to each expert.
            # shape: (batch_size, seq_len, num_experts)
            batched_batch_size_per_expert = ops.batched_histc(expert_indices, self.num_experts)
            # shape: (batch_size, num_experts)
            batched_batch_size_per_expert = batched_batch_size_per_expert.sum(dim=1)
            # shape: (num_experts,)
            batch_size_per_expert = batched_batch_size_per_expert.sum(dim=0)

        # Maybe compute auxiliary losses and accumulate metrics.
        aux_loss: Optional[torch.Tensor] = None
        if self.training and torch.is_grad_enabled():
            with torch.autocast(enabled=False, device_type=x.device.type):
                # Slice to active experts for LB loss, and use effective sizes
                active_idx = self.active_indices
                scores_active = scores.index_select(-1, active_idx)  # (B, S, E_active)
                bbse_active = batched_batch_size_per_expert.index_select(-1, active_idx)  # (B, E_active)
                bse_active = batch_size_per_expert.index_select(0, active_idx)  # (E_active,)
                num_active = active_idx.numel()
                eff_top_k = min(self.top_k, num_active)

                if self.lb_loss_weight is not None:
                    assert self.load_balancing_loss is not None

                    # Make sure scores are normalized, otherwise load balancing loss doesn't work well.
                    if self.gating_function == MoERouterGatingFunction.sigmoid:
                        scores_active = scores_active / scores_active.sum(dim=-1, keepdim=True)

                    lb_loss = load_balancing_loss(
                        num_experts=num_active,
                        top_k=eff_top_k,
                        expert_scores=scores_active,
                        batch_size_per_expert=bse_active,
                        batched_batch_size_per_expert=bbse_active,
                        granularity=self.lb_loss_granularity,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.load_balancing_loss += lb_loss.detach()

                    scaled_lb_loss = self.lb_loss_weight * lb_loss
                    aux_loss = scaled_lb_loss

                if self.z_loss_weight is not None:
                    assert self.z_loss is not None

                    z_loss = router_z_loss(
                        expert_logits=logits,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.z_loss += z_loss.detach()

                    scaled_z_loss = self.z_loss_weight * z_loss
                    aux_loss = scaled_z_loss if aux_loss is None else aux_loss + scaled_z_loss

            self.batch_size_per_expert += batch_size_per_expert
            if self.bias_gamma is not None:
                assert self.score_bias_batch_size_per_expert is not None
                self.score_bias_batch_size_per_expert += batch_size_per_expert

        return expert_weights, expert_indices, batch_size_per_expert, aux_loss

    def extra_repr(self):
        """Add custom parameter to string representation."""
        base_repr = super().extra_repr()
        return f"{base_repr}, prune_keep_k={self.prune_keep_k}, layer_idx={self.layer_idx}"

@dataclass
class PruningMoERouterConfig(MoERouterConfig):
    """
    Config for pruning MoE router.
    """
    prune_keep_k: int = 32
    activation_file: str = ""
    layer_idx: int = 0

    def build(
            self,
            d_model: int,
            num_experts,
            *,
            lb_loss_weight: Optional[float] = None,
            lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
            z_loss_weight: Optional[float] = None,
            dtype: Optional[torch.dtype] = None,
            init_device: str = "cpu",
    ) -> PruningMoELinearRouter:
        """
        Build the pruning router.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")  # Remove name since we're directly instantiating
        kwargs.update(
            d_model=d_model,
            num_experts=num_experts,
            init_device=init_device,
            lb_loss_weight=lb_loss_weight,
            lb_loss_granularity=lb_loss_granularity,
            z_loss_weight=z_loss_weight,
        )
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype

        return PruningMoELinearRouter(**kwargs)