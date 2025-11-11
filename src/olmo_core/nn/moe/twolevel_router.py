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



class MoETwoLevelRouter(MoELinearRouter):
    """
    Custom MoE router with modified forward pass and additional class variables.
    """

    def __init__(
            self,
            *,
            dtype: torch.dtype = torch.float32,
            init_device: str = "cpu",
            document_expert_pool: int,
            eos_token_id: int,
            **kwargs,
    ):
        super().__init__(dtype=dtype, init_device=init_device, **kwargs)

        # the number of experts that each document can select their top-k experts from
        self.document_expert_pool = document_expert_pool
        # the eos token id
        if eos_token_id is None:
            raise OLMoConfigurationError("eos_token_id must be provided for MoETwoLevelRouter")
        self.eos_token_id = eos_token_id

    def forward(
            self,
            x: torch.Tensor,
            *,
            loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
            document_boundaries: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Custom forward pass with modifications to implement two level routing.
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
        logits_mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)

        document_boundaries_cpu = []
        for b in document_boundaries:
            bc = b.detach().cpu().tolist()
            if not bc or bc[-1] != x.size(1):
                bc.append(int(x.size(1)))
            document_boundaries_cpu.append(bc)

        for seq_idx in range(x.size(0)):
            start = 0
            document_boundary = document_boundaries_cpu[seq_idx]
            for end in document_boundary:
                if end <= start:
                    start = end
                    continue
                sequence_logits = logits[seq_idx, start:end, :] # shape: (doc_len, num_experts)
                # calculate the softmax over the experts
                expert_probs = F.softmax(sequence_logits, dim=-1) # shape: (doc_len, num_experts)
                # take the sum across the document
                document_expert_probs = expert_probs.sum(dim=0) # shape: (num_experts,)
                # get the bottom document_expert_pool experts
                bot_document_expert_pool = self.num_experts - self.document_expert_pool
                experts_to_discard = torch.topk(-document_expert_probs, bot_document_expert_pool).indices # shape: (bot_document_expert_pool,)
                # set the logits of these experts to a very large negative value
                # logits[seq_idx, start:end, experts_to_discard] = float('-inf')
                logits_mask[seq_idx, start:end, experts_to_discard] = True
                start = end

        logits.masked_fill_(logits_mask, float('-inf'))

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
            tot_batched_batch_size_per_expert = ops.batched_histc(expert_indices,
                                                              self.num_experts)
            # shape: (batch_size, num_experts)
            tot_batched_batch_size_per_expert = tot_batched_batch_size_per_expert.sum(dim=1)
            # shape: (num_experts,)
            tot_batch_size_per_expert = tot_batched_batch_size_per_expert.sum(dim=0)

        # Maybe compute auxiliary losses and accumulate metrics.
        aux_loss: Optional[torch.Tensor] = None
        if self.training and torch.is_grad_enabled():
            with torch.autocast(enabled=False, device_type=x.device.type):
                if self.lb_loss_weight is not None:
                    assert self.load_balancing_loss is not None

                    doc_lb_losses = []
                    for seq_idx in range(x.size(0)):
                        start = 0
                        document_boundary = document_boundaries_cpu[seq_idx]

                        for end in document_boundary:
                            if end <= start:
                                start=end
                                continue
                            # Get tokens for this document
                            doc_scores = scores[seq_idx, start:end, :]  # (doc_len, num_experts)
                            doc_indices = expert_indices[seq_idx, start:end]  # (doc_len, top_k)

                            # find active experts (not masked)
                            active_experts_mask = (~logits_mask[seq_idx, start:end, :]).any(dim=0)                            doc_scores = doc_scores[:, active_experts_mask]
                            num_active = doc_scores.shape[-1]

                            assert num_active == self.document_expert_pool, f"Number of active experts {num_active} does not match document_expert_pool {self.document_expert_pool}"

                            # we re-assign the expert indices to be in the range of the active experts only
                            expert_id_mapping = torch.zeros(self.num_experts, dtype=torch.long, device=x.device) - 1
                            expert_id_mapping[active_experts_mask] = torch.arange(num_active, device=x.device)

                            doc_indices_local = expert_id_mapping[doc_indices]

                            # Make sure scores are normalized, otherwise load balancing loss doesn't work well.
                            if self.gating_function == MoERouterGatingFunction.sigmoid:
                                doc_scores = doc_scores / doc_scores.sum(dim=-1, keepdim=True)

                            with torch.no_grad():
                                # Histogram the expert ids to identify the number of items/tokens routed to each expert.
                                # shape: (batch_size, seq_len, num_experts)
                                batched_batch_size_per_expert = ops.batched_histc(doc_indices_local.unsqueeze(0), self.document_expert_pool)
                                # shape: (batch_size, num_experts)
                                batched_batch_size_per_expert = batched_batch_size_per_expert.sum(dim=1)
                                # shape: (num_experts,)
                                batch_size_per_expert = batched_batch_size_per_expert.sum(dim=0)

                            # removed loss_div_factor because we are computing by document

                            doc_lb_loss = load_balancing_loss(
                                num_experts=self.document_expert_pool,
                                top_k=self.top_k,
                                expert_scores=doc_scores.unsqueeze(0),
                                batch_size_per_expert=batch_size_per_expert,
                                batched_batch_size_per_expert=batched_batch_size_per_expert,
                                granularity=self.lb_loss_granularity,
                                tp_mesh=self.tp_mesh,
                                cp_mesh=self.cp_mesh,
                            )
                            doc_lb_losses.append(doc_lb_loss)
                            start = end

                    # Combine all document-level LB losses
                    lb_loss = torch.stack(doc_lb_losses).mean()
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

        return expert_weights, expert_indices, tot_batch_size_per_expert, aux_loss

    def extra_repr(self):
        """Add custom parameter to string representation."""
        base_repr = super().extra_repr()
        return f"{base_repr}, document_expert_pool={self.document_expert_pool}, eos_token_id={self.eos_token_id}"

@dataclass
class MoETwoLevelRouterConfig(MoERouterConfig):
    """
    Config for pruning MoE router.
    """
    document_expert_pool: int = 32
    eos_token_id: Optional[int] = None

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
    ) -> MoETwoLevelRouter:
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

        return MoETwoLevelRouter(**kwargs)