import json
import logging
from abc import abstractmethod
from dataclasses import dataclass
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
from olmo_core.nn.moe.router import (
    MoELinearRouter,
    MoELoadBalancingLossGranularity,
    MoERouter,
    MoERouterConfig,
    MoERouterGatingFunction,
    MoERouterType,
)
from olmo_core.nn.moe.twolevel_router import MoETwoLevelRouter, MoETwoLevelRouterConfig
from olmo_core.utils import get_default_device

from .loss import MoELoadBalancingLossGranularity, load_balancing_loss, router_z_loss


class MoETwoLevelBatchLBReduceDPSharedExpRouter(MoETwoLevelRouter):
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
            num_shared_experts: int,
            **kwargs,
    ):
        super().__init__(dtype=dtype, init_device=init_device, document_expert_pool=document_expert_pool, eos_token_id=eos_token_id, **kwargs)

        # the number of experts that each document can select their top-k experts from
        self.document_expert_pool = document_expert_pool
        # the eos token id
        if eos_token_id is None:
            raise OLMoConfigurationError("eos_token_id must be provided for MoETwoLevelRouter")
        self.eos_token_id = eos_token_id

        # check that the number of shared experts is less than the top_k
        if num_shared_experts > self.top_k:
            raise OLMoConfigurationError(f"num_shared_experts ({num_shared_experts}) must be less than or equal to top_k ({self.top_k})")

        self.num_shared_experts = num_shared_experts
        self.num_choose_experts = self.top_k - self.num_shared_experts

    def get_top_k(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ We override the get_top_k to use self.num_choose_experts instead of self.top_k, since we will always activate self.num_shared_experts"""
        expert_weights: torch.Tensor
        expert_indices: torch.Tensor
        if self.bias_gamma is None:
            if self.top_k == 1:
                expert_weights, expert_indices = scores.max(dim=-1, keepdim=True)
            else:
                expert_weights, expert_indices = torch.topk(scores, self.num_choose_experts, dim=-1)
        else:
            assert self.score_bias is not None
            with torch.no_grad():
                _, expert_indices = torch.topk(
                    scores + self.score_bias.unsqueeze(0), self.num_choose_experts, dim=-1  # type: ignore
                )
            expert_weights = scores.gather(-1, expert_indices)

        if self.uniform_expert_assignment:
            raise NotImplementedError("Uniform expert assignment is not supported in MoETwoLevelBatchLBReduceDPSharedExpRouter. (actually very easy - just copy paste the implementation, but too lazy right now since we don't use this)")

        return expert_weights, expert_indices

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        padding_mask: Optional[torch.Tensor] = None,  # shape: (B, S)
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
        # make sure tp and cp are not enabled
        if self.tp_mesh is not None:
            raise NotImplementedError("Tensor parallelism is not supported.")
        if self.cp_mesh is not None:
            raise NotImplementedError("Context parallelism is not supported.")

        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x).float()

        # we remove the last self.num_shared_experts experts (remove from end in case indexing gets weird later?)
        logits = logits[:, :, :self.num_experts - self.num_shared_experts] # shape: (batch_size, seq_len, num_experts - num_shared_experts)
        logits_mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)

        document_boundaries_cpu = []
        for b in document_boundaries:
            bc = b.detach().cpu().tolist()
            if not bc or bc[-1] != x.size(1):
                bc.append(int(x.size(1)))
            document_boundaries_cpu.append(bc)

        # tot_doc_entropy = []
        doc_entropy_sum = logits.new_zeros(())
        doc_entropy_count = 0

        for seq_idx in range(x.size(0)):
            start = 0
            document_boundary = document_boundaries_cpu[seq_idx]
            for end in document_boundary:
                if end <= start:
                    start = end
                    continue
                sequence_logits = logits[seq_idx, start:end, :]  # shape: (doc_len, num_experts - num_shared_experts)
                # calculate the softmax over the experts
                expert_probs = F.softmax(sequence_logits, dim=-1)  # shape: (doc_len, num_experts - num_shared_experts)

                # get the entropy over experts per token
                token_entropies = -torch.sum(
                    expert_probs * torch.log(expert_probs + 1e-10), dim=-1
                )  # shape: (doc_len,)
                # average entropy over the document
                doc_entropy_sum += token_entropies.mean()
                doc_entropy_count += 1

                # take the sum across the document
                document_expert_probs = expert_probs.sum(dim=0)  # shape: (num_experts - num_shared_experts,)
                # get the bottom document_expert_pool experts (including removing the shared experts since we already took that out of the logits)
                bot_document_expert_pool = self.num_experts - self.document_expert_pool - self.num_shared_experts
                experts_to_discard = torch.topk(
                    -document_expert_probs, bot_document_expert_pool
                ).indices  # shape: (bot_document_expert_pool,)
                # set the logits of these experts to a very large negative value
                # logits[seq_idx, start:end, experts_to_discard] = float('-inf')
                logits_mask[seq_idx, start:end, experts_to_discard] = True
                start = end

        logits.masked_fill_(logits_mask, float("-inf"))

        if self.training:
            # log the average document entropy
            # avg_doc_entropy = sum(tot_doc_entropy) / len(tot_doc_entropy) if tot_doc_entropy else 0.0
            # logging.info(f"Average document entropy over experts: {avg_doc_entropy}")
            # self._router_documentlevel_expert_entropy += avg_doc_entropy
            avg_doc_entropy = (doc_entropy_sum / doc_entropy_count).detach()
            self._router_documentlevel_expert_entropy += avg_doc_entropy.item()

        # shape: (batch_size, seq_len, num_experts)
        if self.gating_function == MoERouterGatingFunction.softmax:
            scores = logits.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = F.sigmoid(logits) + 1e-7
        else:
            raise NotImplementedError(self.gating_function)

        # shape: (batch_size, seq_len, self.num_choose_experts)
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
            # shape: (batch_size, seq_len, num_experts - num_shared_experts)
            tot_batched_batch_size_per_expert = ops.batched_histc(expert_indices, self.num_experts - self.num_shared_experts)
            # shape: (batch_size, num_experts - num_shared_experts)
            tot_batched_batch_size_per_expert = tot_batched_batch_size_per_expert.sum(dim=1)
            # shape: (num_experts - num_shared_experts,)
            tot_batch_size_per_expert = tot_batched_batch_size_per_expert.sum(dim=0)

            if self.training:
                valid_expert_indices = expert_indices.view(-1)

                # Update unique experts metric.
                unique_experts = torch.unique(valid_expert_indices)
                num_unique_experts = unique_experts.numel() + self.num_shared_experts # we add the shared experts since they are always active

                self._unique_experts_sum += num_unique_experts
                self._num_batches_tracked += 1

                # Compute router distribution entropy metric
                valid_scores = scores.view(-1, self.num_experts - self.num_shared_experts)
                # get entropy per token
                token_entropies = -torch.sum(valid_scores * torch.log(valid_scores + 1e-10), dim=-1)
                # average entropy over valid tokens
                avg_entropy = token_entropies.mean().item()
                self._router_tokenlevel_expert_entropy += avg_entropy

        # Maybe compute auxiliary losses and accumulate metrics.
        aux_loss: Optional[torch.Tensor] = None
        if self.training and torch.is_grad_enabled():
            with torch.autocast(enabled=False, device_type=x.device.type):
                # use the batch-level load balancing loss
                if self.lb_loss_weight is not None:
                    assert self.load_balancing_loss is not None

                    # Make sure scores are normalized, otherwise load balancing loss doesn't work well.
                    if self.gating_function == MoERouterGatingFunction.sigmoid:
                        scores = scores / scores.sum(dim=-1, keepdim=True)

                    # we now do reduction on the tot_batch_size_per_expert to get a dp-global lb loss (still not full global sinze we don't do across gradient accumulation steps)
                    dp_global_tot_batch_size_per_expert = tot_batch_size_per_expert.clone() # we clone to not interfere with logging or other routing stuff
                    dp_global_loss_div_factor = loss_div_factor.clone()

                    if is_distributed():
                        dist.all_reduce(dp_global_tot_batch_size_per_expert, op=dist.ReduceOp.SUM)
                        dist.all_reduce(dp_global_loss_div_factor, op=dist.ReduceOp.SUM)

                    # collect all non-zero entries of the dp_global_tot_batch_size_per_expert
                    self._reducedp_unique_experts_sum += (dp_global_tot_batch_size_per_expert > 0).sum().item() + self.num_shared_experts # we add the shared experts since they are always active

                    lb_loss = load_balancing_loss(
                        num_experts=self.num_experts - self.num_shared_experts, # we only calculate the load balancing loss over the non-shared experts
                        top_k=self.num_choose_experts, # we only choose self.num_choose_experts
                        expert_scores=scores,
                        batch_size_per_expert=dp_global_tot_batch_size_per_expert,
                        batched_batch_size_per_expert=tot_batched_batch_size_per_expert, # this is not used, so we don't bother reducing it
                        granularity=self.lb_loss_granularity,
                        loss_div_factor=dp_global_loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )

                    self.load_balancing_loss += lb_loss.detach()

                    scaled_lb_loss = self.lb_loss_weight * lb_loss
                    aux_loss = scaled_lb_loss

                if self.z_loss_weight is not None:
                    assert self.z_loss is not None

                    # we don't care about shared expert z_loss since they are always set to 1 (thus logits not used)

                    z_loss = router_z_loss(
                        expert_logits=logits,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.z_loss += z_loss.detach()

                    scaled_z_loss = self.z_loss_weight * z_loss
                    aux_loss = scaled_z_loss if aux_loss is None else aux_loss + scaled_z_loss

            if self.batch_size_per_expert.shape[-1] != tot_batch_size_per_expert.shape[-1]:
                # make sure that the shared expert positions are zero, since it means the parameter was reset
                extra_counts = self.batch_size_per_expert[tot_batch_size_per_expert.shape[-1]:]
                assert torch.all(extra_counts == 0), f"Expected extra counts to be zero, but got {extra_counts}"
                self.batch_size_per_expert = self.batch_size_per_expert[:tot_batch_size_per_expert.shape[-1]]
            self.batch_size_per_expert += tot_batch_size_per_expert
            if self.bias_gamma is not None:
                assert self.score_bias_batch_size_per_expert is not None
                self.score_bias_batch_size_per_expert += tot_batch_size_per_expert

        # in the end, we add on the shared experts to both expert_weights and expert_indices
        if self.num_shared_experts > 0:
            # TODO: need to check this
            expert_weights = F.pad(expert_weights, (0, self.num_shared_experts), value=1.0) # we set the weights of the shared experts to 1 since they are always active
            # we set the indices of the shared experts to the last num_shared_experts indices since we removed those from the logits earlier
            shared_expert_indices = torch.arange(self.num_experts - self.num_shared_experts, self.num_experts, device=expert_indices.device).view(1, 1, self.num_shared_experts).expand(expert_indices.size(0), expert_indices.size(1), self.num_shared_experts)
            expert_indices = torch.cat([expert_indices, shared_expert_indices], dim=-1)
            # we also set tot_batch_size_per_expert for the shared experts to be the batch size since they are always active
            tot_batch_size_per_expert = F.pad(tot_batch_size_per_expert, (0, self.num_shared_experts), value=x.size(0) * x.size(1))

        return expert_weights, expert_indices, tot_batch_size_per_expert, aux_loss

    def extra_repr(self):
        """Add custom parameter to string representation."""
        base_repr = super().extra_repr()
        return f"{base_repr}, document_expert_pool={self.document_expert_pool}, eos_token_id={self.eos_token_id}, num_shared_experts={self.num_shared_experts}"


@dataclass
class MoETwoLevelBatchLBReduceDPSharedExpRouterConfig(MoETwoLevelRouterConfig):
    num_shared_experts: int = 2 # the number of experts to share

    # just update the build to call the correct new class
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
    ) -> MoETwoLevelBatchLBReduceDPSharedExpRouter:
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

        return MoETwoLevelBatchLBReduceDPSharedExpRouter(**kwargs)
