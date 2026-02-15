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


class MoETwoLevelBatchLBReduceDPSharedExpPoolRouter(MoETwoLevelRouter):
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
            num_shared_experts_pool: int,
            shared_exp_lb_loss: Optional[float] = None,
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

        # check that the num_shared_experts is less than the num_shared_experts_pool
        if num_shared_experts > num_shared_experts_pool:
            raise OLMoConfigurationError(f"num_shared_experts ({num_shared_experts}) must be less than or equal to num_shared_experts_pool ({num_shared_experts_pool})")

        self.num_shared_experts = num_shared_experts # number of shared experts that are always activated
        self.num_shared_experts_pool = num_shared_experts_pool # number of shared experts in the shared experts pool
        self.num_choose_experts = self.top_k - self.num_shared_experts # number of experts to choose that are activated
        self.num_choose_experts_pool = self.num_experts - self.num_shared_experts_pool # number of experts to choose from in the pool (excluding the shared experts in the pool) -> this should be bigger than self.document_expert_pool

        if self.num_choose_experts_pool < self.document_expert_pool:
            raise OLMoConfigurationError(f"num_choose_experts_pool ({self.num_choose_experts_pool}) must be greater than or equal to document_expert_pool ({self.document_expert_pool})")

        self.shared_exp_lb_loss_weight = shared_exp_lb_loss

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

    def get_top_k_shared(self, scores: torch.Tensor, k) -> Tuple[torch.Tensor, torch.Tensor]:
        """ We write a function that takes in k (instead of hardcoding the k like before)"""
        expert_weights: torch.Tensor
        expert_indices: torch.Tensor
        if self.bias_gamma is None:
            if k == 1:
                expert_weights, expert_indices = scores.max(dim=-1, keepdim=True)
            else:
                expert_weights, expert_indices = torch.topk(scores, k, dim=-1)
        else:
            assert self.score_bias is not None
            with torch.no_grad():
                _, expert_indices = torch.topk(
                    scores + self.score_bias.unsqueeze(0), k, dim=-1  # type: ignore
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

        # we split the router up into shared experts and standard experts
        logits_standard_exp = logits[:, :, :self.num_choose_experts_pool]
        logits_mask_standard_exp = torch.zeros_like(logits_standard_exp, dtype=torch.bool, device=logits.device)
        logits_shared_exp = logits[:, :, self.num_choose_experts_pool:]
        logits_mask_shared_exp = torch.zeros_like(logits_shared_exp, dtype=torch.bool, device=logits.device)

        assert logits_shared_exp.shape[-1] == self.num_shared_experts_pool, f"Expected the number of shared experts in the logits to be {self.num_shared_experts_pool}, but got {logits_shared_exp.shape[-1]}"

        # # we remove the last self.num_shared_experts experts (remove from end in case indexing gets weird later?)
        # logits = logits[:, :, :self.num_experts - self.num_shared_experts] # shape: (batch_size, seq_len, num_experts - num_shared_experts)
        # logits_mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)

        document_boundaries_cpu = []
        for b in document_boundaries:
            bc = b.detach().cpu().tolist()
            if not bc or bc[-1] != x.size(1):
                bc.append(int(x.size(1)))
            document_boundaries_cpu.append(bc)

        # tot_doc_entropy = []
        doc_entropy_sum = logits_standard_exp.new_zeros(())
        doc_entropy_count = 0

        for seq_idx in range(x.size(0)):
            start = 0
            document_boundary = document_boundaries_cpu[seq_idx]
            for end in document_boundary:
                if end <= start:
                    start = end
                    continue
                sequence_logits_standard_exp = logits_standard_exp[seq_idx, start:end, :]  # shape: (doc_len, num_standard_experts_pool)
                sequence_logits_shared_exp = logits_shared_exp[seq_idx, start:end, :] # shape: (doc_len, num_shared_experts_pool)
                # calculate the softmax over the experts
                expert_probs_standard_exp = F.softmax(sequence_logits_standard_exp, dim=-1)  # shape: (doc_len, num_standard_experts_pool)
                expert_probs_shared_exp = F.softmax(sequence_logits_shared_exp, dim=-1) # shape: (doc_len, num_shared_experts_pool)

                # get the entropy over experts per token (only for the standard experts)
                token_entropies = -torch.sum(
                    expert_probs_shared_exp * torch.log(expert_probs_shared_exp + 1e-10), dim=-1
                )  # shape: (doc_len,)
                # average entropy over the document
                doc_entropy_sum += token_entropies.mean()
                doc_entropy_count += 1

                # take the sum across the document
                document_expert_probs_standard_exp = expert_probs_standard_exp.sum(dim=0)  # shape: (num_standard_experts_pool,)
                document_expert_probs_shared_exp = expert_probs_shared_exp.sum(dim=0) # shape: (num_shared_experts_pool,)

                # get the bottom experts for the standard experts
                bot_document_expert_pool_standard = self.num_choose_experts_pool - self.document_expert_pool # the number of experts to discard for each document from the standard expert pool
                experts_to_discard_standard = torch.topk(
                    -document_expert_probs_standard_exp, bot_document_expert_pool_standard
                ).indices  # shape: (bot_document_expert_pool_standard,)
                # set the logits of these experts to a very large negative value
                logits_mask_standard_exp[seq_idx, start:end, experts_to_discard_standard] = True

                # do the same for the shared experts
                # get the bottom document_expert_pool experts
                bot_document_expert_pool_shared = self.num_shared_experts_pool - self.num_shared_experts  # the number of experts to discard for each document from the shared expert pool
                experts_to_discard_shared = torch.topk(
                    -document_expert_probs_shared_exp, bot_document_expert_pool_shared
                ).indices  # shape: (bot_document_expert_pool,)
                # set the logits of these experts to a very large negative value
                logits_mask_shared_exp[seq_idx, start:end, experts_to_discard_shared] = True

                start = end

        logits_standard_exp.masked_fill_(logits_mask_standard_exp, float("-inf"))
        logits_shared_exp.masked_fill_(logits_mask_shared_exp, float("-inf"))
        # logits.masked_fill_(logits_mask, float("-inf"))

        if self.training:
            # log the average document entropy
            # avg_doc_entropy = sum(tot_doc_entropy) / len(tot_doc_entropy) if tot_doc_entropy else 0.0
            # logging.info(f"Average document entropy over experts: {avg_doc_entropy}")
            # self._router_documentlevel_expert_entropy += avg_doc_entropy
            avg_doc_entropy = (doc_entropy_sum / doc_entropy_count).detach()
            self._router_documentlevel_expert_entropy += avg_doc_entropy.item()

        # shape: (batch_size, seq_len, num_experts)
        if self.gating_function == MoERouterGatingFunction.softmax:
            # scores = logits.softmax(dim=-1)
            scores_standard_exp = logits_standard_exp.softmax(dim=-1)
            scores_shared_exp = logits_shared_exp.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            raise NotImplementedError("Sigmoid gating function is not supported in MoETwoLevelBatchLBReduceDPSharedExpRouter")
            scores = F.sigmoid(logits) + 1e-7
        else:
            raise NotImplementedError(self.gating_function)

        # shape: (batch_size, seq_len, self.num_choose_experts)
        expert_weights_standard_exp, expert_indices_standard_exp = self.get_top_k(scores_standard_exp)
        expert_weights_shared_exp, expert_indices_shared_exp = self.get_top_k_shared(scores_shared_exp, self.num_shared_experts)

        if self.normalize_expert_weights is not None:
            raise NotImplementedError("Expert weight normalization is not supported in MoETwoLevelBatchLBReduceDPSharedExpRouter since it is not clear how to do it with the shared experts (do we normalize over the shared and standard experts together, or separately?)")
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

            # for standard experts
            # shape: (batch_size, seq_len, num_choose_experts_pool)
            tot_batched_batch_size_per_expert_standard = ops.batched_histc(expert_indices_standard_exp, self.num_choose_experts_pool)
            # shape: (batch_size, num_choose_experts_pool)
            tot_batched_batch_size_per_expert_standard = tot_batched_batch_size_per_expert_standard.sum(dim=1)
            # shape: (num_choose_experts_pool,)
            tot_batch_size_per_expert_standard = tot_batched_batch_size_per_expert_standard.sum(dim=0)

            # for shared experts
            # shape: (batch_size, seq_len, num_shared_experts_pool)
            tot_batched_batch_size_per_expert_shared = ops.batched_histc(expert_indices_shared_exp, self.num_shared_experts_pool)
            # shape: (batch_size, num_shared_experts_pool)
            tot_batched_batch_size_per_expert_shared = tot_batched_batch_size_per_expert_shared.sum(dim=1)
            # shape: (num_shared_experts_pool,)
            tot_batch_size_per_expert_shared = tot_batched_batch_size_per_expert_shared.sum(dim=0)

            # merge the two together for logging and returning (lb loss will not use this term)
            tot_batch_size_per_expert = torch.cat([tot_batch_size_per_expert_standard, tot_batch_size_per_expert_shared], dim=0)

            if self.training:
                valid_expert_indices_standard_exp = expert_indices_standard_exp.view(-1)
                valid_expert_indices_shared_exp = expert_indices_shared_exp.view(-1)

                # Update unique experts metric.
                unique_experts_standard_exp = torch.unique(valid_expert_indices_standard_exp)
                unique_experts_shared_exp = torch.unique(valid_expert_indices_shared_exp)
                num_unique_experts = unique_experts_standard_exp.numel() + unique_experts_shared_exp.numel()

                self._unique_experts_sum += num_unique_experts
                self._unique_experts_sum_shared += unique_experts_shared_exp.numel() # also track how many unique experts are used
                self._num_batches_tracked += 1

                # Compute router distribution entropy metric (depricated here - only for standard experts)
                valid_scores = scores_standard_exp.view(-1, self.num_choose_experts_pool)
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
                        raise NotImplementedError("Sigmoid gating function is not supported in MoETwoLevelBatchLBReduceDPSharedExpRouter, so we don't implement load balancing loss for sigmoid gating function")
                        scores = scores / scores.sum(dim=-1, keepdim=True)

                    # we now do reduction on the tot_batch_size_per_expert to get a dp-global lb loss (still not full global sinze we don't do across gradient accumulation steps)
                    dp_global_tot_batch_size_per_expert_standard = tot_batch_size_per_expert_standard.clone() # we clone to not interfere with logging or other routing stuff
                    dp_global_tot_batch_size_per_expert_shared = tot_batch_size_per_expert_shared.clone()
                    dp_global_loss_div_factor = loss_div_factor.clone()

                    if is_distributed():
                        dist.all_reduce(dp_global_tot_batch_size_per_expert_standard, op=dist.ReduceOp.SUM)
                        dist.all_reduce(dp_global_tot_batch_size_per_expert_shared, op=dist.ReduceOp.SUM)
                        dist.all_reduce(dp_global_loss_div_factor, op=dist.ReduceOp.SUM)

                    # collect all non-zero entries of the dp_global_tot_batch_size_per_expert
                    concatenated_bspe = torch.cat([dp_global_tot_batch_size_per_expert_standard, dp_global_tot_batch_size_per_expert_shared], dim=0)
                    self._reducedp_unique_experts_sum += (concatenated_bspe > 0).sum().item() # we add the shared experts since they are always active
                    self._reducedp_unique_experts_sum_shared += (dp_global_tot_batch_size_per_expert_shared > 0).sum().item()

                    lb_loss_standard = load_balancing_loss(
                        num_experts=self.num_choose_experts_pool, # we only calculate the load balancing loss over the non-shared experts
                        top_k=self.num_choose_experts, # we only choose self.num_choose_experts
                        expert_scores=scores_standard_exp,
                        batch_size_per_expert=dp_global_tot_batch_size_per_expert_standard,
                        batched_batch_size_per_expert=tot_batched_batch_size_per_expert_standard, # this is not used, so we don't bother reducing it
                        granularity=self.lb_loss_granularity,
                        loss_div_factor=dp_global_loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )

                    lb_loss_shared = load_balancing_loss(
                        num_experts=self.num_shared_experts_pool,
                        # we only calculate the load balancing loss over the non-shared experts
                        top_k=self.num_shared_experts,  # we only choose self.num_choose_experts
                        expert_scores=scores_shared_exp,
                        batch_size_per_expert=dp_global_tot_batch_size_per_expert_shared,
                        batched_batch_size_per_expert=tot_batched_batch_size_per_expert_shared,
                        # this is not used, so we don't bother reducing it
                        granularity=self.lb_loss_granularity,
                        loss_div_factor=dp_global_loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )

                    self.load_balancing_loss += lb_loss_standard.detach() # this is scaled during logging
                    self.load_balancing_loss_shared += self.shared_exp_lb_loss_weight * lb_loss_shared.detach() # this is scaled now since I don't want to override compute_metrics

                    scaled_lb_loss = self.lb_loss_weight * lb_loss_standard + self.shared_exp_lb_loss_weight * lb_loss_shared
                    aux_loss = scaled_lb_loss

                if self.z_loss_weight is not None:
                    assert self.z_loss is not None

                    # TODO: if there are problems, likely because z_loss_shared and z_loss_standard are weighted differently
                    # we create one z_loss per standard and expert pool
                    z_loss_standard = router_z_loss(
                        expert_logits=logits_standard_exp,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    z_loss_shared = router_z_loss(
                        expert_logits=logits_shared_exp,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.z_loss += z_loss_standard.detach() + z_loss_shared.detach()
                    self.z_loss_shared += z_loss_shared.detach()

                    scaled_z_loss = self.z_loss_weight * (z_loss_standard + z_loss_shared)
                    aux_loss = scaled_z_loss if aux_loss is None else aux_loss + scaled_z_loss

            # merge the shared and standard experts
            self.batch_size_per_expert += tot_batch_size_per_expert
            if self.bias_gamma is not None:
                assert self.score_bias_batch_size_per_expert is not None
                self.score_bias_batch_size_per_expert += tot_batch_size_per_expert

        # in the end, we add on the shared experts to both expert_weights and expert_indices
        if self.num_shared_experts > 0:
            expert_weights = torch.cat([expert_weights_standard_exp, expert_weights_shared_exp], dim=-1)
            expert_indices = torch.cat([expert_indices_standard_exp, expert_indices_shared_exp + self.num_choose_experts_pool], dim=-1) # we need to shift the indices of the shared experts since they come after the standard experts in the original logits

        return expert_weights, expert_indices, tot_batch_size_per_expert, aux_loss

    def extra_repr(self):
        """Add custom parameter to string representation."""
        base_repr = super().extra_repr()
        return f"{base_repr}, document_expert_pool={self.document_expert_pool}, eos_token_id={self.eos_token_id}"


@dataclass
class MoETwoLevelBatchLBReduceDPSharedExpPoolRouterConfig(MoETwoLevelRouterConfig):
    num_shared_experts: int = 2 # the number of experts to share
    num_shared_experts_pool: int = 2
    shared_exp_lb_loss: Optional[float] = None

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
    ) -> MoETwoLevelBatchLBReduceDPSharedExpPoolRouter:
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

        return MoETwoLevelBatchLBReduceDPSharedExpPoolRouter(**kwargs)
