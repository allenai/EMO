from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

import olmo_core.ops.moe as ops
from olmo_core.distributed.utils import is_distributed
from olmo_core.nn.moe.router import (
    MoELinearRouter,
    MoERouterConfig,
    MoERouterGatingFunction,
)

from .loss import MoELoadBalancingLossGranularity, load_balancing_loss, router_z_loss


class MoELinearLBReduceDPRouter(MoELinearRouter):
    """
    Custom MoE router with modified forward pass and additional class variables.
    """

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        padding_mask: Optional[torch.Tensor] = None,  # shape: (B, S)
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Given the input ``x`` of shape ``(B, S, d_model)``, compute the experts assignment.

        :returns: The expert weights of shape ``(B, S, top_k)``,
            the expert indices of shape ``(B, S, top_k)``,
            the total number of items routed to each expert, with shape ``(num_experts,)``,
            and optionally the auxiliary losses.
        """

        # make sure tp and cp are not enabled
        if self.tp_mesh is not None:
            raise NotImplementedError(
                "Tensor parallelism is not supported in MoETwoLevelBatchLBReduceDPRouter."
            )
        if self.cp_mesh is not None:
            raise NotImplementedError(
                "Context parallelism is not supported in MoETwoLevelBatchLBReduceDPRouter."
            )

        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x).float()

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
            # we first make the assertion that we are using granularity of local_batch as opposed to instance, since masking doesn't work with instance
            if self.lb_loss_granularity != MoELoadBalancingLossGranularity.local_batch:
                raise NotImplementedError(
                    "masking with instance-level load balancing loss granularity is not supported yet"
                )

            # Histogram the expert ids to identify the number of items/tokens routed to each expert. This is ONLY USED in
            # the return values used for kernels to route tokens to their corresponding experts, NOT for loss computation
            # shape: (batch_size, seq_len, num_experts)
            batched_batch_size_per_expert_routing = ops.batched_histc(
                expert_indices, self.num_experts
            )
            # shape: (batch_size, num_experts)
            batched_batch_size_per_expert_routing = batched_batch_size_per_expert_routing.sum(dim=1)
            # shape: (num_experts,)
            batch_size_per_expert_routing = batched_batch_size_per_expert_routing.sum(dim=0)

            # we first filter out the padding tokens (also includes masked tokens)
            if padding_mask is not None:
                padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(expert_indices)
                valid_expert_indices = expert_indices.masked_select(padding_mask_expanded).view(
                    -1, expert_indices.size(-1)
                )
                padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(scores)
                valid_scores = scores.masked_select(padding_mask_expanded).view(
                    -1, self.num_experts
                )
                # valid_logits = logits.masked_select(padding_mask_expanded).view(
                #     -1, self.num_experts
                # )

                # (valid_tokens, num_experts)
                batched_batch_size_per_expert = ops.batched_histc(
                    valid_expert_indices, self.num_experts
                )
                # (num_experts)
                batch_size_per_expert = batched_batch_size_per_expert.sum(dim=0)
            else:
                valid_expert_indices = expert_indices.view(-1, expert_indices.size(-1))
                valid_scores = scores.view(-1, self.num_experts)
                # valid_logits = logits.view(-1, self.num_experts)

                batch_size_per_expert = batch_size_per_expert_routing

            # prepare for custom metric
            if self.training:
                # prepare unique experts metric.
                unique_experts = torch.unique(valid_expert_indices.view(-1))
                num_unique_experts = unique_experts.numel()

                self._unique_experts_sum += num_unique_experts
                self._num_batches_tracked += 1

                # Compute router distribution entropy metric
                # calculate entropy of the router distribution over experts
                # get entropy per token
                token_entropies = -torch.sum(valid_scores * torch.log(valid_scores + 1e-10), dim=-1)
                # average entropy over valid tokens
                avg_entropy = token_entropies.mean().item()
                self._router_tokenlevel_expert_entropy += avg_entropy

        # Maybe compute auxiliary losses and accumulate metrics.
        aux_loss: Optional[torch.Tensor] = None
        if self.training and torch.is_grad_enabled():
            with torch.autocast(enabled=False, device_type=x.device.type):
                if self.lb_loss_weight is not None:
                    assert self.load_balancing_loss is not None

                    # we make some extra checks here that gating_function is softmax and that loss_div_factor is set (required for new loss to work)
                    if self.gating_function != MoERouterGatingFunction.softmax:
                        raise NotImplementedError(
                            "load balancing loss currently only supported for softmax gating function"
                        )
                    if loss_div_factor is None:
                        raise ValueError(
                            "loss_div_factor must be set when using load balancing loss"
                        )

                    # Make sure scores are normalized, otherwise load balancing loss doesn't work well. (this SHOULD NOT run)
                    if self.gating_function == MoERouterGatingFunction.sigmoid:
                        scores = scores / scores.sum(dim=-1, keepdim=True)

                    # we now do reduction on the tot_batch_size_per_expert to get a dp-global lb loss (still not full global sinze we don't do across gradient accumulation steps)
                    dp_global_batch_size_per_expert_routing = (
                        batch_size_per_expert_routing.clone()
                    )  # we clone to not interfere with logging or other routing stuff
                    assert isinstance(loss_div_factor, torch.Tensor)
                    dp_global_loss_div_factor = loss_div_factor.clone()

                    if is_distributed():
                        dist.all_reduce(
                            dp_global_batch_size_per_expert_routing, op=dist.ReduceOp.SUM
                        )
                        dist.all_reduce(dp_global_loss_div_factor, op=dist.ReduceOp.SUM)

                    # collect all non-zero entries of the dp_global_batch_size_per_expert_routing
                    self._reducedp_unique_experts_sum += (
                        (dp_global_batch_size_per_expert_routing > 0).sum().item()
                    )

                    lb_loss = load_balancing_loss(
                        num_experts=self.num_experts,
                        top_k=self.top_k,
                        expert_scores=scores,
                        # expert_scores=valid_scores,
                        batch_size_per_expert=dp_global_batch_size_per_expert_routing,
                        # batch_size_per_expert=batch_size_per_expert,
                        batched_batch_size_per_expert=batched_batch_size_per_expert,  # we don't even use this in local_batch granularity, but we pass it anyway
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

                    z_loss = router_z_loss(
                        expert_logits=logits,
                        # expert_logits=valid_logits,
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

        return expert_weights, expert_indices, batch_size_per_expert_routing, aux_loss


@dataclass
class MoELinearLBReduceDPRouterConfig(MoERouterConfig):
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
    ) -> MoELinearLBReduceDPRouter:
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

        return MoELinearLBReduceDPRouter(**kwargs)
