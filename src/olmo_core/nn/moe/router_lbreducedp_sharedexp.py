from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

import olmo_core.ops.moe as ops
from olmo_core.distributed.utils import is_distributed
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.router import (
    MoELinearRouter,
    MoERouterConfig,
    MoERouterGatingFunction,
)

from .loss import MoELoadBalancingLossGranularity, load_balancing_loss, router_z_loss


class MoELinearLBReduceDPSharedExpRouter(MoELinearRouter):
    """
    Standard MoE router with DP-reduced load balancing and shared experts.

    Shared experts are always active for every token (weight=1.0).
    The router selects top-(k - num_shared_experts) from the non-shared expert pool,
    then appends the shared experts. Load balancing loss is computed only over
    the non-shared experts.
    """

    def __init__(
        self,
        *,
        num_shared_experts: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # check that the number of shared experts is less than the top_k
        if num_shared_experts > self.top_k:
            raise OLMoConfigurationError(
                f"num_shared_experts ({num_shared_experts}) must be less than or equal to top_k ({self.top_k})"
            )

        self.num_shared_experts = num_shared_experts
        self.num_choose_experts = self.top_k - self.num_shared_experts

    def get_top_k(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override to select num_choose_experts instead of top_k, since shared experts are always active."""
        if self.bias_gamma is None:
            if self.num_choose_experts == 1:
                expert_weights, expert_indices = scores.max(dim=-1, keepdim=True)
            else:
                expert_weights, expert_indices = torch.topk(scores, self.num_choose_experts, dim=-1)
        else:
            assert self.score_bias is not None
            with torch.no_grad():
                _, expert_indices = torch.topk(
                    scores + self.score_bias.unsqueeze(0), self.num_choose_experts, dim=-1
                )
            expert_weights = scores.gather(-1, expert_indices)

        if self.uniform_expert_assignment:
            raise NotImplementedError(
                "Uniform expert assignment is not supported in MoELinearLBReduceDPSharedExpRouter."
            )

        return expert_weights, expert_indices

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
                "Tensor parallelism is not supported in MoELinearLBReduceDPSharedExpRouter."
            )
        if self.cp_mesh is not None:
            raise NotImplementedError(
                "Context parallelism is not supported in MoELinearLBReduceDPSharedExpRouter."
            )

        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x).float()

        # Remove shared experts from logits before routing (shared experts are the last num_shared_experts)
        num_standard = self.num_experts - self.num_shared_experts
        logits = logits[:, :, :num_standard]  # shape: (batch_size, seq_len, num_standard)

        # shape: (batch_size, seq_len, num_standard)
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
            # shape: (batch_size, seq_len, num_standard)
            batched_batch_size_per_expert_routing = ops.batched_histc(expert_indices, num_standard)
            # shape: (batch_size, num_standard)
            batched_batch_size_per_expert_routing = batched_batch_size_per_expert_routing.sum(dim=1)
            # shape: (num_standard,)
            batch_size_per_expert_routing = batched_batch_size_per_expert_routing.sum(dim=0)

            # Compute batch_size_per_expert excluding padding tokens, using static-shape
            # ops (weighted scatter_add instead of masked_select) to avoid graph breaks
            # in torch.compile.
            if padding_mask is not None:
                # Use padding_mask as weights so padding tokens contribute 0 to the histogram
                weights = (
                    padding_mask.unsqueeze(-1).expand_as(expert_indices).to(expert_indices.dtype)
                )
                hist = torch.zeros(
                    (*expert_indices.shape[:-1], num_standard),
                    dtype=expert_indices.dtype,
                    device=expert_indices.device,
                )
                hist.scatter_add_(-1, expert_indices, weights)
                # (batch_size, num_standard)
                batched_batch_size_per_expert = hist.sum(dim=1)
                # (num_standard,)
                batch_size_per_expert = batched_batch_size_per_expert.sum(dim=0)
            else:
                batch_size_per_expert = batch_size_per_expert_routing
                batched_batch_size_per_expert = batched_batch_size_per_expert_routing

            # prepare for custom metric
            if self.training:
                # Count unique experts via histogram (avoids dynamic-shape torch.unique)
                num_unique_experts = (
                    batch_size_per_expert > 0
                ).sum().item() + self.num_shared_experts

                self._unique_experts_sum += num_unique_experts
                self._num_batches_tracked += 1

                # Compute router distribution entropy metric using masked mean
                # to avoid dynamic-shape masked_select
                token_entropies = -torch.sum(scores * torch.log(scores + 1e-10), dim=-1)  # (B, S)
                if padding_mask is not None:
                    token_entropies = token_entropies * padding_mask.float()
                    num_valid = padding_mask.sum().clamp(min=1)
                    avg_entropy = (token_entropies.sum() / num_valid).item()
                else:
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
                    dp_global_loss_div_factor = loss_div_factor.clone()

                    if is_distributed():
                        dist.all_reduce(
                            dp_global_batch_size_per_expert_routing, op=dist.ReduceOp.SUM
                        )
                        dist.all_reduce(dp_global_loss_div_factor, op=dist.ReduceOp.SUM)

                    # collect all non-zero entries of the dp_global_batch_size_per_expert_routing (add shared experts since they are always active)
                    self._reducedp_unique_experts_sum += (
                        dp_global_batch_size_per_expert_routing > 0
                    ).sum().item() + self.num_shared_experts

                    lb_loss = load_balancing_loss(
                        num_experts=num_standard,  # LB loss over non-shared experts only
                        top_k=self.num_choose_experts,  # only choose num_choose_experts
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

            if self.batch_size_per_expert.shape[-1] != batch_size_per_expert.shape[-1]:
                # make sure that the shared expert positions are zero, since it means the parameter was reset
                extra_counts = self.batch_size_per_expert[batch_size_per_expert.shape[-1] :]
                assert torch.all(
                    extra_counts == 0
                ), f"Expected extra counts to be zero, but got {extra_counts}"
                self.batch_size_per_expert = self.batch_size_per_expert[
                    : batch_size_per_expert.shape[-1]
                ]
            self.batch_size_per_expert += batch_size_per_expert
            if self.bias_gamma is not None:
                assert self.score_bias_batch_size_per_expert is not None
                self.score_bias_batch_size_per_expert += batch_size_per_expert

        # Append shared experts: weight=1.0, indices = last num_shared_experts experts
        if self.num_shared_experts > 0:
            expert_weights = F.pad(expert_weights, (0, self.num_shared_experts), value=1.0)
            shared_expert_indices = (
                torch.arange(num_standard, self.num_experts, device=expert_indices.device)
                .view(1, 1, self.num_shared_experts)
                .expand(expert_indices.size(0), expert_indices.size(1), self.num_shared_experts)
            )
            expert_indices = torch.cat([expert_indices, shared_expert_indices], dim=-1)
            # Shared experts are always active for all tokens
            batch_size_per_expert_routing = F.pad(
                batch_size_per_expert_routing,
                (0, self.num_shared_experts),
                value=x.size(0) * x.size(1),
            )

        return expert_weights, expert_indices, batch_size_per_expert_routing, aux_loss


@dataclass
class MoELinearLBReduceDPSharedExpRouterConfig(MoERouterConfig):
    num_shared_experts: int = 1

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
    ) -> MoELinearLBReduceDPSharedExpRouter:
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

        return MoELinearLBReduceDPSharedExpRouter(**kwargs)
