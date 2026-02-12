from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

import olmo_core.ops.moe as ops
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.router import MoERouterGatingFunction
from olmo_core.nn.moe.twolevel_router import MoETwoLevelRouter, MoETwoLevelRouterConfig

from .loss import MoELoadBalancingLossGranularity, router_z_loss


class MoETwoLevelSamplingNoLBRouter(MoETwoLevelRouter):
    """
    Custom MoE router with modified forward pass and additional class variables.
    """

    # remove lb_loss_weight from init
    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        lb_loss_weight: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, init_device=init_device, **kwargs)

        if lb_loss_weight is not None and lb_loss_weight != 0.0:
            raise OLMoConfigurationError(
                "MoETwoLevelSamplingNoLBRouter does not support load balancing loss. "
                "Please set lb_loss_weight to None or 0.0."
            )

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
        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x).float()
        scores_mask = torch.ones_like(logits, dtype=torch.bool, device=logits.device)

        assert document_boundaries is not None
        document_boundaries_cpu = []
        for b in document_boundaries:
            bc = b.detach().cpu().tolist()
            if not bc or bc[-1] != x.size(1):
                bc.append(int(x.size(1)))
            document_boundaries_cpu.append(bc)

        tot_doc_entropy = []
        for seq_idx in range(x.size(0)):
            start = 0
            document_boundary = document_boundaries_cpu[seq_idx]
            for end in document_boundary:
                if end <= start:
                    start = end
                    continue
                sequence_logits = logits[seq_idx, start:end, :]  # shape: (doc_len, num_experts)
                # calculate the softmax over the experts
                expert_probs = F.softmax(sequence_logits, dim=-1)  # shape: (doc_len, num_experts)

                # get the entropy over experts per token
                token_entropies = -torch.sum(
                    expert_probs * torch.log(expert_probs + 1e-10), dim=-1
                )  # shape: (doc_len,)
                # average entropy over the document
                avg_entropy = token_entropies.mean().item()
                tot_doc_entropy.append(avg_entropy)

                # take the sum across the document
                document_expert_probs = expert_probs.sum(dim=0)  # shape: (num_experts,)
                # sample to select the experts for this document
                experts_to_keep = torch.multinomial(
                    document_expert_probs, self.document_expert_pool, replacement=False
                )  # shape: (document_expert_pool,)
                # we now only keep these experts for this document, set the rest in scores_mask to True
                scores_mask[seq_idx, start:end, experts_to_keep] = False
                start = end

        if self.training:
            # log the average document entropy
            avg_doc_entropy = (
                sum(tot_doc_entropy) / len(tot_doc_entropy) if tot_doc_entropy else 0.0
            )
            # logging.info(f"Average document entropy over experts: {avg_doc_entropy}")
            self._router_documentlevel_expert_entropy += avg_doc_entropy

        # logits.masked_fill_(logits_mask, float('-inf'))

        # shape: (batch_size, seq_len, num_experts)
        if self.gating_function == MoERouterGatingFunction.softmax:
            scores = logits.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = F.sigmoid(logits) + 1e-7
        else:
            raise NotImplementedError(self.gating_function)

        raise NotImplementedError(
            "MoETwoLevelSamplingNoLBRouter is not yet fully implemented. CANNOT mask scores without renormalization"
        )
        # mask out the experts not selected for each document. we mask scores instead of logits to allow z-loss computation
        scores = scores.masked_fill(scores_mask, 0.0)
        # scores.masked_fill_(scores_mask, 0.0)

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
            tot_batched_batch_size_per_expert = ops.batched_histc(expert_indices, self.num_experts)
            # shape: (batch_size, num_experts)
            tot_batched_batch_size_per_expert = tot_batched_batch_size_per_expert.sum(dim=1)
            # shape: (num_experts,)
            tot_batch_size_per_expert = tot_batched_batch_size_per_expert.sum(dim=0)

            # prepare for custom metric
            if self.training:
                # prepare unique experts metric
                if padding_mask is not None:
                    padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(expert_indices)
                    valid_expert_indices = expert_indices.masked_select(padding_mask_expanded)
                else:
                    valid_expert_indices = expert_indices.reshape(-1)

                # Update unique experts metric.
                unique_experts = torch.unique(valid_expert_indices)
                num_unique_experts = unique_experts.numel()

                self._unique_experts_sum += num_unique_experts
                self._num_batches_tracked += 1

                # Compute router distribution entropy metric
                # calculate entropy of the router distribution over experts. NOTE: this should be much lower than document-level, since some experts are already masked out
                if padding_mask is not None:
                    # only consider non-padded tokens
                    padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(scores)
                    valid_scores = scores.masked_select(padding_mask_expanded).view(
                        -1, self.num_experts
                    )
                else:
                    valid_scores = scores.view(-1, self.num_experts)
                # get entropy per token
                token_entropies = -torch.sum(valid_scores * torch.log(valid_scores + 1e-10), dim=-1)
                # average entropy over valid tokens
                avg_entropy = token_entropies.mean().item()
                self._router_tokenlevel_expert_entropy += avg_entropy

        # Maybe compute auxiliary losses and accumulate metrics.
        aux_loss: Optional[torch.Tensor] = None
        if self.training and torch.is_grad_enabled():
            with torch.autocast(enabled=False, device_type=x.device.type):
                if self.z_loss_weight is not None:
                    assert self.z_loss is not None

                    # we use the full logits (without any masking) for z-loss computation
                    z_loss = router_z_loss(
                        expert_logits=logits,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.z_loss += z_loss.detach()

                    scaled_z_loss = self.z_loss_weight * z_loss
                    aux_loss = scaled_z_loss if aux_loss is None else aux_loss + scaled_z_loss

            self.batch_size_per_expert += tot_batch_size_per_expert
            if self.bias_gamma is not None:
                assert self.score_bias_batch_size_per_expert is not None
                self.score_bias_batch_size_per_expert += tot_batch_size_per_expert

        return expert_weights, expert_indices, tot_batch_size_per_expert, aux_loss

    def extra_repr(self):
        """Add custom parameter to string representation."""
        base_repr = super().extra_repr()
        return f"{base_repr}, document_expert_pool={self.document_expert_pool}, eos_token_id={self.eos_token_id}"


@dataclass
class MoETwoLevelSamplingNoLBRouterConfig(MoETwoLevelRouterConfig):
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
    ) -> MoETwoLevelSamplingNoLBRouter:
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

        return MoETwoLevelSamplingNoLBRouter(**kwargs)
