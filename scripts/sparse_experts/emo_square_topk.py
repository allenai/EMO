"""EMO router that keeps fewer experts than it normalises over (repo-side subclass of the pinned
``EmoRouterV2``; user request 2026-11-06, box "sparser squares" of the Disjoint Training Report).

A sub-model of a K-way random split keeps the full model's top-16 while owning only 1/K of the
experts (16 of 64 for K = 8), so it is far denser than the model it is merged back into (16 of
512). Two ways to restore the sparsity inside a sub-model:

* plain top-k (no subclass needed): ``top_k = 4``. The stock router L1-normalises the four
  selected scores and multiplies by 4 (``restore_weight_scale``): weights sum to 4.
* ``norm_top_k = 16`` with ``top_k = 4`` (this class): the weights are computed exactly as in the
  top-16 router (the 16 selected scores L1-normalised, times 16) and only the 4 largest are kept,
  NOT renormalised. Every kept expert enters with the weight it would have in the top-16 model;
  the mass of the 12 dropped experts is simply missing.

Selection, the per-document pool, the auxiliary losses and the load counts are the stock EMO
router's (counts are of the ``top_k`` experts that actually run). Enabled per block from
``olmoe3_275m.py`` (``OLMOE3_SQUARE_TOP_K`` / ``OLMOE3_SQUARE_NORM_TOP_K``, cut-down layers only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

import olmo_core.ops.moe as ops
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.router import MoERouterGatingFunction
from olmo_core.nn.moe.v2.emo_router import EmoRouterV2
from olmo_core.nn.moe.v2.router import MoERouterConfigV2


@dataclass
class SquareTopKRouterConfigV2(MoERouterConfigV2):
    norm_top_k: Optional[int] = None  # None = the stock router

    def build(self, init_device: str = "cpu"):
        if self.norm_top_k is None:
            return super().build(init_device=init_device)
        if self.emo is None:
            raise OLMoConfigurationError("norm_top_k is implemented for the EMO router only")
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        norm_top_k = kwargs.pop("norm_top_k")
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        return NormTopKEmoRouterV2(norm_top_k=norm_top_k, **kwargs, init_device=init_device)


class NormTopKEmoRouterV2(EmoRouterV2):
    def __init__(self, *, norm_top_k: int, init_device: str = "cpu", **kwargs):
        super().__init__(**kwargs, init_device=init_device)
        if not self.top_k <= norm_top_k <= self.emo.min_document_expert_pool:
            raise OLMoConfigurationError(
                f"need top_k ({self.top_k}) <= norm_top_k ({norm_top_k}) <= min_document_expert_pool "
                f"({self.emo.min_document_expert_pool}): the normalisation set must fit in every pool"
            )
        if self.gating_function != MoERouterGatingFunction.softmax:
            raise OLMoConfigurationError("norm_top_k supports the softmax gating function only")
        self.norm_top_k = norm_top_k

    def forward(
        self,
        x: torch.Tensor,
        scores_only: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if scores_only:
            raise OLMoConfigurationError("EMO is only supported for the routed-expert router")
        if segment_ids is None:
            raise OLMoConfigurationError("EMO routing requires per-token segment_ids")
        if segment_ids.shape != x.shape[:2]:
            raise OLMoConfigurationError(
                f"segment_ids shape {tuple(segment_ids.shape)} must match token shape {tuple(x.shape[:2])}"
            )

        x = self.jitter(x)
        logits = self.get_expert_logits(x.float()).float()
        scores = logits.softmax(dim=-1)

        document_scores = ops.doc_sum_scatter(scores, segment_ids)
        keep = ops.pool_keep_mask(document_scores, self._pool_sizes(segment_ids))

        # the stock top-``norm_top_k`` weights (topk returns them largest first) ...
        selection_scores = scores.masked_fill(~keep, float("-inf"))
        _, norm_indices = selection_scores.topk(self.norm_top_k, dim=-1)
        norm_weights = scores.gather(-1, norm_indices)
        if self.normalize_expert_weights is not None:
            norm_weights = F.normalize(norm_weights, p=self.normalize_expert_weights, dim=-1)
        if self.restore_weight_scale:
            norm_weights = norm_weights * self.norm_top_k
        if self.expert_weight_scale is not None:
            norm_weights = norm_weights * self.expert_weight_scale
        # ... of which only the ``top_k`` largest run, with their weights untouched
        expert_indices = norm_indices[..., : self.top_k].contiguous()
        expert_weights = norm_weights[..., : self.top_k].contiguous()

        with torch.no_grad():
            batched_counts = ops.batched_histc(expert_indices, self.num_experts).sum(dim=1)
            counts = batched_counts.sum(dim=0)

        aux_loss_info = (scores, logits, counts, batched_counts, loss_div_factor)
        return expert_weights, expert_indices, counts, aux_loss_info

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, norm_top_k={self.norm_top_k}"
