"""EMO router with a LEARNED per-document expert-pool size d (repo-side subclass of the pinned
OLMo-core EmoRouterV2; nothing in the submodule is modified).

Per MoE layer, each document's mean-pooled hidden state (detached by default) goes through a
``d_head = Linear(d_model, 1)`` to a continuous pool size

    d_soft = d_min + (d_max - d_min) * sigmoid(d_head(mean_tokens(h)))       (d_min = min pool, d_max = max pool)

The forward pass uses the HARD pool ``d_hard = round(d_soft)``: as in EmoRouterV2 the experts are
ranked per document by their summed softmax scores and only the top ``d_hard`` are routable; the
token top-k is unchanged. The pool mask that multiplies the routing scores is a straight-through
estimator over expert ranks r (1-based):

    soft_mask[r] = sigmoid((d_soft - r + 0.5) / T),   hard_mask[r] = 1[r <= d_hard]
    mask = soft_mask + (hard_mask - soft_mask).detach()

so the LM loss reaches d_head only through the experts near the pool boundary ("boundary-only"
gradient). A size penalty pushes the pools small:

    d_loss = lambda_d * mean_docs((d_soft - d_min) / (d_max - d_min))

and is added to the router's auxiliary loss (next to the load-balancing and z losses).

Warm-up (``warmup_steps`` > 0, driven by :class:`LearnedDScheduleCallback`): training starts with NO
pool restriction and phases it in — the hard pool is floored at ``d_max -> d_min`` linearly over the
warm-up and ``lambda_d`` ramps ``0 -> lambda_d`` over the same window (each ramp can be switched off).

Eval: ``eval_mode='predicted'`` (default) uses the head's ``round(d_soft)``; ``'fixed'`` uses the
plain EMO ``eval_document_expert_pool`` (e.g. the full expert set), so a checkpoint can be evaluated
both ways.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import olmo_core.ops.moe as ops
from olmo_core.config import Config
from olmo_core.distributed.utils import get_local_tensor, hide_from_torch, unhide_from_torch
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.router import MoERouterGatingFunction
from olmo_core.nn.moe.v2.emo_router import EmoRouterV2
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.train.callbacks.callback import Callback


@dataclass
class LearnedDConfig(Config):
    temperature: float = 2.0
    """STE softness in units of expert ranks."""
    lambda_d: float = 0.01
    """Weight of the normalised pool-size penalty."""
    warmup_steps: int = 0
    """Steps over which the pool restriction is phased in (0 = restricted from step 1)."""
    floor_warmup: bool = True
    """During warm-up, floor the hard pool at d_max -> d_min (linear)."""
    lambda_warmup: bool = True
    """During warm-up, ramp lambda_d from 0 -> lambda_d (linear)."""
    init_pool: Optional[float] = None
    """d_soft at initialisation (default 0.95 * d_max; d_head bias set accordingly, weights ~0)."""
    detach_doc_embedding: bool = True
    """Detach the mean-pooled hidden state before d_head (no gradient into the trunk from d)."""
    eval_mode: str = "predicted"
    """'predicted' (round(d_soft) from the head) or 'fixed' (EmoRouterConfig.eval_pool_size())."""


@dataclass
class LearnedDRouterConfigV2(MoERouterConfigV2):
    learned_d: Optional[LearnedDConfig] = None

    def num_params(self) -> int:
        return super().num_params() + (self.d_model + 64 if self.learned_d is not None else 0)

    def build(self, init_device: str = "cpu"):
        if self.learned_d is None or self.emo is None:
            return super().build(init_device=init_device)
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        learned_d = kwargs.pop("learned_d")
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        return LearnedDEmoRouterV2(learned_d=learned_d, **kwargs, init_device=init_device)


class LearnedDEmoRouterV2(EmoRouterV2):
    def __init__(self, *, learned_d: LearnedDConfig, init_device: str = "cpu", **kwargs):
        super().__init__(init_device=init_device, **kwargs)
        if learned_d.eval_mode not in ("predicted", "fixed"):
            raise OLMoConfigurationError(f"learned_d.eval_mode must be 'predicted' or 'fixed', got {learned_d.eval_mode!r}")
        if learned_d.temperature <= 0:
            raise OLMoConfigurationError("learned_d.temperature must be > 0")
        self.learned_d = learned_d
        self.d_weight = nn.Parameter(torch.empty(self.d_model, device=init_device, dtype=self.weight.dtype))
        # 64 entries, only [0] is used: the MoE optimizer packs the (bf16) parameters back-to-back in one
        # flat buffer with no padding and inductor requires 16-byte-aligned inputs, so a 1-element
        # parameter would misalign every parameter that follows it (64 keeps the numel divisible by any
        # DP world size as well).
        self.d_bias = nn.Parameter(torch.empty(64, device=init_device, dtype=self.weight.dtype))
        self._d_sched = hide_from_torch(torch.zeros(2, device=self.device))
        self._d_stats = hide_from_torch(torch.zeros(7, device=self.device))
        self._reset_learned_d()

    # -- pool range -------------------------------------------------------------------------
    @property
    def d_min(self) -> int:
        return int(self.emo.min_document_expert_pool)

    @property
    def d_max(self) -> int:
        return int(self.emo.max_document_expert_pool)

    def _reset_learned_d(self):
        init_pool = self.learned_d.init_pool if self.learned_d.init_pool is not None else 0.95 * self.d_max
        init_pool = min(max(float(init_pool), self.d_min + 0.5), self.d_max - 0.5)
        frac = (init_pool - self.d_min) / (self.d_max - self.d_min)
        with torch.no_grad():
            nn.init.trunc_normal_(self.d_weight, std=1e-3, a=-3e-3, b=3e-3)
            self.d_bias.fill_(math.log(frac / (1.0 - frac)))
        # schedule state: [lambda scale, hard-pool floor]; no warm-up -> fully restricted
        self._d_sched = hide_from_torch(torch.tensor([1.0, float(self.d_min)], device=self.device))
        self.set_schedule(step=None)
        self._d_stats = hide_from_torch(torch.zeros(7, device=self.device))

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "d_bias"):  # the base constructor calls this before the head exists
            self._reset_learned_d()

    @torch.no_grad()
    def set_schedule(self, step: Optional[int]):
        """Warm-up state for the given global step (None = past warm-up)."""
        W = self.learned_d.warmup_steps
        frac = 1.0 if (step is None or W <= 0) else min(1.0, max(0.0, step / W))
        lam_scale = frac if self.learned_d.lambda_warmup else 1.0
        floor = self.d_max - (self.d_max - self.d_min) * frac if self.learned_d.floor_warmup else self.d_min
        sched = unhide_from_torch(self._d_sched)
        if sched.device != self.device:
            sched = sched.to(self.device)
            self._d_sched = hide_from_torch(sched)
        sched.copy_(torch.tensor([lam_scale, float(round(floor))], device=sched.device))

    # -- pool size prediction ---------------------------------------------------------------
    def predict_pool(self, x: torch.Tensor, segment_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (d_soft per token, d_soft per document slot, valid document-slot mask)."""
        B, S, D = x.shape
        h = x.detach() if self.learned_d.detach_doc_embedding else x
        h = h.float()
        idx = segment_ids.unsqueeze(-1).expand(-1, -1, D)
        doc_sum = torch.zeros(B, S, D, dtype=h.dtype, device=h.device).scatter_add_(1, idx, h)
        doc_cnt = torch.zeros(B, S, dtype=h.dtype, device=h.device).scatter_add_(1, segment_ids, torch.ones_like(segment_ids, dtype=h.dtype))
        valid = doc_cnt > 0
        doc_mean = doc_sum / doc_cnt.clamp(min=1.0).unsqueeze(-1)
        s = doc_mean @ self.d_weight.float() + self.d_bias[0].float()  # (B, S) one logit per document slot
        d_soft_doc = self.d_min + (self.d_max - self.d_min) * torch.sigmoid(s)
        d_soft_tok = d_soft_doc.gather(1, segment_ids)
        return d_soft_tok, d_soft_doc, valid

    # -- forward ----------------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        scores_only: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ):
        if scores_only:
            raise OLMoConfigurationError("learned-d EMO is only supported for the routed-expert router")
        if segment_ids is None:
            raise OLMoConfigurationError("EMO routing requires per-token segment_ids")
        if segment_ids.shape != x.shape[:2]:
            raise OLMoConfigurationError(f"segment_ids shape {tuple(segment_ids.shape)} must match token shape {tuple(x.shape[:2])}")

        x = self.jitter(x)
        logits = self.get_expert_logits(x.float()).float()
        if self.gating_function in (MoERouterGatingFunction.softmax, MoERouterGatingFunction.topk_softmax):
            scores = logits.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = logits.sigmoid()
            if self.sigmoid_stability_epsilon:
                scores = scores + self.sigmoid_stability_epsilon
        else:
            raise NotImplementedError(self.gating_function)

        document_scores = ops.doc_sum_scatter(scores, segment_ids)
        ranks = document_scores.argsort(dim=-1, descending=True).argsort(dim=-1)  # 0 = best expert of the doc

        d_info = None
        if not self.training and self.learned_d.eval_mode == "fixed":
            pool = torch.full_like(segment_ids, self.emo.eval_pool_size())
            keep = ranks < pool.unsqueeze(-1)
            mask = keep.to(scores.dtype)
            d_soft_tok = pool.to(scores.dtype)
        else:
            d_soft_tok, d_soft_doc, valid = self.predict_pool(x, segment_ids)
            d_hard = d_soft_tok.round().clamp(self.d_min, self.d_max)
            if self.training:
                sched = unhide_from_torch(self._d_sched)
                d_hard = torch.maximum(d_hard, sched[1])
            keep = ranks < d_hard.unsqueeze(-1)
            if self.training and torch.is_grad_enabled():
                soft = torch.sigmoid((d_soft_tok.unsqueeze(-1) - ranks.to(scores.dtype) - 0.5) / self.learned_d.temperature)
                hard = keep.to(scores.dtype)
                mask = soft + (hard - soft).detach()
                d_info = (d_soft_doc, valid, d_hard)
            else:
                mask = keep.to(scores.dtype)

        masked_scores = scores * mask
        if self.gating_function == MoERouterGatingFunction.topk_softmax:
            selection_logits = logits.masked_fill(~keep, float("-inf"))
            _, expert_indices = selection_logits.topk(self.top_k, dim=-1)
            expert_weights = logits.gather(-1, expert_indices).softmax(dim=-1) * mask.gather(-1, expert_indices)
        else:
            selection_scores = masked_scores.masked_fill(~keep, float("-inf"))
            _, expert_indices = selection_scores.topk(self.top_k, dim=-1)
            expert_weights = masked_scores.gather(-1, expert_indices)

        if self.normalize_expert_weights is not None:
            expert_weights = F.normalize(expert_weights, p=self.normalize_expert_weights, dim=-1)
        if self.restore_weight_scale:
            expert_weights = expert_weights * self.top_k
        if self.expert_weight_scale is not None:
            expert_weights = expert_weights * self.expert_weight_scale
        if self.original_top_k is not None and self.top_k != self.original_top_k:
            expert_weights = expert_weights * (self.original_top_k / self.top_k) ** 0.5

        with torch.no_grad():
            batched_counts = ops.batched_histc(expert_indices, self.num_experts).sum(dim=1)
            counts = batched_counts.sum(dim=0)

        self._last_pool = d_soft_tok.detach()  # per-token predicted pool (analysis / extractor)
        aux_loss_info = (scores, logits, counts, batched_counts, loss_div_factor, d_info)
        return expert_weights, expert_indices, counts, aux_loss_info

    # -- auxiliary loss + metrics ----------------------------------------------------------
    def compute_aux_loss(
        self,
        scores,
        logits,
        batch_size_per_expert,
        batched_batch_size_per_expert,
        loss_div_factor,
        d_info=None,
        *,
        accumulate_metrics: bool = True,
    ) -> Optional[torch.Tensor]:
        aux_loss = super().compute_aux_loss(
            scores, logits, batch_size_per_expert, batched_batch_size_per_expert, loss_div_factor,
            accumulate_metrics=accumulate_metrics,
        )
        if d_info is None or not (self.training and torch.is_grad_enabled()):
            return aux_loss
        d_soft_doc, valid, d_hard = d_info
        n_docs = valid.sum().clamp(min=1).to(d_soft_doc.dtype)
        norm = ((d_soft_doc - self.d_min) / (self.d_max - self.d_min)) * valid.to(d_soft_doc.dtype)
        d_mean_norm = norm.sum() / n_docs  # mean over the documents of this micro-batch
        # scale so that summing over micro-batches gives the batch-level mean (as the LM / z losses do)
        n_tok = float(scores.shape[0] * scores.shape[1])
        if loss_div_factor is None:
            loss_div_factor = n_tok
        d_loss = d_mean_norm * (n_tok / loss_div_factor)
        sched = unhide_from_torch(self._d_sched)
        scaled = self.learned_d.lambda_d * sched[0] * d_loss
        if accumulate_metrics:
            stats = self.d_stats
            with torch.no_grad():
                vd = valid.to(d_soft_doc.dtype)
                stats[0] += (d_soft_doc * vd).sum()          # sum of d_soft over documents
                stats[1] += d_hard.sum()                     # sum of the hard pool over tokens
                stats[2] += vd.sum()                         # documents
                stats[3] += get_local_tensor(scaled.detach())
                stats[4] += get_local_tensor(d_loss.detach())
                stats[5] += ((d_soft_doc <= 64) & valid).sum().to(stats.dtype)
                stats[6] += n_tok
        return scaled if aux_loss is None else aux_loss + scaled

    @property
    def d_stats(self) -> torch.Tensor:
        if unhide_from_torch(self._d_stats).device != self.device:
            self._d_stats = hide_from_torch(unhide_from_torch(self._d_stats).to(self.device))
        return unhide_from_torch(self._d_stats)

    def compute_metrics(self, reset: bool = True) -> Dict[str, Tuple[torch.Tensor, Optional[Any]]]:
        from olmo_core.train.common import ReduceType

        out = super().compute_metrics(reset=False)
        stats = self.d_stats
        sched = unhide_from_torch(self._d_sched)
        n = stats[2].clamp(min=1.0)
        out["emo d soft mean"] = (stats[0] / n, ReduceType.mean)
        out["emo d hard mean (tokens)"] = (stats[1] / stats[6].clamp(min=1.0), ReduceType.mean)
        out["emo d frac<=64"] = (stats[5] / n, ReduceType.mean)
        out["emo d loss"] = (stats[3].clone(), ReduceType.mean)
        out["emo d loss unscaled"] = (stats[4].clone(), ReduceType.mean)
        out["emo d lambda scale"] = (sched[0].clone(), ReduceType.mean)
        out["emo d floor"] = (sched[1].clone(), ReduceType.mean)
        if reset:
            self.reset_metrics()
        return out

    def reset_metrics(self):
        super().reset_metrics()
        self.d_stats.zero_()

    def extra_repr(self) -> str:
        ld = self.learned_d
        return (f"{super().extra_repr()}, learned_d(T={ld.temperature}, lambda={ld.lambda_d}, warmup={ld.warmup_steps}, "
                f"init={ld.init_pool}, eval={ld.eval_mode})")


@dataclass
class LearnedDScheduleCallback(Callback):
    """Pushes the trainer's global step into every learned-d router (warm-up schedule)."""

    priority: ClassVar[int] = 10  # before other callbacks on the step

    def _routers(self):
        model = self.trainer.train_module.model
        return [m for m in model.modules() if isinstance(m, LearnedDEmoRouterV2)]

    def pre_train(self):
        for r in self._routers():
            r.set_schedule(self.step)

    def pre_step(self, batch: Dict[str, Any]):
        del batch
        for r in self._routers():
            r.set_schedule(self.step)
