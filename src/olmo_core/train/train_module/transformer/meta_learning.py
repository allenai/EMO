"""
Meta-learning (first-order MAML / FOMAML) train module for EMO pretraining.

Per train step (``meta_mode`` = ``same_tokens`` or ``heldout``):

1. **Inner phase** (selective): forward+backward every inner micro-batch with each document's
   routing restricted to its router top-``inner_pool_size`` experts (the randpool router's
   per-document pool pinned via ``meta_force_pool``). After the backward cycle ``.grad`` holds
   the DP-reduced inner gradient aggregated over all documents in the global batch.
2. **Pseudo-step**: an in-place SGD probe on the expert weights only,
   ``theta'_exp = theta_exp - inner_lr * g_inner`` (router/attention/embeddings untouched).
   The probe is temporary — it determines where the outer gradient is evaluated and is undone
   before the real optimizer step.
3. **Outer phase** (full model): forward+backward the same micro-batches (``same_tokens``) or the
   other half of the rank's micro-batches (``heldout``) with all experts available
   (``meta_force_pool`` = number of non-shared experts => the pool keep-mask keeps everything).
   FSDP2's reduce-scatter *accumulates* into ``.grad``, so after this phase
   ``.grad = lambda_inner * g_inner + g_outer(theta')``.
4. **Restore**: expert weights are restored bitwise; the trainer's normal ``optim_step()`` then
   consumes the accumulated gradients (first-order: the gradient evaluated at ``theta'`` is
   applied to ``theta``).

``meta_mode="vanilla"`` delegates to the parent ``train_batch`` untouched (bit-identical vanilla
EMO — the baseline arm and a correctness oracle). ``meta_mode="outer_only"`` runs a single
full-routing pass (the matching reference for ``inner_lr=0``).

The pseudo-step is isolated in ``_apply_pseudo_step`` so a future second-order implementation can
replace it with a differentiable update without touching the phase structure.

Set ``EMO_META_CHECK_RESTORE=1`` to assert (bitwise) after every step that the expert weights were
restored exactly to their pre-step values.
"""

import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn

from olmo_core.config import DType
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.utils import get_local_tensor, is_distributed
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexp_randpool_router import (
    MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter,
)
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import SkipStepOptimizer
from olmo_core.utils import move_to_device

from ...common import ReduceType
from .config import TransformerTrainModuleConfig
from .train_module import TransformerTrainModule

log = logging.getLogger(__name__)

META_MODES = ("vanilla", "same_tokens", "heldout", "outer_only", "sequential")

_EXPERT_PARAM_SUFFIXES = (
    "feed_forward_moe.experts.mlp.w1",
    "feed_forward_moe.experts.mlp.w2",
    "feed_forward_moe.experts.mlp.w3",
)


class MetaLearningTransformerTrainModule(TransformerTrainModule):
    """
    :class:`TransformerTrainModule` with a two-phase FOMAML train step. See the module docstring.

    :param meta_mode: One of ``vanilla | same_tokens | heldout | outer_only``.
    :param inner_lr: SGD step size of the pseudo-step (raw SGD on expert weights; NOT on the
        AdamW-normalized scale of the outer lr).
    :param inner_pool_size: Per-document expert pool pinned during the inner pass. ``None`` keeps
        the router's vanilla random pool sampling on the inner pass.
    :param lambda_inner: Weight of the direct inner-gradient contribution to the real update
        (``0`` = pure FOMAML: the inner pass influences the update only through where the outer
        gradient is evaluated).
    :param lb_on_inner: Whether the inner pass computes/attaches the router aux losses (LB +
        router z-loss) and accumulates router metrics. Default off: the outer pass alone carries
        them, so per-step metrics aren't double-counted.
    :param inner_grad_clip: Global-norm clip applied to the inner expert gradients before the
        pseudo-step (and before the ``lambda_inner`` scaling of the expert grads). ``None``
        disables.
    :param log_grad_cosine: Record the inner/outer expert-gradient dot product and cosine each
        step (one extra all-reduce of 3 scalars).
    :param outer_expert_update: ``"working_set"`` (default): in the outer backward, each expert's
        WEIGHTS receive gradient only from slots whose expert is in the document's
        top-``inner_pool_size`` working set (weight-only detach: forward values and all
        activation-gradient paths — attention, earlier layers, router — are untouched).
        ``"all"``: every expert receives the full outer gradient (the original, degenerate
        behavior; kept for ablation).
    :param outer_pool: ``"full"`` (default): the outer pass routes over all experts (keep-all).
        ``"random"``: the outer pass samples per-document pool sizes in [min, max] like vanilla
        EMO, so the outer objective also trains restricted-forward (selective-inference)
        operation. Evals are unaffected either way.
    :param inner_optim: How the pseudo-step is computed from the inner gradient. ``"sgd"``
        (default): the raw SGD probe ``theta' = theta - alpha*g_inner``. ``"adam"``: an
        AdamW-preconditioned probe simulated read-only from the LIVE optimizer moments
        (``exp_avg`` / ``exp_avg_sq`` / ``step`` of the expert params) — i.e. the pseudo-step is
        exactly the AdamW step the real optimizer WOULD take on ``g_inner`` at the current
        moments, WITHOUT mutating them (they are consumed unchanged by the real outer step). This
        makes theta' the true "next selective AdamW step" and keeps the perturbation magnitude at
        Adam scale (~alpha per coordinate), immune to the bf16 self-extinction that killed the SGD
        probe.
    :param inner_lr_mode: ``"fixed"`` (default): the pseudo-step base lr is ``inner_lr``.
        ``"match_lr"``: the base lr is the LIVE scheduler lr of the expert param group (so theta'
        tracks the real training step size as the schedule evolves). Requires a scheduler.
    :param inner_lr_scale_min / inner_lr_scale_max: per-step displacement-magnitude range. Each
        train step samples ``s`` log-uniformly in ``[min, max]`` (identical across ranks, seeded by
        the global step) and uses effective lr ``s * base_lr`` for the pseudo-step. ``[1, 1]``
        (default) = a single magnitude; a range (e.g. ``[1, 32]``) trains the transfer property
        across a band of committed-step displacements — the update-magnitude analog of vanilla's
        random pools.
    :param inner_lr_scale_seed: base seed for the per-step scale RNG (only used when min != max).
    """

    def __init__(
        self,
        *,
        meta_mode: str = "same_tokens",
        inner_lr: float = 0.0,
        inner_pool_size: Optional[int] = 32,
        lambda_inner: float = 0.0,
        lb_on_inner: bool = False,
        inner_grad_clip: Optional[float] = 1.0,
        log_grad_cosine: bool = True,
        outer_expert_update: str = "working_set",
        outer_pool: str = "full",
        inner_optim: str = "sgd",
        inner_lr_mode: str = "fixed",
        inner_lr_scale_min: float = 1.0,
        inner_lr_scale_max: float = 1.0,
        inner_lr_scale_seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if meta_mode not in META_MODES:
            raise OLMoConfigurationError(
                f"invalid meta_mode '{meta_mode}', expected one of {META_MODES}"
            )
        if meta_mode in ("same_tokens", "heldout") and inner_lr < 0:
            raise OLMoConfigurationError("'inner_lr' must be >= 0")
        if lambda_inner < 0:
            raise OLMoConfigurationError("'lambda_inner' must be >= 0")
        if inner_optim not in ("sgd", "adam"):
            raise OLMoConfigurationError(
                f"invalid inner_optim '{inner_optim}', expected 'sgd' or 'adam'"
            )
        if inner_lr_mode not in ("fixed", "match_lr"):
            raise OLMoConfigurationError(
                f"invalid inner_lr_mode '{inner_lr_mode}', expected 'fixed' or 'match_lr'"
            )
        if not (0 < inner_lr_scale_min <= inner_lr_scale_max):
            raise OLMoConfigurationError(
                f"require 0 < inner_lr_scale_min ({inner_lr_scale_min}) <= inner_lr_scale_max "
                f"({inner_lr_scale_max})"
            )
        if inner_lr_mode == "match_lr" and self.scheduler is None:
            raise OLMoConfigurationError("inner_lr_mode='match_lr' requires a scheduler")
        if outer_expert_update not in ("working_set", "all"):
            raise OLMoConfigurationError(
                f"invalid outer_expert_update '{outer_expert_update}', expected 'working_set' or 'all'"
            )
        if outer_expert_update == "working_set" and inner_pool_size is None:
            raise OLMoConfigurationError(
                "outer_expert_update='working_set' requires a fixed 'inner_pool_size' "
                "(the working-set size)"
            )
        if meta_mode == "sequential" and inner_pool_size is None:
            raise OLMoConfigurationError("meta_mode='sequential' requires 'inner_pool_size'")
        if outer_pool not in ("full", "random"):
            raise OLMoConfigurationError(
                f"invalid outer_pool '{outer_pool}', expected 'full' or 'random'"
            )

        self.meta_mode = meta_mode
        self.inner_lr = inner_lr
        self.inner_pool_size = inner_pool_size
        self.lambda_inner = lambda_inner
        self.lb_on_inner = lb_on_inner
        self.inner_grad_clip = inner_grad_clip
        self.log_grad_cosine = log_grad_cosine
        self.outer_expert_update = outer_expert_update
        self.outer_pool = outer_pool
        self.inner_optim = inner_optim
        self.inner_lr_mode = inner_lr_mode
        self.inner_lr_scale_min = inner_lr_scale_min
        self.inner_lr_scale_max = inner_lr_scale_max
        self.inner_lr_scale_seed = inner_lr_scale_seed
        self._meta_step_count = 0
        self._last_inner_scale = 1.0
        self._last_inner_eff_lr = 0.0

        # Cache the randpool routers and the fused expert-weight parameters. Both survive FSDP2
        # wrapping and per-block torch.compile (module structure and parameter names are kept).
        self._meta_routers: List[MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter] = [
            m
            for m in self.model.modules()
            if isinstance(m, MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter)
        ]
        self._expert_params: List[nn.Parameter] = [
            p
            for name, p in self.model.named_parameters()
            if name.endswith(_EXPERT_PARAM_SUFFIXES) and p.requires_grad
        ]

        if self.meta_mode != "vanilla":
            if not self._meta_routers:
                raise OLMoConfigurationError(
                    "MetaLearningTransformerTrainModule requires the randpool router "
                    "(MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter) on every MoE block; "
                    "none were found in the model."
                )
            if not self._expert_params:
                raise OLMoConfigurationError(
                    "no trainable expert parameters (*.feed_forward_moe.experts.mlp.w{1,2,3}) "
                    "found in the model"
                )
            # Map each expert param to its optimizer param group (for the Adam-preconditioned
            # pseudo-step and match_lr: lr/betas/eps/weight_decay come from the real group). The
            # parent __init__ has already built self.optim, and param_groups hold the same
            # Parameter identities as model.named_parameters().
            expert_ids = {id(p) for p in self._expert_params}
            self._expert_group_of: Dict[nn.Parameter, Dict[str, Any]] = {}
            expert_groups: List[Dict[str, Any]] = []
            for group in self.optim.param_groups:
                for p in group["params"]:
                    if id(p) in expert_ids:
                        self._expert_group_of[p] = group
                        if group not in expert_groups:
                            expert_groups.append(group)
            self._expert_param_group = expert_groups[0] if expert_groups else None
            if self.inner_optim == "adam" or self.inner_lr_mode == "match_lr":
                missing = [p for p in self._expert_params if p not in self._expert_group_of]
                if missing:
                    raise OLMoConfigurationError(
                        f"{len(missing)} expert param(s) not found in any optimizer param group; "
                        "inner_optim='adam'/inner_lr_mode='match_lr' need the group's "
                        "lr/betas/eps/weight_decay"
                    )
                if self.inner_optim == "adam" and "betas" not in self._expert_param_group:
                    raise OLMoConfigurationError(
                        "inner_optim='adam' requires an AdamW-style optimizer (param group with "
                        "'betas'/'eps'/'weight_decay')"
                    )

            router = self._meta_routers[0]
            self._num_nonshared_experts = router.num_experts - router.num_shared_experts
            if (
                self.inner_pool_size is not None
                and not 0 < self.inner_pool_size <= self._num_nonshared_experts
            ):
                raise OLMoConfigurationError(
                    f"'inner_pool_size' ({self.inner_pool_size}) must be in "
                    f"[1, {self._num_nonshared_experts}] (number of non-shared experts)"
                )
            log.info(
                f"Meta-learning train module: mode={self.meta_mode}, inner_optim={self.inner_optim}, "
                f"inner_lr_mode={self.inner_lr_mode}, inner_lr={self.inner_lr}, "
                f"inner_lr_scale=[{self.inner_lr_scale_min}, {self.inner_lr_scale_max}], "
                f"inner_pool_size={self.inner_pool_size}, lambda_inner={self.lambda_inner}, "
                f"lb_on_inner={self.lb_on_inner}, inner_grad_clip={self.inner_grad_clip}, "
                f"{len(self._meta_routers)} routers, {len(self._expert_params)} expert params, "
                f"{self._num_nonshared_experts} non-shared experts"
            )

    ##########
    # Helpers
    ##########

    def _set_router_meta_state(
        self,
        force_pool: Optional[int],
        skip_aux: bool,
        outer_detach_top_e: Optional[int] = None,
    ):
        for r in self._meta_routers:
            r.meta_force_pool = force_pool
            r.meta_skip_aux = skip_aux
            r.meta_outer_detach_top_e = outer_detach_top_e

    def _phase_loss_div(
        self, micro_batches: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Loss divisor for one phase's micro-batches, replicating the parent's semantics: number of
        non-ignored label tokens, plus the full token count of filtered-out instances added back
        (see the WARN comment in ``TransformerTrainModule.train_batch``).

        Returns ``(div_factor, raw_tokens_for_loss, total_tokens)``.
        """
        raw = move_to_device(torch.tensor(0), self.device)
        div = move_to_device(torch.tensor(0), self.device)
        total = 0
        for mb in micro_batches:
            labels = mb["labels"]
            total += labels.numel()
            n = move_to_device((labels != self.label_ignore_index).sum(), self.device)
            raw += n
            div += n
            if (instance_mask := mb.get("instance_mask")) is not None:
                div += (~instance_mask).sum() * labels.shape[1]
        return div, raw, total

    def _reduce_scalar(self, value: torch.Tensor) -> torch.Tensor:
        if is_distributed():
            dist.all_reduce(value, group=self.dp_process_group)
        return value

    def _expert_global_sq_norm(self, locals_: List[torch.Tensor]) -> torch.Tensor:
        """Global squared L2 norm over dim-0-sharded (disjoint-row) local shards."""
        sq = torch.zeros((), device=self.device, dtype=torch.float32)
        for t in locals_:
            sq += t.float().pow(2).sum()
        return self._reduce_scalar(sq)

    def _sample_inner_scale(self) -> float:
        """Per-step displacement scale, log-uniform in [scale_min, scale_max]. Seeded by the
        global step so every rank draws the SAME scale without communication (the expert weights
        are row-sharded across DP; an inconsistent scale would corrupt the global pseudo-step)."""
        if self.inner_lr_scale_min == self.inner_lr_scale_max:
            return self.inner_lr_scale_min
        # `self._trainer` (not the `self.trainer` property, which raises when unattached) so the
        # sampler is usable in unit tests / before the trainer is attached.
        trainer = getattr(self, "_trainer", None)
        step = int(getattr(trainer, "global_step", 0)) if trainer is not None else 0
        rng = random.Random(self.inner_lr_scale_seed * 1_000_003 + step)
        return math.exp(
            rng.uniform(math.log(self.inner_lr_scale_min), math.log(self.inner_lr_scale_max))
        )

    def _expert_base_lr(self) -> float:
        """Base lr for the pseudo-step. ``match_lr``: the live scheduler lr of the expert group —
        set_lr is idempotent within a global step (the trainer re-sets the identical value in
        optim_step), so reading it here does not perturb the real update."""
        if self.inner_lr_mode == "match_lr":
            assert self.scheduler is not None and self._expert_param_group is not None
            return float(self.scheduler.set_lr(self._expert_param_group, self.trainer))
        return self.inner_lr

    def _apply_pseudo_step(
        self, grad_stash: Dict[nn.Parameter, torch.Tensor]
    ) -> Dict[nn.Parameter, torch.Tensor]:
        """
        Apply the (first-order) pseudo-step in place on the expert weights' local shards and return
        the per-parameter applied delta (fp32 local shards) for the metrics. A future second-order
        variant replaces this (and the surrounding stash/restore) with a differentiable update.

        ``inner_optim="sgd"``: ``delta = -eff_lr * g_inner``.
        ``inner_optim="adam"``: ``delta`` is the AdamW step the real optimizer would take on
        ``g_inner`` at the CURRENT expert moments — simulated read-only (``exp_avg`` /
        ``exp_avg_sq`` / ``step`` are never mutated), so the real outer step consumes them
        unchanged. Matches ``olmo_core.optim.adamw.adamw_step`` with ``step_factor=1``.
        """
        scale = self._sample_inner_scale()
        base_lr = self._expert_base_lr()
        eff_lr = scale * base_lr
        self._last_inner_scale = scale
        self._last_inner_eff_lr = eff_lr

        delta_stash: Dict[nn.Parameter, torch.Tensor] = {}
        if self.inner_optim == "sgd":
            for p, g in grad_stash.items():
                delta = g.mul(-eff_lr)
                get_local_tensor(p.data).add_(delta)
                delta_stash[p] = delta
            return delta_stash

        # inner_optim == "adam": read-only AdamW simulation from the live moments.
        for p, g in grad_stash.items():
            group = self._expert_group_of[p]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            state = self.optim.state.get(p, {})
            w_local = get_local_tensor(p.data)
            g32 = g.float()
            if "exp_avg" in state:
                m = get_local_tensor(state["exp_avg"]).float()
                v = get_local_tensor(state["exp_avg_sq"]).float()
                t = (
                    float(get_local_tensor(state["step"]))
                    if torch.is_tensor(state["step"])
                    else float(state["step"])
                )
            else:  # optimizer hasn't stepped this param yet (very first step)
                m = torch.zeros_like(g32)
                v = torch.zeros_like(g32)
                t = 0.0
            m2 = beta1 * m + (1.0 - beta1) * g32
            v2 = beta2 * v + (1.0 - beta2) * g32 * g32
            bias_correction1 = 1.0 - beta1 ** (t + 1.0)
            bias_correction2 = 1.0 - beta2 ** (t + 1.0)
            denom = (v2.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            update = (m2 / denom).mul_(-(eff_lr / bias_correction1))
            if wd != 0.0:
                # decoupled weight decay: p *= (1 - eff_lr*wd)  ==>  delta_wd = -eff_lr*wd*p
                update.add_(w_local.float(), alpha=-eff_lr * wd)
            delta = update.to(w_local.dtype)
            w_local.add_(delta)
            delta_stash[p] = delta
        return delta_stash

    #############
    # Train step
    #############

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        if self.meta_mode == "vanilla":
            return super().train_batch(batch, dry_run=dry_run)
        if self.meta_mode == "sequential":
            return self._sequential_train_batch(batch, dry_run=dry_run)

        check_restore = os.environ.get("EMO_META_CHECK_RESTORE") == "1"

        self._set_model_mode("train")
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        seq_len = batch["input_ids"].shape[1]
        if self.rank_microbatch_size < seq_len:
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)

        if self.meta_mode == "heldout":
            if len(micro_batches) % 2 != 0:
                raise RuntimeError(
                    f"meta_mode='heldout' needs an even number of micro-batches per rank to split "
                    f"into an inner and an outer half, got {len(micro_batches)}. Adjust "
                    f"global_batch_size / rank_microbatch_size."
                )
            half = len(micro_batches) // 2
            inner_mbs, outer_mbs = micro_batches[:half], micro_batches[half:]
        else:
            inner_mbs = outer_mbs = micro_batches

        run_inner = self.meta_mode != "outer_only"

        # Outer-phase (canonical) loss bookkeeping, mirroring the parent.
        outer_div, outer_raw, outer_total = self._phase_loss_div(outer_mbs)
        self.record_metric(
            "train/masked labels (%)", (outer_total - outer_raw) / outer_total, ReduceType.mean
        )
        instance_masks = [
            mb["instance_mask"] for mb in outer_mbs if mb.get("instance_mask") is not None
        ]
        if instance_masks:
            masked_frac = torch.cat([(~m).float().flatten() for m in instance_masks]).mean()
            self.record_metric("train/masked instances (%)", masked_frac, ReduceType.mean)

        # _prepare_batch pops input_ids/labels out of each micro-batch dict, so prepare each
        # micro-batch exactly once and reuse the prepared tuples across both phases.
        prepared_inner: List[Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]] = []
        inner_div: Optional[torch.Tensor] = None
        if run_inner:
            inner_div, _, _ = self._phase_loss_div(inner_mbs)
            prepared_inner = [self._prepare_batch(mb) for mb in inner_mbs]
        prepared_outer = (
            prepared_inner
            if (run_inner and outer_mbs is inner_mbs)
            else [self._prepare_batch(mb) for mb in outer_mbs]
        )

        check_stash: Dict[nn.Parameter, torch.Tensor] = {}
        if check_restore and run_inner:
            check_stash = {p: get_local_tensor(p.data).clone() for p in self._expert_params}

        inner_ce_loss = move_to_device(torch.tensor(0.0), self.device)
        grad_stash: Dict[nn.Parameter, torch.Tensor] = {}
        weight_stash: Dict[nn.Parameter, torch.Tensor] = {}
        delta_stash: Dict[nn.Parameter, torch.Tensor] = {}
        pre_clip_grad_norm: Optional[torch.Tensor] = None

        if run_inner:
            # ---- Phase 1: inner forward/backward (selective routing) ----
            self._set_router_meta_state(
                force_pool=self.inner_pool_size, skip_aux=not self.lb_on_inner
            )
            num_inner = len(prepared_inner)
            for i, (input_ids, labels, model_kwargs) in enumerate(prepared_inner):
                with self._train_microbatch_context(i, num_inner):
                    _, loss, ce_loss, z_loss = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        # The inner objective is pure CE unless L_inner contributes directly to
                        # the real update; the z-loss would otherwise leak into the pseudo-step
                        # gradient.
                        z_loss_multiplier=self.z_loss_multiplier if self.lambda_inner > 0 else None,
                        loss_div_factor=inner_div,
                        return_logits=False,
                        **model_kwargs,
                    )
                    inner_ce_loss += get_local_tensor(ce_loss.detach())
                    del ce_loss, z_loss
                    loss.backward()
            # `.grad` now holds the DP-reduced sharded inner gradients for all params.

            # ---- Pseudo-step (expert params only) ----
            expert_grad_locals = []
            for p in self._expert_params:
                if p.grad is None:
                    raise RuntimeError(
                        "an expert parameter received no gradient in the meta inner pass"
                    )
                expert_grad_locals.append(get_local_tensor(p.grad))
            pre_clip_grad_norm = self._expert_global_sq_norm(expert_grad_locals).sqrt()
            if self.inner_grad_clip is not None:
                # Sync-free clip: scale by min(1, clip / norm) in place on `.grad` so the
                # pseudo-step and the lambda_inner path share the same clipped expert grads.
                clip_coef = (self.inner_grad_clip / (pre_clip_grad_norm + 1e-6)).clamp(max=1.0)
                for g in expert_grad_locals:
                    g.mul_(clip_coef)
            for p in self._expert_params:
                assert p.grad is not None
                grad_stash[p] = get_local_tensor(p.grad).clone()
                weight_stash[p] = get_local_tensor(p.data).clone()

            if self.lambda_inner == 0.0:
                # Pure FOMAML: the inner grads must not leak into the real update.
                self.optim.zero_grad(set_to_none=True)
            else:
                for q in self.model.parameters():
                    if q.grad is not None:
                        get_local_tensor(q.grad).mul_(self.lambda_inner)

            delta_stash = self._apply_pseudo_step(grad_stash)

        # ---- Phase 2: outer forward/backward at theta' ----
        # outer_pool="full": routing over all experts (keep-all). outer_pool="random": the router
        # samples per-document pool sizes in [min, max] exactly like vanilla EMO, so the outer
        # objective also trains restricted-forward operation (selective-inference robustness).
        # With outer_expert_update="working_set", each expert's weights receive gradient only
        # from slots inside the document's top-`inner_pool_size` working set (weight-only detach;
        # activation gradients and the router are untouched).
        self._set_router_meta_state(
            force_pool=self._num_nonshared_experts if self.outer_pool == "full" else None,
            skip_aux=False,
            outer_detach_top_e=(
                self.inner_pool_size if self.outer_expert_update == "working_set" else None
            ),
        )
        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        num_outer = len(prepared_outer)
        for i, (input_ids, labels, model_kwargs) in enumerate(prepared_outer):
            with self._train_microbatch_context(i, num_outer):
                _, loss, ce_loss, z_loss = self.model_forward(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=outer_div,
                    return_logits=False,
                    **model_kwargs,
                )
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss
                # FSDP2's reduce-scatter ACCUMULATES into `.grad`:
                # `.grad = lambda_inner * g_inner + g_outer(theta')`.
                loss.backward()

        # ---- Meta metrics (before the restore frees the stashes) + restore ----
        if run_inner:
            self._meta_step_count += 1
            self._record_meta_metrics(
                inner_ce_loss, pre_clip_grad_norm, grad_stash, weight_stash, delta_stash, dry_run
            )
            for p in self._expert_params:
                get_local_tensor(p.data).copy_(weight_stash[p])
            if check_restore:
                for p in self._expert_params:
                    assert torch.equal(
                        get_local_tensor(p.data), check_stash[p]
                    ), "expert weights were NOT restored bitwise after the pseudo-step"
                log.info("EMO_META_CHECK_RESTORE: expert weights restored bitwise")
            grad_stash.clear()
            weight_stash.clear()
            delta_stash.clear()
            check_stash.clear()

        # Leave the routers clean for eval / the next step.
        self._set_router_meta_state(force_pool=None, skip_aux=False)

        del batch, prepared_inner, prepared_outer

        self.model.post_batch(dry_run=dry_run)

        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        # Record the canonical (outer) loss metrics, mirroring the parent.
        if isinstance(self.optim, SkipStepOptimizer):
            if is_distributed():
                ce_batch_loss.div_(self._reduce_divide_factor)
                dist.all_reduce(ce_batch_loss)
                ce_batch_loss.div_(self.world_size)
                ce_batch_loss.mul_(self._reduce_divide_factor)
            self.record_ce_loss(ce_batch_loss)
            self.optim.latest_loss = ce_batch_loss
        else:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            assert self.z_loss_multiplier is not None
            self.record_metric("Z loss", z_batch_loss, ReduceType.mean, namespace="train")
            self.record_metric(
                "Z loss unscaled",
                z_batch_loss / self.z_loss_multiplier,
                ReduceType.mean,
                namespace="train",
            )

        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            reset=True
        ).items():
            self.record_metric(metric_name, metric_val, reduction, namespace="train")

    def _sequential_train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        """
        The SEQUENTIAL ablation of the same-tokens FOMAML arm: two REAL optimizer steps per
        token batch instead of a temporary probe.

        1. Selective pass: forward+backward all micro-batches under per-doc top-``inner_pool_size``
           routing (an ordinary selective training step — router/attention/experts all update),
           then an internal clip + LR-set + AdamW step.
        2. Full pass on the SAME tokens at the post-step weights: full routing, expert-weight
           gradients masked to each document's working set (same machinery as the ws arms);
           the trainer's normal ``optim_step()`` commits this second update.

        Differences vs ``same_tokens`` FOMAML: the selective step is COMMITTED (not undone), it is
        AdamW-sized/-shaped (not a large raw-SGD probe), and there is no evaluation-at-theta'
        meta term. The trainer still counts ONE step per token batch, so token axes stay
        comparable across arms. During dry runs the internal optimizer step is skipped.
        """
        self._set_model_mode("train")
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        seq_len = batch["input_ids"].shape[1]
        if self.rank_microbatch_size < seq_len:
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)

        div, raw, total = self._phase_loss_div(micro_batches)
        self.record_metric("train/masked labels (%)", (total - raw) / total, ReduceType.mean)
        prepared = [self._prepare_batch(mb) for mb in micro_batches]
        num_mbs = len(prepared)

        # ---- Step 1: selective pass (real update; aux losses on) ----
        self._set_router_meta_state(force_pool=self.inner_pool_size, skip_aux=False)
        sel_ce_loss = move_to_device(torch.tensor(0.0), self.device)
        for i, (input_ids, labels, model_kwargs) in enumerate(prepared):
            with self._train_microbatch_context(i, num_mbs):
                _, loss, ce_loss, z_loss = self.model_forward(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=div,
                    return_logits=False,
                    **model_kwargs,
                )
                sel_ce_loss += get_local_tensor(ce_loss.detach())
                del ce_loss, z_loss
                loss.backward()

        if not dry_run:
            if self.max_grad_norm is not None:
                self._clip_grad_norm(self.max_grad_norm)
            if self.scheduler is not None:
                for group in self.optim.param_groups:
                    self.scheduler.set_lr(group, self.trainer)
            self.optim.step()
            self.model.post_optim_step()
        self.optim.zero_grad(set_to_none=True)

        # ---- Step 2: full pass on the same tokens at the post-step weights (ws-masked) ----
        self._set_router_meta_state(
            force_pool=self._num_nonshared_experts,
            skip_aux=False,
            outer_detach_top_e=(
                self.inner_pool_size if self.outer_expert_update == "working_set" else None
            ),
        )
        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        for i, (input_ids, labels, model_kwargs) in enumerate(prepared):
            with self._train_microbatch_context(i, num_mbs):
                _, loss, ce_loss, z_loss = self.model_forward(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=div,
                    return_logits=False,
                    **model_kwargs,
                )
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss
                loss.backward()

        self._set_router_meta_state(force_pool=None, skip_aux=False)
        del batch, prepared
        self.model.post_batch(dry_run=dry_run)
        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        if isinstance(self.optim, SkipStepOptimizer):
            if is_distributed():
                ce_batch_loss.div_(self._reduce_divide_factor)
                dist.all_reduce(ce_batch_loss)
                ce_batch_loss.div_(self.world_size)
                ce_batch_loss.mul_(self._reduce_divide_factor)
            self.record_ce_loss(ce_batch_loss)
            self.optim.latest_loss = ce_batch_loss
        else:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        # Reuse the meta metric name so cross-arm plots align: the selective step's CE.
        self.record_metric("meta inner CE loss", sel_ce_loss, ReduceType.mean, namespace="train")
        if z_batch_loss is not None:
            assert self.z_loss_multiplier is not None
            self.record_metric("Z loss", z_batch_loss, ReduceType.mean, namespace="train")
        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            reset=True
        ).items():
            self.record_metric(metric_name, metric_val, reduction, namespace="train")

    def _record_meta_metrics(
        self,
        inner_ce_loss: torch.Tensor,
        pre_clip_grad_norm: Optional[torch.Tensor],
        grad_stash: Dict[nn.Parameter, torch.Tensor],
        weight_stash: Dict[nn.Parameter, torch.Tensor],
        delta_stash: Dict[nn.Parameter, torch.Tensor],
        dry_run: bool,
    ):
        if dry_run:
            return
        self.record_metric("meta inner CE loss", inner_ce_loss, ReduceType.mean, namespace="train")
        assert pre_clip_grad_norm is not None
        # These norms/dots are already globally reduced, hence reduce_type=None.
        self.record_metric(
            "meta inner grad norm (experts)",
            pre_clip_grad_norm,
            reduce_type=None,
            namespace="train",
        )
        self.record_metric(
            "meta inner scale", self._last_inner_scale, reduce_type=None, namespace="train"
        )
        self.record_metric(
            "meta inner eff lr", self._last_inner_eff_lr, reduce_type=None, namespace="train"
        )
        stash_sq = self._expert_global_sq_norm(list(grad_stash.values()))
        # Actual applied pseudo-step magnitude (optimizer-agnostic: for adam it is NOT
        # inner_lr*|g|, so read it from the delta the pseudo-step actually applied).
        delta_norm = self._expert_global_sq_norm(list(delta_stash.values())).sqrt()
        self.record_metric(
            "meta pseudo step delta norm", delta_norm, reduce_type=None, namespace="train"
        )
        weight_sq = self._expert_global_sq_norm(
            [get_local_tensor(p.data) for p in self._expert_params]
        )
        self.record_metric(
            "meta delta/weight norm",
            delta_norm / (weight_sq.sqrt() + 1e-12),
            reduce_type=None,
            namespace="train",
        )
        if self._meta_step_count % 10 == 1:
            # Measured bf16 survival of the pseudo-step: what the outer pass's bf16 all-gather
            # actually sees of the perturbation, computed exactly on the fp32 local shards
            # (dim-0 sharding = whole rows, so the local simulation matches the real cast).
            surv_dot = torch.zeros((), device=self.device, dtype=torch.float32)
            surv_sq = torch.zeros((), device=self.device, dtype=torch.float32)
            delta_sq = torch.zeros((), device=self.device, dtype=torch.float32)
            for p in self._expert_params:
                w = weight_stash[p]
                delta = delta_stash[p]
                survived = (w + delta).bfloat16().float() - w.bfloat16().float()
                d32 = delta.float()
                s32 = survived.float()
                surv_dot += (s32 * d32).sum()
                surv_sq += s32.pow(2).sum()
                delta_sq += d32.pow(2).sum()
            self._reduce_scalar(surv_dot)
            self._reduce_scalar(surv_sq)
            self._reduce_scalar(delta_sq)
            self.record_metric(
                "meta bf16 survival cosine",
                surv_dot / (surv_sq.sqrt() * delta_sq.sqrt() + 1e-20),
                reduce_type=None,
                namespace="train",
            )
            self.record_metric(
                "meta bf16 survival norm ratio",
                surv_sq.sqrt() / (delta_sq.sqrt() + 1e-20),
                reduce_type=None,
                namespace="train",
            )

        if self.log_grad_cosine:
            # g_outer over expert shards = `.grad` minus the lambda_inner * g_inner contribution.
            dot = torch.zeros((), device=self.device, dtype=torch.float32)
            outer_sq = torch.zeros((), device=self.device, dtype=torch.float32)
            for p in self._expert_params:
                assert p.grad is not None
                g_inner = grad_stash[p].float()
                g_outer = get_local_tensor(p.grad).float()
                if self.lambda_inner != 0.0:
                    g_outer = g_outer - self.lambda_inner * g_inner
                dot += (g_inner * g_outer).sum()
                outer_sq += g_outer.pow(2).sum()
            self._reduce_scalar(dot)
            self._reduce_scalar(outer_sq)
            self.record_metric(
                "meta inner-outer grad dot (experts)", dot, reduce_type=None, namespace="train"
            )
            cosine = dot / (stash_sq.sqrt() * outer_sq.sqrt() + 1e-12)
            self.record_metric(
                "meta inner-outer grad cosine (experts)",
                cosine,
                reduce_type=None,
                namespace="train",
            )


_META_CONFIG_FIELDS = (
    "meta_mode",
    "inner_lr",
    "inner_pool_size",
    "lambda_inner",
    "lb_on_inner",
    "inner_grad_clip",
    "log_grad_cosine",
    "outer_expert_update",
    "outer_pool",
    "inner_optim",
    "inner_lr_mode",
    "inner_lr_scale_min",
    "inner_lr_scale_max",
    "inner_lr_scale_seed",
)


@dataclass
class MetaLearningTransformerTrainModuleConfig(TransformerTrainModuleConfig):
    """
    Configuration for :class:`MetaLearningTransformerTrainModule`. All meta knobs are
    CLI-overridable via ``--train_module.<knob>=...``.
    """

    meta_mode: str = "same_tokens"
    inner_lr: float = 0.0
    inner_pool_size: Optional[int] = 32
    lambda_inner: float = 0.0
    lb_on_inner: bool = False
    inner_grad_clip: Optional[float] = 1.0
    log_grad_cosine: bool = True
    outer_expert_update: str = "working_set"
    outer_pool: str = "full"
    inner_optim: str = "sgd"
    inner_lr_mode: str = "fixed"
    inner_lr_scale_min: float = 1.0
    inner_lr_scale_max: float = 1.0
    inner_lr_scale_seed: int = 0

    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
    ) -> MetaLearningTransformerTrainModule:
        if self.pp_config is not None:
            raise OLMoConfigurationError(
                "pipeline parallelism is not supported by the meta-learning train module"
            )

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # `exclude_none=True` would silently drop meaningful None values (e.g.
        # `inner_pool_size=null` meaning "keep random pool sampling on the inner pass"), so pop
        # every meta field and pass them explicitly from self.
        for f in _META_CONFIG_FIELDS:
            kwargs.pop(f, None)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)

        return MetaLearningTransformerTrainModule(
            model=model,
            device=device,
            meta_mode=self.meta_mode,
            inner_lr=self.inner_lr,
            inner_pool_size=self.inner_pool_size,
            lambda_inner=self.lambda_inner,
            lb_on_inner=self.lb_on_inner,
            inner_grad_clip=self.inner_grad_clip,
            log_grad_cosine=self.log_grad_cosine,
            outer_expert_update=self.outer_expert_update,
            outer_pool=self.outer_pool,
            inner_optim=self.inner_optim,
            inner_lr_mode=self.inner_lr_mode,
            inner_lr_scale_min=self.inner_lr_scale_min,
            inner_lr_scale_max=self.inner_lr_scale_max,
            inner_lr_scale_seed=self.inner_lr_scale_seed,
            **kwargs,
        )
