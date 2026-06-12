from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

import olmo_core.ops.moe as ops
from olmo_core.distributed.utils import is_distributed
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.router import MoERouterGatingFunction
from olmo_core.nn.moe.twolevel_router import MoETwoLevelRouter, MoETwoLevelRouterConfig

from .loss import MoELoadBalancingLossGranularity, load_balancing_loss, router_z_loss


class MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter(MoETwoLevelRouter):
    """
    Same as MoETwoLevelBatchLBReduceDPSharedExpRouter but with a random document_expert_pool
    sampled uniformly from [min_document_expert_pool, max_document_expert_pool] per document
    during training. At eval time, uses a fixed eval_document_expert_pool.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        min_document_expert_pool: int,
        max_document_expert_pool: int,
        eval_document_expert_pool: Optional[int] = None,
        eos_token_id: int,
        num_shared_experts: int,
        num_forced_experts: int = 0,
        extension_finetune_mode: bool = False,
        extension_finetune_top_e: int = 0,
        extension_finetune_detach_router: bool = False,
        ghost_extend_mode: bool = False,
        ghost_extend_num: int = 1,
        ghost_extend_coeff_mode: str = "usage",
        ghost_extend_random_k: int = 8,
        ghost_extend_route: str = "always",
        ghost_extend_detach_coeff: bool = False,
        **kwargs,
    ):
        # Pass max_document_expert_pool as document_expert_pool to satisfy parent constructor
        # Pop document_expert_pool from kwargs if present (from config's as_dict) to avoid duplicate
        kwargs.pop("document_expert_pool", None)
        super().__init__(
            dtype=dtype,
            init_device=init_device,
            document_expert_pool=max_document_expert_pool,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.min_document_expert_pool = min_document_expert_pool
        self.max_document_expert_pool = max_document_expert_pool
        self.eval_document_expert_pool = (
            eval_document_expert_pool
            if eval_document_expert_pool is not None
            else (min_document_expert_pool + max_document_expert_pool) // 2
        )

        # the eos token id
        if eos_token_id is None:
            raise OLMoConfigurationError("eos_token_id must be provided for MoETwoLevelRouter")
        self.eos_token_id = eos_token_id

        # check that the number of shared experts is less than the top_k
        if num_shared_experts > self.top_k:
            raise OLMoConfigurationError(
                f"num_shared_experts ({num_shared_experts}) must be less than or equal to top_k ({self.top_k})"
            )

        self.num_shared_experts = num_shared_experts
        self.num_choose_experts = self.top_k - self.num_shared_experts

        # Number of experts (last N non-shared) that are always forced into the document pool.
        # Useful when extending the model with new experts that need guaranteed routing.
        self.num_forced_experts = num_forced_experts

        # Extension finetune mode: when True, per-slot expert MLP outputs are detached for slots
        # whose chosen expert is not in the doc's top-e set (and for the shared expert slots).
        # See plan in /root/.claude/plans/bright-weaving-prism.md.
        self.extension_finetune_mode = extension_finetune_mode
        self.extension_finetune_top_e = extension_finetune_top_e
        # Additional flag: when True, also detach router gradient paths for non-top-e experts
        # (CE-via-expert_weights, LB via scores cols, router z-loss via logits cols). The
        # softmax cross-coupling from kept (top-e) slots is left intact (acceptable leak).
        # When False (default), router still updates freely from all paths even with MLP detach on.
        # Read by downstream callers (MoE.forward) via getattr — default False if absent.
        self.extension_finetune_detach_router = extension_finetune_detach_router
        if self.extension_finetune_detach_router and not self.extension_finetune_mode:
            raise OLMoConfigurationError(
                "extension_finetune_detach_router=True requires extension_finetune_mode=True; "
                "the router-detach paths are no-ops without the parent mode active."
            )
        # Per-forward stash; read+cleared by MoE.forward.
        self._detach_mask: Optional[torch.Tensor] = None

        # Ghost-expert training (models_fullextend). For each document a "ghost" expert is
        # simulated as a linear combination of the document pool's expert weights; the model
        # is trained with this perpetually-blended new expert present so that, at the end of
        # training, instantiating and adding a real new expert is well-conditioned.
        self.ghost_extend_mode = ghost_extend_mode
        self.ghost_extend_num = ghost_extend_num
        self.ghost_extend_coeff_mode = ghost_extend_coeff_mode
        self.ghost_extend_random_k = ghost_extend_random_k
        self.ghost_extend_route = ghost_extend_route
        self.ghost_extend_detach_coeff = ghost_extend_detach_coeff
        if self.ghost_extend_mode:
            if self.ghost_extend_route != "always":
                raise OLMoConfigurationError(
                    f"ghost_extend_route={self.ghost_extend_route!r} is not implemented yet; "
                    "only 'always' is currently supported."
                )
            if self.ghost_extend_coeff_mode not in ("usage", "uniform", "random"):
                raise OLMoConfigurationError(
                    f"ghost_extend_coeff_mode={self.ghost_extend_coeff_mode!r} must be one of "
                    "'usage', 'uniform', 'random'."
                )
        # Per-forward stash consumed by MoE.forward: (doc_sizes, coeff_list, gate_list) or None,
        # where coeff_list/gate_list hold one entry per ghost (ghost_extend_num).
        self._ghost_extend_stash: Optional[Tuple[torch.Tensor, list, list]] = None

    def get_top_k(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """We override the get_top_k to use self.num_choose_experts instead of self.top_k, since we will always activate self.num_shared_experts"""
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
            raise NotImplementedError(
                "Uniform expert assignment is not supported in MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter."
            )

        return expert_weights, expert_indices

    def _build_ghost_alpha(
        self,
        kept_mask: torch.Tensor,
        document_expert_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the blend coefficients alpha (over non-shared experts) for one document's ghost.

        :param kept_mask: Bool tensor ``(num_non_shared_experts,)``; ``True`` for experts in the
            document pool (the candidates the ghost is blended from).
        :param document_expert_probs: ``(num_non_shared_experts,)`` document-level summed softmax
            probabilities. Carries grad to the router; used by the "usage" coefficient mode.
        """
        E = document_expert_probs.shape[0]
        mode = self.ghost_extend_coeff_mode
        if mode == "usage":
            # Document-usage-weighted average over the pool. Stays in the graph (unless detached)
            # so the router rows are trained to be averageable.
            weights = document_expert_probs * kept_mask
            alpha = weights / (weights.sum() + 1e-9)
            if self.ghost_extend_detach_coeff:
                alpha = alpha.detach()
            return alpha
        elif mode == "uniform":
            kf = kept_mask.to(document_expert_probs.dtype)
            return kf / kf.sum().clamp_min(1.0)
        elif mode == "random":
            # Uniform average over a random sample of ghost_extend_random_k pool experts.
            kept_idx = kept_mask.nonzero(as_tuple=False).flatten()
            k = min(self.ghost_extend_random_k, int(kept_idx.numel()))
            alpha = torch.zeros(E, dtype=document_expert_probs.dtype, device=kept_mask.device)
            if k > 0:
                perm = torch.randperm(int(kept_idx.numel()), device=kept_idx.device)[:k]
                alpha[kept_idx[perm]] = 1.0 / k
            return alpha
        else:
            raise OLMoConfigurationError(
                f"unknown ghost_extend_coeff_mode {self.ghost_extend_coeff_mode!r}"
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
        Custom forward pass with modifications to implement two level routing
        with random document expert pool sizes.
        """
        # make sure tp and cp are not enabled
        if self.tp_mesh is not None:
            raise NotImplementedError("Tensor parallelism is not supported.")
        if self.cp_mesh is not None:
            raise NotImplementedError("Context parallelism is not supported.")

        # Clear any stale detach_mask from a previous forward (read+consumed by MoE.forward).
        self._detach_mask = None
        _ef_active = bool(self.extension_finetune_mode) and int(self.extension_finetune_top_e) > 0

        # Ghost-expert training (models_fullextend). Only active during training; eval measures the
        # base model (no ghost). Per-document blend specs are collected in the doc loop below and
        # assembled into self._ghost_extend_stash, which MoE.forward reads to run the ghost(s).
        self._ghost_extend_stash = None
        _ghost_active = bool(self.ghost_extend_mode) and self.training
        ghost_doc_specs: list = []  # (token_count, kept_mask, document_expert_probs) per document

        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x).float()

        # we remove the last self.num_shared_experts experts (remove from end in case indexing gets weird later?)
        logits = logits[
            :, :, : self.num_experts - self.num_shared_experts
        ]  # shape: (batch_size, seq_len, num_experts - num_shared_experts)
        logits_mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)

        assert document_boundaries is not None
        document_boundaries_cpu = []
        for b in document_boundaries:
            bc = b.detach().cpu().tolist()
            if not bc or bc[-1] != x.size(1):
                bc.append(int(x.size(1)))
            document_boundaries_cpu.append(bc)

        doc_entropy_sum = logits.new_zeros(())
        doc_entropy_count = 0

        num_non_shared_experts = self.num_experts - self.num_shared_experts

        # extension_finetune_mode: per-token boolean indicating whether each non-shared expert
        # is in the token's document's top-e set. Filled inside the per-doc loop below.
        if _ef_active:
            in_top_e = torch.zeros(
                x.size(0),
                x.size(1),
                num_non_shared_experts,
                dtype=torch.bool,
                device=x.device,
            )

        for seq_idx in range(x.size(0)):
            start = 0
            document_boundary = document_boundaries_cpu[seq_idx]
            for end in document_boundary:
                if end <= start:
                    start = end
                    continue
                sequence_logits = logits[
                    seq_idx, start:end, :
                ]  # shape: (doc_len, num_experts - num_shared_experts)
                # calculate the softmax over the experts
                expert_probs = F.softmax(
                    sequence_logits, dim=-1
                )  # shape: (doc_len, num_experts - num_shared_experts)

                # get the entropy over experts per token
                token_entropies = -torch.sum(
                    expert_probs * torch.log(expert_probs + 1e-10), dim=-1
                )  # shape: (doc_len,)
                # average entropy over the document
                doc_entropy_sum += token_entropies.mean()
                doc_entropy_count += 1

                # take the sum across the document
                document_expert_probs = expert_probs.sum(
                    dim=0
                )  # shape: (num_experts - num_shared_experts,)

                # Record per-doc top-e for extension_finetune_mode before any pool-sampling control flow
                # (some branches below early-`continue`, but top-e still applies to those docs).
                if _ef_active:
                    top_e = min(int(self.extension_finetune_top_e), num_non_shared_experts)
                    top_e_indices = torch.topk(document_expert_probs, top_e).indices
                    in_top_e[seq_idx, start:end, top_e_indices] = True

                # Sample random pool size per document
                if self.training:
                    document_expert_pool = torch.randint(
                        self.min_document_expert_pool,
                        self.max_document_expert_pool + 1,
                        (1,),
                    ).item()
                else:
                    document_expert_pool = self.eval_document_expert_pool

                # get the bottom document_expert_pool experts (including removing the shared experts since we already took that out of the logits)
                bot_document_expert_pool = num_non_shared_experts - document_expert_pool

                # Determine which experts to discard from the document pool (None => pool covers
                # all non-shared experts, nothing discarded).
                experts_to_discard: Optional[torch.Tensor] = None
                if bot_document_expert_pool > 0:
                    if self.num_forced_experts > 0:
                        # Forced experts (last num_forced_experts non-shared) are always in the pool.
                        # Only discard from the non-forced experts.
                        num_candidates = num_non_shared_experts - self.num_forced_experts
                        bot_to_discard = min(bot_document_expert_pool, num_candidates)
                        if bot_to_discard > 0:
                            candidate_probs = document_expert_probs[:num_candidates]
                            experts_to_discard = torch.topk(
                                -candidate_probs, bot_to_discard
                            ).indices  # shape: (bot_to_discard,)
                    else:
                        experts_to_discard = torch.topk(
                            -document_expert_probs, bot_document_expert_pool
                        ).indices  # shape: (bot_document_expert_pool,)

                # set the logits of these experts to a very large negative value
                if experts_to_discard is not None:
                    logits_mask[seq_idx, start:end, experts_to_discard] = True

                # Ghost-expert bookkeeping: record this document's pool (kept experts) and its
                # token span so MoE.forward can blend a ghost expert from the pool weights.
                if _ghost_active:
                    kept_mask = torch.ones(
                        num_non_shared_experts, dtype=torch.bool, device=logits.device
                    )
                    if experts_to_discard is not None:
                        kept_mask[experts_to_discard] = False
                    ghost_doc_specs.append((end - start, kept_mask, document_expert_probs))

                start = end

        logits.masked_fill_(logits_mask, float("-inf"))

        if self.training:
            avg_doc_entropy = (doc_entropy_sum / doc_entropy_count).detach()
            self._router_documentlevel_expert_entropy += avg_doc_entropy.item()

        # Ghost-expert routing (models_fullextend): each ghost is a new expert whose router row is an
        # alpha-blend of its document pool's router rows, so its logit is ghost_logit = sum_i alpha_i
        # * logit_i. The ghost logits join the routing-softmax denominator below, so the real pool
        # experts and the ghost(s) form a single distribution (the real experts shrink to make room).
        ghost_coeff_list: list = []
        ghost_logits_list: list = []  # each (B, S)
        if _ghost_active and len(ghost_doc_specs) > 0:
            counts = [c for (c, _, _) in ghost_doc_specs]
            # alpha is 0 off-pool, so zero out the -inf pool-mask entries to keep the dot finite.
            logits_clean_flat = torch.where(
                torch.isinf(logits), torch.zeros_like(logits), logits
            ).reshape(
                -1, num_non_shared_experts
            )  # (N, E')
            for _ in range(self.ghost_extend_num):
                # One blend per document; "random" re-samples per ghost, "usage"/"uniform" are
                # deterministic across ghosts.
                alpha_rows = [
                    self._build_ghost_alpha(kept_mask, dep)
                    for (_, kept_mask, dep) in ghost_doc_specs
                ]
                ghost_coeff_list.append(torch.stack(alpha_rows, dim=0))  # (G, E')
                # Expand each doc's alpha over its tokens (flattened doc order == token order).
                token_alpha = torch.cat(
                    [a.unsqueeze(0).expand(c, -1) for a, c in zip(alpha_rows, counts)], dim=0
                )  # (N, E')
                ghost_logits_list.append(
                    (token_alpha * logits_clean_flat).sum(dim=-1).reshape(x.size(0), x.size(1))
                )  # (B, S)

        # Routing scores. `scores_pool` is the softmax over the real (non-shared) pool experts only;
        # `scores` is renormalized so the pool experts and the ghost(s) form one distribution. They
        # are identical when no ghost is active. get_top_k / expert_weights use `scores`; the
        # auxiliary lb-loss and the entropy metric use `scores_pool` (the ghost is a transient blend,
        # not a real expert to load-balance).
        ghost_gates: list = []  # each (N,) per-token ghost routing weight
        if self.gating_function == MoERouterGatingFunction.softmax:
            lse_pool = torch.logsumexp(logits, dim=-1)  # (B, S)
            scores_pool = torch.exp(logits - lse_pool.unsqueeze(-1))
            if ghost_logits_list:
                # logsumexp over pool ∪ ghosts == logsumexp([lse_pool, ghost_logit_1, ...]).
                lse_aug = torch.logsumexp(
                    torch.stack([lse_pool, *ghost_logits_list], dim=-1), dim=-1
                )  # (B, S)
                scores = torch.exp(logits - lse_aug.unsqueeze(-1))
                ghost_gates = [torch.exp(gl - lse_aug).reshape(-1) for gl in ghost_logits_list]
            else:
                scores = scores_pool
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            if ghost_logits_list:
                raise NotImplementedError(
                    "ghost_extend_mode currently requires softmax gating; the ghost "
                    "renormalization is defined for the routing softmax."
                )
            scores = F.sigmoid(logits) + 1e-7
            scores_pool = scores
        else:
            raise NotImplementedError(self.gating_function)

        # shape: (batch_size, seq_len, self.num_choose_experts)
        expert_weights, expert_indices = self.get_top_k(scores)

        # Stash the ghost specs (blend coefficients + per-token gates) for MoE.forward.
        if _ghost_active and ghost_logits_list:
            self._ghost_extend_stash = (
                torch.tensor(counts, dtype=torch.long, device=x.device),
                ghost_coeff_list,
                ghost_gates,
            )

        # extension_finetune_mode: build per-slot detach mask based on top-e membership.
        # True = detach (backward-disabled) for this (token, k) slot.
        _ef_detach_mask_nonshared: Optional[torch.Tensor] = None
        if _ef_active:
            # expert_indices: (B, S, num_choose), values in [0, num_non_shared_experts)
            in_top_e_selected = torch.gather(in_top_e, dim=-1, index=expert_indices.long())
            _ef_detach_mask_nonshared = ~in_top_e_selected  # (B, S, num_choose)

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
            tot_batched_batch_size_per_expert = ops.batched_histc(
                expert_indices, self.num_experts - self.num_shared_experts
            )
            # shape: (batch_size, num_experts - num_shared_experts)
            tot_batched_batch_size_per_expert = tot_batched_batch_size_per_expert.sum(dim=1)
            # shape: (num_experts - num_shared_experts,)
            tot_batch_size_per_expert = tot_batched_batch_size_per_expert.sum(dim=0)

            if self.training:
                valid_expert_indices = expert_indices.view(-1)

                # Update unique experts metric.
                unique_experts = torch.unique(valid_expert_indices)
                num_unique_experts = (
                    unique_experts.numel() + self.num_shared_experts
                )  # we add the shared experts since they are always active

                self._unique_experts_sum += num_unique_experts
                self._num_batches_tracked += 1

                # Compute router distribution entropy metric (over the real pool experts).
                valid_scores = scores_pool.view(-1, self.num_experts - self.num_shared_experts)
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
                        scores_pool = scores_pool / scores_pool.sum(dim=-1, keepdim=True)

                    # we now do reduction on the tot_batch_size_per_expert to get a dp-global lb loss (still not full global sinze we don't do across gradient accumulation steps)
                    dp_global_tot_batch_size_per_expert = (
                        tot_batch_size_per_expert.clone()
                    )  # we clone to not interfere with logging or other routing stuff
                    assert isinstance(loss_div_factor, torch.Tensor)
                    dp_global_loss_div_factor = loss_div_factor.clone()

                    if is_distributed():
                        dist.all_reduce(dp_global_tot_batch_size_per_expert, op=dist.ReduceOp.SUM)
                        dist.all_reduce(dp_global_loss_div_factor, op=dist.ReduceOp.SUM)

                    # collect all non-zero entries of the dp_global_tot_batch_size_per_expert
                    self._reducedp_unique_experts_sum += (
                        dp_global_tot_batch_size_per_expert > 0
                    ).sum().item() + self.num_shared_experts  # we add the shared experts since they are always active

                    # extension_finetune_detach_router: cut LB-loss gradient path into router rows
                    # for non-top-e experts by detaching those columns of the scores tensor.
                    # Top-e columns still flow grad → router can still learn via LB for kept experts.
                    if _ef_active and self.extension_finetune_detach_router:
                        scores_for_lb = torch.where(in_top_e, scores_pool, scores_pool.detach())
                    else:
                        scores_for_lb = scores_pool

                    lb_loss = load_balancing_loss(
                        num_experts=self.num_experts
                        - self.num_shared_experts,  # we only calculate the load balancing loss over the non-shared experts
                        top_k=self.num_choose_experts,  # we only choose self.num_choose_experts
                        expert_scores=scores_for_lb,
                        batch_size_per_expert=dp_global_tot_batch_size_per_expert,
                        batched_batch_size_per_expert=tot_batched_batch_size_per_expert,  # this is not used, so we don't bother reducing it
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

                    # extension_finetune_detach_router: cut router-z-loss gradient path into
                    # router rows for non-top-e experts by detaching those columns of `logits`.
                    if _ef_active and self.extension_finetune_detach_router:
                        logits_for_zloss = torch.where(in_top_e, logits, logits.detach())
                    else:
                        logits_for_zloss = logits

                    z_loss = router_z_loss(
                        expert_logits=logits_for_zloss,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.z_loss += z_loss.detach()

                    scaled_z_loss = self.z_loss_weight * z_loss
                    aux_loss = scaled_z_loss if aux_loss is None else aux_loss + scaled_z_loss

            if self.batch_size_per_expert.shape[-1] != tot_batch_size_per_expert.shape[-1]:
                # make sure that the shared expert positions are zero, since it means the parameter was reset
                extra_counts = self.batch_size_per_expert[tot_batch_size_per_expert.shape[-1] :]
                assert torch.all(
                    extra_counts == 0
                ), f"Expected extra counts to be zero, but got {extra_counts}"
                self.batch_size_per_expert = self.batch_size_per_expert[
                    : tot_batch_size_per_expert.shape[-1]
                ]
            self.batch_size_per_expert += tot_batch_size_per_expert
            if self.bias_gamma is not None:
                assert self.score_bias_batch_size_per_expert is not None
                self.score_bias_batch_size_per_expert += tot_batch_size_per_expert

        # in the end, we add on the shared experts to both expert_weights and expert_indices
        if self.num_shared_experts > 0:
            expert_weights = F.pad(
                expert_weights, (0, self.num_shared_experts), value=1.0
            )  # we set the weights of the shared experts to 1 since they are always active
            # we set the indices of the shared experts to the last num_shared_experts indices since we removed those from the logits earlier
            shared_expert_indices = (
                torch.arange(
                    self.num_experts - self.num_shared_experts,
                    self.num_experts,
                    device=expert_indices.device,
                )
                .view(1, 1, self.num_shared_experts)
                .expand(expert_indices.size(0), expert_indices.size(1), self.num_shared_experts)
            )
            expert_indices = torch.cat([expert_indices, shared_expert_indices], dim=-1)
            # we also set tot_batch_size_per_expert for the shared experts to be the batch size since they are always active
            tot_batch_size_per_expert = F.pad(
                tot_batch_size_per_expert, (0, self.num_shared_experts), value=x.size(0) * x.size(1)
            )

        # extension_finetune_mode: pad the detach mask with True for the shared-expert slots
        # (shared expert is always detached in this mode; see plan "What exactly gets detached").
        if _ef_active and _ef_detach_mask_nonshared is not None:
            if self.num_shared_experts > 0:
                pad = torch.ones(
                    _ef_detach_mask_nonshared.size(0),
                    _ef_detach_mask_nonshared.size(1),
                    self.num_shared_experts,
                    dtype=torch.bool,
                    device=_ef_detach_mask_nonshared.device,
                )
                detach_mask = torch.cat([_ef_detach_mask_nonshared, pad], dim=-1)
            else:
                detach_mask = _ef_detach_mask_nonshared
            # shape: (B, S, top_k), True = detach this (token, k) slot in backward
            self._detach_mask = detach_mask

        return expert_weights, expert_indices, tot_batch_size_per_expert, aux_loss

    def extra_repr(self):
        """Add custom parameter to string representation."""
        base_repr = super().extra_repr()
        return f"{base_repr}, min_document_expert_pool={self.min_document_expert_pool}, max_document_expert_pool={self.max_document_expert_pool}, eval_document_expert_pool={self.eval_document_expert_pool}, eos_token_id={self.eos_token_id}, num_shared_experts={self.num_shared_experts}, num_forced_experts={self.num_forced_experts}, extension_finetune_mode={self.extension_finetune_mode}, extension_finetune_top_e={self.extension_finetune_top_e}, extension_finetune_detach_router={self.extension_finetune_detach_router}, ghost_extend_mode={self.ghost_extend_mode}, ghost_extend_num={self.ghost_extend_num}, ghost_extend_coeff_mode={self.ghost_extend_coeff_mode}, ghost_extend_random_k={self.ghost_extend_random_k}, ghost_extend_route={self.ghost_extend_route}, ghost_extend_detach_coeff={self.ghost_extend_detach_coeff}"


@dataclass
class MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouterConfig(MoETwoLevelRouterConfig):
    num_shared_experts: int = 1
    min_document_expert_pool: int = 8
    max_document_expert_pool: int = 128
    eval_document_expert_pool: Optional[int] = None  # defaults to midpoint of min/max
    num_forced_experts: int = 0  # last N non-shared experts always included in pool
    extension_finetune_mode: bool = False
    extension_finetune_top_e: int = 0
    extension_finetune_detach_router: bool = False
    # --- ghost-expert training (models_fullextend) ---
    ghost_extend_mode: bool = False
    ghost_extend_num: int = 1
    ghost_extend_coeff_mode: str = "usage"  # "usage" | "uniform" | "random"
    ghost_extend_random_k: int = 8
    ghost_extend_route: str = "always"  # "always" | "topk" (topk not yet implemented)
    ghost_extend_detach_coeff: bool = False

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
    ) -> MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter:
        """
        Build the router.
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

        return MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter(**kwargs)
