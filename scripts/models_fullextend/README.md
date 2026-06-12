# models_fullextend

Extendability experiment: can we pretrain EMO so that **adding a brand-new
expert after training is well-conditioned** — i.e. a freshly instantiated
expert slots into the model and is useful with little or no degradation, rather
than landing in a dead region of weight space?

The hypothesis is that if, *throughout pretraining*, the model is perpetually
exposed to a simulated "newly added expert" that is born as an average of the
experts the current document already uses, then the trained expert/router
weight space becomes smooth enough that a real averaged-initialized expert
later drops in cleanly. We call the simulated expert a **ghost expert**.

## The ghost-expert method

For every document we already have a document-level expert pool (the EMO
two-level router keeps the top `document_expert_pool` experts for the doc and
prunes the rest). On top of the normal forward pass we add one (or more) ghost
experts. A ghost is a **full new expert** whose router row *and* MLP weights are
the **same linear combination** of the document pool's experts:

```
alpha_i  = blend coefficients over the document pool (sum to 1)
r_ghost  = sum_i alpha_i * r_i        # router row  = blend of pool router rows
W_ghost  = sum_i alpha_i * W_i        # MLP weights = blend of pool MLP weights
```

Key properties:

- **Ghost, not instantiated.** `r_ghost` / `W_ghost` are never stored or
  initialized as parameters; they are recomputed per document from the existing
  experts on every forward. There are zero new parameters.
- **Routes like a real expert.** The ghost's logit is `sum_i alpha_i * logit_i`
  (the blended router row applied to the token), and it joins the routing
  **softmax denominator alongside the real pool experts**, so the pool experts
  and the ghost(s) form a single distribution that sums to one — the real
  experts shrink to make room for the ghost. (This is the *renormalized*
  variant; the base forward is intentionally **not** preserved.)
- **Backprop updates the originals.** Because the ghost is a differentiable
  blend of the real experts, autograd routes the ghost's gradient straight back
  into the constituent experts' MLPs *and* their router rows
  (`dL/dW_i += alpha_i * dL/dW_ghost`, likewise for `r_i`). The model is thus
  trained to make averaged experts useful — for every coefficient mode, not
  just the usage-weighted one.
- **Training-only.** Ghosts are added only in `train()` mode; eval/inference
  measure the model with no ghost.

### Choosing the blend coefficients `alpha`

`ghost_extend_coeff_mode` selects how the ghost is composed from the pool:

- `usage` — document-usage-weighted: `alpha_i` ∝ the document-level summed
  routing probability of pool expert `i`. The new expert is the average of what
  the document actually routes to. (Adds an extra router-gradient path through
  `alpha`.)
- `uniform` — equal weight over the whole pool.
- `random` — uniform average over a random sample of `ghost_extend_random_k`
  pool experts (the mode where `ghost_extend_num > 1` is meaningful, since each
  ghost re-samples; `usage`/`uniform` are deterministic across ghosts).

## Hyperparameters

All are router-config fields, set in the launch script via dotted CLI
overrides (`--model.block.feed_forward_moe.router.<name>=...`):

| Name | Default | Meaning |
|---|---|---|
| `ghost_extend_mode` | `false` | Master switch. When `true` and in training mode, every document gets ghost expert(s). |
| `ghost_extend_num` | `1` | Number of ghost experts simulated per document (summed into the output). |
| `ghost_extend_coeff_mode` | `usage` | Blend-coefficient scheme: `usage` / `uniform` / `random`. |
| `ghost_extend_random_k` | `8` | Sample size for `coeff_mode="random"` (clamped to pool size; ignored otherwise). |
| `ghost_extend_route` | `always` | How the ghost is routed. Only `always` is implemented (every doc token passes through the ghost, weighted by its renormalized routing share). `topk` (ghost competes in the per-token top-k via slot displacement) is **deferred** and currently raises. |
| `ghost_extend_detach_coeff` | `false` | If `true`, detach `alpha` from the graph, cutting the *extra* router-grad path that only `usage` adds. The blended-router-row gate path still trains the router rows. No-op for `uniform`/`random` (already constant). |

Notes:
- Requires **softmax** gating (renormalization is defined for the routing
  softmax); the router raises on `sigmoid` gating with ghost enabled.
- The ghost's mixing weight is its renormalized routing share, **not** a tunable
  scalar — there is deliberately no `gate_scale` knob.
- The load-balancing loss and entropy metric are computed on the real-expert
  pool distribution only (the ghost is a transient blend, not an expert to
  balance); the z-loss is on raw logits and is unaffected.

## Implementation

The mechanism lives in the published EMO router and MoE layer (no new
model-type or argparse args):

- `src/olmo_core/nn/moe/twolevel_batchlb_reducedp_sharedexp_randpool_router.py`
  — builds per-document blend coefficients, the blended ghost logits, the
  renormalized routing scores, and the per-token ghost gates; stashes them for
  the MoE layer.
- `src/olmo_core/nn/moe/mlp.py` (`DroplessMoEMLP.ghost_forward`) — materializes
  `W_ghost` per document via an einsum over the expert axis and runs the
  grouped SwiGLU as a grouped-GEMM grouped by document.
- `src/olmo_core/nn/moe/parallel_mlp.py` (`ParallelDroplessMLP.compute_ghost`)
  — passthrough (not supported under expert/tensor parallelism).
- `src/olmo_core/nn/moe/moe.py` (`MoEBase.forward`) — reads the stash and adds
  `gate * ghost_out` to the MoE output.

## Scripts

- `emo_1b14b_130b.sh` — **baseline**: the unmodified EMO 1B/14B 130B-token
  randpool recipe (carried over from `models_sizescaling`, repointed to this
  experiment) for an apples-to-apples reference.
- `emo_1b14b_130b_ghostexpert.sh` — the ghost-expert run. Same recipe plus the
  `ghost_extend_*` overrides (currently `coeff_mode=usage`, `route=always`,
  `num=1`).

See the top-level `CLAUDE.md` for `MODE=local|beaker` launch usage and the
experiment conventions (WandB project `emo-extension`, tag `models_fullextend`,
save root `/weka/oe-training-default/ryanwang/EMO/models_fullextend`,
`DATA_ROOT=s3://ai2-llm`).

## Status / next steps

- Implemented and unit-tested: `always` route, all three coefficient modes,
  renormalized routing, gradients verified to reach both the constituent
  experts and the router rows in every mode.
- Not yet done: the downstream "actually add a new expert and measure
  degradation" evaluation; the `topk` route; coefficient-mode / `num` sweeps.
