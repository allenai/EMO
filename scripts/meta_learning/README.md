# meta_learning — FOMAML-style EMO pretraining (32-of-128 selective inner step → full-model outer step)

## Motivation

The modular_extension **k=32 CPT (no extension)** result showed that post-hoc selective-expert CPT
on a vanilla-EMO model buys almost nothing on its own cluster and degrades the pool broadly —
plain CPT dominates on every metric. This experiment changes **EMO's own pretraining objective**
(the first 100B-token phase, from scratch, 128-expert model) so the model is trained to have the
property that *updates made in 32-expert selective mode also improve the full model*.

## Mechanism (first-order MAML / FOMAML)

Per train step (`MetaLearningTransformerTrainModule`, in
`src/olmo_core/train/train_module/transformer/meta_learning.py`):

1. **Inner phase** (selective): forward+backward the whole rank batch with each document
   restricted to its router top-`inner_pool_size` (32) experts (randpool router's per-doc pool
   pinned via `meta_force_pool`). The DP-reduced gradient aggregates every document in the
   global batch — each expert's inner grad comes from the docs that routed to it.
2. **Pseudo-step**: in-place SGD probe on expert weights only:
   `θ'_exp = θ_exp − inner_lr · g_inner`. Temporary — it determines *where* the outer gradient
   is evaluated and is undone before the real optimizer step.
3. **Outer phase** (full): forward+backward the same tokens (`same_tokens`) or the other half of
   the rank's micro-batches (`heldout`) with all 128 experts. FSDP2's reduce-scatter accumulates,
   so `.grad = λ·g_inner + g_outer(θ')` (default `lambda_inner=0` = pure FOMAML).
4. **Restore** expert weights bitwise; AdamW consumes the accumulated grads.

`meta_mode=vanilla` delegates to the stock train module (baseline arm + correctness oracle);
`outer_only` is a single full-routing pass (the α=0 reference).

Knobs (all `--train_module.<knob>=...`): `meta_mode`, `inner_lr`, `inner_pool_size`
(`null` ⇒ keep random pool sampling on the inner pass), `lambda_inner`, `lb_on_inner`,
`inner_grad_clip`, `log_grad_cosine`.

**Known caveats** (watched, not yet addressed):
- At `lambda_inner=0` the router never receives selective-mode gradient (vanilla EMO's selective
  robustness comes from training WITH random pools). Contingency arm: `lambda_inner=0.5
  lb_on_inner=true`. Watch `train/unique experts` for collapse.
- bf16 all-gather can quantize away pseudo-step deltas ≪0.4% relative — keep
  `train/meta delta/weight norm` ≥ ~1e-3.
- `heldout` halves the effective outer batch (512 instances/step).

## Pipeline

```bash
# 0. Mechanism-correctness gate (tiny model, fp32, must pass before any Beaker launch):
PYTHONPATH=src torchrun --nproc-per-node=1 scripts/meta_learning/verify_meta_step.py
PYTHONPATH=src torchrun --nproc-per-node=2 scripts/meta_learning/verify_meta_step.py

# 1. Local smoke (tiny model on real data, all modes + inner-lr sweep, restore assert on):
bash scripts/meta_learning/smoke_local.sh
#    -> pick inner_lr with train/meta delta/weight norm ~1e-3..1e-2; update the pilot scripts.

# 2. Pilot (Beaker, 8 nodes/arm; commit AND push first — gantry clones from origin):
MODE=beaker bash scripts/meta_learning/emo128_baseline_20b.sh   # vanilla 128e, 20B
MODE=beaker bash scripts/meta_learning/meta_sametok_10b.sh      # FOMAML same-tokens, 10B (~2.1x/step)
MODE=beaker bash scripts/meta_learning/meta_heldout_10b.sh      # FOMAML held-out split, 10B (~1x/step)
```

## Headline metric

Two pool-pinned LM evaluators (`PoolPinnedLMEvaluatorCallbackConfig`) run every 500 steps on the
v3-small ppl validation mix: `eval/lm-full/*` (model-default eval pool = 128) and
`eval/lm-pool32/*` (pool pinned to 32). The **selective-vs-full CE gap** (pool32 − full) per arm,
against the vanilla baseline's gap, is the pilot's go/no-go: the meta arms should shrink the gap
without materially regressing full-routing CE (compare at matched tokens = 10B, and vs the
baseline's 20B point for matched compute against `same_tokens`).

Per-step diagnostics in W&B (`train/` namespace): `meta inner CE loss`,
`meta inner grad norm (experts)`, `meta pseudo step delta norm`, `meta delta/weight norm`,
`meta inner-outer grad dot/cosine (experts)`.

Downstream (later, on finished checkpoints): the modular_extension k=32 CPT protocol on the
meta-pretrained model vs the baseline — does selective-expert CPT transfer now?

## Files

| file | role |
|---|---|
| `src/olmo_core/train/train_module/transformer/meta_learning.py` | the two-phase train module + config |
| `src/olmo_core/nn/moe/twolevel_batchlb_reducedp_sharedexp_randpool_router.py` | `meta_force_pool` / `meta_skip_aux` flags |
| `src/olmo_core/train/callbacks/pool_pinned_lm_evaluator.py` | pool-pinned ppl evaluator |
| `src/scripts/train/olmoe-1B-7B_fsl_meta.py` | entry script (randpool-only clone of the parent) |
| `verify_meta_step.py` | correctness gate (oracles, finite-difference, manual-reference grads) |
| `smoke_local.sh` | tiny-model smoke on real data + inner-lr sweep |
| `emo128_baseline_20b.sh`, `meta_sametok_10b.sh`, `meta_heldout_10b.sh` | pilot arms |
| `build_report.py` | report → `claude_outputs/meta_learning/report.html` |

Outputs land in `/weka/oe-training-default/ryanwang/EMO/meta_learning/` (= `./meta_learning/` in
a GPU-attached session; gitignored — never `git add -A`).
