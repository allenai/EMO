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
PYTHONPATH=src torchrun --nproc-per-node=1 scripts/meta_learning/model_scripts/verify_meta_step.py
PYTHONPATH=src torchrun --nproc-per-node=2 scripts/meta_learning/model_scripts/verify_meta_step.py

# 1. Local smoke (tiny model on real data, all modes + inner-lr sweep, restore assert on):
bash scripts/meta_learning/model_scripts/smoke_local.sh
#    -> pick inner_lr with train/meta delta/weight norm ~1e-3..1e-2; update the pilot scripts.

# 2. Pilot (Beaker, 8 nodes/arm; commit AND push first — gantry clones from origin):
MODE=beaker bash scripts/meta_learning/model_scripts/emo128_baseline_20b.sh   # vanilla 128e, 20B
MODE=beaker bash scripts/meta_learning/model_scripts/meta_sametok_10b.sh      # FOMAML same-tokens, 10B (~2.1x/step)
MODE=beaker bash scripts/meta_learning/model_scripts/meta_heldout_10b.sh      # FOMAML held-out split, 10B (~1x/step)
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

Layout: `model_scripts/` holds phase-1 pretraining (the per-arm launch scripts, the local
smoke, the correctness gate); `eval_scripts/` holds the phase-2 cluster-wise CPT + eval
pipeline; report infra (`build_report.py`, `fetch_report_data.py`) stays at the top level
(`scripts/publish_reports.sh` expects it there).

| file | role |
|---|---|
| `src/olmo_core/train/train_module/transformer/meta_learning.py` | the two-phase train module + config |
| `src/olmo_core/nn/moe/twolevel_batchlb_reducedp_sharedexp_randpool_router.py` | `meta_force_pool` / `meta_skip_aux` flags |
| `src/olmo_core/train/callbacks/pool_pinned_lm_evaluator.py` | pool-pinned ppl evaluator |
| `src/scripts/train/olmoe-1B-7B_fsl_meta.py` | entry script (randpool-only clone of the parent) |
| `model_scripts/verify_meta_step.py` | correctness gate (oracles, finite-difference, manual-reference grads) |
| `model_scripts/smoke_local.sh` | tiny-model smoke on real data + inner-lr sweep |
| `model_scripts/emo128_baseline_20b.sh`, `model_scripts/meta_sametok_10b.sh`, `model_scripts/meta_heldout_10b.sh` | pilot arms |
| `build_report.py` | report → `claude_outputs/meta_learning/report.html` |

Outputs land in `/weka/oe-training-default/ryanwang/EMO/meta_learning/` (= `./meta_learning/` in
a GPU-attached session; gitignored — never `git add -A`).

## Phase 2: per-arm k=32 cluster-wise CPT on tokens 20B–40B

The modular_extension k=32-CPT (no extension) protocol, applied per arm to the 20B
checkpoints (`step4768`), on the shared extracted 20B–40B doc window
(`meta_learning/data/meta128_20B-40B/`, one extraction serves all arms). Everything is
**per-arm**: each model clusters the docs with its OWN router and selects experts from its
own concentration. CPT stages are **standard 32-expert training** (pool pinned
min=max=eval=33 — no random-pool EMO objective, since the meta arms never trained across
pool sizes), **carry** optimizer (Adam moments sliced from the 20B checkpoint), 4-node
Beaker jobs, sequential clusters 0..31. Budget: 20B train tokens per arm (final model =
40B). Arms: `sametok_ws_lam05` first, then `vanilla`.

Stage order (per arm, `MODEL=vanilla|sametok_ws_lam05`):

```bash
MODEL=... bash scripts/meta_learning/eval_scripts/convert_20b_to_hf.sh          # 1. step4768 -> HF (local GPU)
MODEL=... bash scripts/meta_learning/eval_scripts/launch_embed_docs.sh          # 2. pilot shards 0-15 (Beaker)
MODEL=... SHARDS="$(seq -s, 0 127)" JOBS=4 bash scripts/meta_learning/eval_scripts/launch_embed_docs.sh  # full
MODEL=... bash scripts/meta_learning/eval_scripts/cluster_docs.sh               # 3. k=32 spherical k-means (local CPU)
PYTHONPATH=.:src python scripts/meta_learning/eval_scripts/cluster_expert_concentration.py --model ...  # 4. top-32/layer selection
MODEL=... bash scripts/meta_learning/eval_scripts/build_cluster_token_data.sh   # 5. per-cluster token shards (20B budget)
CLUSTERS=0 MODEL=... bash scripts/meta_learning/eval_scripts/run_k32cpt_arm.sh  # 6. pilot stage, then full 0..31
```

Phase-2 files (all under `eval_scripts/`): `convert_20b_to_hf.sh`, `launch_embed_docs.sh`,
`cluster_docs.sh`, `cluster_expert_concentration.py` (127 standard experts),
`build_cluster_token_data.sh` (parameterized `src.scripts.clustering.build_cluster_token_data`),
`expert_subset_surgery.py` (128-expert pool, `--conc` per arm, bf16 snapshots opt-in),
`k32cpt_stage.sh` (4 nodes), `run_k32cpt_arm.sh` (sequential driver, resumable).
