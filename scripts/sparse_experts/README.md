# sparse_experts — quarter-sized experts, 8 active of {128, 256, 512, 1024}

## Question

Holding the *active* parameter budget roughly fixed (top_k=8 = 7 routed + 1 shared) and shrinking
each expert to a quarter of the EMO-128e baseline's size (expert `hidden_size` 1024 → 256), how do
EMO's quality and selective-expert behaviour change as the total expert count grows
128 → 256 → 512 → 1024?

## Arms

All four scripts are `scripts/meta_learning/model_scripts/emo128_baseline_20b.sh` with two
changes: `--model.block.feed_forward_moe.hidden_size=256` and `num_experts` (the randpool
`max_document_expert_pool` / `eval_document_expert_pool` follow `num_experts`). Everything else
is identical: OLMoE-mix-0824, 10B tokens (2385 steps; flat WSD trunk so it can be extended later by resuming with a larger `max_duration`), lr 2e-3 flat WSD trunk (warmup 500, decay_steps=1),
lb 1e-1, 1 shared expert, min pool 8, 8 nodes / 64 GPUs, `meta_mode=vanilla` (stock trainer plus
the pool-pinned `lm-full` / `lm-pool32` evaluators), fixed checkpoints every ~2.5B tokens.

| script | experts (active/total) | expert hidden | total params | active params | expert params / layer |
|---|---|---|---|---|---|
| `emo128_baseline_20b.sh` (reference, meta_learning) | 8 / 128 | 1024 | 13.569B | 1.489B | 805.3M |
| `model_scripts/sparse_8of128_10b.sh` | 8 / 128 | 256 | 3.905B | 0.885B | 201.3M |
| `model_scripts/sparse_8of256_10b.sh` | 8 / 256 | 256 | 7.130B | 0.889B | 402.7M |
| `model_scripts/sparse_8of512_10b.sh` | 8 / 512 | 256 | 13.581B | 0.898B | 805.3M |
| `model_scripts/sparse_8of1024_10b.sh` | 8 / 1024 | 256 | 26.483B | 0.914B | 1610.6M |

Counts are `TransformerConfig.num_params` / `num_active_params` (dolma2 padded vocab 100,352,
d_model 2048, 16 layers, 411.0M embedding + LM-head params, router = d_model × num_experts per
layer). Regenerate with `PYTHONPATH=src python scripts/sparse_experts/model_sizes.py`.

Note the 8-of-512 arm has the same total parameter count as the baseline (4× as many experts,
each a quarter the size), and the 8-of-1024 arm is 2× the baseline.

## Launch

```bash
# Verify the built config without training (prints and exits):
MODE=local NPROC=1 PYTHONPATH=src bash scripts/sparse_experts/model_scripts/sparse_8of128_10b.sh --dry-run

# Beaker (commit AND push first — gantry clones from origin):
MODE=beaker bash scripts/sparse_experts/model_scripts/sparse_8of128_10b.sh
```

Checkpoints land under `/weka/oe-training-default/ryanwang/EMO/sparse_experts/<runname>/`;
W&B project `emo-extension`, tags `[pretraining, sparse_experts, sparse_h256]`.

## Launched runs

| run | Beaker experiment | commit | submitted |
|---|---|---|---|
| `sparse_8of1024_10b` | https://beaker.org/ex/01M1HYDT9HCAHFR6FW75ATNAJM (allocated, urgent, 8 nodes) | f7813f54 | 2026-09-02 |
