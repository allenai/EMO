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

| run | Beaker experiment | commit | submitted | status |
|---|---|---|---|---|
| `sparse_8of1024_10b` | https://beaker.org/ex/01M1HYDT9HCAHFR6FW75ATNAJM (allocated, urgent, 8 nodes) | f7813f54 | 2026-09-02 | FAILED at first step: grouped_gemm 0.3.0 `At most 512 experts are supported when batch_sizes is a CUDA tensor` (Beaker image kernel limit). Needs a fix in `MoEMLP.gmm` before relaunch. |
| `sparse_8of512_10b` | https://beaker.org/ex/01M1M32WG6VSPKDXWVPDNZ5661 (allocated, urgent, 8 nodes) | 0235822 | 2026-09-03 | DONE 2026-09-04: 2385 steps, final train CE 2.98, ckpts step{596,1192,1788,2384,2385} |
| `sparse_8of1024_10b` (relaunch) | https://beaker.org/ex/01M1NG1HKZCH03MHEN8CDXND6A (allocated, urgent, 8 nodes) | af71ff93 | 2026-09-04 | DONE 2026-09-04: 2385 steps, final train CE 2.97 (512: 2.98), 10.9k tok/s/GPU, ckpts step{596,1192,1500,1788,2000,2384,2385} (~297 GB each). grouped_gemm >512 fix verified first: check 01M1M3TVYYANS2K4DE4YR0R4M6 (all OK), smoke 01M1NFBSA4SHX8CMJERFAC3S9K (20 steps, exit 0) |

## Co-activation analysis of sparse_8of512_10b (2026-09-04)

Design: `coactivation/PROPOSAL.md`. Pipeline: `coactivation/sample_docs.py` (40k held-out docs from the
20B-40B window, stratified 24k dclm / 4k each starcoder, pes2o, proof-pile-2, olmo-mix; truncated to
4,095 tokens + EOS; 50.9M tokens) -> `coactivation/launch_beaker.sh` (1 allocated node x 8 GPUs, native
forward, router hooks; pilot 01M1NKCXQ4ETAB0TX1HDV71XHF, full 01M1NS78PJJXY5KJW8GE8ABE88, ~80k tok/s/GPU)
-> `coactivation/analyze_coactivation.py` -> `build_report.py` (claude_outputs/sparse_experts/report.html).
Data on weka: `sparse_experts/coactivation/sparse_8of512_10b_step2384/pool_{config,64}/` (counts.npz,
topk.npy int16 (N,16,7), tok_doc/tok_pos, info.json; 50 GB total).

Headline (full routing, eval pool 512): modularity of the token co-activation graph Q = 0.25-0.36 in
layers 1-15 and 0.48 in layer 0 (shuffled null 0); Louvain 4-10 communities/layer; effective #experts
411-435 of 511, none unused; only 0-2% of pairs never co-activate beyond layer 2 (45% in layer 0);
median lift 0.15-0.40 with p99 9-31 (heavy tail = the structure); token-vs-doc lift Spearman rises 0.09
(layer 0) -> 0.5 (late); docs touch ~400 experts/layer early -> ~230 late. Pool 64: Q 0.30-0.35,
token-vs-doc agreement 0.7-0.9, effective experts 331-413.
