# Proposal: expert co-activation analysis of `sparse_8of512_10b` (DRAFT, awaiting approval)

## Goal

For the 8-of-512 quarter-size-expert EMO model, measure how often every pair of experts is
active together, per layer, on held-out documents. Output is one 512×512 count matrix per
layer (token-level), plus a document-level variant, plus the raw per-token expert indices so any
later analysis (cross-layer, sequential, per-domain) needs no re-run.

## Inputs

| item | choice | why |
|---|---|---|
| checkpoint | `sparse_experts/sparse_8of512_10b/step2384` (10B-token fixed ckpt) | last full-LR checkpoint; `step2385` differs by one lr≈0 step |
| loading | native OLMo-core (`TransformerConfig.from_dict(config.json).build` + `load_model_and_optim_state`), bf16 autocast, 1 GPU | no HF conversion needed; identical pattern to `scripts/modular_extension/eval_k32cpt_ce.py` |
| routing at eval | as configured: `eval_document_expert_pool = 512` = full pool | the run already trains with eval pool = all experts, so this is exactly inference-time routing: per token, top-7 of the 511 standard experts + shared expert 511 always on |
| documents | `meta_learning/data/meta128_20B-40B/` (18.3M whole docs, 36.9B tokens, same seed-0 data order the 512 run used) | already extracted; entirely past the 512 model's 10B-token horizon, so never seen |

Window composition (from the cluster-label file, all 18.3M docs):

| source | token share | doc share | avg doc len |
|---|---|---|---|
| dclm | 93.4% | 95.6% | 1,966 |
| proof-pile-2 | 3.2% | 0.4% | 14,945 |
| starcoder | 1.8% | 2.5% | 1,461 |
| pes2o | 1.6% | 1.3% | 2,408 |
| olmo-mix (wiki etc.) | 0.1% | 0.2% | 764 |

Doc length percentiles 50/90/99 = 729 / 3,800 / 21,517 tokens; 9% of docs exceed 4,096 tokens but they hold 59% of all tokens.

## Sampling design

- **Stratified by source with quotas**: dclm 24,000 docs; starcoder, pes2o, proof-pile-2, olmo-mix 4,000 docs each → **40,000 docs**. Uniform random within source (seeded). Selection is done on the compact cluster-label file (has `source_path`, `doc_start_offset`, `doc_len` for every doc), then the chosen docs' `token_ids` are pulled in one parallel pass over the 16 `docs-*.jsonl.gz` shards.
- **Truncate each doc to its first 4,096 tokens** (the model's context length). Keeps per-doc influence bounded and avoids the 9% of long docs contributing 59% of tokens. Alternative if preferred: chunk long docs into several 4,096 windows (more tokens from code/math).
- Expected volume: **≈60–80M tokens**. With 21 expert pairs per token per layer that is >1e9 pair observations per layer against 130k pairs, so ~10k observations per pair on average; counts are far past sampling noise.
- **Two aggregates from the same pass** (token counts are additive): (a) *natural mix* — per-source matrices reweighted to the window's token shares, i.e. what the training distribution induces; (b) *per-source* matrices kept separately for domain comparison. The uniform-quota sample exists so minority domains have enough tokens for (b).

## Forward pass

- Pack docs EOS-separated into 4,096-token sequences and pass `doc_lens` / `max_doc_lens`, exactly as the trainer and the CE eval scripts do → intra-document attention masking and per-document `seg_id` for the randpool router. Batch of 8 sequences.
- **Capture**: `register_forward_hook` on each of the 16 `MoE.router` modules; element 1 of the router's return tuple is `expert_indices` of shape `(B, S, 8)`, last column = shared expert 511. No source edits. Padding tokens are masked out.

## Statistics accumulated (per layer, on GPU, exact int64)

1. **Token-level co-activation** `C_tok[l, i, j]` = number of tokens at which experts *i* and *j* are both in the top-8. Diagonal = per-expert token usage. Computed as `A^T A` on the token×expert 0/1 matrix per batch (fp32, exact at batch scale) and added into an int64 accumulator. Shape `(16, 512, 512)`; the shared row/col is kept for completeness and excluded from analysis.
2. **Document-level co-activation** `C_doc[l, i, j]` = number of documents in which both *i* and *j* are active at least once. This is the modularity notion EMO's randpool training acts on (per-document expert pools).
3. **Per-source copies** of 1 and 2: `(5, 16, 512, 512)`.
4. **Raw per-token top-7 indices**, int16, shape `(N_tokens, 16, 7)` ≈ 18 GB for 80M tokens, plus a token→(doc, position) index and the doc metadata (source, offset, length). Router weights not stored by default (would double it).

## Analysis and report

- Normalizations from the counts: lift / PMI `C_ij·N / (C_i·C_j)`, Jaccard, conditional `P(j|i)`; comparison against the independent-routing null with the same marginals.
- Structure: spectral / Louvain clustering of the lift graph per layer; heatmaps reordered by cluster; modularity score per layer; how block structure changes with depth; token-level vs document-level agreement; per-source overlap of the strongest pairs.
- Report page under `claude_outputs/sparse_experts/coactivation/` built by a `build_report.py`, registered in `scripts/publish_reports.sh` (you deploy).
- The extraction script is parametric in checkpoint path, so the same pass can later run on `sparse_8of1024_10b`, the 128-expert `meta128_vanilla/step2384`, and future 128/256 arms for a like-for-like comparison across expert counts.

## Compute plan

- Model in bf16 is 27 GB; fits on the local A100 with room for batch 8×4,096. Forward-only throughput guess 20–40k tok/s → **≈35–70 min for the full pass on the single local GPU**. No Beaker needed; if it turns out slower, shard by source over 8 GPUs on Beaker (allocated).
- **Staged**: (1) pilot with 500 docs locally (~2 min) to validate hooks, shapes, packing and throughput; (2) full 40k-doc pass in the background; (3) analysis + report.

## Storage

`sparse_experts/coactivation/sparse_8of512_10b_step2384/` on weka: count arrays (~350 MB), per-token indices (~18 GB), metadata. Nothing goes to S3 except the report HTML under `claude_outputs/`.

## Optional extensions (off by default; say if wanted)

- A second pass with the eval pool pinned to 32 or 64 (via the router's `eval_document_expert_pool`) to see co-activation under selective-expert routing.
- Adjacent-layer cross co-activation `(15, 512, 512)` (expert *i* at layer *l* with *j* at *l+1*); derivable later from the saved per-token indices.
- Weighted variant using router weights instead of 0/1 membership.

## Decision points for you

1. Checkpoint `step2384` and full-pool eval routing — OK?
2. 40k docs with the 24k/4k/4k/4k/4k source quotas, truncated at 4,096 tokens — OK, or chunk long docs instead?
3. Store all per-token indices (~18 GB on weka) — OK, or only a 10% subset?
4. Run locally on the 1 A100 (staged pilot then full) — OK?
5. Any of the optional extensions wanted in the first pass?
