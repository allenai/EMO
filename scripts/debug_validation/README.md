# debug_validation

Validation-perplexity view of the OLMoE3-ladder 275M 512-expert models (standard vs EMO routing) and
a pool-size-distribution ablation of EMO.

## 1. Offline v3-small ppl validation of the existing 512-expert models

The two `sparse_experts` runs `olmoe3_275m_10b` (standard top-16 router) and `olmoe3_275m_emo_10b`
(EMO, uniform per-document pool in [16, 512]) were trained without in-loop evals. `eval_ppl_validation.py`
reproduces OLMo-core's in-loop LM eval (`LMEvaluatorCallbackConfig` on `DataMix.v3_small_ppl_validation`)
on every permanent checkpoint (steps 5000 / 10000 / 15000 / 19000 / 19074 = 2.6B ... 10B tokens):

| set | val docs (= padded instances) | val tokens |
|---|---|---|
| c4_en | 2,099 | 995k |
| dolma_books | 4 | 487k |
| dolma_common-crawl | 796 | 480k |
| dolma_pes2o | 212 | 513k |
| dolma_reddit | 2,315 | 482k |
| dolma_stack | 404 | 418k |
| dolma_wiki | 869 | 497k |
| ice | 291 | 905k |
| m2d2_s2orc | 10,914 | 990k |
| pile | 560 | 669k |
| wikitext_103 | 59 | 250k |

Data: `s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/<set>/val/part-0-00000.npy` (raw
uint32 token streams + `.csv.gz` document index; not on the weka mirror). Each document is one instance
truncated to the training sequence length 8192 and padded; the metric is the mean per-token CE over the
label-masked positions (identical to the in-loop `eval/lm/<set>/CE loss`) plus a labeled-tokens-only
variant. Model loading = `scripts/sparse_experts/olmoe3_routing/extract_routing.py` (pinned submodule,
bf16). The instance indices are cached under the shared work dir
`/weka/oe-training-default/ryanwang/dataset-cache/dataset-common/instance-indices-8192-*` (pre-warmed
from a GPU session; the parallel prep can race on the shared S3 client -> the script preps serially).

```bash
bash scripts/debug_validation/launch_ppl_validation.sh        # 2 allocated 1-GPU Beaker jobs (one per model)
python scripts/debug_validation/aggregate_ppl_validation.py    # -> claude_outputs/debug_validation/ppl_validation/{summary.json,table_*.md}
```

Results: `claude_outputs/debug_validation/ppl_validation/<run>/step<N>.json`.

## 2. Beta-distributed document pool sizes (EMO ablation)

`model_scripts/olmoe3_275m_emo_beta{2,4}_10b.sh`: the identical EMO model / data / recipe / 4-node
allocated topology as `sparse_experts/model_scripts/olmoe3_275m_emo_10b.sh`, but each training document's
pool size `d` over the same range [16, 512] is drawn as

    u ~ U(0, 1);  d = round(16 + (512 - 16) * u ** (1 / alpha))        # fraction ~ Beta(alpha, 1)

with alpha = 2 (mean pool 347; 24% of docs >= 448) and alpha = 4 (mean 413; 43% >= 448), vs uniform
(mean 264). Eval pool stays 512. Implemented as `OLMOE3_EMO_POOL_DIST=beta:<alpha>` in
`scripts/sparse_experts/olmoe3_275m.py`, which patches `EmoRouterV2._pool_sizes` in-process (the pinned
`external/OLMo-core` is untouched). Both arms run the in-loop v3-small ppl validation every 1000 steps
and at the end (`OLMOE3_PPL_EVAL_INTERVAL=1000`, logged to W&B as `eval/lm/<set>/CE loss`).

Checkpoints: `/weka/oe-training-default/ryanwang/EMO/debug_validation/<run>/` (permanent every 5000 steps
+ final). W&B project `emo-extension`, tags `[pretraining, debug_validation, olmoe3_275m, 512e, emo,
pool_beta<alpha>, jupiter, 10b]`.

## Launched

| run | Beaker | commit | submitted | status |
|---|---|---|---|---|
| `olmoe3_275m_emo_beta2_10b` | https://beaker.org/ex/01M23NCFD03S129HJZN5RC25YW (allocated, 4 nodes) | 20ed25409 | 2026-09-09 18:03 UTC | DONE 19:41 UTC (19,074 steps, ~1h35 incl. 20 in-loop evals of ~20 s; ckpts step{5000,10000,15000,19000,19074}) |
| `olmoe3_275m_emo_beta4_10b` | https://beaker.org/ex/01M23ND2QPD1ZD1J0V1HHMCE7A (allocated, 4 nodes) | 20ed25409 | 2026-09-09 18:03 UTC | DONE 19:41 UTC (same) |
| ppl validation `olmoe3_275m_10b` (5 ckpts) | https://beaker.org/ex/01M23NN9KTC9CWXEN2CQBR7BTZ (allocated, 1 GPU) | 20ed25409 | 2026-09-09 18:13 UTC | DONE 19:01 UTC (5/5 ckpts, ~9 min each) |
| ppl validation `olmoe3_275m_emo_10b` (5 ckpts) | https://beaker.org/ex/01M23NNPBAG459EMB0Z0K2J4S3 (allocated, 1 GPU) | 20ed25409 | 2026-09-09 18:13 UTC | DONE 19:11 UTC (5/5 ckpts) |
| offline-vs-in-loop cross-check: `olmoe3_275m_emo_beta4_10b/step19000` | https://beaker.org/ex/01M23V3EXP4NSKDTFHW1VK8SZ2 (allocated, 1 GPU) | 332970e2d | 2026-09-09 19:47 UTC | DONE: offline == in-loop within 0.0006 CE on every set (bf16 noise) |

## Results: offline v3-small ppl validation (CE loss, in-loop convention; tables in `claude_outputs/debug_validation/ppl_validation/`)

Mean over the 11 sets, standard vs EMO (uniform pool): 3.241 / 3.259 (step 5000), 3.075 / 3.097 (10000),
3.005 / 3.029 (15000), 2.973 / 2.996 (19000), 2.974 / 3.000 (19074). EMO trails by +0.017 to +0.025 CE on
average, at every checkpoint and on every set (largest on m2d2_s2orc / reddit / c4 / common-crawl, +0.03 to
+0.05; smallest on pes2o / stack / books, ~+0.01); the gap does not close with tokens. Step 19000 -> 19074 is
flat (WSD trunk at constant LR).

## Results: four arms (`claude_outputs/debug_validation/ppl_validation/table_four_arms.md`; Beta arms from their in-loop evals, `inloop_beta_arms.json`)

Mean CE over the 11 sets:

| step | standard | EMO uniform [16,512] | EMO Beta(2,1) | EMO Beta(4,1) |
|---|---|---|---|---|
| 5000 | 3.241 | 3.259 | 3.248 | 3.238 |
| 10000 | 3.075 | 3.097 | 3.085 | 3.076 |
| 15000 | 3.005 | 3.029 | 3.011 | 3.004 |
| 19000 | 2.973 | 2.996 | 2.982 | 2.971 |
| 19074 | 2.974 | 3.000 | 2.985 | 2.979 |

Skewing the per-document pool toward large pools closes EMO's full-routing CE gap to the standard router
monotonically: uniform +0.022, Beta(2,1) +0.009, Beta(4,1) -0.003 (step 19000, per-set differences all
within +-0.016). The ordering holds at every checkpoint from step 5000 on. Not measured here: what the
Beta arms give up in selective (small-pool) routing quality, which is EMO's purpose - that needs a pool-32/64
eval of the same checkpoints (the routing extractor's `--restrict` can do it).
