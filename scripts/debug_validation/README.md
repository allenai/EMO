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
