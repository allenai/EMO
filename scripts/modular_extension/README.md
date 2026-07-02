# modular_extension

Experiment folder for the modular-extension line of work, building on the `models_v2`
runs (same W&B project `emo-extension`, same conventions from `CLAUDE.md`:
`MODELS_DIR=/weka/oe-training-default/ryanwang/EMO/modular_extension`,
`DATA_ROOT=s3://ai2-llm`, tags `[pretraining, modular_extension]`). A dedicated web
report will be registered in `scripts/publish_reports.sh` once there are runs.

## Training-data window extractor

`extract_training_doc_window.py` extracts the exact set of **whole documents** a training
run consumes in a token window, e.g. everything `emo_64exp_50b_wsd_lr2e-3` trains on
between its 100B and 110B token marks (deterministic even beyond the run's current step,
so it applies to the in-place `extend1t` continuation).

**Semantics** (any-overlap superset): the window `[start_tokens, end_tokens)` is mapped to
instance positions (position = token // 4096); every 4096-token instance the run consumes
at those positions is found via the run's own shuffled global-index file, and every
document overlapping any of those instances is reconstructed **whole** (stitched from
neighboring dataset offsets when an instance straddles a document boundary) and deduped by
`(source_path, doc_start_offset)`. Total doc-tokens therefore lands slightly above the
window size. A document's trailing EOS separator (dolma2: `100257`) is not included in
`token_ids`.

**Correctness notes** (see the script docstring for details):

- The loader must use `chunk_size = max_target_sequence_length // sequence_length` (= 2
  for these runs) to reproduce the training order — `NumpyDataLoaderBase.wrap_numpy_dataset`
  sets this, and it is why the run's cache contains `global_indices_chunk2_*.npy`. (The
  same fix was applied to `src/scripts/dump_training_batch.py`, which previously assumed
  chunk_size=1.)
- With `--work-dir` pointing at the training run's dataset cache (default
  `~/dataset-cache`), the extractor memmaps the **exact global-indices file the run
  itself wrote**, and it hard-asserts the dataset fingerprint from the checkpoint.
- `instance_filter_config` only loss-masks instances (never drops/reorders them); the
  extractor recomputes that mask as provenance (`filtered_touch_positions`).

**Output** (`--output-dir`): `docs-<shard>.jsonl.gz` with one JSON per document —
`source_path`, `doc_start_offset`, `doc_len`, `n_touching_instances`,
`first_touch_position`, `touch_positions`, optional `filtered_touch_positions` /
`truncated_left/right`, and `token_ids` — ordered by `(source file, doc_start_offset)`;
plus `manifest-<shard>.json` with the run/fingerprint/window/stats.

**Run it** (sharded by file, so no cross-shard dedup is needed):

```bash
# small validation slice (one step's worth of instances)
PYTHONPATH=src python scripts/modular_extension/extract_training_doc_window.py \
    --checkpoint models_v2/emo_64exp_50b_wsd_lr2e-3/step11921 \
    --start-tokens 100000595968 --end-tokens 100004790272 \
    --output-dir modular_extension/data/validation --dump-instances /tmp/val_instances.npz

# full 10B window, 16 parallel shards
for i in $(seq 0 15); do
  PYTHONPATH=src python scripts/modular_extension/extract_training_doc_window.py \
      --checkpoint models_v2/emo_64exp_50b_wsd_lr2e-3/step11921 \
      --start-tokens 100e9 --end-tokens 110e9 \
      --shard $i --num-shards 16 \
      --output-dir modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-110B \
      > /tmp/extract_shard$i.log 2>&1 &
done
```

Outputs land under `modular_extension/data/` on weka (not `claude_outputs/` — too large
for the S3 report sync).

## Document-level router clustering (k=64 partition of the window)

Partitions the extracted documents into k groups by clustering **document-level router
embeddings** from the EMO **100B checkpoint** (`step23842`): the published pretraining
clustering recipe (`probs` → `mean_pca_l2` → `spherical_kmeans`; see
`scripts/clustering/`), but with each document represented by its router softmax
probabilities **averaged over its first 2048 tokens** (in-forward pooling, math identical
to `src/scripts/clustering/extract_document.py`; `doc_topk_freq` is saved too).

Stages (all idempotent; sharding is docs `i::128` of the global doc enumeration, so a
small run's shards count toward a later full sweep):

```bash
# 1. one-time: convert the 100B checkpoint to HF (local GPU validates logits)
bash scripts/modular_extension/convert_100b_to_hf.sh

# 2. embed docs on Beaker (default: shards 0-15 = ~12.5% ≈ 1.2B capped tokens, 2 jobs x 8 GPUs)
bash scripts/modular_extension/launch_embed_docs.sh
# full sweep (only with explicit approval -- ~10B capped tokens):
#   SHARDS="$(seq -s, 0 127)" JOBS=4 bash scripts/modular_extension/launch_embed_docs.sh

# 3. merge + cluster + export partition (CPU, local)
SHARDS=0-15 bash scripts/modular_extension/cluster_docs.sh
```

Outputs: embeddings under `modular_extension/cluster/emo100b_step23842/embeddings/`
(`doc_probs-*.npy` fp16 + ids + per-shard info), cluster.py artifacts under
`modular_extension/cluster/emo100b_step23842/doc_probs_mean_pca_l2_spherical_kmeans_k64/`
(assignments, centroids, metrics, summary), and the partition at
`modular_extension/data/<run>_100B-110B/doc_clusters_k64.jsonl.gz`
(`{source_path, doc_start_offset, doc_len, cluster}` — joinable back onto the doc data).
Re-clustering at another k reuses the saved embeddings: `K=32 SHARDS=... bash
scripts/modular_extension/cluster_docs.sh`.
