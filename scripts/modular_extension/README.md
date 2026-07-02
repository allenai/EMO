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
