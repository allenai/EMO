#!/usr/bin/env bash
# Full extraction of the documents emo_64exp_50b_wsd_lr2e-3 trains on between its
# 100B and 110B token marks (any-overlap whole-document superset). 16 file-sharded
# local workers (pure S3 byte-range I/O; no GPUs / Beaker needed), then a manifest merge.
#
#   bash scripts/modular_extension/run_extract_100b_110b.sh
#
# Output: modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-110B/
#   docs-{000..015}.jsonl.gz + manifest-*.json + manifest.json (merged) + logs/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CHECKPOINT=models_v2/emo_64exp_50b_wsd_lr2e-3/step11921
OUT=modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-110B
NUM_SHARDS=16
LOGS="$OUT/logs"
mkdir -p "$LOGS"

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    PYTHONPATH=src python scripts/modular_extension/extract_training_doc_window.py \
        --checkpoint "$CHECKPOINT" \
        --start-tokens 100e9 --end-tokens 110e9 \
        --shard "$i" --num-shards "$NUM_SHARDS" \
        --output-dir "$OUT" \
        > "$LOGS/shard$i.log" 2>&1 &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [[ $fail -ne 0 ]]; then
    echo "ERROR: one or more shards failed; see $LOGS/" >&2
    exit 1
fi

OUT="$OUT" python - <<'EOF'
import glob, json, os

out = os.environ["OUT"]
shards = sorted(glob.glob(os.path.join(out, "manifest-*.json")))
merged = None
for p in shards:
    m = json.load(open(p))
    if merged is None:
        merged = {k: v for k, v in m.items() if k not in ("shard", "stats", "elapsed_seconds", "docs_file")}
        merged["stats"] = dict(m["stats"])
        merged["shard_elapsed_seconds"] = [m["elapsed_seconds"]]
    else:
        assert m["dataset_fingerprint"] == merged["dataset_fingerprint"]
        for k, v in m["stats"].items():
            merged["stats"][k] += v
        merged["shard_elapsed_seconds"].append(m["elapsed_seconds"])
merged["num_shard_manifests"] = len(shards)
with open(os.path.join(out, "manifest.json"), "w") as f:
    json.dump(merged, f, indent=2)
s = merged["stats"]
print(f"MERGED: {len(shards)} shards, {s['instances']:,} instances, {s['docs']:,} docs, "
      f"{s['doc_tokens']:,} doc-tokens ({s['doc_tokens']/(s['instances']*4096):.3f}x window), "
      f"{s['requests']:,} requests, {s['bytes_read']/1e9:.1f}GB read, "
      f"{s['truncated_docs']} truncated, {s['masked_instances']} filter-masked instances")
EOF
