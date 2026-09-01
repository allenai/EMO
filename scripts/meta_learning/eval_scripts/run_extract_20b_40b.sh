# PARENT: "scripts/modular_extension/run_extract_100b_110b.sh"
# DESCRIPTION:
#     Full extraction of the documents the meta_learning 128e runs consume between their 20B and
#     40B token marks — the phase-2 window for the k=32-CPT-style cluster-wise selective CPT
#     (phase 1 = 20B pretraining; phase 2 = per-cluster CPT on the following 20B). All
#     meta_learning arms + the vanilla baseline share the same dataset config / seed /
#     sequence-length, hence the same deterministic data order, so ONE extraction serves every
#     arm; the checkpoint arg is used only for its dataset config + loader state
#     (meta128_vanilla_100b/step4768 = the 20B fixed checkpoint). 16 file-sharded local workers
#     (pure S3 byte-range I/O; no GPUs / Beaker needed), then a manifest merge.
#
#   bash scripts/meta_learning/eval_scripts/run_extract_20b_40b.sh
#
# Output: meta_learning/data/meta128_20B-40B/
#   docs-{000..015}.jsonl.gz + manifest-*.json + manifest.json (merged) + logs/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

CHECKPOINT=meta_learning/meta128_vanilla/step4768
OUT=meta_learning/data/meta128_20B-40B
NUM_SHARDS=16
LOGS="$OUT/logs"
mkdir -p "$LOGS"

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    PYTHONPATH=src python scripts/modular_extension/extract_training_doc_window.py \
        --checkpoint "$CHECKPOINT" \
        --start-tokens 20e9 --end-tokens 40e9 \
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
