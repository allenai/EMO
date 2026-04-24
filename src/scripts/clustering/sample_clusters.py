"""
Sample token contexts per cluster to aid manual labeling.

For each cluster, writes a text file with N sampled token contexts
(decoded +/- ctx_win tokens around the target) plus summary stats:
source breakdown, top-frequent within-cluster tokens, and top-enriched
tokens (ratio of within-cluster frequency to background frequency).

Usage:
    python -m src.scripts.clustering.sample_clusters \\
        --cluster-dir .../probs_mean_pca_l2_spherical_kmeans_k32 \\
        --n-samples 200
"""

import argparse
import gzip
import json
import logging
import os
from collections import Counter

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Per-cluster token sampling for labeling")
    parser.add_argument("--cluster-dir", required=True, help="Dir with assignments.npy, summary.json")
    parser.add_argument("--data-dir", default=None, help="Dir with metadata_tokens, documents, etc.")
    parser.add_argument("--n-samples", type=int, default=200, help="Samples per cluster (default 200)")
    parser.add_argument("--context-window", type=int, default=10, help="Context tokens before/after")
    parser.add_argument("--top-k-tokens", type=int, default=30, help="Top tokens to report")
    parser.add_argument("--min-token-count", type=int, default=3,
                        help="Min within-cluster count for enrichment calc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cluster_dir = args.cluster_dir
    data_dir = args.data_dir or os.path.dirname(os.path.normpath(cluster_dir))

    logger.info(f"Cluster dir: {cluster_dir}")
    logger.info(f"Data dir:    {data_dir}")

    assignments = np.load(os.path.join(cluster_dir, "assignments.npy"))
    with open(os.path.join(cluster_dir, "run_info.json")) as f:
        run_info = json.load(f)
    with open(os.path.join(data_dir, "info.json")) as f:
        info = json.load(f)
    summary_path = os.path.join(cluster_dir, "summary.json")
    summary = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    summary_by_id = {c["cluster"]: c for c in summary}

    logger.info("Loading token metadata...")
    meta = []
    with gzip.open(os.path.join(data_dir, "metadata_tokens.jsonl.gz"), "rt") as f:
        for line in f:
            meta.append(json.loads(line))

    documents = np.load(os.path.join(data_dir, "documents.npy"))
    boundaries = np.load(os.path.join(data_dir, "doc_boundaries.npy"))

    from transformers import AutoTokenizer

    model_path = info.get("model_path", "")
    logger.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    k = run_info["k"]
    rng = np.random.RandomState(args.seed)
    ctx_win = args.context_window

    logger.info("Computing global token counts...")
    global_token_counts: Counter = Counter()
    for m in meta:
        global_token_counts[m["token_id"]] += 1
    n_total = len(meta)

    def get_context(token_idx: int):
        m = meta[token_idx]
        doc_idx = m["doc_index"]
        pos = m["token_position"]
        doc_start = int(boundaries[doc_idx])
        doc_end = int(boundaries[doc_idx + 1])
        doc_tokens = documents[doc_start:doc_end]
        ctx_start = max(0, pos - ctx_win)
        ctx_end = min(len(doc_tokens), pos + ctx_win + 1)
        before = (
            tokenizer.decode(doc_tokens[ctx_start:pos].tolist(), skip_special_tokens=True)
            if pos > ctx_start
            else ""
        )
        target = tokenizer.decode(doc_tokens[pos : pos + 1].tolist(), skip_special_tokens=True)
        after = (
            tokenizer.decode(doc_tokens[pos + 1 : ctx_end].tolist(), skip_special_tokens=True)
            if pos + 1 < ctx_end
            else ""
        )
        return m["source"], before, target, after

    out_dir = os.path.join(cluster_dir, "cluster_samples")
    os.makedirs(out_dir, exist_ok=True)

    def decode_compact(tid: int) -> str:
        s = tokenizer.decode([int(tid)], skip_special_tokens=False)
        return s.replace("\n", "\\n")

    for c in range(k):
        idx = np.where(assignments == c)[0]
        size = len(idx)
        if size == 0:
            logger.warning(f"cluster {c}: empty, skipping")
            continue

        n_sample = min(args.n_samples, size)
        sampled = rng.choice(idx, size=n_sample, replace=False)

        cluster_token_counts: Counter = Counter()
        for i in idx:
            cluster_token_counts[meta[i]["token_id"]] += 1

        enrichments = []
        for tid, cnt in cluster_token_counts.most_common():
            if cnt < args.min_token_count:
                continue
            p_in = cnt / size
            p_all = global_token_counts[tid] / n_total
            if p_all <= 0:
                continue
            enrichments.append((tid, cnt, p_in / p_all))
        enrichments.sort(key=lambda x: -x[2])
        top_enriched = enrichments[: args.top_k_tokens]
        top_freq = cluster_token_counts.most_common(args.top_k_tokens)

        sc = summary_by_id.get(c, {}).get("source_counts", {})
        src_total = sum(sc.values()) or 1

        out_path = os.path.join(out_dir, f"cluster_{c:02d}.txt")
        with open(out_path, "w") as f:
            f.write(f"=== Cluster {c} (size {size}) ===\n")
            f.write("Source distribution:\n")
            for src, cnt in sorted(sc.items(), key=lambda kv: -kv[1]):
                f.write(f"  {src:<30s} {cnt:>8,}  ({100 * cnt / src_total:5.1f}%)\n")
            f.write("\n")
            f.write(f"Top {args.top_k_tokens} frequent tokens (within-cluster):\n  ")
            f.write(
                ", ".join(f"{decode_compact(tid)!r}:{cnt}" for tid, cnt in top_freq)
                + "\n\n"
            )
            f.write(f"Top {args.top_k_tokens} enriched tokens (within/background ratio):\n  ")
            f.write(
                ", ".join(
                    f"{decode_compact(tid)!r}:{e:.1f}x(n={cnt})" for tid, cnt, e in top_enriched
                )
                + "\n\n"
            )
            f.write(f"=== {n_sample} Sampled contexts ===\n")
            for si in sampled:
                src, before, target, after = get_context(int(si))
                before = before.replace("\n", " ")[-100:]
                target_oneline = target.replace("\n", "\\n")
                after = after.replace("\n", " ")[:100]
                f.write(f"[{src}] ...{before}[[{target_oneline}]]{after}...\n")
        logger.info(f"cluster {c:2d}: size={size:>7,}, wrote {n_sample} samples -> {out_path}")

    logger.info(f"Done. Samples in {out_dir}/")


if __name__ == "__main__":
    main()
