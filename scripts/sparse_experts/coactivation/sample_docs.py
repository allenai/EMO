#!/usr/bin/env python3
"""
Sample a stratified, held-out document set for the expert co-activation analysis.

Source: the 20B-40B training-stream doc window `meta_learning/data/meta128_20B-40B/`
(18.3M whole docs; same seed-0 data order as `sparse_8of512_10b`, which trained on 10B tokens,
so every doc here is unseen by that model). Selection is done on the compact cluster-label
file (one line per doc: source_path, doc_start_offset, doc_len, cluster), then the chosen
docs' token_ids are pulled in ONE parallel pass over the 16 `docs-*.jsonl.gz` shards, matching
on the (source_path, doc_start_offset) key from the line prefix (no JSON parse for misses).

Output (`--out-dir`, on weka so Beaker workers can read it):
  docs.jsonl.gz    one line per doc, seeded-shuffled order: {doc_id, source, token_ids[:max_tokens]}
  doc_meta.jsonl   doc_id, source, source_path, doc_start_offset, doc_len, kept_len, cluster_k32_vanilla
  summary.json     quotas, counts, token totals per source

Usage:
  PYTHONPATH=src python scripts/sparse_experts/coactivation/sample_docs.py \
      --out-dir sparse_experts/coactivation/docs_40k
"""

import argparse
import gzip
import json
import random
import re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_WINDOW = REPO / "meta_learning/data/meta128_20B-40B"
DEFAULT_QUOTAS = {
    "dclm": 24000,
    "starcoder": 4000,
    "pes2o": 4000,
    "proof-pile-2": 4000,
    "olmo-mix": 4000,
}
KEY_RE = re.compile(r'^\{"source_path": "([^"]+)", "doc_start_offset": (\d+),')


def source_of(path: str) -> str:
    return path.split("/preprocessed/")[1].split("/")[0]


def select(window: Path, quotas: dict, seed: int):
    """Seeded stratified sample from the cluster-label file. Returns {key: meta}."""
    per_source = defaultdict(list)
    t0 = time.time()
    with gzip.open(window / "doc_clusters_k32_vanilla.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            per_source[source_of(d["source_path"])].append(
                (d["source_path"], d["doc_start_offset"], d["doc_len"], d["cluster"])
            )
    print(
        f"read cluster file in {time.time() - t0:.0f}s: "
        + ", ".join(f"{s}={len(v)}" for s, v in per_source.items()),
        flush=True,
    )
    rng = random.Random(seed)
    chosen = {}
    for src, n in quotas.items():
        pool = per_source[src]
        assert len(pool) >= n, (src, len(pool), n)
        for sp, off, ln, cl in rng.sample(pool, n):
            chosen[(sp, off)] = dict(
                source=src, source_path=sp, doc_start_offset=off, doc_len=ln, cluster_k32_vanilla=cl
            )
    return chosen


def scan_shard(args):
    shard_path, keys, max_tokens = args
    hits = {}
    n = 0
    with gzip.open(shard_path, "rt") as f:
        for line in f:
            n += 1
            m = KEY_RE.match(line)
            if m is None:
                raise RuntimeError(f"unexpected line prefix in {shard_path}: {line[:80]}")
            key = (m.group(1), int(m.group(2)))
            if key in keys:
                d = json.loads(line)
                assert d["doc_len"] == len(d["token_ids"]), key
                hits[key] = d["token_ids"][:max_tokens]
    return str(shard_path), n, hits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window", default=str(DEFAULT_WINDOW))
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--max-tokens", type=int, default=4096, help="truncate each doc to this many tokens"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quotas", default=json.dumps(DEFAULT_QUOTAS), help="JSON {source: n_docs}")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    window = Path(args.window)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    quotas = json.loads(args.quotas)

    chosen = select(window, quotas, args.seed)
    print(f"selected {len(chosen)} docs", flush=True)

    shards = sorted(window.glob("docs-*.jsonl.gz"))
    keys = set(chosen)
    t0 = time.time()
    with Pool(min(args.workers, len(shards))) as pool:
        results = pool.map(scan_shard, [(s, keys, args.max_tokens) for s in shards])
    found = {}
    for sp, n, hits in results:
        found.update(hits)
        print(f"  {Path(sp).name}: scanned {n} lines, {len(hits)} hits", flush=True)
    print(f"scan done in {time.time() - t0:.0f}s; found {len(found)}/{len(chosen)}", flush=True)
    missing = keys - set(found)
    assert (
        not missing
    ), f"{len(missing)} selected docs not found in shards, e.g. {list(missing)[:3]}"

    order = sorted(chosen)  # deterministic base order
    random.Random(args.seed + 1).shuffle(order)
    tok_per_source = Counter()
    docs_per_source = Counter()
    with gzip.open(out / "docs.jsonl.gz", "wt") as fd, open(out / "doc_meta.jsonl", "w") as fm:
        for doc_id, key in enumerate(order):
            meta = dict(doc_id=doc_id, **chosen[key], kept_len=len(found[key]))
            fm.write(json.dumps(meta) + "\n")
            fd.write(
                json.dumps(dict(doc_id=doc_id, source=meta["source"], token_ids=found[key])) + "\n"
            )
            tok_per_source[meta["source"]] += meta["kept_len"]
            docs_per_source[meta["source"]] += 1
    summary = dict(
        window=str(window),
        seed=args.seed,
        max_tokens=args.max_tokens,
        quotas=quotas,
        num_docs=len(order),
        docs_per_source=dict(docs_per_source),
        kept_tokens_per_source=dict(tok_per_source),
        kept_tokens_total=sum(tok_per_source.values()),
    )
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
