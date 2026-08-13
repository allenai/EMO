"""
Build per-cluster training token shards + held-out sets for the k=32 CPT experiment.

Joins the frozen k=32 partition (doc_clusters_k32.jsonl.gz) with the extraction JSONLs
(token_ids per doc) and writes, per cluster:

  <out>/cluster{c:02d}/train/part-w{W}s{S}.npy    uint32 token streams, one EOS (100257)
                                                  appended after every document
  <out>/cluster{c:02d}/heldout.npy                held-out docs (never trained on)
  <out>/cluster{c:02d}/manifest.json              doc/token counts per role

Selection (phase 1, from the partition file alone, fully deterministic):
  - rows are deduped by (source_path, doc_start_offset), first occurrence wins; each key
    remembers which window (0=100B-110B, 1=110B-130B) provides its tokens, so boundary
    docs that appear in both windows are written exactly once (phase-2 workers only emit
    docs whose assigned window matches their shard's window -- no cross-worker sync).
  - held-out: hash-based (blake2b of the key) fraction of docs, capped at HELDOUT_CAP
    tokens per cluster;
  - train: seeded shuffle of the remaining docs, taken until the cluster's proportional
    share of the 30B budget (budget_c = 30e9 * cluster_tokens_c / total_tokens) is reached
    (the first doc to cross the budget is included, so budgets are met within one doc).

Phase 2 parallelizes over the 32 extraction shard files (16 per window).

Run:
    PYTHONPATH=.:src python -m src.scripts.clustering.build_cluster_token_data \\
        --out modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-130B/k32_cpt_tokens
"""

import argparse
import glob
import gzip
import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, "modular_extension/data")
PART = os.path.join(DATA, "emo_64exp_50b_wsd_lr2e-3_100B-130B/doc_clusters_k32.jsonl.gz")
WINDOW_DIRS = [
    os.path.join(DATA, "emo_64exp_50b_wsd_lr2e-3_100B-110B"),
    os.path.join(DATA, "emo_64exp_50b_wsd_lr2e-3_110B-130B"),
]
EOS = 100257
K = 32
TOTAL_BUDGET = 30_000_000_000
HELDOUT_CAP = 50_000_000  # tokens per cluster
HELDOUT_FRAC_HASH = 0.02  # candidate fraction by key hash (then capped)
SEED = 1234


def key_of(source_path: str, off: int) -> str:
    return f"{source_path}|{off}"


def is_heldout_candidate(key: str) -> bool:
    h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    return (h % 10_000) < int(HELDOUT_FRAC_HASH * 10_000)


def phase1(out_dir: str) -> dict:
    """key -> (cluster, window, role) with role in {train, heldout}; skipped keys absent."""
    win0_docs = json.load(open(os.path.join(WINDOW_DIRS[0], "manifest.json")))["stats"]["docs"]
    rows_by_cluster: list = [[] for _ in range(K)]
    seen: dict = {}
    n_dup = 0
    with gzip.open(PART, "rt") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            key = key_of(d["source_path"], d["doc_start_offset"])
            if key in seen:
                n_dup += 1
                continue
            window = 0 if i < win0_docs else 1
            seen[key] = True
            rows_by_cluster[d["cluster"]].append((key, window, d["doc_len"]))
    logger.info(f"phase1: {sum(map(len, rows_by_cluster)):,} unique docs ({n_dup:,} boundary dups)")

    cluster_tokens = [sum(r[2] for r in rows) for rows in rows_by_cluster]
    total = sum(cluster_tokens)
    roles: dict = {}
    manifests = []
    rng = np.random.default_rng(SEED)
    for c in range(K):
        rows = rows_by_cluster[c]
        budget = int(TOTAL_BUDGET * cluster_tokens[c] / total)
        held_tok = 0
        train_pool = []
        for key, window, dl in rows:
            if is_heldout_candidate(key) and held_tok < HELDOUT_CAP:
                roles[key] = (c, window, "heldout")
                held_tok += dl
            else:
                train_pool.append((key, window, dl))
        order = rng.permutation(len(train_pool))
        train_tok = 0
        n_train = 0
        for j in order:
            if train_tok >= budget:
                break
            key, window, dl = train_pool[j]
            roles[key] = (c, window, "train")
            train_tok += dl
            n_train += 1
        manifests.append({
            "cluster": c, "docs_total": len(rows), "tokens_total": cluster_tokens[c],
            "budget": budget, "train_docs": n_train, "train_tokens": train_tok,
            "heldout_tokens": held_tok,
        })
        logger.info(f"  cluster {c:2d}: budget {budget / 1e9:.2f}B -> train {train_tok / 1e9:.2f}B "
                    f"({n_train:,} docs), heldout {held_tok / 1e6:.0f}M")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "selection_manifest.json"), "w") as f:
        json.dump({"seed": SEED, "total_budget": TOTAL_BUDGET, "clusters": manifests}, f, indent=2)
    return roles


def phase2_shard(args):
    """Stream one extraction shard, appending selected docs' tokens (+EOS) per cluster/role."""
    shard_path, window, out_dir, roles_path = args
    import pickle

    with open(roles_path, "rb") as f:
        roles = pickle.load(f)
    tag = os.path.basename(shard_path).replace("docs-", "").replace(".jsonl.gz", "")
    bufs: dict = {}
    counts: dict = {}
    with gzip.open(shard_path, "rt") as f:
        for line in f:
            d = json.loads(line)
            r = roles.get(key_of(d["source_path"], d["doc_start_offset"]))
            if r is None or r[1] != window:
                continue
            c, _, role = r
            bufs.setdefault((c, role), []).extend(d["token_ids"] + [EOS])
            k = (c, role)
            counts[k] = counts.get(k, 0) + 1
            # flush large buffers to bound memory
            if len(bufs[k]) > 50_000_000:
                _flush(out_dir, c, role, window, tag, bufs.pop(k))
    for (c, role), buf in bufs.items():
        _flush(out_dir, c, role, window, tag, buf)
    return {f"{c}/{role}": n for (c, role), n in counts.items()}


def _flush(out_dir, c, role, window, tag, tokens):
    sub = "train" if role == "train" else "heldout_parts"
    d = os.path.join(out_dir, f"cluster{c:02d}", sub)
    os.makedirs(d, exist_ok=True)
    n_existing = len(glob.glob(os.path.join(d, f"part-w{window}s{tag}-*.npy")))
    # RAW binary (no npy header), matching the ai2-llm token files the loader memmaps.
    np.asarray(tokens, dtype=np.uint32).tofile(
        os.path.join(d, f"part-w{window}s{tag}-{n_existing:02d}.npy"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    roles = phase1(args.out)
    import pickle

    roles_path = os.path.join(args.out, "roles.pkl")
    with open(roles_path, "wb") as f:
        pickle.dump(roles, f)
    logger.info(f"phase1 done: {len(roles):,} selected docs; roles cached")

    jobs = []
    for w, wd in enumerate(WINDOW_DIRS):
        for sp in sorted(glob.glob(os.path.join(wd, "docs-*.jsonl.gz"))):
            jobs.append((sp, w, args.out, roles_path))
    logger.info(f"phase2: {len(jobs)} shard files, {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(phase2_shard, jobs)):
            logger.info(f"  shard {i + 1}/{len(jobs)} done ({sum(res.values()):,} docs written)")

    # merge heldout parts into one file per cluster; verify totals against selection manifest
    sel = json.load(open(os.path.join(args.out, "selection_manifest.json")))["clusters"]
    for c in range(K):
        cdir = os.path.join(args.out, f"cluster{c:02d}")
        parts = sorted(glob.glob(os.path.join(cdir, "heldout_parts", "*.npy")))
        held = (np.concatenate([np.fromfile(p, dtype=np.uint32) for p in parts])
                if parts else np.array([], np.uint32))
        held.tofile(os.path.join(cdir, "heldout.npy"))
        for pth in parts:
            os.remove(pth)
        os.rmdir(os.path.join(cdir, "heldout_parts"))
        train_tok = sum(os.path.getsize(p) // 4
                        for p in glob.glob(os.path.join(cdir, "train", "*.npy")))
        expect_train = sel[c]["train_tokens"] + sel[c]["train_docs"]  # +1 EOS per doc
        expect_held = held.shape[0]
        assert train_tok == expect_train, \
            f"cluster {c}: train tokens {train_tok:,} != expected {expect_train:,}"
        with open(os.path.join(cdir, "manifest.json"), "w") as f:
            json.dump({**sel[c], "train_tokens_with_eos": train_tok,
                       "heldout_tokens_with_eos": int(expect_held)}, f, indent=2)
        logger.info(f"cluster {c:2d}: train {train_tok / 1e9:.2f}B (with EOS), "
                    f"heldout {expect_held / 1e6:.0f}M ok")
    logger.info("DONE")


if __name__ == "__main__":
    main()
