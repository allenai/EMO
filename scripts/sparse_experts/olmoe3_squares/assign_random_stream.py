#!/usr/bin/env python3
"""Random document assignment straight from a training-stream dump, no model pass (from-scratch squares, user request 2026-11-13):
every document (EOS-segment of an instance, the same document table as assign_docs.py / extract_routing.doc_table) of the stream gets a
uniform random group. Writes records_<shard>.npz in the layout pack_groups.py reads (inst, start, end, group, share, ce, mass, label);
ce and mass are zeros (there is no model to route with), so the pack's routing-share statistics are zeros too.
  python assign_random_stream.py --stream <stream dir> --out <dir> --k 4 [--seed 0] [--procs 16]"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

EOS = 100_257


def one(args):
    stream, out, sk, base, k, seed = args
    tokens = np.load(Path(stream) / f"shard_{sk}.tokens.npy", mmap_mode="r"); meta = np.load(Path(stream) / f"shard_{sk}.meta.npz")
    N, S = tokens.shape; rng = np.random.default_rng([seed, sk]); R = dict(inst=[], start=[], end=[], label=[])
    for b0 in range(0, N, 2048):
        tb = np.asarray(tokens[b0:b0 + 2048]); eos = tb == EOS; eos[:, 0] = False
        rows, cols = np.nonzero(eos)                                   # a document starts at column 0 and at every later EOS
        inst = np.concatenate([np.arange(len(tb)), rows]); start = np.concatenate([np.zeros(len(tb), np.int64), cols])
        o = np.lexsort((start, inst)); inst = inst[o]; start = start[o]
        end = np.empty_like(start); end[:-1] = start[1:]; last = np.r_[inst[1:] != inst[:-1], True]; end[last] = S
        R["inst"].append(base + b0 + inst); R["start"].append(start.astype(np.int32)); R["end"].append(end.astype(np.int32))
        R["label"].append(meta["label_id"][b0:b0 + len(tb)][inst].astype(np.int16))
    R = {key: np.concatenate(v) for key, v in R.items()}; n = len(R["inst"])
    grp = rng.integers(0, k, size=n).astype(np.int8)
    np.savez(Path(out) / f"records_{sk}.npz", **R, group=grp, share=np.full(n, 1.0 / k, np.float16), ce=np.zeros(n, np.float16), mass=np.zeros((n, 8, 1), np.int8))
    return n, np.bincount(grp, minlength=k)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stream", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--procs", type=int, default=16); a = ap.parse_args()
    man = json.load(open(a.stream / "manifest.json")); a.out.mkdir(parents=True, exist_ok=True); jobs = []; off = 0
    for s in man["shards"]: jobs.append((str(a.stream), str(a.out), s["k"], off, a.k, a.seed)); off += s["n_instances"]
    with Pool(a.procs) as pool: res = pool.map(one, jobs)
    print(f"{sum(r[0] for r in res):,} documents assigned uniformly at random to {a.k} groups: {sum(r[1] for r in res).tolist()}")


if __name__ == "__main__":
    main()
