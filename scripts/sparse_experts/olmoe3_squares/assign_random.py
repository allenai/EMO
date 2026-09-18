#!/usr/bin/env python3
"""Random document assignment (control for olmoe3_squares): copies the assignment records of a routing-based run
(assign_docs.py output, one records_<rank>.npz per rank, one row per document of the window) and replaces every
document's group by a uniform random draw, so pack_groups.py can build the four random packs from the same documents.
  python assign_random.py --src <assign dir> --out <dir> [--k 4] [--seed 0]"""
import argparse, glob
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    rng = np.random.default_rng(a.seed); a.out.mkdir(parents=True, exist_ok=True); n_tot = 0; counts = np.zeros(a.k, int)
    for f in sorted(glob.glob(str(a.src / "records_*.npz"))):
        z = dict(np.load(f)); n = len(z["inst"]); g = rng.integers(0, a.k, size=n).astype(np.int8)
        z["group"] = g; z["share"] = np.full(n, 1.0 / a.k, dtype=np.float16)
        np.savez(a.out / Path(f).name, **z); n_tot += n; counts += np.bincount(g, minlength=a.k)
    print(f"{n_tot:,} documents assigned uniformly at random to {a.k} groups: {counts.tolist()}")


if __name__ == "__main__":
    main()
