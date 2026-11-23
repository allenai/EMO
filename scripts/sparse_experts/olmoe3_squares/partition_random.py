#!/usr/bin/env python3
"""Random expert partition (control for olmoe3_squares): in every partitioned layer the experts are shuffled into k groups
of equal size; layer 1 stays whole in every group, exactly like partition.py. Writes the same groups.json layout
(layer9_agreement empty, no preview).   python partition_random.py --out <groups.json> [--k 4] [--seed 0]"""
import argparse, json
from pathlib import Path
import numpy as np

LAYERS = list(range(1, 10)); PARTITIONED = list(range(2, 10)); E = 512


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--num-experts", type=int, default=E)
    ap.add_argument("--whole-layers", type=int, default=1, help="expert layers 1..N stay whole in every group (default 1); layers N+1..9 are partitioned (2026-11-23: 4 -> only layers 5-9 are split)")
    a = ap.parse_args(); PARTITIONED = list(range(a.whole_layers + 1, 10)); WHOLE = list(range(1, a.whole_layers + 1))
    rng = np.random.default_rng(a.seed); k = a.k; n = a.num_experts   # n % k != 0 (12 groups of 512, 2026-11-23): near-equal groups, the first n % k one larger
    groups = {l: [list(range(n)) for _ in range(k)] for l in WHOLE}
    for l in PARTITIONED:
        perm = rng.permutation(n); groups[l] = [sorted(x.tolist()) for x in np.array_split(perm, k)]
    out = dict(k=k, num_experts=n, partitioned_layers=PARTITIONED, untouched_layers=WHOLE, source=f"random(seed={a.seed})", size_normalized=False,
               groups={str(l): groups[l] for l in LAYERS}, sizes={str(l): [len(x) for x in groups[l]] for l in LAYERS}, layer9_agreement={}, preview=None)
    a.out.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}: {k} random groups of {n // k} experts in layers {PARTITIONED[0]}-{PARTITIONED[-1]}, layer{'s 1-' + str(WHOLE[-1]) if len(WHOLE) > 1 else ' 1'} whole")


if __name__ == "__main__":
    main()
