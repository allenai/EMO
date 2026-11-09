#!/usr/bin/env python3
"""Random OVERLAPPING expert partition (user request 2026-11-09): K groups of E*c/K experts per partitioned layer in which every expert
belongs to exactly c groups. Built from c independent random c-way... no: from c independent random (K/c)-way partitions of the experts,
stacked (groups 0..K/c-1 come from the first permutation, the next K/c from the second, ...), so group sizes are exact and every expert
appears once in every block of K/c groups. Layer 1 stays whole in every group, exactly like partition_random.py. Same groups.json layout
plus "copies".   python partition_overlap.py --out <groups.json> [--k 16] [--copies 2] [--seed 3]"""
import argparse, json
from pathlib import Path
import numpy as np

LAYERS = list(range(1, 10)); PARTITIONED = list(range(2, 10)); E = 512


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--copies", type=int, default=2); ap.add_argument("--seed", type=int, default=3); ap.add_argument("--num-experts", type=int, default=E); a = ap.parse_args()
    rng = np.random.default_rng(a.seed); k, c, n = a.k, a.copies, a.num_experts; assert k % c == 0 and n % (k // c) == 0
    per_block = k // c; size = n // per_block
    groups = {1: [list(range(n)) for _ in range(k)]}
    for l in PARTITIONED:
        gl = []
        for _ in range(c):
            perm = rng.permutation(n); gl += [sorted(perm[j * size:(j + 1) * size].tolist()) for j in range(per_block)]
        cnt = np.bincount(np.concatenate([np.array(g) for g in gl]), minlength=n); assert (cnt == c).all(), cnt
        groups[l] = gl
    out = dict(k=k, copies=c, num_experts=n, partitioned_layers=PARTITIONED, untouched_layers=[1], source=f"random_overlap(seed={a.seed}, copies={c})", size_normalized=False,
               groups={str(l): groups[l] for l in LAYERS}, sizes={str(l): [len(x) for x in groups[l]] for l in LAYERS}, layer9_agreement={}, preview=None)
    a.out.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"))
    ov = [len(set(groups[2][i]) & set(groups[2][j])) for i in range(per_block) for j in range(per_block, k)]
    print(f"wrote {a.out}: {k} random groups of {size} experts in layers {PARTITIONED[0]}-{PARTITIONED[-1]}, every expert in {c} groups, layer 1 whole; "
          f"layer-2 overlap between a first-block and a second-block group: min {min(ov)} / mean {np.mean(ov):.1f} / max {max(ov)} experts")


if __name__ == "__main__":
    main()
