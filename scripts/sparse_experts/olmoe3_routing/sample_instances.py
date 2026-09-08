#!/usr/bin/env python3
"""Stratified sample of raw 8192-token instances from an extract_stream.py output.

Groups the Dolma 3.5 mix labels (source:topic) into coarse sources and draws a seeded uniform
sample per group with quotas (defaults sum to 8000), then writes one npz in seeded-shuffled order
so any prefix (e.g. the first 1000 = pilot) is itself approximately stratified.

Output: --out npz with tokens (N, 8192) int32, step, slot, dataset_index, label (str), group (str),
        instance_mask; plus <out>.summary.json.
Usage: python scripts/sparse_experts/olmoe3_routing/sample_instances.py \
         --stream sparse_experts/olmoe3_routing/stream_10b_20b --out sparse_experts/olmoe3_routing/sample_8k.npz
"""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np

GROUP_RULES = [  # (prefix substring, group)
    ("cc_all_dressed", "web"), ("hplt-project", "web"),
    ("dolma4pdfs", "pdf"),
    ("the-stack-v2", "code"), ("swallow-code", "code"), ("sponge", "code"), ("codetextish", "code"),
    ("finemath", "math"), ("swallow-math", "math"),
]
DEFAULT_QUOTAS = {"web": 4000, "pdf": 1500, "code": 1200, "math": 800, "other": 500}


def group_of(label: str) -> str:
    for key, g in GROUP_RULES:
        if key in label:
            return g
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quotas", default=json.dumps(DEFAULT_QUOTAS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-filtered", action="store_true", default=True, help="skip instances the run's repetition filter rejected")
    args = ap.parse_args()
    quotas = json.loads(args.quotas)
    man = json.load(open(args.stream / "manifest.json"))
    labels = json.load(open(args.stream / "labels.json")); inv = {v: k for k, v in labels.items()}
    rows = []  # (shard, row, step, slot, dataset_index, label, group, mask)
    for sh in man["shards"]:
        m = np.load(args.stream / f"shard_{sh['k']}.meta.npz")
        for i in range(len(m["step"])):
            lab = inv[int(m["label_id"][i])]
            rows.append((sh["k"], i, int(m["step"][i]), int(m["slot"][i]), int(m["dataset_index"][i]), lab, group_of(lab), bool(m["instance_mask"][i])))
    print(f"{len(rows):,} instances in stream; group counts: {Counter(r[6] for r in rows)}")
    if args.drop_filtered:
        n0 = len(rows); rows = [r for r in rows if r[7]]; print(f"dropped {n0-len(rows)} filtered instances")
    rng = np.random.default_rng(args.seed)
    chosen = []
    for g, q in quotas.items():
        pool = [r for r in rows if r[6] == g]
        take = min(q, len(pool))
        pick = rng.choice(len(pool), size=take, replace=False)
        chosen += [pool[i] for i in pick]
        print(f"group {g}: {take}/{len(pool)} chosen")
    order = rng.permutation(len(chosen)); chosen = [chosen[i] for i in order]
    S = man["seq_len"]
    tokens = np.empty((len(chosen), S), np.int32)
    shards = {}
    for j, r in enumerate(chosen):
        k = r[0]
        if k not in shards:
            shards[k] = np.load(args.stream / f"shard_{k}.tokens.npy", mmap_mode="r")
        tokens[j] = shards[k][r[1]]
    np.savez(args.out, tokens=tokens, step=np.array([r[2] for r in chosen], np.int32), slot=np.array([r[3] for r in chosen], np.int16),
             dataset_index=np.array([r[4] for r in chosen], np.int64), label=np.array([r[5] for r in chosen]),
             group=np.array([r[6] for r in chosen]), instance_mask=np.array([r[7] for r in chosen]))
    summ = {"n": len(chosen), "groups": dict(Counter(r[6] for r in chosen)), "labels": dict(Counter(r[5] for r in chosen).most_common(40)),
            "steps": [min(r[2] for r in chosen), max(r[2] for r in chosen)], "seed": args.seed, "quotas": quotas,
            "stream_counts": dict(Counter(r[6] for r in rows))}
    json.dump(summ, open(str(args.out) + ".summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "labels"}, indent=1))


if __name__ == "__main__":
    main()
