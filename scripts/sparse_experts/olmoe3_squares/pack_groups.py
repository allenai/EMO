#!/usr/bin/env python3
"""Stage 1b of olmoe3_squares: re-pack the assigned documents into one token stream per group.

Reads records_<rank>.npz from assign_docs.py, sorts documents by (instance, start) = the training
order the full model would have consumed them in, and writes for each group g a flat token file
group<g>/part-0.npy (uint32 memmap, no npy header, i.e. the dolma/OLMo-core tokenized format) made
of the documents' tokens with EOS between documents. Also writes stats.json: docs/tokens per group,
in-group routing share by group / layer / source, and the full model's CE by group.

Usage: python pack_groups.py --records <assign dir> --stream <stream dir> --out <dir> [--header]
"""
import argparse, glob, json
from pathlib import Path
import numpy as np

EOS = 100_257


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True); ap.add_argument("--stream", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--header", action="store_true", help="write real .npy files (with header) instead of raw memmaps")
    a = ap.parse_args()
    G = json.load(open(a.groups)); k = G["k"]; PL = [int(l) for l in G["partitioned_layers"]]
    labels = {v: kname for kname, v in json.load(open(a.stream / "labels.json")).items()}
    man = json.load(open(a.stream / "manifest.json")); n_per = [s["n_instances"] for s in man["shards"]]; bases = np.concatenate([[0], np.cumsum(n_per)[:-1]])
    R = {}
    for f in sorted(glob.glob(str(a.records / "records_*.npz"))):
        z = np.load(f)
        for key in z.files: R.setdefault(key, []).append(z[key])
    R = {key: np.concatenate(v) for key, v in R.items()}
    order = np.lexsort((R["start"], R["inst"])); R = {key: v[order] for key, v in R.items()}
    n = len(R["inst"]); length = (R["end"] - R["start"]).astype(np.int64); grp = R["group"].astype(int)
    print(f"{n:,} docs, {length.sum():,} tokens; groups (docs) {np.bincount(grp, minlength=k).tolist()}")
    a.out.mkdir(parents=True, exist_ok=True)
    # token files: per group, one flat stream in training order, EOS-separated
    shard_of = np.searchsorted(np.cumsum(n_per), R["inst"], side="right"); local = R["inst"] - bases[shard_of]
    tok_files = {}; cur = None
    outs = {}
    for g in range(k):
        d = a.out / f"group{g}"; d.mkdir(exist_ok=True)
        ntok = int(length[grp == g].sum() + (grp == g).sum())  # + one EOS per doc (upper bound; trimmed below)
        outs[g] = dict(path=d / "part-0.npy", buf=np.lib.format.open_memmap(d / "part-0.npy", mode="w+", dtype=np.uint32, shape=(ntok,)) if a.header else np.memmap(d / "part-0.npy", mode="w+", dtype=np.uint32, shape=(ntok,)), pos=0, docs=0)
    for i in range(n):
        sk = int(shard_of[i])
        if cur != sk:
            tok_files = np.load(a.stream / f"shard_{sk}.tokens.npy", mmap_mode="r"); cur = sk
        seg = np.asarray(tok_files[local[i], R["start"][i]:R["end"][i]]).astype(np.uint32)
        o = outs[grp[i]]
        if seg[0] != EOS and o["pos"] > 0:
            o["buf"][o["pos"]] = EOS; o["pos"] += 1
        o["buf"][o["pos"]:o["pos"] + len(seg)] = seg; o["pos"] += len(seg); o["docs"] += 1
        if i % 500000 == 0 and i: print(f"  {i:,}/{n:,} docs packed")
    sizes = {}
    for g, o in outs.items():
        o["buf"].flush(); sizes[g] = o["pos"]
        if a.header:
            arr = np.load(o["path"], mmap_mode="r")[: o["pos"]].copy(); np.save(o["path"], arr)
        else:
            with open(o["path"], "r+b") as fh: fh.truncate(o["pos"] * 4)
    # stats: what a sub-model would miss = 1 - in-group share
    mass = R["mass"].astype(np.float64)  # (n, PL, k)
    own = mass[np.arange(n), :, grp]      # (n, PL)
    share_layer = own.sum(0) / np.maximum(mass.sum(2).sum(0), 1)
    share_doc = own.sum(1) / np.maximum(mass.sum((1, 2)), 1)
    stats = dict(n_docs=int(n), n_tokens=int(length.sum()), docs_per_group=np.bincount(grp, minlength=k).tolist(), tokens_per_group=[int(sizes[g]) for g in range(k)],
                 token_share=(np.array([sizes[g] for g in range(k)]) / sum(sizes.values())).round(4).tolist(),
                 in_group_share_mean=float(share_doc.mean()), in_group_share_token_weighted=float((share_doc * length).sum() / length.sum()),
                 in_group_share_pct=[float(np.percentile(share_doc, p)) for p in (10, 25, 50, 75, 90)],
                 in_group_share_by_layer={str(l): float(share_layer[j]) for j, l in enumerate(PL)},
                 in_group_share_by_group={str(g): float(share_doc[grp == g].mean()) for g in range(k)},
                 ce_by_group={str(g): float(np.average(R["ce"][grp == g].astype(float), weights=length[grp == g])) for g in range(k)},
                 ce_all=float(np.average(R["ce"].astype(float), weights=length)),
                 by_source={})
    src = R["label"].astype(int)
    for s in np.unique(src):
        m = src == s; stats["by_source"][labels.get(int(s), str(s))] = dict(docs=int(m.sum()), tokens=int(length[m].sum()), groups=np.bincount(grp[m], minlength=k).tolist(), in_group_share=float(share_doc[m].mean()))
    json.dump(stats, open(a.out / "stats.json", "w"), indent=1)
    print(json.dumps({key: stats[key] for key in ("docs_per_group", "tokens_per_group", "token_share", "in_group_share_mean", "in_group_share_by_layer", "ce_by_group")}))


if __name__ == "__main__":
    main()
