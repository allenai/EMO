#!/usr/bin/env python3
"""Merge rank shards of extract_routing.py (<cond>/rank<k>/) into <cond>/ by summing counts and
concatenating per-document arrays (docs are disjoint across ranks; document ids are re-based).

Usage: python scripts/sparse_experts/olmoe3_routing/merge_routing.py <cond_dir> [--keep-ranks]
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond", type=Path)
    ap.add_argument("--keep-ranks", action="store_true")
    a = ap.parse_args()
    ranks = sorted(p for p in a.cond.glob("rank*") if (p / "DONE").exists())
    metas = [json.load(open(r / "meta.json")) for r in ranks]
    assert len(ranks) == metas[0]["world"], f"{len(ranks)} DONE shards but world={metas[0]['world']}"
    usage = coact_tok = coact_doc = cross = None; n_tok = 0
    du, ds, st, ce, cl, dl, di, sel, extra = [], [], [], [], [], [], [], [], {}
    for r in ranks:
        c = np.load(r / "counts.npz")
        usage = c["usage"] if usage is None else usage + c["usage"]
        coact_tok = c["coact_tok"] if coact_tok is None else coact_tok + c["coact_tok"]
        coact_doc = c["coact_doc"] if coact_doc is None else coact_doc + c["coact_doc"]
        n_tok += int(c["n_tok"])
        x = np.load(r / "cross.npy"); cross = x if cross is None else cross + x
        du.append(np.load(r / "doc_usage.npy")); ds.append(np.load(r / "doc_scores.npy"))
        s = np.load(r / "doc_stats.npz"); st.append(s["stats_sum"]); ce.append(s["ce_sum"]); cl.append(s["ce_len"]); dl.append(s["doc_len"])
        d = np.load(r / "docs.npz", allow_pickle=True); off = sum(len(x) for x in sel)
        di.append(d["doc_inst"] + off); sel.append(d["sel"])
        for k in d.files:
            if k not in ("doc_inst", "sel"): extra.setdefault(k, []).append(d[k])
    np.savez(a.cond / "counts.npz", usage=usage, coact_tok=coact_tok, coact_doc=coact_doc, n_tok=n_tok)
    np.save(a.cond / "cross.npy", cross)
    np.save(a.cond / "doc_usage.npy", np.concatenate(du)); np.save(a.cond / "doc_scores.npy", np.concatenate(ds))
    np.savez(a.cond / "doc_stats.npz", stats_sum=np.concatenate(st), ce_sum=np.concatenate(ce), ce_len=np.concatenate(cl), doc_len=np.concatenate(dl))
    np.savez(a.cond / "docs.npz", doc_inst=np.concatenate(di), sel=np.concatenate(sel), **{k: np.concatenate(v) for k, v in extra.items()})
    raws = [r / "raw_topk.npy" for r in ranks if (r / "raw_topk.npy").exists()]
    if raws:
        arrs = [np.load(p, mmap_mode="r") for p in raws]
        out = np.lib.format.open_memmap(a.cond / "raw_topk.npy", mode="w+", dtype=np.int16, shape=(sum(len(x) for x in arrs),) + arrs[0].shape[1:])
        p = 0
        for x in arrs: out[p:p+len(x)] = x; p += len(x)
        out.flush()
    m = dict(metas[0]); m.update(n_instances=sum(x["n_instances"] for x in metas), n_tokens=n_tok, n_docs=int(sum(len(x) for x in dl)),
                                 mean_ce=float(np.concatenate(ce).sum() / np.concatenate(cl).sum()), ranks=len(ranks), raw_instances=sum(x["raw_instances"] for x in metas))
    m.pop("rank", None); json.dump(m, open(a.cond / "meta.json", "w"), indent=1)
    (a.cond / "DONE").touch()
    if not a.keep_ranks:
        for r in ranks: shutil.rmtree(r)
    print(f"merged {len(ranks)} ranks: {m['n_instances']} instances, {n_tok:,} tokens, {m['n_docs']} docs, CE {m['mean_ce']:.4f}")


if __name__ == "__main__":
    main()
