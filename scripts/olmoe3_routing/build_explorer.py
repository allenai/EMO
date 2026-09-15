#!/usr/bin/env python3
"""Data for the report's Explorer tab: sample documents of the held-out 20B-window set (sample_8k_20b.npz)
as the k=4 squares pipeline partitions them, for the EMO 512e, standard MoE 512e and learned-pool models,
plus (learned-pool model) the predicted per-document pool sizes bucketed from 16 to 512.

Per model: groups.json of its squares run + its start-model held-out pass (doc_usage.npy: per-document
expert counts per layer). A document goes to the group receiving most of its layer 2-9 selections (per
expert of the group when groups.json says size_normalized), exactly as in assign_docs / piecewise_eval.
Text is decoded with the dolma2 tokenizer (tokenizers library, cached file).

  python scripts/olmoe3_routing/build_explorer.py [--per-group 30] [--per-bucket 25]
Writes claude_outputs/olmoe3_routing/explorer.json
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "sparse_experts"
EOS = 100_257
LAYERS = list(range(1, 10))
MODELS = {
    "emo": dict(label="EMO 512e (uniform pools)", sq="olmoe3_squares", held="runs_heldout20b/emo_step19074/none"),
    "std": dict(label="Standard MoE 512e", sq="olmoe3_squares_std", held="runs_heldout20b_std/std_step19074/none"),
    "learnedd": dict(label="EMO 512e, learned pools", sq="olmoe3_squares_learnedd", held="runs_heldout20b_learnedd/learnedd_step19074/none",
                     pool="runs_heldout20b_learnedd/learnedd_step19074_docpool/none/rank0/doc_pool.npy"),
}
BUCKETS = [(16, 16, "16 (top-k only)"), (17, 32, "17–32"), (33, 64, "33–64"), (65, 128, "65–128"), (129, 256, "129–256"), (257, 512, "257–512")]


def doc_spans(tokens):
    """(doc -> (instance, start, end)) with the extractor's rule: every EOS starts the next segment."""
    eos = (tokens == EOS).astype(np.int64); eos[:, 0] = 0
    seg = eos.cumsum(1); spans = []
    for i in range(len(tokens)):
        b = np.flatnonzero(np.diff(seg[i]) != 0) + 1
        starts = np.concatenate([[0], b]); ends = np.concatenate([b, [tokens.shape[1]]])
        spans += [(i, int(a), int(z)) for a, z in zip(starts, ends)]
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=30); ap.add_argument("--per-bucket", type=int, default=25)
    ap.add_argument("--snippet", type=int, default=500); ap.add_argument("--full", type=int, default=1800)
    ap.add_argument("--out", type=Path, default=ROOT / "claude_outputs/olmoe3_routing/explorer.json")
    a = ap.parse_args()
    tok = Tokenizer.from_file(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--allenai--dolma2-tokenizer/snapshots/*/tokenizer.json"))[0])
    z = np.load(S / "olmoe3_routing/sample_8k_20b.npz"); tokens = z["tokens"]; spans = doc_spans(tokens)
    rng = np.random.default_rng(0)

    def text_of(d):
        i, s, e = spans[d]; ids = [int(t) for t in tokens[i, s:e] if t != EOS]
        t = tok.decode(ids); return t[: a.snippet], t[: a.full], len(t) > a.full

    def doc_record(d, label, family, length, share, pools=None):
        sn, full, more = text_of(d)
        r = dict(id=int(d), source=str(label), family=str(family), tokens=int(length), share=round(float(share), 3), snippet=sn, full=full, truncated=bool(more))
        if pools is not None: r["pools"] = [int(round(float(p))) for p in pools]
        return r

    out = dict(models={}, buckets=None)
    for key, m in MODELS.items():
        gpath = S / m["sq"] / "groups.json"; held = S / "olmoe3_routing" / m["held"]
        if not gpath.exists() or not (held / "doc_usage.npy").exists(): print(f"skip {key}: missing inputs"); continue
        G = json.load(open(gpath)); k = G["k"]; PL = [int(l) for l in G["partitioned_layers"]]
        du = np.load(held / "doc_usage.npy", mmap_mode="r"); st = np.load(held / "doc_stats.npz"); doc_len = st["doc_len"]
        docs = np.load(held / "docs.npz", allow_pickle=True); di = docs["doc_inst"]; label = docs["label"][di]; family = docs["group"][di]
        assert len(spans) == len(di), (len(spans), len(di))
        mass = np.zeros((du.shape[0], k))
        for l in PL:
            oh = np.zeros((G["num_experts"], k))
            for g in range(k): oh[G["groups"][str(l)][g], g] = 1
            mass += np.asarray(du[:, l - 1, :]).astype(float) @ oh
        n_exp = np.array([sum(len(G["groups"][str(l)][g]) for l in PL) for g in range(k)], float)
        grp = (mass / (n_exp if G.get("size_normalized") else 1.0)).argmax(1); share = mass[np.arange(len(grp)), grp] / np.maximum(mass.sum(1), 1)
        pools = np.load(S / "olmoe3_routing" / m["pool"]) if m.get("pool") and (S / "olmoe3_routing" / m["pool"]).exists() else None
        ok = doc_len >= 64
        groups = []
        for g in range(k):
            idx = np.flatnonzero((grp == g) & ok); fam = Counter(family[idx].tolist()); tot = max(1, len(idx))
            pick = rng.choice(idx, size=min(a.per_group, len(idx)), replace=False)
            pick = pick[np.argsort(-share[pick])]  # most typical documents first
            groups.append(dict(group=g, n_docs=int((grp == g).sum()), token_share=float(doc_len[grp == g].sum() / doc_len.sum()),
                               mean_share=float(share[idx].mean()), families={f: round(c / tot, 3) for f, c in fam.most_common()},
                               experts_per_layer={str(l): len(G["groups"][str(l)][g]) for l in PL},
                               docs=[doc_record(d, label[d], family[d], doc_len[d], share[d], pools[d] if pools is not None else None) for d in pick]))
        out["models"][key] = dict(label=m["label"], size_normalized=bool(G.get("size_normalized")), groups=groups)
        if pools is not None:
            mp = pools[:, 1:].mean(1)  # layers 2-9
            bres = []
            for lo, hi, name in BUCKETS:
                sel = (mp >= lo - 0.5) & (mp < hi + 0.5) & ok if lo > 16 else (mp < 16.5) & ok
                idx = np.flatnonzero(sel); tot = max(1, len(idx))
                pick = rng.choice(idx, size=min(a.per_bucket, len(idx)), replace=False) if len(idx) else np.array([], int)
                bres.append(dict(name=name, lo=lo, hi=hi, n_docs=int(len(idx)), frac_docs=float(len(idx) / ok.sum()), frac_tokens=float(doc_len[idx].sum() / doc_len[ok].sum()),
                                 mean_tokens=float(doc_len[idx].mean()) if len(idx) else None, median_tokens=float(np.median(doc_len[idx])) if len(idx) else None,
                                 families={f: round(c / tot, 3) for f, c in Counter(family[idx].tolist()).most_common()},
                                 docs=[doc_record(d, label[d], family[d], doc_len[d], share[d], pools[d]) for d in pick]))
            by_family = {}
            for f in sorted(set(family[ok].tolist())):
                sel = ok & (family == f)
                by_family[f] = dict(n=int(sel.sum()), mean_pool=float(mp[sel].mean()), median_pool=float(np.median(mp[sel])), frac_le16=float((mp[sel] < 16.5).mean()), frac_le64=float((mp[sel] < 64.5).mean()), mean_tokens=float(doc_len[sel].mean()))
            by_len = {}
            for lo, hi in ((64, 128), (128, 512), (512, 2048), (2048, 100000)):
                sel = ok & (doc_len >= lo) & (doc_len < hi)
                by_len[f"{lo}–{hi if hi < 100000 else 'max'}"] = dict(n=int(sel.sum()), mean_pool=float(mp[sel].mean()), frac_le16=float((mp[sel] < 16.5).mean()), frac_le64=float((mp[sel] < 64.5).mean()))
            layer_mean = pools[ok].mean(0).tolist()
            out["buckets"] = dict(model=key, buckets=bres, by_family=by_family, by_length=by_len, layer_mean=[round(v, 1) for v in layer_mean],
                                  n_docs=int(ok.sum()), note="mean predicted pool over layers 2–9 per document; documents with at least 64 tokens")
        print(f"{key}: groups {[g['n_docs'] for g in groups]}, pools {'yes' if pools is not None else 'no'}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
