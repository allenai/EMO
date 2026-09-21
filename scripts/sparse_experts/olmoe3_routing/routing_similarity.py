#!/usr/bin/env python3
"""Routing of the merged random-split squares vs the continued baseline on the 300B-token held-out sample, across the matched
checkpoints 10B -> 130B (user request 2026-09-21). Inputs: the held-out passes' doc_usage.npy (per-document expert counts per layer,
all documents) and raw_topk.npy (token-level top-16 for the first 200 instances). Per matched point:
  overlap    experts used by the merged model that the baseline also uses, per document and layer: absolute count and recall
             (fraction of the baseline's expert set), mean over documents
  kl         KL(P_baseline || Q_merged) of the per-document routing distributions (selection counts over experts, per layer);
             Q smoothed with half a count per expert; mean over documents
  coverage   document-level share of the document's routing that falls on its most-used square (max over squares), merged and baseline
  token      token-level: fraction of a document's tokens whose 16 experts come exclusively (16/16), >= 90% (>= 15/16) or >= 80% (>= 13/16)
             from the document's dominant square; merged and baseline; first 200 instances only
Layers 2-9 (layer 1 is not partitioned); documents with >= 64 tokens. The 10B point is the start model on both sides (identical).
  python scripts/sparse_experts/olmoe3_routing/routing_similarity.py [--control std] -> claude_outputs/olmoe3_routing/<control sqn>/routing_similarity.json
"""
import argparse, json
from pathlib import Path
import numpy as np

CONTROLS = {"std": dict(sqn="olmoe3_squares_stdrand", hr="runs_heldout300b_stdrand", hrb="runs_heldout300b_std", start="std_step19074", sample="sample_8k_300b.npz"),
            "emo": dict(sqn="olmoe3_squares_emorand", hr="runs_heldout300b_emorand", hrb="runs_heldout300b_emo", start="emo_step19074", sample="sample_8k_300b.npz")}
STEPS = [20000, 25000, 30000, 35000, 38148, 39073, 44073, 49073, 54073, 57221, 66481, 116479, 166478, 216477, 247956]
EOS = 100_257; LAYERS = list(range(2, 10)); K_TOP = 16


def doc_spans(tokens):
    eos = (tokens == EOS).astype(np.int64); eos[:, 0] = 0; seg = eos.cumsum(1); spans = []
    for i in range(len(tokens)):
        b = np.flatnonzero(np.diff(seg[i]) != 0) + 1; starts = np.concatenate([[0], b]); ends = np.concatenate([b, [tokens.shape[1]]])
        spans += [(i, int(a), int(z)) for a, z in zip(starts, ends)]
    return spans


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--control", default="std"); ap.add_argument("--min-tokens", type=int, default=64); a = ap.parse_args()
    C = CONTROLS[a.control]; R = Path("sparse_experts/olmoe3_routing"); G = json.load(open(Path("sparse_experts") / C["sqn"] / "groups.json")); k = G["k"]; E = G["num_experts"]
    grp_of = {l: np.full(E, -1, np.int64) for l in LAYERS}
    for l in LAYERS:
        for g in range(k): grp_of[l][G["groups"][str(l)][g]] = g
    onehot = {l: np.eye(k)[grp_of[l]] for l in LAYERS}  # (E, k)
    sample = np.load(R / C["sample"]); spans = doc_spans(sample["tokens"])
    out = dict(control=a.control, k=k, layers=LAYERS, min_tokens=a.min_tokens, points=[])
    points = [("start", R / C["hrb"] / C["start"] / "none", R / C["hrb"] / C["start"] / "none", 19074)] + \
             [(f"match{s}", R / C["hrb"] / f"baseline_step{s}" / "none", R / C["hr"] / f"merged_match{s}" / "none", s) for s in STEPS]
    for name, db, dm, step in points:
        if not (db / "doc_usage.npy").exists() or not (dm / "doc_usage.npy").exists(): continue
        ub = np.load(db / "doc_usage.npy", mmap_mode="r"); um = np.load(dm / "doc_usage.npy", mmap_mode="r")
        dl = np.load(db / "doc_stats.npz")["doc_len"]; keep = np.flatnonzero(dl >= a.min_tokens)
        rec = dict(name=name, step=step, tokens_b=step * 524288 / 1e9, n_docs=int(len(keep)), per_layer={})
        for l in LAYERS:
            b = np.asarray(ub[keep, l - 1, :]).astype(np.float64); m = np.asarray(um[keep, l - 1, :]).astype(np.float64)
            bs, ms = b > 0, m > 0; nb = bs.sum(1); inter = (bs & ms).sum(1)
            pb = b / np.maximum(b.sum(1, keepdims=True), 1); qm = (m + 0.5) / (m.sum(1, keepdims=True) + 0.5 * E)
            with np.errstate(divide="ignore", invalid="ignore"): kl = np.where(pb > 0, pb * np.log(pb / qm), 0).sum(1)
            covm = (m @ onehot[l]).max(1) / np.maximum(m.sum(1), 1); covb = (b @ onehot[l]).max(1) / np.maximum(b.sum(1), 1)
            # mass-based expert sets: the baseline's top-64 experts by routing mass, and its smallest set covering 90% of its routing
            ob = np.argsort(-b, 1); om = np.argsort(-m, 1); top64 = 64
            tb = np.zeros_like(bs); np.put_along_axis(tb, ob[:, :top64], True, 1); tm = np.zeros_like(ms); np.put_along_axis(tm, om[:, :top64], True, 1)
            rec64 = ((tb & tm).sum(1) / top64)
            csum = np.take_along_axis(pb, ob, 1).cumsum(1); n90 = (csum < 0.9).sum(1) + 1  # size of the 90%-mass set
            b90 = np.zeros_like(bs); m90 = np.zeros_like(ms)
            for i, n in enumerate(n90): b90[i, ob[i, :n]] = True; m90[i, om[i, :n]] = True
            rec90 = (b90 & m90).sum(1) / np.maximum(n90, 1)
            rec["per_layer"][str(l)] = dict(overlap_abs=float(inter.mean()), recall=float((inter / np.maximum(nb, 1)).mean()), n_experts_baseline=float(nb.mean()),
                                            n_experts_merged=float(ms.sum(1).mean()), kl=float(kl.mean()), coverage_merged=float(covm.mean()), coverage_baseline=float(covb.mean()),
                                            recall_top64=float(rec64.mean()), overlap_top64=float((rec64 * top64).mean()), recall_mass90=float(rec90.mean()), n_mass90=float(n90.mean()), overlap_mass90=float((rec90 * n90).mean()))
        # token level on the raw subset (first 200 instances)
        rb = np.load(db / "raw_topk.npy", mmap_mode="r"); rm = np.load(dm / "raw_topk.npy", mmap_mode="r"); n_raw = rb.shape[0]
        tok = {"merged": {}, "baseline": {}}
        for l in LAYERS:
            gb = grp_of[l][np.asarray(rb[:, :, l - 1, :])]; gm = grp_of[l][np.asarray(rm[:, :, l - 1, :])]  # (n_raw, S, 16) square ids
            for key, gg in (("baseline", gb), ("merged", gm)):
                ex = ge90 = ge80 = share = cnt = 0
                for (i, s0, s1) in spans:
                    if i >= n_raw: break
                    if s1 - s0 < a.min_tokens: continue
                    seg = gg[i, s0:s1]  # (T, 16)
                    per_sq = np.stack([(seg == g).sum(1) for g in range(k)], 1)  # (T, k)
                    dom = per_sq.sum(0).argmax(); c = per_sq[:, dom]
                    ex += (c == K_TOP).mean(); ge90 += (c >= 15).mean(); ge80 += (c >= 13).mean(); share += (c / K_TOP).mean(); cnt += 1
                tok[key][str(l)] = dict(exclusive=ex / max(cnt, 1), ge90=ge90 / max(cnt, 1), ge80=ge80 / max(cnt, 1), share=share / max(cnt, 1), n_docs=cnt)
        rec["token"] = tok
        avg = lambda key, side=None: float(np.mean([rec["per_layer"][str(l)][key] for l in LAYERS]))
        rec["avg"] = {key: avg(key) for key in ("overlap_abs", "recall", "n_experts_baseline", "n_experts_merged", "kl", "coverage_merged", "coverage_baseline", "recall_top64", "overlap_top64", "recall_mass90", "n_mass90", "overlap_mass90")}
        rec["avg"].update({f"token_{side}_{key}": float(np.mean([tok[side][str(l)][key] for l in LAYERS])) for side in ("merged", "baseline") for key in ("exclusive", "ge90", "ge80", "share")})
        out["points"].append(rec)
        print(f"{name:12s} {rec['tokens_b']:6.1f}B docs {len(keep):5d} | overlap {rec['avg']['overlap_abs']:6.1f} of {rec['avg']['n_experts_baseline']:6.1f} (recall {rec['avg']['recall']:.3f}) KL {rec['avg']['kl']:.3f} top64 {rec['avg']['recall_top64']:.3f} mass90 {rec['avg']['recall_mass90']:.3f} ({rec['avg']['n_mass90']:.0f}) | "
              f"cov merged {rec['avg']['coverage_merged']:.3f} base {rec['avg']['coverage_baseline']:.3f} | tok excl/ge90/ge80 merged {rec['avg']['token_merged_exclusive']:.3f}/{rec['avg']['token_merged_ge90']:.3f}/{rec['avg']['token_merged_ge80']:.3f} share {rec['avg']['token_merged_share']:.3f} base {rec['avg']['token_baseline_share']:.3f}", flush=True)
    od = Path("claude_outputs/olmoe3_routing") / C["sqn"]; od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "routing_similarity.json", "w"), indent=1); print("wrote", od / "routing_similarity.json")


if __name__ == "__main__":
    main()
