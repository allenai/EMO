#!/usr/bin/env python3
"""Routing drift from the 10B start model on the 300B-token held-out sample (user request 2026-09-24): for the merged 4 squares, the merged
8 squares and the jointly trained baseline, KL(P_start || Q_model) of the per-document routing distribution (selection counts over experts,
per layer 2-9; Q smoothed with half a count per expert; mean over documents with >= 64 tokens) at every matched checkpoint 10B -> 130B.
Input: the held-out passes' doc_usage.npy. The start model's routing is the k=4 merge at 10B (the untrained slices merge back to exactly
the start model; the start pass's own arrays were pruned). A model whose arrays are missing at a point gets null there.
  python scripts/sparse_experts/olmoe3_routing/routing_drift.py [--control std] -> claude_outputs/olmoe3_routing/<control sqn>/routing_drift.json
"""
import argparse, json
from pathlib import Path
import numpy as np

CONTROLS = {"std": dict(sqn="olmoe3_squares_stdrand", hr4="runs_heldout300b_stdrand", hr8="runs_heldout300b_stdrand8", hrb="runs_heldout300b_std"),
            "emo": dict(sqn="olmoe3_squares_emorand", hr4="runs_heldout300b_emorand", hr8="runs_heldout300b_emorand8", hrb="runs_heldout300b_emo")}
STEPS = [19074, 20000, 25000, 30000, 35000, 38148, 39073, 44073, 49073, 54073, 57221, 66481, 116479, 166478, 216477, 247956]
LAYERS = list(range(2, 10))


def usage_path(d):
    """doc_usage.npy of a held-out pass: consolidated (none/), the worker's own folder (none/rank0/) or a re-extraction (none/rerun/)."""
    for sub in ("", "rank0", "rerun"):
        p = d / sub / "doc_usage.npy"
        if p.exists(): return p
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--control", default="std"); ap.add_argument("--min-tokens", type=int, default=64); a = ap.parse_args()
    C = CONTROLS[a.control]; R = Path("sparse_experts/olmoe3_routing")
    start = usage_path(R / C["hr4"] / "merged_match19074" / "none"); assert start, "start-model arrays missing"
    us = np.load(start, mmap_mode="r"); E = us.shape[-1]
    dl = np.load(start.parent / "doc_stats.npz")["doc_len"]; keep = np.flatnonzero(dl >= a.min_tokens)
    P = {l: None for l in LAYERS}
    for l in LAYERS:
        s = np.asarray(us[keep, l - 1, :]).astype(np.float64); P[l] = s / np.maximum(s.sum(1, keepdims=True), 1)
    out = dict(control=a.control, layers=LAYERS, min_tokens=a.min_tokens, n_docs=int(len(keep)), models=["m4", "m8", "baseline"], points=[])
    for s in STEPS:
        rec = dict(step=s, tokens_b=s * 524288 / 1e9, kl={})
        for key, d in (("m4", R / C["hr4"] / f"merged_match{s}" / "none"), ("m8", R / C["hr8"] / f"merged_match{s}" / "none"),
                       ("baseline", R / C["hrb"] / (f"baseline_step{s}" if s != 19074 else "merged_match19074") / "none")):
            if key == "baseline" and s == 19074: d = R / C["hr4"] / "merged_match19074" / "none"   # the baseline at 10B is the start model itself
            p = usage_path(d)
            if not p: rec["kl"][key] = None; continue
            um = np.load(p, mmap_mode="r"); per = {}
            for l in LAYERS:
                m = np.asarray(um[keep, l - 1, :]).astype(np.float64); q = (m + 0.5) / (m.sum(1, keepdims=True) + 0.5 * E); pb = P[l]
                with np.errstate(divide="ignore", invalid="ignore"): kl = np.where(pb > 0, pb * np.log(pb / q), 0).sum(1)
                per[str(l)] = float(kl.mean())
            per["avg"] = float(np.mean([per[str(l)] for l in LAYERS])); rec["kl"][key] = per
        out["points"].append(rec)
        print(f"step {s:6d} {rec['tokens_b']:6.1f}B | " + " ".join(f"{k} {'-' if v is None else f'{v['avg']:.3f}'}" for k, v in rec["kl"].items()), flush=True)
    od = Path("claude_outputs/olmoe3_routing") / C["sqn"]; od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "routing_drift.json", "w"), indent=1); print("wrote", od / "routing_drift.json")


if __name__ == "__main__":
    main()
