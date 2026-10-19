#!/usr/bin/env python3
"""Expert-load statistics of the jointly trained EMO baselines (512e / 1000e / 2000e) from the held-out routing passes (user question
2026-10-19: box 5H of Q5). For every model and matched checkpoint, the per-layer token counts per expert (counts.npz 'usage', free routing,
eval pool = all experts, 65.5M held-out tokens) give expert-count-free measures of balance:
  norm_entropy   H(load) / log E           (1 = perfectly uniform)
  eff_frac       exp(H) / E                (effective number of experts as a fraction of E)
  cv             std(load) / mean(load)    (0 = uniform; the same relative imbalance gives the same CV at any E)
  max_rel        max(load) / mean(load)    (the busiest expert, in multiples of its fair share)
  top10_share    share of tokens on the busiest 10% of the experts (uniform: 0.10)
  under_half     fraction of experts receiving < 1/2 of their fair share
  under_tenth    fraction of experts receiving < 1/10 of their fair share ('near-dead')
All are reported per layer (1-9) and as the mean over layers, for the joint baselines and for the merged four-square LR 4e-4 models at the same points.   python scripts/olmoe3_routing/load_balance.py --out claude_outputs/olmoe3_routing/load_balance.json
"""
import argparse, json
from pathlib import Path
import numpy as np

R = Path("sparse_experts/olmoe3_routing")
MODELS = {  # model -> (held-out dir of the baseline passes, start tag, held-out dir of the four-square LR 4e-4 merges)
    "EMO 512e": ("runs_heldout300b_emo", "emo_step19074", "runs_heldout300b_emorand_lr4e-4"), "EMO 1000e": ("runs_heldout300b_emo1000", "emo1000_step19074", "runs_heldout300b_emo1000rand4"),
    "EMO 2000e": ("runs_heldout300b_emo2000", "emo2000_step19074", "runs_heldout300b_emo2000rand4")}
STEPS = [19074, 30000, 38148, 57221, 116479]


def stats(u):
    u = u.astype(np.float64); p = u / u.sum(); E = len(u); mean = u.mean()
    nz = p[p > 0]; H = float(-(nz * np.log(nz)).sum())
    order = np.sort(u)[::-1]; k10 = max(1, int(round(0.1 * E)))
    return dict(norm_entropy=H / np.log(E), eff_frac=float(np.exp(H) / E), cv=float(u.std() / mean), max_rel=float(u.max() / mean), top10_share=float(order[:k10].sum() / u.sum()),
                under_half=float((u < 0.5 * mean).mean()), under_tenth=float((u < 0.1 * mean).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    out = {}
    def one(d):
        if not (d / "counts.npz").exists(): return None
        z = np.load(d / "counts.npz"); usage = z["usage"]; meta = json.load(open(d / "meta.json"))
        per_layer = [stats(usage[i]) for i in range(usage.shape[0])]
        return dict(E=int(meta["E"]), n_tokens=int(meta["n_tokens"]), mean_ce=meta["mean_ce"], layers=per_layer, mean={k: float(np.mean([pl[k] for pl in per_layer])) for k in per_layer[0]})
    for model, (hd, start, hm) in MODELS.items():
        out[model] = {"baseline": {}, "merged4": {}}
        for st in STEPS:
            r = one(R / hd / (start if st == 19074 else f"baseline_step{st}") / "none")
            if r: out[model]["baseline"][str(st)] = r; print(f"{model} baseline {st}: " + " ".join(f"{k}={v:.3f}" for k, v in r["mean"].items()))
            if st != 19074:
                r = one(R / hm / f"merged_match{st}" / "none")
                if r: out[model]["merged4"][str(st)] = r; print(f"{model} merged4  {st}: " + " ".join(f"{k}={v:.3f}" for k, v in r["mean"].items()))
    a.out.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
