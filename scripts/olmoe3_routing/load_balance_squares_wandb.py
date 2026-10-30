#!/usr/bin/env python3
"""In-loop load-balancing loss of the random-control SQUARES at sub-model LR 4e-4 (four and eight squares of EMO 512e / 1000e / 2000e), expressed as
the loss of the COMBINED (merged) model (box 5H of Q5, user request 2026-10-30).
Each square trains its own E/K experts on its own documents, so in the merged model a token only ever routes inside its square's experts. The merged
model's loss (E / k) * sum_e f_e P_e over all E experts and all K squares' tokens then decomposes exactly: with every square consuming the same batch per
step (token share 1/K each), f_e and P_e of an expert are 1/K times its values inside its square, and the merged loss is the plain MEAN of the K
squares' own losses at the same step. Steps map to the joint run's token axis as joint_step = window_start + K * square_step (window 3 runs carry the
joint counter already: joint_step = 57221 + K * (step - run_start)).
W&B logs weight x loss (weight 0.01); the stored values are divided by the weight (1 = perfect balance). Mean over the nine MoE blocks.
  python scripts/olmoe3_routing/load_balance_squares_wandb.py --out claude_outputs/olmoe3_routing/load_balance_squares_wandb.json
"""
import argparse, json, re
from pathlib import Path
import numpy as np, wandb

ARMS = {  # model -> K -> (run prefix, windows)
    "EMO 512e": {4: ("olmoe3_275m_emorand_lr4e-4_square", 3), 8: ("olmoe3_275m_emorand8_lr4e-4_square", 2)},
    "EMO 1000e": {4: ("olmoe3_275m_emo1000rand4_square", 3), 8: ("olmoe3_275m_emo1000rand8_square", 2)},
    "EMO 2000e": {4: ("olmoe3_275m_emo2000rand4_square", 3), 8: ("olmoe3_275m_emo2000rand8_square", 2)}}
WSTART = {1: 19074, 2: 38148, 3: 57221}
BIN = 1000  # joint steps per bin
WEIGHT = 0.01


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    api = wandb.Api(timeout=120); out = {}
    keys = ["_step"] + [f"train/block {b:02d}/load balancing loss" for b in range(1, 10)]
    for model, arms in ARMS.items():
        out[model] = {}
        for K, (rp, windows) in arms.items():
            names = [f"{rp}{g}{'' if w == 1 else f'_w{w}'}" for w in range(1, windows + 1) for g in range(K)]
            per_square = {}  # (w, g) -> {square_step: loss}
            for r in api.runs("ryanyxw/emo-extension", filters={"display_name": {"$in": names}}):
                m = re.match(rf"{re.escape(rp)}(\d+)(?:_w(\d))?$", r.name); g = int(m.group(1)); w = int(m.group(2) or 1)
                d = per_square.setdefault((w, g), {})
                for h in r.history(keys=keys, samples=2000, pandas=False):
                    vals = [h.get(k) for k in keys[1:]]; vals = [v for v in vals if v is not None]
                    if h.get("_step") is not None and vals: d[int(h["_step"])] = float(np.mean(vals)) / WEIGHT
                print(model, K, r.name, r.state, len(d), flush=True)
            # bin every square on the joint-step axis, then average the squares per bin (= the merged model's loss)
            bins = {}  # joint bin -> {g: [values]}
            for (w, g), d in per_square.items():
                run_start = min(d) if w == 3 else 0
                for s, v in d.items():
                    js = WSTART[w] + K * (s - run_start); bins.setdefault(js // BIN * BIN, {}).setdefault(g, []).append(v)
            series = [[b, float(np.mean([np.mean(v) for v in gs.values()])), len(gs)] for b, gs in sorted(bins.items()) if len(gs) == K]
            out[model][str(K)] = dict(runs=names, found=sorted(f"w{w}g{g}" for w, g in per_square), series=series)
            print(model, K, "bins with all squares:", len(series), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w")); print("wrote", a.out)


if __name__ == "__main__":
    main()
