#!/usr/bin/env python3
"""In-loop load-balancing metrics of the jointly trained EMO baselines (512e / 1000e / 2000e) from W&B (box 5H of Q5, 2026-10-19).
Per MoE block (1-9) the trainer logs, every 10 steps on the training batch:
  load balancing loss   (E / k) * sum_e f_e * P_e  (f_e = share of routed tokens on expert e, P_e = its mean router probability): 1 at perfect balance
                        for ANY expert count, 1 + (E^2 / k) cov_e(f, P) otherwise, i.e. a dimensionless relative-imbalance measure (comparable across E)
  load imbalance        max_e(tokens on e) / mean_e(tokens on e) in the local batch, max over ranks (1 = uniform)
  global load imbalance the same with the counts summed over all data-parallel ranks (the quantity EMO's global balancing acts on)
Each run's history is downsampled to <= 2000 rows; the mean over blocks is stored per row, and the values at the matched checkpoints are the mean over the
1,000 steps before each point.   python scripts/olmoe3_routing/load_balance_wandb.py --out claude_outputs/olmoe3_routing/load_balance_wandb.json
"""
import argparse, json
from pathlib import Path
import numpy as np, wandb

RUNS = {"EMO 512e": ["olmoe3_275m_emo_10b", "olmoe3_275m_emo_20b_1node", "olmoe3_275m_emo_30b_1node", "olmoe3_275m_emo_130b"],
        "EMO 1000e": ["olmoe3_275m_1000e_emo_10b", "olmoe3_275m_1000e_emo_30b", "olmoe3_275m_1000e_emo_61b"],
        "EMO 2000e": ["olmoe3_275m_2000e_emo_10b", "olmoe3_275m_2000e_emo_30b", "olmoe3_275m_2000e_emo_61b"]}
METRICS = {"lb_loss": "load balancing loss", "imbalance": "load imbalance", "global_imbalance": "global load imbalance"}
POINTS = [19074, 30000, 38148, 57221, 116479]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    api = wandb.Api(timeout=120); out = {}
    for model, names in RUNS.items():
        rows = {}  # step -> {metric: mean over blocks}
        for r in api.runs("ryanyxw/emo-extension", filters={"display_name": {"$in": names}}):
            keys = ["_step"] + [f"train/block {b:02d}/{m}" for b in range(1, 10) for m in METRICS.values()]
            for h in r.history(keys=keys, samples=2000, pandas=False):
                st = h.get("_step")
                if st is None: continue
                rec = {}
                for mk, mname in METRICS.items():
                    vals = [h.get(f"train/block {b:02d}/{mname}") for b in range(1, 10)]; vals = [v for v in vals if v is not None]
                    if vals: rec[mk] = float(np.mean(vals)); rec[mk + "_blocks"] = [float(v) for v in vals]
                if rec and (st not in rows or len(rec) > len(rows[st])): rows[st] = rec
            print(model, r.name, r.state, "rows so far", len(rows), flush=True)
        steps = sorted(rows)
        series = {mk: [[s, rows[s][mk]] for s in steps if mk in rows[s]] for mk in METRICS}
        at = {}
        for p in POINTS:
            win = [s for s in steps if p - 1000 <= s <= p]
            at[str(p)] = {mk: float(np.mean([rows[s][mk] for s in win if mk in rows[s]])) if any(mk in rows[s] for s in win) else None for mk in METRICS}
            at[str(p)]["blocks"] = {mk: [float(np.mean(x)) for x in zip(*[rows[s][mk + "_blocks"] for s in win if mk + "_blocks" in rows[s]])] if any(mk + "_blocks" in rows[s] for s in win) else None for mk in METRICS}
        out[model] = dict(runs=names, series=series, at=at)
        print(model, {p: {k: (round(v, 3) if isinstance(v, float) else None) for k, v in d.items() if k != "blocks"} for p, d in at.items()}, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w")); print("wrote", a.out)


if __name__ == "__main__":
    main()
