"""Summarise the learned-d sweep from W&B (project emo-extension): per run the train CE averaged over the
last 100 steps, the in-loop v3-small eval CE at steps 1000 / 2000 (predicted-d routing), per-layer mean
predicted pool d at the end, the fraction of documents with d <= 64, and stability signals (skipped optimizer
steps, max grad norm, max train-CE jump between consecutive logged steps).

  python scripts/sparse_experts/learnedd/sweep_table.py [--out sparse_experts/learnedd_sweep/sweep_table.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import wandb

ENTITY_PROJECT = "ryanyxw/emo-extension"
PREFIX = "olmoe3_275m_emo_learnedd_sweep_"
MOE_LAYERS = list(range(1, 10))


def _hist(run, keys, step_key="_step"):
    rows = list(run.scan_history(keys=[step_key, *keys], page_size=2000))
    return [r for r in rows if r.get(step_key) is not None]


def summarise(run):
    out = {"run": run.name, "id": run.id, "state": run.state, "config_tags": run.tags}
    ce = _hist(run, ["train/CE loss", "optim/total grad norm", "optim/step skipped"])
    ce = sorted(ce, key=lambda r: r["_step"])
    steps = [r["_step"] for r in ce]
    out["last_step"] = steps[-1] if steps else None
    vals = [(r["_step"], r["train/CE loss"]) for r in ce if r.get("train/CE loss") is not None]
    tail = [v for s, v in vals if s > (steps[-1] - 100)] if steps else []
    out["train_ce_last100"] = sum(tail) / len(tail) if tail else None
    out["train_ce_at"] = {s: v for s, v in vals if s in (500, 1000, 1500, 2000)}
    jumps = [abs(b[1] - a[1]) for a, b in zip(vals, vals[1:])]
    out["max_ce_jump"] = max(jumps) if jumps else None
    out["nan_ce"] = any(isinstance(v, float) and math.isnan(v) for _, v in vals)
    gn = [r["optim/total grad norm"] for r in ce if r.get("optim/total grad norm") is not None]
    out["max_grad_norm"] = max(gn) if gn else None
    sk = [r["optim/step skipped"] for r in ce if r.get("optim/step skipped") is not None]
    out["skipped_steps"] = int(sum(sk)) if sk else 0
    # eval
    ev = _hist(run, ["eval/v3-small-ppl/CE loss", "eval/v3-small-ppl/PPL"]) if any(
        k.startswith("eval/") for k in run.summary.keys()) else []
    ev_keys = [k for k in run.summary.keys() if k.startswith("eval/") and k.endswith("CE loss")]
    out["eval_keys"] = ev_keys
    out["eval_ce"] = {}
    if ev_keys:
        rows = sorted(_hist(run, ev_keys), key=lambda r: r["_step"])
        for r in rows:
            out["eval_ce"][r["_step"]] = {k.split("/")[1]: r[k] for k in ev_keys if r.get(k) is not None}
    # learned-d metrics (last logged value per layer)
    dkeys = [f"train/block {l:02d}/emo d soft mean" for l in MOE_LAYERS]
    fkeys = [f"train/block {l:02d}/emo d frac<=64" for l in MOE_LAYERS]
    present = [k for k in dkeys + fkeys if k in run.summary]
    if present:
        rows = sorted(_hist(run, present), key=lambda r: r["_step"])
        last = {}
        for r in rows:
            for k in present:
                if r.get(k) is not None:
                    last[k] = r[k]
        out["d_soft_mean"] = {l: last.get(f"train/block {l:02d}/emo d soft mean") for l in MOE_LAYERS}
        out["d_frac_le64"] = {l: last.get(f"train/block {l:02d}/emo d frac<=64") for l in MOE_LAYERS}
        # d trajectory of the middle layer for a quick look
        traj = [(r["_step"], r.get("train/block 05/emo d soft mean")) for r in rows if r.get("train/block 05/emo d soft mean") is not None]
        out["d_traj_block05"] = traj[:: max(1, len(traj) // 20)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("sparse_experts/learnedd_sweep/sweep_table.json"))
    ap.add_argument("--prefix", default=PREFIX)
    args = ap.parse_args()
    api = wandb.Api(timeout=120)
    runs = [r for r in api.runs(ENTITY_PROJECT, filters={"display_name": {"$regex": f"^{args.prefix}"}})]
    table = []
    for r in sorted(runs, key=lambda r: r.name):
        s = summarise(r)
        table.append(s)
        ev = " ".join(f"eval@{k}={v.get('v3-small-ppl', list(v.values())[0] if v else float('nan')):.4f}" for k, v in sorted(s["eval_ce"].items()))
        dm = s.get("d_soft_mean") or {}
        dstr = " ".join(f"L{l}:{dm[l]:.0f}" for l in MOE_LAYERS if dm.get(l) is not None)
        print(f"{s['run'][len(args.prefix):]:28s} {s['state']:9s} step={s['last_step']} trainCE(last100)={s['train_ce_last100'] and round(s['train_ce_last100'], 4)} "
              f"{ev} skipped={s['skipped_steps']} maxgn={s['max_grad_norm'] and round(s['max_grad_norm'], 2)} maxjump={s['max_ce_jump'] and round(s['max_ce_jump'], 3)} d: {dstr}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(table, open(args.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
