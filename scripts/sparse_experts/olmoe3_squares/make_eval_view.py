#!/usr/bin/env python3
"""Evaluation view of a checkpoint with a different activation rule in MoE layers 2-9 (sparser squares, 2026-11-06): a directory holding a
modified config.json and a relative symlink to the checkpoint's model_and_optim, so a full model (merged, joint baseline, start) can be run the
way the sparser squares were trained: top_k experts per token in layers 2-9 (layer 1 keeps its own top_k), with either the stock weights
(the kept scores normalised among themselves) or, with --norm-top-k 16, the top-16 weights of which only the top_k largest are kept
(emo_square_topk.py). No weights are copied or changed.
  python scripts/sparse_experts/olmoe3_squares/make_eval_view.py --src <checkpoint dir> --out <view dir> --top-k 4 [--norm-top-k 16]
"""
import argparse, copy, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, required=True); ap.add_argument("--norm-top-k", type=int, default=None); ap.add_argument("--layers", default="2-9"); a = ap.parse_args()
    lo, hi = (int(x) for x in a.layers.split("-")); cfg = json.load(open(a.src / "config.json")); m = cfg["model"]; ov = m.setdefault("block_overrides", {})
    for l in range(lo, hi + 1):
        blk = ov.get(str(l)) or copy.deepcopy(m["block"]); r = blk["routed_experts_router"]; r["top_k"] = a.top_k
        if a.norm_top_k is not None: r["norm_top_k"] = a.norm_top_k; r["_CLASS_"] = "emo_square_topk.SquareTopKRouterConfigV2"
        ov[str(l)] = blk
    a.out.mkdir(parents=True, exist_ok=True); json.dump(cfg, open(a.out / "config.json", "w"))
    link = a.out / "model_and_optim"
    if link.is_symlink(): link.unlink()
    os.symlink(os.path.relpath(a.src.resolve() / "model_and_optim", a.out.resolve()), link)
    assert (link / ".metadata").exists(), f"{link} does not resolve to a checkpoint"
    print(f"{a.out}: layers {lo}-{hi} top_k {a.top_k}" + (f", weights normalised over the top {a.norm_top_k}" if a.norm_top_k else "") + f" -> {os.readlink(link)}")


if __name__ == "__main__":
    main()
