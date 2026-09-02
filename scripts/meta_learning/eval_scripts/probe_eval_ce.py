#!/usr/bin/env python3
# PARENT: "scripts/modular_extension/eval_k32cpt_ce.py" (eval machinery, reused via import)
"""Step-granularity probe eval for the phase-2 diagnostic: full-128-expert CE on cluster
held-out sets, where the model = the arm's 20B base with ONE cluster's 32-expert subset
(at training step k) patched in, composed IN MEMORY -- no pool writebacks, no disk churn.

Subcommands:
  make-base   distill the base checkpoint's model half into a bf16 weights-only .pt
      python probe_eval_ce.py make-base --checkpoint <step4768 dir> --out base_bf16.pt
  eval        evaluate base (+ optional subset patch) on clusters
      python probe_eval_ce.py eval --base-snapshot base_bf16.pt --config-from <step4768 dir> \\
          [--subset-ckpt <stage stepK dir> --selection <subset>/selection.json] \\
          --tokens-root <k32_cpt_tokens_...> --clusters 5 --max-tokens-per-cluster 4194304 \\
          --out ce_...json

Output JSON shape matches eval_k32cpt_ce.py ({"results": {"<c>": {"ce", "positions", ...}}}).
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "eval_k32cpt_ce", REPO / "scripts/modular_extension/eval_k32cpt_ce.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

E_POOL, NS_POOL, N_LAYERS = 128, 127, 16
EXPERT_SUFFIXES = ("feed_forward_moe.experts.mlp.w1", "feed_forward_moe.experts.mlp.w2",
                   "feed_forward_moe.experts.mlp.w3")


def make_base(args):
    from olmo_core.distributed.checkpoint import load_keys

    if os.path.exists(args.out):
        print(f"SKIP: {args.out} exists")
        return
    (sd,) = list(load_keys(str(Path(args.checkpoint) / "model_and_optim"), ["model"]))
    tmp = args.out + ".tmp"
    torch.save({k: v.to(torch.bfloat16) for k, v in sd.items()}, tmp)
    os.rename(tmp, args.out)
    print(f"base snapshot: {args.out} ({os.path.getsize(args.out) / 1e9:.1f} GB)")


def build_patched_model(args, device):
    from olmo_core.nn.transformer import TransformerConfig

    tcfg = json.load(open(Path(args.config_from) / "config.json"))["model"]
    moe = tcfg["block"]["feed_forward_moe"]
    router = moe.get("router") or {}
    if "eval_document_expert_pool" in router:
        router["eval_document_expert_pool"] = moe["num_experts"]  # full routing
    model = TransformerConfig.from_dict(tcfg).build(init_device="meta")
    model.to_empty(device=device)

    sd = torch.load(args.base_snapshot, map_location="cpu")
    if args.subset_ckpt:
        from olmo_core.distributed.checkpoint import load_keys

        (sub,) = list(load_keys(str(Path(args.subset_ckpt) / "model_and_optim"), ["model"]))
        sel = json.load(open(args.selection))["slots_per_layer"]
        n = 0
        for k in list(sd.keys()):
            if not any(k.endswith(s) for s in EXPERT_SUFFIXES):
                continue  # router frozen during probe training; base rows stay
            layer = int(k.split(".")[1])
            slot_ids = sel[str(layer)] + [NS_POOL]  # subset slot order; last = shared
            per = sd[k].shape[0] // E_POOL
            pv = sd[k].reshape(E_POOL, per, *sd[k].shape[1:]).clone()
            sv = sub[k].reshape(len(slot_ids), per, *sub[k].shape[1:])
            for i, slot in enumerate(slot_ids):
                pv[slot] = sv[i].to(pv.dtype)
            sd[k] = pv.reshape(E_POOL * per, *sd[k].shape[1:])
            n += 1
        assert n == 3 * N_LAYERS, f"patched {n} tensors, expected {3 * N_LAYERS}"
    missing, unexpected = model.load_state_dict(
        {k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    return model


def run_eval(args):
    device = torch.device("cuda")
    model = build_patched_model(args, device)
    results = {}
    for c in E.parse_clusters(args.clusters):
        hp = Path(args.tokens_root) / f"cluster{c:02d}" / "heldout.npy"
        ce, n, dt = E.eval_cluster(model, hp, device, args.batch_size,
                                   args.max_tokens_per_cluster)
        results[str(c)] = {"ce": round(ce, 5), "positions": n, "seconds": round(dt, 1)}
        print(f"cluster {c:2d}: CE {ce:.4f} over {n / 1e6:.1f}M positions ({dt:.0f}s)",
              flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"base": args.base_snapshot, "subset": args.subset_ckpt,
                   "results": results}, f, indent=2)
    print(f"Wrote {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("make-base")
    pm.add_argument("--checkpoint", required=True)
    pm.add_argument("--out", required=True)
    pe = sub.add_parser("eval")
    pe.add_argument("--base-snapshot", required=True)
    pe.add_argument("--config-from", required=True)
    pe.add_argument("--subset-ckpt", default=None)
    pe.add_argument("--selection", default=None)
    pe.add_argument("--tokens-root", required=True)
    pe.add_argument("--clusters", default="0-31")
    pe.add_argument("--out", required=True)
    pe.add_argument("--batch-size", type=int, default=4)
    pe.add_argument("--max-tokens-per-cluster", type=int, default=4_194_304)
    args = p.parse_args()
    if args.cmd == "eval" and args.subset_ckpt:
        assert args.selection, "--selection required with --subset-ckpt"
    make_base(args) if args.cmd == "make-base" else run_eval(args)


if __name__ == "__main__":
    main()
