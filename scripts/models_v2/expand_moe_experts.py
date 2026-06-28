#!/usr/bin/env python3
"""Expert upcycling: expand a trained stdMoE (``moe_lbreducedp_sharedexp``) checkpoint from
``--from-experts`` to ``--to-experts`` experts and write a checkpoint a normal training run can
auto-resume from. CPU-only, single process (no GPU / no torch.distributed init needed).

Why this is slot-aware (NOT a naive tensor doubling)
----------------------------------------------------
For model-type ``moe_lbreducedp_sharedexp`` the "shared expert" is the LAST routed slot, not a
separate module (see ``src/olmo_core/nn/moe/router_lbreducedp_sharedexp.py``: the router slices
logits to ``[:, :, :num_standard]`` with ``num_standard = num_experts - num_shared_experts`` and
pins the last ``num_shared_experts`` slot(s) to weight 1.0). So a 64-expert checkpoint is
63 standard experts (slots 0..62) + 1 shared (slot 63); a 128-expert target is 127 standard
(0..126) + 1 shared (127). We therefore:

  new slot 0..62    <- old standard 0..62           (kept verbatim)
  new slot 63..125  <- old standard 0..62           (one copy each; 63 new experts)
  new slot 126      <- old standard <seeded random>  (the one extra new expert)
  new slot 127      <- old shared (old slot 63)      (shared moved to last slot)

For ``--init-mode random`` the 64 new standard slots (63..126) are freshly initialized with the
exact module init (``kaiming_uniform_(a=sqrt(5))`` for experts, ``trunc_normal_(std=0.02)`` for the
router) instead of copied.

Tensors expanded per MoE block (keys end with these):
  feed_forward_moe.experts.mlp.w1   [E*d_model, hidden]   (per-expert rows = d_model)
  feed_forward_moe.experts.mlp.w2   [E*hidden, d_model]   (per-expert rows = hidden)
  feed_forward_moe.experts.mlp.w3   [E*d_model, hidden]
  feed_forward_moe.router.weight    [E*d_model]           (per-expert rows = d_model, 1-D)
The matching optimizer moments (exp_avg / exp_avg_sq) under optim["state"][key] are expanded the
same way. Adam ``step`` is per-tensor (one scalar per param), so it is left as-is (carry) or zeroed
(reset) -- never per-slot.

Optimizer axes
--------------
  --kept-optim carry|reset : kept experts + non-MoE params keep their moments (carry) or all moments
                             are zeroed + per-param step set to 0 (reset = fresh Adam at the branch).
  --new-optim  copy|zero   : NEW upcycled experts inherit their source expert's moments (copy) or
                             start at zero. N/A for --init-mode random (always zero); irrelevant
                             under --kept-optim reset (everything zeroed).

I/O round-trips purely through the checkpoint state-dict (no model/FSDP rebuild):
  load_keys(src, ["model","optim"]) -> reshape tensors -> save_state_dict(out, {"model","optim"}).
This is loadable by the trainer's resume path (load_model_and_optim_state rebuilds the same
{"model","optim"} structure and fills by key; shapes now match the 128-expert model).
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-checkpoint", required=True, help="source step dir (has model_and_optim/, train/, config.json)")
    p.add_argument("--output-checkpoint", required=True, help="target step dir to create")
    p.add_argument("--from-experts", type=int, default=64)
    p.add_argument("--to-experts", type=int, default=128)
    p.add_argument("--num-shared", type=int, default=1)
    p.add_argument("--init-mode", choices=["random", "upcycle", "upcycle_jitter"], required=True)
    p.add_argument("--kept-optim", choices=["carry", "reset"], required=True)
    p.add_argument("--new-optim", choices=["copy", "zero"], default="copy",
                   help="new-expert moments: copy source (upcycle modes) or zero. Ignored for random/reset.")
    p.add_argument("--jitter-std", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_plan(e_old, e_new, num_shared, gen):
    """Return per-new-slot (src, kind). kind in {'keep','new','shared'}; src is the old slot to
    copy from (used for keep/shared and for upcycle 'new')."""
    ns_old = e_old - num_shared
    ns_new = e_new - num_shared
    num_new = ns_new - ns_old
    assert num_new > 0, f"to-experts must exceed from-experts (got {e_new} <= {e_old})"

    src = [0] * e_new
    kind = [""] * e_new
    # kept standard
    for e in range(ns_old):
        src[e], kind[e] = e, "keep"
    # new standard: copy each old standard once, then fill the remainder by seeded-random source
    for j, e in enumerate(range(ns_old, ns_new)):
        if j < ns_old:
            src[e] = j
        else:
            src[e] = int(torch.randint(0, ns_old, (1,), generator=gen).item())
        kind[e] = "new"
    # shared (moved to the last num_shared slots)
    for k in range(num_shared):
        src[ns_new + k], kind[ns_new + k] = ns_old + k, "shared"
    return src, kind, ns_old, ns_new


def expand_weight(old, e_old, e_new, plan, *, is_router, init_mode, jitter_std, gen):
    """old: flat tensor [E_old*per, *rest] (router: 1-D [E_old*d_model]). Returns [E_new*per, *rest]."""
    src, kind, ns_old, ns_new = plan
    per = old.shape[0] // e_old
    rest = tuple(old.shape[1:])
    oldv = old.reshape(e_old, per, *rest)
    newv = torch.empty((e_new, per, *rest), dtype=old.dtype)

    for e in range(e_new):
        if kind[e] in ("keep", "shared"):
            newv[e] = oldv[src[e]]
        elif init_mode == "upcycle":
            newv[e] = oldv[src[e]]
        elif init_mode == "upcycle_jitter":
            w = oldv[src[e]].clone()
            noise = torch.randn(w.shape, generator=gen, dtype=torch.float32).to(w.dtype)
            newv[e] = w * (1.0 + jitter_std * noise)
        # init_mode == "random": filled in bulk below

    if init_mode == "random":
        block = newv[ns_old:ns_new]  # [num_new, per, *rest]
        num_new = block.shape[0]
        if is_router:
            flat = torch.empty(num_new * per, dtype=torch.float32)
            torch.nn.init.trunc_normal_(flat, std=0.02, a=-3 * 0.02, b=3 * 0.02, generator=gen)
            newv[ns_old:ns_new] = flat.reshape(num_new, per).to(old.dtype)
        else:
            flat = torch.empty(num_new * per, *rest, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(flat, a=math.sqrt(5), generator=gen)
            newv[ns_old:ns_new] = flat.reshape(num_new, per, *rest).to(old.dtype)

    return newv.reshape(e_new * per, *rest)


def expand_moment(old, e_old, e_new, plan, *, init_mode, new_optim):
    """Expand an Adam moment (exp_avg / exp_avg_sq) with the slot plan: keep/shared copy their
    source moment; new slots copy source moment (upcycle + new_optim==copy) else zero."""
    src, kind, ns_old, ns_new = plan
    per = old.shape[0] // e_old
    rest = tuple(old.shape[1:])
    oldv = old.reshape(e_old, per, *rest)
    newv = torch.zeros((e_new, per, *rest), dtype=old.dtype)
    copy_new = (init_mode != "random") and (new_optim == "copy")
    for e in range(e_new):
        if kind[e] in ("keep", "shared"):
            newv[e] = oldv[src[e]]
        elif copy_new:
            newv[e] = oldv[src[e]]
        # else leave zero
    return newv.reshape(e_new * per, *rest)


MOE_SUFFIXES = (
    "feed_forward_moe.experts.mlp.w1",
    "feed_forward_moe.experts.mlp.w2",
    "feed_forward_moe.experts.mlp.w3",
    "feed_forward_moe.router.weight",
)


def is_moe_key(k):
    return any(k.endswith(s) for s in MOE_SUFFIXES)


def is_router_key(k):
    return k.endswith("feed_forward_moe.router.weight")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))

    from olmo_core.distributed.checkpoint import load_keys, save_state_dict

    src_dir = Path(args.input_checkpoint)
    out_dir = Path(args.output_checkpoint)
    src_mao = src_dir / "model_and_optim"
    out_mao = out_dir / "model_and_optim"
    assert src_mao.is_dir(), f"missing {src_mao}"
    if (out_mao / ".metadata").exists():
        print(f"=== output already exists, nothing to do: {out_mao} ===")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    e_old, e_new, num_shared = args.from_experts, args.to_experts, args.num_shared
    gen = torch.Generator().manual_seed(args.seed)
    plan = build_plan(e_old, e_new, num_shared, gen)
    src_slots, kind, ns_old, ns_new = plan
    print(f"=== expand {e_old}->{e_new} experts (shared={num_shared}); standard {ns_old}->{ns_new}")
    print(f"    init-mode={args.init_mode} kept-optim={args.kept_optim} new-optim={args.new_optim} "
          f"jitter-std={args.jitter_std} seed={args.seed}")
    print(f"    new-standard slot 126 (the one extra) copies old standard {src_slots[ns_new-1]}")

    print(f"=== loading model+optim state from {src_mao} (single process, no dist) ===")
    model_sd, optim_sd = list(load_keys(str(src_mao), ["model", "optim"]))
    assert isinstance(model_sd, dict) and isinstance(optim_sd, dict)
    # NOTE: load_keys returns FLAT dotted dicts. model_sd keys are FQNs ("blocks.0...w1").
    # optim_sd keys are flattened: "state.<fqn>.{exp_avg,exp_avg_sq,step}" and
    # "param_groups.<fqn>.<hyperparam>". (The on-disk checkpoint uses flatten_optimizer_state.)
    n_state = sum(1 for k in optim_sd if k.endswith(".exp_avg"))
    print(f"    model keys: {len(model_sd)}   optim keys: {len(optim_sd)}   params with state: {n_state}")

    moe_keys = [k for k in model_sd if is_moe_key(k)]
    print(f"    MoE tensor keys to expand: {len(moe_keys)} (expect 4*n_layers)")
    assert len(moe_keys) > 0, "no MoE keys found -- key naming mismatch"

    # --- expand model weights + their optimizer moments (flat optim keys) ---
    for k in moe_keys:
        is_r = is_router_key(k)
        old_w = model_sd[k]
        assert old_w.shape[0] % e_old == 0, f"{k}: dim0 {old_w.shape[0]} not divisible by {e_old}"
        model_sd[k] = expand_weight(
            old_w, e_old, e_new, plan,
            is_router=is_r, init_mode=args.init_mode, jitter_std=args.jitter_std, gen=gen,
        )
        for mkey in ("exp_avg", "exp_avg_sq"):
            sk = f"state.{k}.{mkey}"
            assert sk in optim_sd, f"missing optimizer moment {sk}"
            assert torch.is_tensor(optim_sd[sk]), f"{sk} is not a tensor"
            optim_sd[sk] = expand_moment(
                optim_sd[sk], e_old, e_new, plan,
                init_mode=args.init_mode, new_optim=args.new_optim,
            )

    # --- sanity: no other model tensor secretly depends on num_experts ---
    for k, v in model_sd.items():
        if is_moe_key(k) or not torch.is_tensor(v):
            continue
        if v.numel() == e_old:  # e.g. a stray [num_experts] buffer
            print(f"    WARNING: non-MoE tensor {k} has numel {v.numel()} == from-experts; inspect!")

    # --- kept-optim reset: zero ALL moments + per-param step ---
    if args.kept_optim == "reset":
        print("=== kept-optim=reset: zeroing all optimizer moments + steps ===")
        for sk in list(optim_sd):
            if sk.endswith(".exp_avg") or sk.endswith(".exp_avg_sq"):
                if torch.is_tensor(optim_sd[sk]):
                    optim_sd[sk] = torch.zeros_like(optim_sd[sk])
            elif sk.endswith(".step"):
                s = optim_sd[sk]
                optim_sd[sk] = torch.zeros_like(s) if torch.is_tensor(s) else 0.0

    # --- write the expanded checkpoint ---
    print(f"=== saving expanded checkpoint to {out_mao} ===")
    save_state_dict(str(out_mao), {"model": model_sd, "optim": optim_sd}, save_overwrite=True)

    # --- copy trainer state (preserves global_step + data cursor) verbatim ---
    src_train, out_train = src_dir / "train", out_dir / "train"
    print(f"=== copying trainer state {src_train} -> {out_train} ===")
    if out_train.exists():
        shutil.rmtree(out_train)
    shutil.copytree(src_train, out_train)

    # --- config.json: copy, patch num_experts (informational; resume builds model from run config) ---
    src_cfg, out_cfg = src_dir / "config.json", out_dir / "config.json"
    if src_cfg.exists():
        import json
        cfg = json.load(open(src_cfg))
        try:
            cfg["model"]["block"]["feed_forward_moe"]["num_experts"] = e_new
        except Exception:
            pass
        json.dump(cfg, open(out_cfg, "w"), indent=2)
    for extra in ("data_paths.txt",):
        if (src_dir / extra).exists():
            shutil.copy2(src_dir / extra, out_dir / extra)

    # --- self-check: reload one block's w1 + router and verify shapes / kept-slot identity ---
    print("=== self-check: reloading expanded tensors ===")
    w1_key = next(k for k in moe_keys if k.endswith("experts.mlp.w1"))
    r_key = next(k for k in moe_keys if is_router_key(k))
    new_w1, new_r, new_w1_m = list(load_keys(
        str(out_mao), [f"model.{w1_key}", f"model.{r_key}", f"optim.state.{w1_key}.exp_avg"]))
    per_w1 = new_w1.shape[0] // e_new
    per_r = new_r.shape[0] // e_new
    assert new_w1.shape[0] == e_new * per_w1, new_w1.shape
    assert new_r.shape[0] == e_new * per_r, new_r.shape
    assert tuple(new_w1_m.shape) == tuple(new_w1.shape), (new_w1_m.shape, new_w1.shape)
    # kept slot 0 identity vs source
    src_w1, src_r = list(load_keys(str(src_mao), [f"model.{w1_key}", f"model.{r_key}"]))
    ov = src_w1.reshape(e_old, per_w1, -1)
    nv = new_w1.reshape(e_new, per_w1, -1)
    assert torch.equal(nv[0], ov[0]), "kept slot 0 mismatch"
    assert torch.equal(nv[62], ov[62]), "kept slot 62 mismatch"
    assert torch.equal(nv[ns_new], ov[ns_old]), "shared slot not moved correctly"
    if args.init_mode == "upcycle":
        assert torch.equal(nv[63], ov[0]), "upcycle new slot 63 should copy old standard 0"
    print(f"    OK: w1 {tuple(new_w1.shape)} router {tuple(new_r.shape)}; kept/shared slots verified")
    print(f"=== DONE: {out_dir} ===")


if __name__ == "__main__":
    main()
