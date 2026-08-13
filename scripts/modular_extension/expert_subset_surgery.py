#!/usr/bin/env python3
"""Expert-subset surgery for the k=32 CPT experiment (CPU, single process).

Two modes, both round-tripping purely through checkpoint state dicts (pattern:
scripts/models_v2/expand_moe_experts.py):

extract   Build a (32 standard + 1 shared)-expert checkpoint from a 64-expert pool
          checkpoint, using a PER-LAYER selection of 32 standard experts (from
          expert_concentration.json, frozen at the 100B analysis). Subset slot order is
          the sorted original slot ids; the shared expert (pool slot 63) becomes subset
          slot 32 (last, as the router requires). Expert weights (w1/w2/w3), router rows,
          and (carry mode) Adam moments + per-tensor step are sliced identically.
          Fresh mode zeroes moments and step for the expert tensors.
          Output: <out>/model_and_optim + <out>/selection.json.

writeback Scatter a trained subset checkpoint's standard-expert slots (and shared slot)
          back into the pool checkpoint at their original per-layer slot ids, including
          Adam moments + step for the expert tensors (the router is frozen during
          training and is NOT written back). Edits the pool's model_and_optim in place
          by rewriting it (load -> patch -> save).

The non-expert parameters are frozen during subset training, so only expert tensors ever
move; everything else in the pool stays byte-identical across the sequential sweep.

Usage:
  python scripts/modular_extension/expert_subset_surgery.py extract \\
      --pool <pool_dir> --out <subset_dir> --cluster 5 [--fresh-optim]
  python scripts/modular_extension/expert_subset_surgery.py writeback \\
      --pool <pool_dir> --trained <trained_step_dir> --selection <subset_dir>/selection.json
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

CONC = (REPO / "modular_extension/cluster/emo100b_step23842_100B-130B/k32_cpt"
        / "expert_concentration.json")
E_POOL = 64          # 63 standard + 1 shared (shared = last slot)
NS_POOL = 63
E_SUB = 33           # 32 standard + 1 shared
NS_SUB = 32
N_LAYERS = 16

EXPERT_SUFFIXES = ("feed_forward_moe.experts.mlp.w1", "feed_forward_moe.experts.mlp.w2",
                   "feed_forward_moe.experts.mlp.w3")
ROUTER_SUFFIX = "feed_forward_moe.router.weight"


def layer_of(key: str) -> int:
    assert key.startswith("blocks."), key
    return int(key.split(".")[1])


def slots_for_layer(selection, layer: int) -> list:
    """Pool slot ids that map to subset slots 0..32 for this layer (last = shared 63)."""
    return selection[str(layer)] + [NS_POOL]


def slice_slots(t: torch.Tensor, e_from: int, slot_ids: list) -> torch.Tensor:
    per = t.shape[0] // e_from
    v = t.reshape(e_from, per, *t.shape[1:])
    return v[torch.tensor(slot_ids)].reshape(len(slot_ids) * per, *t.shape[1:]).clone()


def scatter_slots(pool_t, sub_t, slot_ids: list) -> torch.Tensor:
    per = pool_t.shape[0] // E_POOL
    pv = pool_t.reshape(E_POOL, per, *pool_t.shape[1:]).clone()
    sv = sub_t.reshape(len(slot_ids), per, *sub_t.shape[1:])
    for sub_slot, pool_slot in enumerate(slot_ids):
        pv[pool_slot] = sv[sub_slot]
    return pv.reshape(E_POOL * per, *pool_t.shape[1:])


def load_selection_for_cluster(cluster: int) -> dict:
    conc = json.load(open(CONC))
    entry = next(c for c in conc["clusters"] if c["cluster"] == cluster)
    # canonical subset order = sorted original slot ids (stable, order-independent)
    return {str(layer): sorted(ids)
            for layer, ids in enumerate(entry["top32_experts_per_layer"])}


def extract(args):
    from olmo_core.distributed.checkpoint import load_keys, save_state_dict

    out = Path(args.out)
    if (out / "model_and_optim" / ".metadata").exists():
        print(f"SKIP: {out}/model_and_optim exists")
        return
    selection = load_selection_for_cluster(args.cluster)
    out.mkdir(parents=True, exist_ok=True)

    pool_mao = str(Path(args.pool) / "model_and_optim")
    print(f"extract: cluster {args.cluster} from {pool_mao} (fresh_optim={args.fresh_optim})")
    if args.fresh_optim:
        # the fresh arm trains with --load_optim_state=false, so the subset checkpoint's
        # optimizer half is never read: skip loading the pool's ~52GB of moments entirely
        # and write a model-only checkpoint (~14GB instead of ~45GB).
        (model_sd,) = list(load_keys(pool_mao, ["model"]))
        optim_sd = None
    else:
        model_sd, optim_sd = list(load_keys(pool_mao, ["model", "optim"]))

    n_sliced = 0
    for k in list(model_sd):
        is_exp = any(k.endswith(s) for s in EXPERT_SUFFIXES)
        is_rtr = k.endswith(ROUTER_SUFFIX)
        if not (is_exp or is_rtr):
            continue
        slot_ids = slots_for_layer(selection, layer_of(k))
        model_sd[k] = slice_slots(model_sd[k], E_POOL, slot_ids)
        if optim_sd is not None:
            for mkey in ("exp_avg", "exp_avg_sq"):
                sk = f"state.{k}.{mkey}"
                assert sk in optim_sd, f"missing {sk}"
                optim_sd[sk] = slice_slots(optim_sd[sk], E_POOL, slot_ids)
        n_sliced += 1
    assert n_sliced == 4 * N_LAYERS, f"sliced {n_sliced} tensors, expected {4 * N_LAYERS}"

    to_save = {"model": model_sd} if optim_sd is None else {"model": model_sd, "optim": optim_sd}
    save_state_dict(str(out / "model_and_optim"), to_save, save_overwrite=True)
    with open(out / "selection.json", "w") as f:
        json.dump({"cluster": args.cluster, "num_experts": E_SUB, "num_shared": 1,
                   "pool": str(args.pool), "fresh_optim": bool(args.fresh_optim),
                   "slots_per_layer": selection}, f, indent=2)
    # config.json purely informational for humans; the training run builds its model from CLI.
    src_cfg = Path(args.pool) / "config.json"
    if src_cfg.exists():
        cfg = json.load(open(src_cfg))
        try:
            cfg["model"]["block"]["feed_forward_moe"]["num_experts"] = E_SUB
        except Exception:
            pass
        json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print(f"extract done -> {out}")


def writeback(args):
    from olmo_core.distributed.checkpoint import load_keys, save_state_dict

    sel = json.load(open(args.selection))
    selection = sel["slots_per_layer"]
    pool_mao = str(Path(args.pool) / "model_and_optim")
    trained_mao = str(Path(args.trained) / "model_and_optim")
    marker = Path(args.pool) / f"wroteback_cluster{sel['cluster']:02d}.json"
    if marker.exists():
        print(f"SKIP: {marker} exists")
        return

    print(f"writeback: cluster {sel['cluster']} {trained_mao} -> {pool_mao}")
    pool_model, pool_optim = list(load_keys(pool_mao, ["model", "optim"]))
    sub_model, sub_optim = list(load_keys(trained_mao, ["model", "optim"]))

    # --- sanity: ONLY experts may have changed during training -------------------
    # Every non-expert tensor of the trained subset must be bit-identical to the pool's
    # (proves freeze_params held); the frozen router rows must equal their pool slices.
    n_checked = 0
    for k, v in sub_model.items():
        if any(k.endswith(s) for s in EXPERT_SUFFIXES):
            continue
        if k.endswith(ROUTER_SUFFIX):
            slot_ids = slots_for_layer(selection, layer_of(k))
            expect = slice_slots(pool_model[k], E_POOL, slot_ids)
            assert torch.equal(v, expect), f"FROZEN ROUTER CHANGED: {k}"
        else:
            assert torch.equal(v, pool_model[k]), f"FROZEN PARAM CHANGED: {k}"
        n_checked += 1
    print(f"verified {n_checked} non-expert tensors unchanged (freeze held)")

    n_scattered = 0
    for k in list(pool_model):
        if not any(k.endswith(s) for s in EXPERT_SUFFIXES):
            continue  # router frozen -> not written back; nothing else is trainable
        slot_ids = slots_for_layer(selection, layer_of(k))
        assert k in sub_model, f"missing {k} in trained subset"
        pool_model[k] = scatter_slots(pool_model[k], sub_model[k], slot_ids)
        for mkey in ("exp_avg", "exp_avg_sq"):
            pk, skk = f"state.{k}.{mkey}", f"state.{k}.{mkey}"
            if skk in sub_optim and pk in pool_optim:
                pool_optim[pk] = scatter_slots(pool_optim[pk], sub_optim[skk], slot_ids)
        stk = f"state.{k}.step"
        if stk in sub_optim and stk in pool_optim:
            pool_optim[stk] = sub_optim[stk]
        n_scattered += 1
    assert n_scattered == 3 * N_LAYERS, f"scattered {n_scattered}, expected {3 * N_LAYERS}"

    tmp = Path(args.pool) / "model_and_optim.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    save_state_dict(str(tmp), {"model": pool_model, "optim": pool_optim}, save_overwrite=True)
    old = Path(args.pool) / "model_and_optim.old"
    Path(pool_mao).rename(old)
    tmp.rename(pool_mao)
    shutil.rmtree(old)

    # weights-only bf16 snapshot of the pool after this stage (standalone; for later
    # per-stage eval / forgetting analysis)
    snap_dir = Path(args.pool) / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    snap = snap_dir / f"pool_after_c{sel['cluster']:02d}.pt"
    torch.save({k: v.to(torch.bfloat16) for k, v in pool_model.items()}, str(snap))
    print(f"snapshot: {snap} ({snap.stat().st_size / 1e9:.1f} GB)")

    with open(marker, "w") as f:
        json.dump({"trained": str(args.trained)}, f)
    print(f"writeback done: {marker}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("extract")
    pe.add_argument("--pool", required=True, help="64-expert pool step dir")
    pe.add_argument("--out", required=True, help="output subset dir")
    pe.add_argument("--cluster", type=int, required=True)
    pe.add_argument("--fresh-optim", action="store_true",
                    help="zero expert Adam moments + step (fresh-optimizer arm)")
    pw = sub.add_parser("writeback")
    pw.add_argument("--pool", required=True)
    pw.add_argument("--trained", required=True, help="trained subset step dir")
    pw.add_argument("--selection", required=True, help="selection.json from extract")
    args = p.parse_args()
    extract(args) if args.cmd == "extract" else writeback(args)


if __name__ == "__main__":
    main()
