#!/usr/bin/env python3
"""Stage 2a of olmoe3_squares: cut a group's sub-model checkpoint out of the full EMO 512e checkpoint.

OLMoDDP checkpoints store every parameter as flat fp32 optimizer tensors `module.<name>.{main,exp_avg,
exp_avg_sq,step}` (see olmo_core.optim.moe_optimizer). For the partitioned layers the routed-expert
tensors `w_up_gate (E, 2H, Dl)`, `w_down (E, H, Dl)` and the router weight `(E, d_model)` are sliced to
the group's expert ids (ascending), all three optimizer tensors alike; every other tensor is copied.
The result is written as a model+optimizer-only DCP checkpoint (`.metadata` + *.distcp at the top
level, no train/ state), which the trainer loads with --trainer.load_path + load_trainer_state=false.

Usage: python slice_checkpoint.py --checkpoint <step dir> --groups groups.json --group g --out <dir> [--merge-check]
"""
import argparse, json, re, shutil
from pathlib import Path
import numpy as np, torch

SUFFIXES = ("main", "exp_avg", "exp_avg_sq", "step")
EXPERT_KEYS = {"routed_experts.w_up_gate": 3, "routed_experts.w_down": 3, "routed_experts_router.weight": 2}  # -> ndim


def expert_shape(key: str, numel: int, E: int, mcfg: dict):
    H = mcfg["block"]["routed_experts"]["hidden_size"]; Dl = mcfg["block"]["routed_experts"]["d_model"]; D = mcfg["d_model"]
    if key.endswith("routed_experts.w_up_gate"): shp = (E, 2 * H, Dl)
    elif key.endswith("routed_experts.w_down"): shp = (E, H, Dl)
    elif key.endswith("routed_experts_router.weight"): shp = (E, D)
    else: raise KeyError(key)
    assert int(np.prod(shp)) == numel, (key, shp, numel)
    return shp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True); ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--group", type=int, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    import torch.distributed.checkpoint as dcp
    from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
    G = json.load(open(a.groups)); E = G["num_experts"]; g = a.group
    mcfg = json.load(open(a.checkpoint / "config.json"))["model"]
    src = a.checkpoint / "model_and_optim"
    meta = get_checkpoint_metadata(str(src)).state_dict_metadata
    keys = [k for k in meta if k.startswith("module.")]
    layer_of = lambda k: int(re.match(r"module\.blocks\.(\d+)\.", k).group(1)) if k.startswith("module.blocks.") else None
    part = {int(l) for l in G["partitioned_layers"]}
    if a.out.exists():
        assert a.overwrite, f"{a.out} exists"; shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    out_sd, n_sliced, kept_numel = {}, 0, 0
    for key in sorted(keys):
        val = next(load_keys(str(src), [key]))
        suffix = key.rsplit(".", 1)[1]; base = key[len("module."):-len(suffix) - 1]
        l = layer_of(key)
        ek = next((ek for ek in EXPERT_KEYS if base.endswith(ek)), None)
        if ek is not None and l in part and suffix != "step":
            ids = torch.tensor(G["groups"][str(l)][g], dtype=torch.long)
            full = val.reshape(expert_shape(base, val.numel(), E, mcfg))
            val = full.index_select(0, ids).reshape(-1).contiguous(); n_sliced += 1
        out_sd[key] = val.clone(); kept_numel += val.numel() if suffix == "main" else 0
    dcp.save(out_sd, storage_writer=dcp.FileSystemWriter(str(a.out)))
    # read-back check on a few sliced tensors
    chk = get_checkpoint_metadata(str(a.out)).state_dict_metadata
    for l in sorted(part)[:2]:
        key = f"module.blocks.{l}.routed_experts.w_down.main"; ids = G["groups"][str(l)][g]
        full = next(load_keys(str(src), [key])).reshape(expert_shape(key[7:-5], meta[key].size.numel(), E, mcfg))
        got = next(load_keys(str(a.out), [key])).reshape(len(ids), *full.shape[1:])
        assert torch.equal(got, full[ids]), f"read-back mismatch at {key}"
    info = dict(group=g, source=str(a.checkpoint), n_keys=len(out_sd), n_sliced_tensors=n_sliced, main_numel=int(kept_numel),
                experts_per_layer={str(l): len(G["groups"][str(l)][g]) for l in range(1, 10)})
    json.dump(info, open(a.out / "slice_info.json", "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
