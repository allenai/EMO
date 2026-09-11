#!/usr/bin/env python3
"""Stage 4 of olmoe3_squares: merge the four trained sub-models back into one 512-expert model.

Inputs: one checkpoint step dir per group (their model_and_optim holds flat fp32 `module.<name>.main`
tensors), groups.json and the groups' token shares. For the partitioned layers (2..9) every group's
expert tensors (w_up_gate, w_down, router rows) are scattered back to their global expert ids, so the
merged layer has all 512 experts again, each owned by exactly one group. Everything else (attention,
embeddings, dense layer 0, latent projections, shared expert, norms, lm head, and the whole of
layer 1) is averaged over the four sub-models with token-share weights. Output: a model-only DCP
checkpoint (`module.<name>.main` keys) + config.json of the full model, readable by the routing
extractor's loader for evaluation.

Usage: python merge_models.py --groups groups.json --full <full step dir> --subs g0dir,g1dir,g2dir,g3dir \
           --weights 0.2,0.4,0.18,0.22 --out <dir>
"""
import argparse, json, re, shutil
from pathlib import Path
import numpy as np, torch

sys_path_note = None


def expert_shape(base: str, E: int, mcfg: dict):
    H = mcfg["block"]["routed_experts"]["hidden_size"]; Dl = mcfg["block"]["routed_experts"]["d_model"]; D = mcfg["d_model"]
    if base.endswith("routed_experts.w_up_gate"): return (E, 2 * H, Dl)
    if base.endswith("routed_experts.w_down"): return (E, H, Dl)
    if base.endswith("routed_experts_router.weight"): return (E, D)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=Path, required=True); ap.add_argument("--full", type=Path, required=True, help="full-model step dir (for config.json and key list)")
    ap.add_argument("--subs", required=True, help="comma-separated sub-model step dirs, group order"); ap.add_argument("--weights", required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    import torch.distributed.checkpoint as dcp
    from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
    G = json.load(open(a.groups)); E = G["num_experts"]; k = G["k"]; part = {int(l) for l in G["partitioned_layers"]}
    subs = [Path(p) for p in a.subs.split(",")]; w = np.array([float(x) for x in a.weights.split(",")]); w = w / w.sum()
    assert len(subs) == k == len(w)
    mcfg = json.load(open(a.full / "config.json"))["model"]
    full_meta = get_checkpoint_metadata(str(a.full / "model_and_optim")).state_dict_metadata
    keys = sorted(kk for kk in full_meta if kk.startswith("module.") and kk.endswith(".main"))
    sub_dirs = [str(p / "model_and_optim") if (p / "model_and_optim").exists() else str(p) for p in subs]  # step dir or bare DCP dir (init slices)
    layer_of = lambda kk: int(re.match(r"module\.blocks\.(\d+)\.", kk).group(1)) if kk.startswith("module.blocks.") else None
    if a.out.exists():
        assert a.overwrite, f"{a.out} exists"; shutil.rmtree(a.out)
    a.out.mkdir(parents=True); (a.out / "model_and_optim").mkdir()
    out_sd, n_scatter, n_avg = {}, 0, 0
    for key in keys:
        base = key[len("module."):-len(".main")]; l = layer_of(key); shp = expert_shape(base, E, mcfg)
        vals = [next(load_keys(d, [key])) for d in sub_dirs]
        if shp is not None and l in part:
            merged = torch.empty(shp, dtype=vals[0].dtype)
            for g in range(k):
                ids = torch.tensor(G["groups"][str(l)][g], dtype=torch.long)
                merged[ids] = vals[g].reshape(len(ids), *shp[1:])
            out_sd[key] = merged.reshape(-1).contiguous(); n_scatter += 1
        else:
            assert all(v.numel() == vals[0].numel() for v in vals), key
            out_sd[key] = sum(float(w[g]) * vals[g] for g in range(k)).to(vals[0].dtype); n_avg += 1
        assert out_sd[key].numel() == full_meta[key].size.numel(), (key, out_sd[key].numel(), full_meta[key].size)
    dcp.save(out_sd, storage_writer=dcp.FileSystemWriter(str(a.out / "model_and_optim")))
    shutil.copy(a.full / "config.json", a.out / "config.json")
    info = dict(subs=[str(p) for p in subs], weights=w.round(4).tolist(), n_scattered=n_scatter, n_averaged=n_avg, n_keys=len(out_sd))
    json.dump(info, open(a.out / "merge_info.json", "w"), indent=1); print(json.dumps(info))


if __name__ == "__main__":
    main()
