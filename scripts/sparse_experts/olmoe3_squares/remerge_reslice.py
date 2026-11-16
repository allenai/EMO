#!/usr/bin/env python3
"""Merge K sub-models and cut the merged model into K NEW sub-models in one pass, in memory (2026-11-15, re-partition every ~100M tokens:
merge_models.py + 8 x slice_checkpoint.py take ~40 min per cycle through a 30 GB merged checkpoint on disk; this takes a few minutes).
Same rules as those two tools, optimizer state (exp_avg / exp_avg_sq / step) always carried:
  merge: experts of the partitioned layers (w_up_gate, w_down, router rows) are scattered back to their global ids; every other tensor
         is the weighted mean of the sub-models (bit-exact copy when identical in all of them);
  slice: sub-model g of the new grouping keeps its experts' rows (ascending ids) of those tensors, everything else whole.
Source is either --subs (sub-model step dirs, order of --old-groups) or --full (a full-model step dir: the first cut).
Outputs: <--out>/group<g>/model_and_optim (+ config.json of --config-from/group<g> when given) unless --no-slice; --save-merged <dir>
writes the merged model too (model_and_optim + the full model's config.json + merge_info.json; a resumable/evaluable step dir).
  python remerge_reslice.py --full-config <full step dir> (--full <step dir> | --old-groups G --subs d0,..,d7 --weights w0,..,w7)
         [--new-groups G' --out <dir>] [--save-merged <dir>] [--no-slice]"""
import argparse, json, os, re, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch


def expert_shape(base, E, mcfg):
    H = mcfg["block"]["routed_experts"]["hidden_size"]; Dl = mcfg["block"]["routed_experts"]["d_model"]; D = mcfg["d_model"]
    if base.endswith("routed_experts.w_up_gate"): return (E, 2 * H, Dl)
    if base.endswith("routed_experts.w_down"): return (E, H, Dl)
    if base.endswith("routed_experts_router.weight"): return (E, D)
    return None


def save_dcp(sd, out):
    import torch.distributed.checkpoint as dcp
    out = Path(out)
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True); dcp.save(sd, storage_writer=dcp.FileSystemWriter(str(out)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-config", type=Path, required=True, help="full-model step dir: config.json and the key list")
    ap.add_argument("--full", type=Path); ap.add_argument("--old-groups", type=Path); ap.add_argument("--subs"); ap.add_argument("--weights")
    ap.add_argument("--new-groups", type=Path); ap.add_argument("--out", type=Path); ap.add_argument("--save-merged", type=Path); ap.add_argument("--no-slice", action="store_true")
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args(); t0 = time.time()
    from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
    mcfg = json.load(open(a.full_config / "config.json"))["model"]
    full_meta = get_checkpoint_metadata(str(a.full_config / "model_and_optim")).state_dict_metadata
    keys = sorted(k for k in full_meta if k.startswith("module.") and k.endswith((".main", ".exp_avg", ".exp_avg_sq", ".step")))
    layer_of = lambda k: int(re.match(r"module\.blocks\.(\d+)\.", k).group(1)) if k.startswith("module.blocks.") else None
    dcp_dir = lambda p: str(p / "model_and_optim") if (p / "model_and_optim").exists() else str(p)
    info = dict(n_keys=len(keys))
    if a.full is not None: srcs = [dcp_dir(a.full)]; Go = None; info["source"] = str(a.full)
    else:
        Go = json.load(open(a.old_groups)); subs = [Path(p) for p in a.subs.split(",")]; w = np.array([float(x) for x in a.weights.split(",")]); w = w / w.sum()
        assert len(subs) == Go["k"] == len(w); srcs = [dcp_dir(p) for p in subs]; info.update(subs=[str(p) for p in subs], weights=w.round(5).tolist())
    Gn = None if a.no_slice else json.load(open(a.new_groups)); kn = 0 if Gn is None else Gn["k"]
    E = (Gn or Go)["num_experts"] if (Gn or Go) else None
    outs = [dict() for _ in range(kn)]; merged_sd = {} if a.save_merged else None; n_sc = n_avg = n_same = n_sl = 0
    with ThreadPoolExecutor(len(srcs)) as ex:
        for c0 in range(0, len(keys), a.chunk):
            ck = keys[c0:c0 + a.chunk]; loaded = list(ex.map(lambda d: list(load_keys(d, ck)), srcs))
            for i, key in enumerate(ck):
                suffix = "." + key.rsplit(".", 1)[1]; base = key[len("module."):-len(suffix)]; l = layer_of(key)
                vals = [ld[i] for ld in loaded]; shp = expert_shape(base, E, mcfg) if suffix != ".step" else None
                if Go is None: m = vals[0]
                elif shp is not None and l in {int(x) for x in Go["partitioned_layers"]}:
                    owners = np.bincount(np.concatenate([np.asarray(Go["groups"][str(l)][g]) for g in range(Go["k"])]), minlength=E); assert (owners == 1).all(), f"layer {l}: not a partition"
                    m = torch.empty(shp, dtype=vals[0].dtype)
                    for g in range(Go["k"]):
                        ids = torch.tensor(Go["groups"][str(l)][g], dtype=torch.long); m[ids] = vals[g].reshape(len(ids), *shp[1:])
                    m = m.reshape(-1); n_sc += 1
                else:
                    assert all(v.numel() == vals[0].numel() for v in vals), key
                    if all(torch.equal(v, vals[0]) for v in vals[1:]): m = vals[0]; n_same += 1
                    else: m = sum(float(w[g]) * vals[g] for g in range(len(vals))).to(vals[0].dtype); n_avg += 1
                assert m.numel() == full_meta[key].size.numel(), (key, m.numel(), full_meta[key].size)
                if merged_sd is not None: merged_sd[key] = m
                if Gn is not None:
                    if shp is not None and l in {int(x) for x in Gn["partitioned_layers"]}:
                        fm = m.reshape(shp)
                        for g in range(kn): outs[g][key] = fm.index_select(0, torch.tensor(Gn["groups"][str(l)][g], dtype=torch.long)).reshape(-1).contiguous()
                        n_sl += 1
                    else:
                        for g in range(kn): outs[g][key] = m
            del loaded
    info.update(n_scattered=n_sc, n_averaged=n_avg, n_identical_copied=n_same, n_sliced=n_sl, load_merge_seconds=round(time.time() - t0, 1))
    # writes in forked children (the state is shared copy-on-write): one per new sub-model + one for the merged model
    jobs = [(outs[g], a.out / f"group{g}" / "model_and_optim") for g in range(kn)] + ([(merged_sd, a.save_merged / "model_and_optim")] if a.save_merged else [])
    pids = []
    for sd, out in jobs:
        pid = os.fork()
        if pid == 0:
            try: torch.set_num_threads(4); save_dcp(sd, out); os._exit(0)
            except BaseException as e: print("save failed:", out, repr(e), file=sys.stderr, flush=True); os._exit(1)
        pids.append(pid)
    assert all(os.waitpid(p, 0)[1] == 0 for p in pids), "a checkpoint write failed"
    for g in range(kn):   # read-back check of one sliced tensor per sub-model against the merged tensor in memory
        key = "module.blocks.2.routed_experts.w_down.main"; got = next(load_keys(str(a.out / f"group{g}" / "model_and_optim"), [key])); assert torch.equal(got, outs[g][key]), f"read-back mismatch group {g}"
        json.dump(dict(group=g, experts_per_layer={str(l): len(Gn["groups"][str(l)][g]) for l in range(1, 10)}), open(a.out / f"group{g}" / "model_and_optim" / "slice_info.json", "w"))
    if a.save_merged:
        shutil.copy(a.full_config / "config.json", a.save_merged / "config.json"); json.dump(info, open(a.save_merged / "merge_info.json", "w"), indent=1)
    info["seconds"] = round(time.time() - t0, 1); print(json.dumps(info))


if __name__ == "__main__":
    main()
