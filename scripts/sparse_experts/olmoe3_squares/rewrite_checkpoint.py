#!/usr/bin/env python3
"""Rewrite an OLMoDDP model+optimizer checkpoint as a single-file DCP dir holding only the
`module.<name>.{main,exp_avg,exp_avg_sq,step}` tensors (drops the SkipStepAdamW rolling windows and
the multi-rank sharding), i.e. the same layout merge_models.py writes. Used to give the baseline
finetunes a start checkpoint structurally identical to the merged models'.

Usage: python rewrite_checkpoint.py --src <step dir> --out <dir>   (writes <out>/model_and_optim + config.json)
"""
import argparse, json, shutil
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    import torch.distributed.checkpoint as dcp
    from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
    src = a.src / "model_and_optim"; meta = get_checkpoint_metadata(str(src)).state_dict_metadata
    keys = sorted(k for k in meta if k.startswith("module.") and k.endswith((".main", ".exp_avg", ".exp_avg_sq", ".step")))
    if a.out.exists():
        assert a.overwrite, a.out; shutil.rmtree(a.out)
    (a.out / "model_and_optim").mkdir(parents=True)
    sd = {k: next(load_keys(str(src), [k])).clone() for k in keys}
    dcp.save(sd, storage_writer=dcp.FileSystemWriter(str(a.out / "model_and_optim")))
    shutil.copy(a.src / "config.json", a.out / "config.json")
    print(json.dumps(dict(src=str(a.src), out=str(a.out), n_keys=len(sd), dropped=[k for k in meta if k not in sd])))


if __name__ == "__main__":
    main()
