#!/usr/bin/env python3
"""Shrink trainer checkpoints to their weights (2026-11-15: the branch-point experiment filled weka twice): an OLMoDDP checkpoint stores every
parameter as `module.<name>.{main,exp_avg,exp_avg_sq,step}`; merging (merge_models.py without --with-optim) and held-out scoring
(extract_routing.py) read only `.main`, so a checkpoint that will only be merged and scored keeps a third of its size. Rewrites
<step dir>/model_and_optim in place (new folder written first, then swapped); idempotent (marker file .main_only). NOT for checkpoints a
run must resume or continue training from.
  python strip_optim.py <step dir> [<step dir> ...]"""
import shutil, sys
from pathlib import Path


def main():
    import torch.distributed.checkpoint as dcp
    from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
    for d in map(Path, sys.argv[1:]):
        src = d / "model_and_optim"
        if (d / ".main_only").exists() or not (src / ".metadata").exists(): continue
        keys = sorted(k for k in get_checkpoint_metadata(str(src)).state_dict_metadata if k.startswith("module.") and k.endswith(".main"))
        sd = {k: v.clone() for k, v in zip(keys, load_keys(str(src), keys))}
        tmp = d / "model_and_optim_main"
        if tmp.exists(): shutil.rmtree(tmp)
        dcp.save(sd, storage_writer=dcp.FileSystemWriter(str(tmp)))
        shutil.rmtree(src); tmp.rename(src); (d / ".main_only").write_text("weights only (strip_optim.py)\n")
        print(f"stripped {d}: {len(keys)} tensors kept")


if __name__ == "__main__":
    main()
