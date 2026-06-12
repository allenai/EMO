"""Materialize a "ghost" variant of a converted HF EMO checkpoint.

The variant is a sibling dir ``<src>-ghost`` that symlinks every file from the
standard HF checkpoint (weights, tokenizer, modeling code) EXCEPT ``config.json``,
which is rewritten with ``ghost_extend_eval=true`` (+ coeff mode / random_k). This
makes the ghost toggle live in config.json, so eval reads it reliably regardless of
how the eval harness forwards --model-args.

Symlinks are RELATIVE so they resolve both in this session (~/EMO/...) and on Beaker
workers (/weka/oe-training-default/...), which mount the same weka at different roots.

    python scripts/models_fullextend/make_ghost_hf_variant.py --src <hf_dir> \
        [--coeff-mode usage] [--random-k 8]
"""

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Standard (ghost-off) HF checkpoint dir")
    ap.add_argument("--dst", default=None, help="Defaults to <src>-ghost")
    ap.add_argument("--coeff-mode", default="usage", choices=["usage", "uniform", "random"])
    ap.add_argument("--random-k", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve() if args.dst else src.parent / (src.name + "-ghost")
    if not (src / "config.json").is_file():
        raise SystemExit(f"{src} is not an HF checkpoint dir (no config.json)")
    dst.mkdir(parents=True, exist_ok=True)

    for f in src.iterdir():
        link = dst / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        if f.name == "config.json":
            continue
        link.symlink_to(os.path.relpath(f, dst))  # relative symlink

    cfg = json.loads((src / "config.json").read_text())
    cfg["ghost_extend_eval"] = True
    cfg["ghost_extend_coeff_mode"] = args.coeff_mode
    cfg["ghost_extend_random_k"] = args.random_k
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote ghost variant ({args.coeff_mode}) -> {dst}")


if __name__ == "__main__":
    main()
