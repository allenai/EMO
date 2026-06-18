"""Verify router-freezing for the models_routerfixed smoke test.

Given two checkpoints saved a few steps apart from a frozen-router run, assert:
  - every ``*.router.weight`` is **bit-identical** across the two steps (the router never moved), and
  - at least one sampled non-router weight **changed** (training actually happened).

Optionally (``--expect-router-equals <ckpt>``) assert the run's routers still equal a reference
checkpoint's routers -- used to confirm the grafted init survived save -> load -> train.

Runs single-process (plain ``python``); ``load_keys`` returns full unsharded tensors.

    python scripts/models_routerfixed/check_frozen.py --a <ckptA> --b <ckptB> \
        [--expect-router-equals <init_ckpt>]
"""

import argparse

import torch

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys


def _mo(d: str) -> str:
    d = d.rstrip("/")
    return d if d.endswith("model_and_optim") else d + "/model_and_optim"


def _load(dir_: str, keys):
    keys = list(keys)
    return dict(zip(keys, load_keys(dir_, keys)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", required=True, help="earlier checkpoint dir (or its model_and_optim/)")
    ap.add_argument("--b", required=True, help="later checkpoint dir")
    ap.add_argument(
        "--expect-router-equals", help="reference checkpoint whose routers should match"
    )
    args = ap.parse_args()

    a, b = _mo(args.a), _mo(args.b)
    keys = set(get_checkpoint_metadata(a).state_dict_metadata.keys())
    router = sorted(k for k in keys if k.startswith("model.") and k.endswith("router.weight"))
    assert router, "no model router.weight keys found in checkpoint A"

    # Sample a few non-router model params that should change under training.
    model_keys = [k for k in keys if k.startswith("model.") and "router" not in k]
    sample = []
    for needle in ("embeddings.weight", "experts", "att"):
        hit = next((k for k in sorted(model_keys) if needle in k), None)
        if hit and hit not in sample:
            sample.append(hit)
    assert sample, "could not find any non-router model params to sample"

    A = _load(a, router + sample)
    B = _load(b, router + sample)

    # 1. routers frozen: bit-identical across steps.
    moved = [k for k in router if not torch.equal(A[k], B[k])]
    assert not moved, f"FAIL: {len(moved)} router tensor(s) CHANGED between steps, e.g. {moved[:2]}"

    # 2. training happening: at least one sampled non-router weight changed.
    changed = [k for k in sample if not torch.equal(A[k], B[k])]
    assert changed, (
        f"FAIL: none of the sampled non-router weights changed ({sample}); training is not updating "
        "weights"
    )

    print(f"PASS: all {len(router)} routers bit-identical across steps")
    print(f"PASS: {len(changed)}/{len(sample)} sampled non-router weights changed: {changed}")

    if args.expect_router_equals:
        r = _mo(args.expect_router_equals)
        R = _load(r, router)
        mismatch = [k for k in router if not torch.equal(A[k], R[k])]
        assert not mismatch, f"FAIL: routers diverged from grafted init {r}, e.g. {mismatch[:2]}"
        print(f"PASS: routers still equal grafted init {r}")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
