"""Build the router-fixed step-0 init checkpoint for the ``models_routerfixed`` experiment.

Question under test: does the router need to be *learned* during pretraining, or can a
good router (found once) be frozen while the rest of the model trains around it?

This script produces the starting checkpoint for that experiment. It takes a *source*
checkpoint (the fully-trained ``emo_1b14b_50bof130b`` run, step 11921 = 50B tokens) and writes a
new, model-only checkpoint whose weights are:

  - **non-router params**: a *fresh* EMO init, **bit-identical** to the original run's step-0
    init. The init is topology-independent and seed-deterministic -- each weight is drawn from
    a ``torch.Generator`` seeded by the model-level ``init_seed`` (=0 in the config), and
    ``_apply_init`` initialises the full tensor before sharding -- so rebuilding the same
    config single-process reproduces the original init exactly (see
    ``src/olmo_core/nn/transformer/init.py``).
  - **router params** (``*.router.weight``, one flat tensor per MoE layer): the *trained*
    routers grafted in from the source checkpoint.

The downstream training run loads this with ``--load_path=<out>/model_and_optim
--load_trainer_state=false --load_optim_state=false`` (fresh optimizer, step 0) and freezes the
routers via ``--model.freeze_params='[blocks.*.feed_forward_moe.router.*]'``.

Mirrors ``scripts/models_fullextend/add_expert_to_checkpoint.py``: runs single-process on CPU
(plain ``python``, no torchrun / process group), rebuilds the exact model config from the
source ``config.json``, and writes model weights only (``optim=None``).

    python scripts/models_routerfixed/build_step0_routerfixed.py \
        --src-checkpoint models_routerfixed/emo_1b14b_50bof130b/step11921 \
        --out-dir       models_routerfixed/init_routerfixed_step0
"""

import argparse
import json
import logging
import os

import torch

from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    load_keys,
    save_model_and_optim_state,
)
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig
from olmo_core.utils import seed_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROUTER_SUFFIX = "router.weight"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--src-checkpoint",
        required=True,
        help="Source checkpoint dir (contains config.json + model_and_optim/). Routers are "
        "grafted from here; its config.json defines the exact model to rebuild.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Destination dir. Weights are written to <out-dir>/model_and_optim/.",
    )
    ap.add_argument(
        "--init-seed",
        type=int,
        default=12536,
        help="ExperimentConfig.init_seed used by the original run's seed_all(). The actual "
        "weight init is driven by the model-level init_seed in config.json; this just matches "
        "the harness RNG order.",
    )
    args = ap.parse_args()

    src_cfg_path = os.path.join(args.src_checkpoint, "config.json")
    src_mo = os.path.join(args.src_checkpoint, "model_and_optim")
    with open(src_cfg_path) as f:
        cfg = json.load(f)

    # Rebuild the EXACT model config the source run used (qk_norm, num_experts, router class,
    # init_seed, ...). Force the attention backend to torch so init runs on CPU without flash-attn;
    # the backend does not affect any saved weight.
    mc = TransformerConfig.from_dict(cfg["model"])
    assert isinstance(mc.block, TransformerBlockConfig) and mc.block.feed_forward_moe is not None
    mc.block.sequence_mixer.backend = AttentionBackendName.torch

    # Fresh init, byte-exact to the original run's step-0 init (see module docstring).
    seed_all(args.init_seed)
    model = mc.build(init_device="meta")
    model.init_weights(device=torch.device("cpu"))

    param_names = [n for n, _ in model.named_parameters() if n.endswith(ROUTER_SUFFIX)]
    assert param_names, "no router.weight params found in the model"
    params = dict(model.named_parameters())
    # Checkpoint stores model state under a "model." prefix (named_parameters() drops it).
    ckpt_keys = ["model." + n for n in param_names]

    # Sanity: every router key we intend to load actually exists in the source checkpoint.
    src_keys = set(get_checkpoint_metadata(src_mo).state_dict_metadata.keys())
    missing = [k for k in ckpt_keys if k not in src_keys]
    assert not missing, f"router keys missing from source checkpoint: {missing[:3]} ..."
    logger.info(f"Grafting {len(param_names)} trained router tensors from {src_mo}")

    # Snapshot the fresh routers so we can prove the graft actually replaced them.
    fresh = {n: params[n].detach().clone() for n in param_names}

    # load_keys must run non-distributed (plain python); returns full unsharded tensors in key order.
    loaded = list(load_keys(src_mo, ckpt_keys))
    assert len(loaded) == len(param_names)

    with torch.no_grad():
        for n, t in zip(param_names, loaded):
            t = t.reshape(params[n].shape).to(params[n].dtype)
            params[n].copy_(t)

    # Verify: routers now == trained source; routers now != fresh init (graft was a real swap).
    for n, t in zip(param_names, loaded):
        t = t.reshape(params[n].shape).to(params[n].dtype)
        assert torch.equal(params[n].cpu(), t.cpu()), f"router {n} did not take the grafted value"
        assert not torch.equal(
            params[n].cpu(), fresh[n].cpu()
        ), f"router {n} equals the fresh init -- graft was a no-op"
    for n in param_names[:2]:
        logger.info(
            f"  {n}: grafted_norm={params[n].norm().item():.4f} "
            f"fresh_norm={fresh[n].norm().item():.4f}"
        )
    # Non-router params are fresh init by construction (we never load anything but routers into
    # `model`), so there is nothing else to overwrite.

    out_mo = os.path.join(args.out_dir, "model_and_optim")
    os.makedirs(args.out_dir, exist_ok=True)
    save_model_and_optim_state(dir=out_mo, model=model, optim=None, save_overwrite=True)
    logger.info(f"Saved router-fixed step-0 (model-only) checkpoint to {out_mo}")


if __name__ == "__main__":
    main()
