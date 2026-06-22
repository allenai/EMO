"""Grow an EMO OLMo-core checkpoint by instantiating brand-new expert(s).

The mirror of ``src/scripts/eval/prune_moe_checkpoint.py`` (which *removes* experts):
this *adds* ``--num-new-experts`` real, instantiated experts to a trained
``models_fullextend`` ghost checkpoint so the model can be continually pretrained
with everything frozen except the new expert(s). See the experiment README.

Layout / shared-expert gotcha
-----------------------------
In the randpool router the **shared experts are the last ``num_shared_experts``
indices** (the forward strips ``logits[..., :num_experts - num_shared_experts]``).
So a new expert cannot be appended at the very end (it would be misread as the
shared expert). Instead each new expert is inserted as the **last non-shared
expert**, pushing the shared expert(s) to the end. For 128->129 (1 shared):

    old (128): [ nonshared_0 .. nonshared_126 | shared_127 ]
    new (129): [ nonshared_0 .. nonshared_126 | NEW_127 | shared_128 ]

Initialization
--------------
Each new expert is the **uniform average of the original non-shared experts** --
both the MLP rows (``w1``/``w2``/``w3``) and the router row. This is the natural
"averaged new expert" init and matches the uniform ghost the model was trained
with. (For ``num_new_experts > 1`` the new experts are identical clones of that
average; the default is 1.)

The new config sets ``num_experts += num_new_experts``, and on the router
``num_forced_experts = num_new_experts`` (the new expert is forced into every
document pool so it has a chance to be routed) and ``num_new_experts`` (enables
the token/document activation metric during continual pretraining).

Only model weights are written (``optim=None``); the continual-pretrain run loads
them with a fresh optimizer/schedule.

    python scripts/models_fullextend/add_expert_to_checkpoint.py \
        --checkpoint-path models_fullextend/emo_1b14b_50bof130b_ghost_uniform_always_detachF/step11921 \
        --save-path     models_fullextend/emo_1b14b_50bof130b_ghost_uniform_always_detachF/step11921-plus1 \
        --num-new-experts 1
"""

import argparse
import json
import logging
import os
import shutil

import torch

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def expand_param(src: torch.Tensor, old_E: int, num_shared: int, num_new: int) -> torch.Tensor:
    """Insert ``num_new`` averaged experts as the last non-shared experts.

    Handles both router rows (1-D, viewed as ``(E, d)``) and expert MLP weights
    (2-D, viewed as ``(E, b, cols)``). The new experts are the mean over the
    original non-shared experts; the shared experts are shifted to the end.
    """
    nonshared = old_E - num_shared
    new_E = old_E + num_new
    if src.dim() == 1:
        d = src.shape[0] // old_E
        sm = src.view(old_E, d)
        nm = src.new_zeros(new_E, d)
        nm[:nonshared] = sm[:nonshared]
        nm[nonshared : nonshared + num_new] = sm[:nonshared].mean(dim=0, keepdim=True)
        nm[nonshared + num_new :] = sm[nonshared:]  # shared experts, shifted to end
        return nm.reshape(new_E * d)
    else:
        rows, cols = src.shape
        b = rows // old_E
        sm = src.view(old_E, b, cols)
        nm = src.new_zeros(new_E, b, cols)
        nm[:nonshared] = sm[:nonshared]
        nm[nonshared : nonshared + num_new] = sm[:nonshared].mean(dim=0, keepdim=True)
        nm[nonshared + num_new :] = sm[nonshared:]
        return nm.reshape(new_E * b, cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-path", required=True, help="Source checkpoint dir (contains config.json + model_and_optim/).")
    ap.add_argument("--save-path", required=True, help="Destination checkpoint dir.")
    ap.add_argument("--num-new-experts", type=int, default=1, help="Number of new experts to add.")
    args = ap.parse_args()

    cfg_path = os.path.join(args.checkpoint_path, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Build config (override attention backend to torch so the surgery runs on CPU
    # without flash-attn; the saved config.json keeps the original backend untouched).
    mc = TransformerConfig.from_dict(cfg["model"])
    assert isinstance(mc.block, TransformerBlockConfig) and mc.block.feed_forward_moe is not None
    mc.block.sequence_mixer.backend = AttentionBackendName.torch

    moe = mc.block.feed_forward_moe
    old_E = moe.num_experts
    num_shared = int(getattr(moe.router, "num_shared_experts", 0))
    num_new = args.num_new_experts
    new_E = old_E + num_new
    logger.info(f"experts {old_E} -> {new_E}  (num_shared={num_shared}, num_new={num_new})")

    logger.info(f"Loading weights from {args.checkpoint_path}")
    model = mc.build(init_device="cpu")
    load_model_and_optim_state(dir=os.path.join(args.checkpoint_path, "model_and_optim"), model=model, optim=None)

    # New model config: more experts, new expert forced-routed + activation-tracked.
    new_mc = mc.copy()
    assert isinstance(new_mc.block, TransformerBlockConfig) and new_mc.block.feed_forward_moe is not None
    new_mc.block.feed_forward_moe.num_experts = new_E
    new_mc.block.feed_forward_moe.router.num_forced_experts = num_new
    new_mc.block.feed_forward_moe.router.num_new_experts = num_new
    new_model = new_mc.build(init_device="cpu")

    # Copy/expand every parameter. Non-expert params shape-match and copy verbatim;
    # router rows + expert MLPs are expanded with the averaged new expert(s).
    new_sd = new_model.state_dict()
    written = set()
    with torch.no_grad():
        for name, p in model.named_parameters():
            assert name in new_sd, f"{name} missing from new model"
            np_ = new_sd[name]
            if p.shape == np_.shape:
                np_.copy_(p)
            elif "router" in name or "experts" in name:
                expanded = expand_param(p.data, old_E, num_shared, num_new)
                assert expanded.shape == np_.shape, f"{name}: {tuple(expanded.shape)} != {tuple(np_.shape)}"
                np_.copy_(expanded)
            else:
                raise ValueError(f"shape mismatch for non-router/expert param {name}: {tuple(p.shape)} vs {tuple(np_.shape)}")
            written.add(name)
    missing = set(n for n, _ in new_model.named_parameters()) - written
    assert not missing, f"new params never written: {sorted(missing)[:5]}"
    logger.info(f"Wrote {len(written)} params into the {new_E}-expert model")

    # Save model-only checkpoint + config.json (original dict with the count fields bumped,
    # preserving the original attention backend etc. for downstream conversion/eval).
    os.makedirs(args.save_path, exist_ok=True)
    save_model_and_optim_state(dir=os.path.join(args.save_path, "model_and_optim"), model=new_model, optim=None)

    cfg["model"]["block"]["feed_forward_moe"]["num_experts"] = new_E
    cfg["model"]["block"]["feed_forward_moe"]["router"]["num_forced_experts"] = num_new
    cfg["model"]["block"]["feed_forward_moe"]["router"]["num_new_experts"] = num_new
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    meta_src = os.path.join(args.save_path, "model_and_optim", ".metadata")
    if os.path.exists(meta_src):
        shutil.copy2(meta_src, os.path.join(args.save_path, ".metadata"))
    logger.info(f"Saved {new_E}-expert checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
