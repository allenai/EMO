#!/usr/bin/env python3
"""Per-cluster held-out CE for k=32 CPT checkpoints (one GPU per process).

Evaluates a 64-expert checkpoint on each cluster's held-out token stream, replicating
training-time instance semantics: 4096-token instances, doc lengths from EOS boundaries
(intra-document masking + per-document routing), eval_document_expert_pool forced to
num_experts (route over all experts). CE is the mean next-token cross entropy over all
scored positions, bf16 autocast.

Accepts either a training checkpoint dir (config.json + model_and_optim) or a bf16
weights-only snapshot (.pt from expert_subset_surgery writeback) plus a --config-from
checkpoint for the model config.

Usage (shard clusters across GPUs):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:src python scripts/modular_extension/eval_k32cpt_ce.py \\
      --checkpoint models_v2/emo_64exp_50b_wsd_lr2e-3/step23842 \\
      --clusters 0-7 --out <out>/ce_step23842_shard0.json
"""
import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO / "src"))

EOS = 100257
SEQ_LEN = 4096
TOKENS_ROOT = (REPO / "modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-130B"
               / "k32_cpt_tokens")


def parse_clusters(spec: str):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def build_model(args, device):
    from olmo_core.nn.hf.convert_checkpoint import load_config
    from olmo_core.nn.transformer import TransformerConfig

    cfg_src = args.config_from or args.checkpoint
    experiment_config = load_config(cfg_src)
    tcfg = experiment_config["model"]
    moe = tcfg["block"]["feed_forward_moe"]
    router = moe.get("router") or {}
    if "eval_document_expert_pool" in router:
        router["eval_document_expert_pool"] = moe["num_experts"]
    model = TransformerConfig.from_dict(tcfg).build(init_device="meta")
    model.to_empty(device=device)

    if args.snapshot:
        sd = torch.load(args.snapshot, map_location="cpu")
        missing, unexpected = model.load_state_dict(
            {k: v.to(torch.float32) for k, v in sd.items()}, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
    else:
        from olmo_core.distributed.checkpoint import load_model_and_optim_state

        with TemporaryDirectory() as wd:
            load_model_and_optim_state(str(Path(args.checkpoint) / "model_and_optim"),
                                       model, work_dir=wd)
    model.eval()
    return model


@torch.no_grad()
def eval_cluster(model, heldout_path, device, batch_size, max_tokens):
    from olmo_core.data.utils import get_document_lengths

    stream = np.fromfile(heldout_path, dtype=np.uint32)
    if max_tokens:
        stream = stream[:max_tokens]
    n_inst = len(stream) // SEQ_LEN
    inst = torch.tensor(stream[: n_inst * SEQ_LEN].astype(np.int64)).view(n_inst, SEQ_LEN)

    tot_loss, tot_pos = 0.0, 0
    t0 = time.time()
    for lo in range(0, n_inst, batch_size):
        batch = inst[lo:lo + batch_size].to(device)
        doc_lens = [get_document_lengths(row.cpu(), EOS) for row in batch]
        max_docs = max(len(d) for d in doc_lens)
        dl = torch.zeros(len(doc_lens), max_docs, dtype=torch.int32)
        for i, d in enumerate(doc_lens):
            dl[i, : len(d)] = d
        mdl = torch.max(dl, dim=-1).values.tolist()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=batch, doc_lens=dl.to(device), max_doc_lens=mdl)
            logits = out.logits if hasattr(out, "logits") else out
            loss = F.cross_entropy(
                logits[:, :-1].flatten(0, 1).float(), batch[:, 1:].flatten())
        tot_loss += float(loss) * batch.shape[0] * (SEQ_LEN - 1)
        tot_pos += batch.shape[0] * (SEQ_LEN - 1)
    return tot_loss / tot_pos, tot_pos, time.time() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None, help="step dir (config.json + model_and_optim)")
    p.add_argument("--snapshot", default=None, help="bf16 weights-only .pt (needs --config-from)")
    p.add_argument("--config-from", default=None, help="step dir supplying config.json")
    p.add_argument("--clusters", default="0-31")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens-per-cluster", type=int, default=None)
    args = p.parse_args()
    assert args.checkpoint or (args.snapshot and args.config_from)

    device = torch.device("cuda")
    model = build_model(args, device)

    results = {}
    for c in parse_clusters(args.clusters):
        hp = TOKENS_ROOT / f"cluster{c:02d}" / "heldout.npy"
        ce, n, dt = eval_cluster(model, hp, device, args.batch_size,
                                 args.max_tokens_per_cluster)
        results[str(c)] = {"ce": round(ce, 5), "positions": n, "seconds": round(dt, 1)}
        print(f"cluster {c:2d}: CE {ce:.4f} over {n / 1e6:.1f}M positions ({dt:.0f}s)",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"checkpoint": args.checkpoint or args.snapshot, "results": results}, f,
                  indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
