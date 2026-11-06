#!/usr/bin/env python3
"""Router weights and router logits of an olmoe3_275m checkpoint on fixed held-out tokens (user request 2026-11-06).

Hypothesis under test: during disjoint training the squares' routers drift in scale or offset (each square's softmax only sees its own
experts, so a per-token offset shared by a square's experts is invisible to it), and the merged model's routing diverges because the
squares' logits no longer live on one scale. For one checkpoint (start model, joint baseline, merged model, or a single square) this
script runs the first --max-instances instances of the held-out sample and writes, per MoE layer:

  weight_L{l}        (E_l, d_model) float16   the routed-expert router rows (no bias in this family)
  logit_mean_L{l}    (E_l,)  float32          per-expert mean logit over ALL tokens of the pass
  logit_std_L{l}     (E_l,)  float32          per-expert std of the logit over all tokens
  sel_count_L{l}     (E_l,)  int64            tokens that routed to the expert (the router's own top-k)
  logits_L{l}        (n, E_l) float16         the full logit vector at n fixed token positions (same positions for every checkpoint,
                                              so merged / baseline / square can be compared token by token)
  tok_L{l}           (4,) float64             means over all tokens of: per-token logit std across experts, max logit, logsumexp,
                                              softmax mass of the router's top-k
  sample_pos         (n, 2) int32             (instance, position) of the sampled tokens

plus meta.json (checkpoint, per-layer expert counts and top-k, tokens, mean CE). Expert indices are the checkpoint's own (a square's
layer has only its group's experts, in groups.json order); the analysis maps them back through groups.json.

  PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_routing/router_logits.py --checkpoint <step dir> \
      --instances sparse_experts/olmoe3_routing/sample_8k_300b.npz --out-dir <dir> [--max-instances 64] [--batch-size 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_routing import MOE_LAYERS, build_model, log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--instances", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-instances", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sample-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-first-ce", type=float, default=4.5, help="guard against mis-loaded weights")
    a = ap.parse_args()
    if (a.out_dir / "meta.json").exists():
        log(f"{a.out_dir}: done already"); return
    from olmo_core.data.utils import get_labels

    device = torch.device("cuda")
    tokens = np.load(a.instances)["tokens"][: a.max_instances]
    N, S = tokens.shape
    rng = np.random.default_rng(a.seed)   # the SAME positions for every checkpoint (depends only on N, S, n and the seed)
    flat = np.sort(rng.choice(N * S, size=min(a.sample_tokens, N * S), replace=False))
    pos = np.stack([flat // S, flat % S], axis=1).astype(np.int32)
    model, routers, mcfg = build_model(a.checkpoint, device, None, False)
    E = {l: int(routers[l].num_experts) for l in MOE_LAYERS}; K = {l: int(routers[l].top_k) for l in MOE_LAYERS}
    log(f"{a.checkpoint}: experts {E}, top_k {K}; {N} instances x {S} tokens, {len(flat)} sampled positions")
    s1 = {l: torch.zeros(E[l], dtype=torch.float64, device=device) for l in MOE_LAYERS}; s2 = {l: torch.zeros(E[l], dtype=torch.float64, device=device) for l in MOE_LAYERS}
    cnt = {l: torch.zeros(E[l], dtype=torch.int64, device=device) for l in MOE_LAYERS}; tok = {l: torch.zeros(4, dtype=torch.float64, device=device) for l in MOE_LAYERS}
    samples = {l: [] for l in MOE_LAYERS}; cur = {}

    def make_hook(l):
        def hook(mod, inp, out):
            with torch.no_grad():
                lg = mod.get_expert_logits(inp[0].float()).float()                      # (B, S, E_l)
                f = lg.reshape(-1, lg.shape[-1]).double()
                s1[l] += f.sum(0); s2[l] += (f * f).sum(0)
                idx = out[1].reshape(-1)
                cnt[l] += torch.bincount(idx, minlength=E[l])
                p = lg.softmax(-1)
                tok[l] += torch.stack([lg.std(-1).sum(), lg.max(-1).values.sum(), torch.logsumexp(lg, -1).sum(), p.gather(-1, out[1].long()).sum(-1).sum()]).double()
                bi, bp = cur["sel"]
                if len(bi):
                    samples[l].append(lg[bi, bp].to(torch.float16).cpu())
        return hook

    hooks = [routers[l].register_forward_hook(make_hook(l)) for l in MOE_LAYERS]
    ce_sum, ce_n, t0 = 0.0, 0, time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, N, a.batch_size):
            ids = torch.from_numpy(tokens[b0 : b0 + a.batch_size].astype(np.int64)).to(device)
            m = (pos[:, 0] >= b0) & (pos[:, 0] < b0 + ids.shape[0])
            cur["sel"] = (torch.from_numpy(pos[m, 0] - b0).long().to(device), torch.from_numpy(pos[m, 1]).long().to(device))
            out = model(ids, labels=get_labels({"input_ids": ids}), loss_reduction="none", return_logits=False)
            loss = out.ce_loss if hasattr(out, "ce_loss") else out
            ce_sum += float(loss.float().sum()); ce_n += loss.numel()
            if b0 == 0:
                first = ce_sum / ce_n; log(f"first batch CE {first:.3f}")
                assert first < a.max_first_ce, f"first-batch CE {first:.3f}: weights mis-loaded?"
    for h in hooks: h.remove()
    T = N * S; arrays = {"sample_pos": pos}
    for l in MOE_LAYERS:
        mean = (s1[l] / T); var = (s2[l] / T - mean * mean).clamp_min(0)
        arrays[f"weight_L{l}"] = routers[l].weight.detach().float().reshape(E[l], -1).to(torch.float16).cpu().numpy()
        arrays[f"logit_mean_L{l}"] = mean.float().cpu().numpy(); arrays[f"logit_std_L{l}"] = var.sqrt().float().cpu().numpy()
        arrays[f"sel_count_L{l}"] = cnt[l].cpu().numpy(); arrays[f"tok_L{l}"] = (tok[l] / T).cpu().numpy()
        arrays[f"logits_L{l}"] = torch.cat(samples[l]).numpy()
        assert arrays[f"logits_L{l}"].shape == (len(flat), E[l]), arrays[f"logits_L{l}"].shape
    a.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(a.out_dir / "router_logits.npz", **arrays)
    json.dump({"checkpoint": str(a.checkpoint), "E_layer": E, "top_k": K, "n_instances": N, "n_tokens": T, "n_sample": int(len(flat)), "seed": a.seed,
               "mean_ce": ce_sum / ce_n, "seconds": round(time.time() - t0, 1)}, open(a.out_dir / "meta.json", "w"))
    log(f"done: mean CE {ce_sum / ce_n:.4f}, {time.time() - t0:.0f}s -> {a.out_dir}")


if __name__ == "__main__":
    main()
