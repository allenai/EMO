#!/usr/bin/env python3
"""Router statistics of an olmoe3_275m checkpoint under prefix-restricted document pools.

Runs the pinned-OLMo-core model natively on raw 8192-token training-stream instances (see
extract_stream.py / sample_instances.py) with a chosen routing restriction and accumulates, per MoE
layer, everything the analysis needs WITHOUT storing per-token records for the whole pass:

  usage[l, e]                 tokens routed to expert e (top-16 slots)
  coact_tok[l, e, f]          token-level co-activation counts (both in a token's top-16), sym., diag=usage
  coact_doc[l, e, f]          document-level co-activation (both used at least once in a document)
  cross[l, m, e, f]           tokens with expert e at layer l AND expert f at layer m (all layer pairs)
  doc_usage[d, l, e]          per-document routed-token counts (int32)
  doc_scores[d, l, e]         per-document summed softmax router scores (fp32) -> doc pool rankings
  doc_stats[d, l, :]          per-document mean router entropy, mean top-16 mass, mean top-1 prob
  doc_ce[d], doc_len[d]       per-document mean CE (next-token loss) and token count
  raw/topk.npy                top-16 expert indices for the first --raw-instances instances only

Restriction (--restrict): 'none' or 'A-B:P' = apply a document pool of size P to MoE layers A..B
(inclusive, 1-based block indices; layers 1..9 are MoE) and leave the others unrestricted. The pool
rule is the EMO one for BOTH models: per document, sum softmax scores over the document's tokens,
keep the top-P experts, route top-16 within them. EMO checkpoints implement it natively
(eval_document_expert_pool per layer); standard checkpoints get a PoolMaskedRouter that reproduces
EmoRouterV2's selection exactly (self-tested, --selftest).

Sharding: --rank/--world split the instance list; merge_routing.py sums the shards.

Usage (Beaker 1-GPU, PYTHONPATH=external/OLMo-core/src):
  python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint <step dir> \
      --instances sparse_experts/olmoe3_routing/sample_8k.npz --out-dir <dir>/<model>/<cond> \
      --restrict 1-3:64 [--max-instances 1000] [--rank 0 --world 4]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

EOS = 100_257
MOE_LAYERS = list(range(1, 10))  # block 0 is dense in this family


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------
# pool-masked router for the STANDARD model: EmoRouterV2's selection with a fixed pool size
# --------------------------------------------------------------------------------------------
def make_pool_masked_router_class():
    import torch.nn.functional as F
    from olmo_core.nn.moe.v2.router import MoERouterV2
    from olmo_core.ops import moe as ops

    class PoolMaskedRouter(MoERouterV2):
        """MoERouterV2 whose top-k is restricted to each document's top-`pool` experts.

        `pool=None` -> the untouched parent forward. Mirrors EmoRouterV2.forward (softmax gating,
        pool_keep_mask on document-summed scores, normalize, restore_weight_scale).
        """

        pool: Optional[int] = None
        requires_segment_ids = True  # makes the model pass token->document ids (from EOS)
        eos_token_id = EOS

        def forward(self, x, scores_only, *, loss_div_factor=None, segment_ids=None):
            if self.pool is None or scores_only:
                return MoERouterV2.forward(self, x, scores_only, loss_div_factor=loss_div_factor)
            assert segment_ids is not None and segment_ids.shape == x.shape[:2]
            logits = self.get_expert_logits(x.float()).float()
            scores = logits.softmax(dim=-1)
            document_scores = ops.doc_sum_scatter(scores, segment_ids)
            keep = ops.pool_keep_mask(document_scores, torch.full_like(segment_ids, self.pool))
            selection_scores = scores.masked_fill(~keep, float("-inf"))
            _, expert_indices = selection_scores.topk(self.top_k, dim=-1)
            expert_weights = scores.gather(-1, expert_indices)
            if self.normalize_expert_weights is not None:
                expert_weights = F.normalize(expert_weights, p=self.normalize_expert_weights, dim=-1)
            if self.restore_weight_scale:
                expert_weights = expert_weights * self.top_k
            if self.expert_weight_scale is not None:
                expert_weights = expert_weights * self.expert_weight_scale
            with torch.no_grad():
                batched_counts = ops.batched_histc(expert_indices, self.num_experts).sum(dim=1)
                counts = batched_counts.sum(dim=0)
            return expert_weights, expert_indices, counts, (scores, logits, counts, batched_counts, loss_div_factor)

    return PoolMaskedRouter


def parse_restrict(spec: str) -> Tuple[List[int], Optional[int]]:
    if spec == "none":
        return [], None
    m = re.fullmatch(r"(\d+)-(\d+):(\d+)", spec)
    if not m:
        raise ValueError(f"--restrict must be 'none' or 'A-B:P', got {spec!r}")
    a, b, p = map(int, m.groups())
    layers = [l for l in range(a, b + 1)]
    if not layers or any(l not in MOE_LAYERS for l in layers):
        raise ValueError(f"layers {layers} not within MoE layers {MOE_LAYERS}")
    return layers, p


# --------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------
def build_model(checkpoint: Path, device, attn_backend: Optional[str], use_cute_kda: bool):
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.transformer import OLMoDDPModelConfig

    mcfg = json.load(open(checkpoint / "config.json"))["model"]
    for blk in [mcfg["block"], *mcfg.get("block_overrides", {}).values()]:
        sm = blk.get("sequence_mixer") or {}
        if attn_backend and "backend" in sm:
            sm["backend"] = attn_backend
        if "use_cute_kernel" in sm:
            sm["use_cute_kernel"] = use_cute_kda
    model = OLMoDDPModelConfig.from_dict(mcfg).build(init_device="meta")
    model.to(torch.bfloat16)  # fwd precision used in training; DCP casts the fp32 shards on load
    model.to_empty(device=device)
    # OLMoDDP checkpoints keep the weights as flattened fp32 optimizer `module.<name>.main` tensors;
    # the pinned HF converter's loader handles that layout (asserts every parameter is found and
    # numel matches). Falls back to the plain DCP loader for conventional checkpoints.
    from olmo_core.nn.hf.convert_checkpoint import _load_ddp_optimizer_model_state as load_ddp_ckpt
    with TemporaryDirectory() as wd:
        sd = load_ddp_ckpt(str(checkpoint / "model_and_optim"), model, work_dir=wd, return_state_dict=False)
        if sd is None:
            load_model_and_optim_state(str(checkpoint / "model_and_optim"), model, work_dir=wd)
    model.eval()
    routers = {}
    for name, blk in model.blocks.items():
        r = getattr(blk, "routed_experts_router", None)
        if r is not None:
            routers[int(name)] = r
    assert sorted(routers) == MOE_LAYERS, sorted(routers)
    return model, routers, mcfg


def apply_restriction(routers: Dict[int, torch.nn.Module], layers: List[int], pool: Optional[int], num_experts: int):
    from olmo_core.nn.moe.v2.emo_router import EmoRouterV2

    PoolMasked = make_pool_masked_router_class()
    kinds = {}
    for l, r in routers.items():
        p = pool if l in layers else None
        if isinstance(r, EmoRouterV2):
            # Routers built from the same block config SHARE one EmoRouterConfig object; give this
            # router its own copy before editing, or every layer ends up with the last value set.
            r.emo = copy.deepcopy(r.emo)
            r.emo.eval_document_expert_pool = p if p is not None else num_experts
            r.emo.validate_for_router(num_experts=num_experts, top_k=r.top_k)
            kinds[l] = f"emo(eval_pool={r.emo.eval_pool_size()})"
        else:
            r.__class__ = PoolMasked
            r.pool = p
            kinds[l] = f"std(pool={p})"
    return kinds


# --------------------------------------------------------------------------------------------
# accumulators
# --------------------------------------------------------------------------------------------
class Accum:
    def __init__(self, L: int, E: int, k: int, n_doc: int, device):
        self.L, self.E, self.k = L, E, k
        z = lambda *s, dt=torch.int64: torch.zeros(*s, dtype=dt, device=device)
        self.usage = z(L, E)
        self.coact_tok = z(L, E, E)
        self.coact_doc = z(L, E, E)
        self.cross = z(L, L, E, E)
        self.doc_usage = z(n_doc, L, E, dt=torch.int32)
        self.doc_scores = z(n_doc, L, E, dt=torch.float32)
        self.doc_stats = z(n_doc, L, 3, dt=torch.float64)  # sums: entropy, top16 mass, top1 prob
        self.doc_ce = z(n_doc, dt=torch.float64)
        self.doc_len = z(n_doc, dt=torch.int64)
        self.doc_ce_len = z(n_doc, dt=torch.int64)
        self.n_tok = 0

    def add_batch(self, idx: torch.Tensor, scores: List[torch.Tensor], doc: torch.Tensor, loss: Optional[torch.Tensor], loss_doc: Optional[torch.Tensor]):
        """idx: (T, L, k) long; scores[l]: (T, E) fp32 softmax; doc: (T,) global doc id."""
        T = idx.shape[0]; L, E, k = self.L, self.E, self.k
        self.n_tok += T
        onehot_l = []
        for l in range(L):
            oh = torch.zeros(T, E, dtype=torch.float32, device=idx.device)
            oh.scatter_(1, idx[:, l], 1.0)  # (T, E) 0/1, k ones per row
            onehot_l.append(oh)
            self.usage[l] += oh.sum(0).long()
            self.coact_tok[l] += (oh.T @ oh).long()
            # per-doc usage / scores
            self.doc_usage[:, l].index_add_(0, doc, oh.int())
            self.doc_scores[:, l].index_add_(0, doc, scores[l])
            p = scores[l]
            ent = -(p * (p + 1e-30).log()).sum(-1)
            mass = p.gather(1, idx[:, l]).sum(-1)
            top1 = p.max(-1).values
            self.doc_stats[:, l].index_add_(0, doc, torch.stack([ent, mass, top1], -1).double())
        for l in range(L):
            for m in range(L):
                self.cross[l, m] += (onehot_l[l].T @ onehot_l[m]).long()
        self.doc_len.index_add_(0, doc, torch.ones_like(doc))
        if loss is not None:
            self.doc_ce.index_add_(0, loss_doc, loss.double())
            self.doc_ce_len.index_add_(0, loss_doc, torch.ones_like(loss_doc))

    def finalize_doc_coact(self):
        # document-level co-activation from per-doc usage (expert used >=1 token in the doc)
        for l in range(self.L):
            u = (self.doc_usage[:, l] > 0).float()  # (n_doc, E)
            self.coact_doc[l] = (u.T @ u).long()

    def save(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "counts.npz", usage=self.usage.cpu().numpy(), coact_tok=self.coact_tok.cpu().numpy(),
                 coact_doc=self.coact_doc.cpu().numpy(), n_tok=self.n_tok)
        np.save(out / "cross.npy", self.cross.cpu().numpy())
        np.save(out / "doc_usage.npy", self.doc_usage.cpu().numpy())
        np.save(out / "doc_scores.npy", self.doc_scores.cpu().numpy())
        np.savez(out / "doc_stats.npz", stats_sum=self.doc_stats.cpu().numpy(), ce_sum=self.doc_ce.cpu().numpy(),
                 ce_len=self.doc_ce_len.cpu().numpy(), doc_len=self.doc_len.cpu().numpy())


# --------------------------------------------------------------------------------------------
# main pass
# --------------------------------------------------------------------------------------------
def doc_table(tokens: np.ndarray):
    """Global doc ids per token: doc = (instance, EOS-segment). Returns (doc_of (N,S) int64, n_doc, doc_inst)."""
    eos = (tokens == EOS).astype(np.int64)
    eos[:, 0] = 0
    seg = eos.cumsum(1)  # same rule as ops.segment_ids_from_eos
    n_seg = seg[:, -1] + 1
    base = np.concatenate([[0], np.cumsum(n_seg)[:-1]])
    doc_of = seg + base[:, None]
    doc_inst = np.repeat(np.arange(len(tokens)), n_seg)
    return doc_of, int(n_seg.sum()), doc_inst


def run(args):
    from olmo_core.data.utils import get_labels

    device = torch.device("cuda")
    layers, pool = parse_restrict(args.restrict)
    z = np.load(args.instances)
    tokens = z["tokens"]
    sel = np.arange(len(tokens))
    if args.max_instances:
        sel = sel[: args.max_instances]
    sel = sel[args.rank :: args.world]
    tokens = tokens[sel]
    meta = {k: z[k][sel] for k in z.files if k != "tokens"}
    N, S = tokens.shape
    doc_of, n_doc, doc_inst = doc_table(tokens)
    log(f"rank {args.rank}/{args.world}: {N} instances, {N*S:,} tokens, {n_doc} docs; restrict={args.restrict}")

    model, routers, mcfg = build_model(args.checkpoint, device, args.attn_backend, args.use_cute_kda)
    E = mcfg["block"]["routed_experts"]["num_experts"]; k = mcfg["block"]["routed_experts_router"]["top_k"]
    L = len(MOE_LAYERS)
    kinds = apply_restriction(routers, layers, pool, E)
    log("routers: " + ", ".join(f"L{l}:{kinds[l]}" for l in MOE_LAYERS))
    for l, r in routers.items():  # read-back check of the effective per-layer pool
        eff = r.emo.eval_pool_size() if hasattr(r, "emo") and r.emo is not None else (r.pool or E)
        want = pool if l in layers else E
        assert eff == want, f"layer {l}: effective pool {eff} != wanted {want}"

    captured: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    def make_hook(l):
        def hook(mod, inp, out):
            x = inp[0]
            with torch.no_grad():
                scores = mod.get_expert_logits(x.float()).float().softmax(-1)
            captured[l] = (out[1], scores)
        return hook
    hooks = [routers[l].register_forward_hook(make_hook(l)) for l in MOE_LAYERS]

    acc = Accum(L, E, k, n_doc, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_n = min(args.raw_instances, N)
    raw = np.lib.format.open_memmap(args.out_dir / "raw_topk.npy", mode="w+", dtype=np.int16, shape=(raw_n, S, L, k)) if raw_n else None
    t0 = time.time(); B = args.batch_size
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, N, B):
            ids = torch.from_numpy(tokens[b0 : b0 + B].astype(np.int64)).to(device)
            labels = get_labels({"input_ids": ids})
            out = model(ids, labels=labels, loss_reduction="none", return_logits=False)
            loss = out.ce_loss if hasattr(out, "ce_loss") else out  # LMOutputWithLoss -> per-token CE (B, S)
            assert loss.shape == ids.shape, loss.shape
            idx = torch.stack([captured[l][0] for l in MOE_LAYERS], dim=2).long()  # (B, S, L, k)
            scores = [captured[l][1] for l in MOE_LAYERS]
            # sanity: restricted layers must route inside a pool of size <= P per document
            doc = torch.from_numpy(doc_of[b0 : b0 + B]).to(device)
            if pool is not None and b0 == 0:
                for l in layers:
                    li = MOE_LAYERS.index(l)
                    for d in torch.unique(doc[0])[:3]:
                        used = torch.unique(idx[0][doc[0] == d][:, li])
                        assert len(used) <= pool, f"layer {l}: doc used {len(used)} experts > pool {pool}"
            Tn = idx.shape[0] * S
            # per-token loss at position t predicts token t+1 -> attribute to the doc of t+1
            loss_doc = torch.roll(doc, shifts=-1, dims=1)
            valid = labels != -100
            acc.add_batch(idx.view(Tn, L, k), [s.view(Tn, E) for s in scores], doc.view(Tn),
                          loss[valid], loss_doc[valid])
            if raw is not None and b0 < raw_n:
                n = min(B, raw_n - b0)
                raw[b0 : b0 + n] = idx[:n].to(torch.int16).cpu().numpy()
            if b0 == 0:
                ce0 = loss[valid].mean().item()
                assert ce0 < args.max_first_ce, f"first-batch CE {ce0:.3f} > {args.max_first_ce}: weights probably mis-loaded"
            if (b0 // B) % args.log_every == 0:
                done = b0 + idx.shape[0]
                el = time.time() - t0
                log(f"{done}/{N} instances, {done*S/el/1e3:.0f}k tok/s, loss {loss[valid].mean().item():.3f}, eta {(N-done)*S/(done*S/el)/60:.1f} min")
    for h in hooks: h.remove()
    acc.finalize_doc_coact()
    acc.save(args.out_dir)
    if raw is not None: raw.flush()
    np.savez(args.out_dir / "docs.npz", doc_inst=doc_inst, sel=sel, **meta)
    json.dump({"checkpoint": str(args.checkpoint), "restrict": args.restrict, "layers": layers, "pool": pool,
               "kinds": kinds, "n_instances": int(N), "n_tokens": int(N * S), "n_docs": n_doc,
               "mean_ce": float(acc.doc_ce.sum() / acc.doc_ce_len.sum()), "E": E, "top_k": k,
               "rank": args.rank, "world": args.world, "raw_instances": raw_n,
               "elapsed_s": time.time() - t0}, open(args.out_dir / "meta.json", "w"), indent=1)
    (args.out_dir / "DONE").touch()
    log(f"done: mean CE {acc.doc_ce.sum()/acc.doc_ce_len.sum():.4f}; saved to {args.out_dir}")


# --------------------------------------------------------------------------------------------
# self-test (CPU): pool-masked standard router == EmoRouterV2; accumulators == brute force
# --------------------------------------------------------------------------------------------
def selftest():
    from olmo_core.config import DType
    from olmo_core.nn.moe import EmoRouterConfig, MoELoadBalancingLossGranularity, MoERouterGatingFunction
    from olmo_core.nn.moe.v2.router import MoERouterConfigV2
    from olmo_core.ops import moe as ops

    torch.manual_seed(0)
    E, k, D, B, S = 64, 4, 32, 2, 96
    common = dict(d_model=D, num_experts=E, top_k=k, bias=False, normalize_expert_weights=1.0,
                  gating_function=MoERouterGatingFunction.softmax, dtype=DType.float32, lb_loss_weight=0.01,
                  lb_loss_granularity=MoELoadBalancingLossGranularity.instance, z_loss_weight=1e-5,
                  restore_weight_scale=True, use_recompute_fp32_cast=False)
    std = MoERouterConfigV2(**common).build(init_device="cpu")
    emo = MoERouterConfigV2(**common, emo=EmoRouterConfig(eos_token_id=EOS, min_document_expert_pool=k,
                            max_document_expert_pool=E, eval_document_expert_pool=16)).build(init_device="cpu")
    emo.load_state_dict(std.state_dict()); std.eval(); emo.eval()
    x = torch.randn(B, S, D)
    ids = torch.randint(0, 1000, (B, S)); ids[0, 30] = EOS; ids[0, 70] = EOS; ids[1, 50] = EOS
    seg = ops.segment_ids_from_eos(ids, EOS)
    # 1) unrestricted: pool-masked std with pool=None == plain std
    w0, i0, *_ = std(x, False)
    std.__class__ = make_pool_masked_router_class(); std.pool = None
    w1, i1, *_ = std(x, False, segment_ids=seg)
    assert torch.equal(i0, i1) and torch.allclose(w0, w1), "pool=None changed routing"
    # 2) pool=16 == EmoRouterV2 eval pool 16
    std.pool = 16
    w2, i2, *_ = std(x, False, segment_ids=seg)
    w3, i3, *_ = emo(x, False, segment_ids=seg)
    assert torch.equal(i2, i3) and torch.allclose(w2, w3), "pool-masked std != EmoRouterV2"
    # 3) pool=E == unrestricted
    std.pool = E
    w4, i4, *_ = std(x, False, segment_ids=seg)
    assert torch.equal(i4, i0), "pool=E should equal unrestricted"
    # 4) every doc uses <= 16 experts under pool 16
    for b in range(B):
        for d in seg[b].unique():
            assert len(i2[b][seg[b] == d].unique()) <= 16
    # 5) accumulators vs brute force
    tok = ids.numpy().astype(np.int32); doc_of, n_doc, _ = doc_table(tok)
    assert np.array_equal(doc_of, seg.numpy() + np.array([[0], [seg[0, -1].item() + 1]])), "doc_table != segment_ids_from_eos"
    L = 2; acc = Accum(L, E, k, n_doc, torch.device("cpu"))
    idx = torch.stack([i2, i0], dim=2)  # (B,S,L,k)
    sc = [torch.rand(B * S, E).softmax(-1) for _ in range(L)]
    doc = torch.from_numpy(doc_of).view(-1)
    acc.add_batch(idx.view(B * S, L, k), sc, doc, None, None); acc.finalize_doc_coact()
    ref_u = np.zeros((L, E)); ref_ct = np.zeros((L, E, E)); ref_x = np.zeros((L, L, E, E)); ref_du = np.zeros((n_doc, L, E))
    I = idx.view(B * S, L, k).numpy(); D_ = doc.numpy()
    for t in range(B * S):
        for l in range(L):
            for e in I[t, l]:
                ref_u[l, e] += 1; ref_du[D_[t], l, e] += 1
                for f in I[t, l]: ref_ct[l, e, f] += 1
            for m in range(L):
                for e in I[t, l]:
                    for f in I[t, m]: ref_x[l, m, e, f] += 1
    assert np.array_equal(acc.usage.numpy(), ref_u) and np.array_equal(acc.coact_tok.numpy(), ref_ct)
    assert np.array_equal(acc.cross.numpy(), ref_x) and np.array_equal(acc.doc_usage.numpy(), ref_du)
    used0 = (ref_du[:, 0] > 0).astype(np.int64)
    ref_dc = used0.T @ used0
    assert np.array_equal(acc.coact_doc[0].numpy(), ref_dc)
    assert np.allclose(acc.doc_scores[:, 0].numpy(), np.stack([sc[0].numpy()[D_ == d].sum(0) for d in range(n_doc)]))
    log("selftest OK: pool-masked router == EmoRouterV2; accumulators == brute force")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--instances", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--restrict", default="none")
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--raw-instances", type=int, default=200)
    ap.add_argument("--attn-backend", default=None, help="override, e.g. flash_3 / torch")
    ap.add_argument("--use-cute-kda", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-first-ce", type=float, default=4.0, help="guard against mis-loaded weights")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    for a in ("checkpoint", "instances", "out_dir"):
        assert getattr(args, a) is not None, f"--{a.replace('_','-')} required"
    if (args.out_dir / "DONE").exists():
        log(f"{args.out_dir} already DONE, skipping"); return
    run(args)


if __name__ == "__main__":
    main()
