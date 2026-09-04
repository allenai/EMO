#!/usr/bin/env python3
"""
Expert co-activation extraction for an EMO (randpool router) OLMo-core checkpoint.

Runs the model natively (no HF conversion) over a packed document set and records, per layer,
which experts each token was routed to. Accumulates:

  C_tok[src, l, i, j]  #tokens (of source src) at which experts i and j are both in the top-k
                       (diagonal = per-expert token usage)
  C_doc[src, l, i, j]  #docs (of source src) in which both i and j are active at least once
  topk[t, l, :]        raw routed expert indices per token (int16), + token -> (doc, pos) maps

Routing is whatever the model does at eval: per token, top-(k - n_shared) of the standard
experts inside the document's expert pool + the shared expert(s) (always the last indices).
`--eval-pool N` pins the randpool router's eval pool to N experts (document-level top-N by
doc-summed router prob, then per-token top-k inside it); `config` keeps the checkpoint's value
(`sparse_8of512_10b`: 512 = full routing). Several pools can be run in one job (`a,b,...`);
the model is loaded once.

Packing mirrors training: docs are EOS-terminated and packed whole into `--seq-len` sequences
(first-fit-decreasing), remaining slots padded with EOS; `doc_lens` are passed so attention is
intra-document and the router sees per-document `seg_id`. EOS/pad tokens are excluded from all
statistics (an EOS is routed as part of the *following* document per `segment_ids_from_eos`).

Multi-GPU: run under torchrun; rank r takes docs with doc_id % WORLD_SIZE == r, writes
`rank{r}/`, and rank 0 merges once every rank has written its `DONE` flag (no process group
needed). Idempotent per (pool, rank): existing `DONE` -> skip.

  torchrun --nproc-per-node=8 scripts/sparse_experts/coactivation/extract_coactivation.py run \\
      --checkpoint sparse_experts/sparse_8of512_10b/step2384 \\
      --docs sparse_experts/coactivation/docs_40k/docs.jsonl.gz \\
      --out-dir sparse_experts/coactivation/sparse_8of512_10b_step2384 --eval-pools config,64

`--selftest` checks the accumulation kernels against a naive loop on random data (no model).
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

EOS = 100257


# --------------------------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------------------------
def pack_docs(docs: List[dict], seq_len: int):
    """First-fit-decreasing packing of EOS-terminated docs into seq_len rows.

    Returns (tokens int64 (R, S), doc_id int32 (R, S) with -1 for pad, valid bool (R, S)).
    """
    items = []
    for d in docs:
        toks = list(d["token_ids"][: seq_len - 1]) + [EOS]
        items.append((d["doc_id"], toks))
    items.sort(key=lambda x: -len(x[1]))
    bins: List[List[tuple]] = []
    free: List[int] = []
    for doc_id, toks in items:
        n = len(toks)
        placed = False
        for b, f in enumerate(free):
            if f >= n:
                bins[b].append((doc_id, toks))
                free[b] -= n
                placed = True
                break
        if not placed:
            bins.append([(doc_id, toks)])
            free.append(seq_len - n)
    R = len(bins)
    tokens = np.full((R, seq_len), EOS, dtype=np.int64)
    doc_of = np.full((R, seq_len), -1, dtype=np.int32)
    valid = np.zeros((R, seq_len), dtype=bool)
    for r, b in enumerate(bins):
        pos = 0
        for doc_id, toks in b:
            n = len(toks)
            tokens[r, pos : pos + n] = toks
            doc_of[r, pos : pos + n - 1] = doc_id  # exclude the terminating EOS
            valid[r, pos : pos + n - 1] = True
            pos += n
    return tokens, doc_of, valid


# --------------------------------------------------------------------------------------------
# accumulation
# --------------------------------------------------------------------------------------------
class Accumulator:
    def __init__(self, num_sources: int, num_layers: int, num_experts: int, device):
        shape = (num_sources, num_layers, num_experts, num_experts)
        self.c_tok = torch.zeros(shape, dtype=torch.int64, device=device)
        self.c_doc = torch.zeros(shape, dtype=torch.int64, device=device)
        self.n_tok = torch.zeros(num_sources, dtype=torch.int64, device=device)
        self.n_doc = torch.zeros(num_sources, dtype=torch.int64, device=device)
        self.E = num_experts
        self.L = num_layers

    @torch.no_grad()
    def add(self, routed: torch.Tensor, src: torch.Tensor, doc: torch.Tensor):
        """routed: (T, L, k) long expert ids of VALID tokens; src: (T,) source idx; doc: (T,) doc ids
        (each doc belongs to exactly one source)."""
        T = routed.shape[0]
        if T == 0:
            return
        # one-hot membership (L, T, E) as float32 -> exact integer matmuls at batch scale
        onehot = torch.zeros(self.L, T, self.E, dtype=torch.float32, device=routed.device)
        onehot.scatter_(2, routed.transpose(0, 1), 1.0)
        doc_u, doc_local = torch.unique(doc, return_inverse=True)
        D = doc_u.numel()
        # doc-level membership (L, D, E): any token of the doc used the expert
        dm = torch.zeros(self.L, D, self.E, dtype=torch.float32, device=routed.device)
        dm.index_reduce_(1, doc_local, onehot, "amax", include_self=True)
        doc_src = torch.zeros(D, dtype=torch.long, device=routed.device)
        doc_src.scatter_(0, doc_local, src)
        for s in torch.unique(src).tolist():
            tm = src == s
            a = onehot[:, tm]  # (L, T_s, E)
            self.c_tok[s] += torch.bmm(a.transpose(1, 2), a).long()
            self.n_tok[s] += int(tm.sum())
            dmask = doc_src == s
            b = dm[:, dmask]
            self.c_doc[s] += torch.bmm(b.transpose(1, 2), b).long()
            self.n_doc[s] += int(dmask.sum())


def selftest():
    torch.manual_seed(0)
    L, E, k, T = 3, 16, 4, 500
    routed = torch.stack([torch.randperm(E)[:k] for _ in range(T * L)]).view(T, L, k).cuda()
    doc = torch.randint(0, 20, (T,)).cuda()
    src_of_doc = torch.randint(0, 2, (20,)).cuda()
    src = src_of_doc[doc]
    acc = Accumulator(2, L, E, "cuda")
    # two chunks split BY DOCUMENT (a doc never spans batches in the real run: docs are whole
    # within a packed row)
    first = doc < 10
    acc.add(routed[first], src[first], doc[first])
    acc.add(routed[~first], src[~first], doc[~first])
    # naive
    c_tok = torch.zeros(2, L, E, E, dtype=torch.long)
    c_doc = torch.zeros(2, L, E, E, dtype=torch.long)
    docsets = {}
    for t in range(T):
        s = int(src[t])
        for li in range(L):
            ids = routed[t, li].tolist()
            for i in ids:
                for j in ids:
                    c_tok[s, li, i, j] += 1
            docsets.setdefault((int(doc[t]), li), set()).update(ids)
    for (d, li), ids in docsets.items():
        s = int(src_of_doc[d])
        for i in ids:
            for j in ids:
                c_doc[s, li, i, j] += 1
    assert torch.equal(acc.c_tok.cpu(), c_tok), "token-level mismatch"
    assert torch.equal(acc.c_doc.cpu(), c_doc), "doc-level mismatch"
    assert acc.n_tok.sum().item() == T and acc.n_doc.sum().item() == 20
    # packing
    docs = [
        dict(doc_id=i, token_ids=list(range(1, 1 + n)))
        for i, n in enumerate([5, 3000, 4100, 7, 1000])
    ]
    tok, doc_of, valid = pack_docs(docs, 4096)
    assert tok.shape[1] == 4096 and (tok[~valid] == EOS).all()
    assert valid.sum() == sum(min(len(d["token_ids"]), 4095) for d in docs)
    assert set(np.unique(doc_of[valid]).tolist()) == {0, 1, 2, 3, 4}
    print("selftest OK: accumulation (token + doc level, 2 sources, 2 chunks) and packing")


# --------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------
class _NoHead(nn.Module):
    def forward(self, h, **kwargs):  # skip the 100k-vocab logits; we only need routing
        return h


def build_model(checkpoint: Path, device, model_dtype):
    # config.json at the step-dir level is the experiment config (same recipe as
    # scripts/modular_extension/eval_k32cpt_ce.py; olmo_core.nn.hf.load_config is avoided on purpose).
    from olmo_core.nn.transformer import TransformerConfig

    tcfg = json.load(open(checkpoint / "config.json"))["model"]
    model = TransformerConfig.from_dict(tcfg).build(init_device="meta")
    model.to_empty(device=device)
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    with TemporaryDirectory() as wd:
        load_model_and_optim_state(str(checkpoint / "model_and_optim"), model, work_dir=wd)
    if model_dtype != torch.float32:
        model.to(model_dtype)
    model.lm_head = _NoHead()
    model.eval()
    routers = [blk.feed_forward_moe.router for blk in model.blocks.values()]
    return model, routers, tcfg


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def run_pass(args, model, routers, tcfg, pool_label: str, pool_value, docs, sources, device, log):
    from olmo_core.data.utils import get_document_lengths

    out = Path(args.out_dir) / f"pool_{pool_label}"
    rank_dir = out / f"rank{args.rank}"
    if (rank_dir / "DONE").exists():
        log(f"[pool {pool_label}] rank {args.rank} already done, skipping")
        return out
    rank_dir.mkdir(parents=True, exist_ok=True)

    moe = tcfg["block"]["feed_forward_moe"]
    E = moe["num_experts"]
    n_shared = routers[0].num_shared_experts
    k_routed = routers[0].top_k - n_shared
    L = len(routers)
    for r in routers:
        r.eval_document_expert_pool = (
            pool_value if pool_value is not None else r.eval_document_expert_pool
        )
    log(
        f"[pool {pool_label}] eval_document_expert_pool={routers[0].eval_document_expert_pool} "
        f"E={E} k_routed={k_routed} n_shared={n_shared} L={L}"
    )

    captured: Dict[int, torch.Tensor] = {}
    hooks = []
    for li, r in enumerate(routers):

        def _hook(mod, inp, outp, li=li):
            captured[li] = outp[1]  # expert_indices (B, S, top_k), shared appended last

        hooks.append(r.register_forward_hook(_hook))

    tokens, doc_of, valid = pack_docs(docs, args.seq_len)
    src_of_doc = {d["doc_id"]: sources.index(d["source"]) for d in docs}
    src_lut = torch.full((max(src_of_doc) + 1,), -1, dtype=torch.long)
    for d, s in src_of_doc.items():
        src_lut[d] = s
    src_lut = src_lut.to(device)
    R = tokens.shape[0]
    n_valid = int(valid.sum())
    log(
        f"[pool {pool_label}] rank {args.rank}: {len(docs)} docs -> {R} rows of {args.seq_len}, "
        f"{n_valid} valid tokens ({n_valid / (R * args.seq_len):.1%} fill)"
    )

    acc = Accumulator(len(sources), L, E, device)
    topk_out = np.lib.format.open_memmap(
        rank_dir / "topk.npy", mode="w+", dtype=np.int16, shape=(n_valid, L, k_routed)
    )
    tok_doc = np.empty(n_valid, dtype=np.int32)
    tok_pos = np.empty(n_valid, dtype=np.int16)
    write_ptr = 0
    t0 = time.time()
    tokens_t = torch.from_numpy(tokens)
    with torch.no_grad():
        for lo in range(0, R, args.batch_size):
            batch = tokens_t[lo : lo + args.batch_size].to(device)
            B = batch.shape[0]
            doc_lens = [get_document_lengths(row, EOS) for row in tokens_t[lo : lo + B]]
            max_docs = max(len(d) for d in doc_lens)
            dl = torch.zeros(B, max_docs, dtype=torch.int32)
            for i, d in enumerate(doc_lens):
                dl[i, : len(d)] = d
            mdl = torch.max(dl, dim=-1).values.tolist()
            captured.clear()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(input_ids=batch, doc_lens=dl.to(device), max_doc_lens=mdl)
            assert len(captured) == L, f"captured {len(captured)} of {L} routers"
            idx = torch.stack([captured[li] for li in range(L)], dim=2)  # (B, S, L, top_k)
            if n_shared:
                shared_ok = (
                    idx[..., k_routed:] == torch.arange(E - n_shared, E, device=device)
                ).all()
                assert bool(shared_ok), "shared expert(s) not in the trailing top_k slots"
            routed = idx[..., :k_routed]  # (B, S, L, k_routed)
            v = torch.from_numpy(valid[lo : lo + B]).to(device)
            dof = torch.from_numpy(doc_of[lo : lo + B]).to(device).long()
            rv = routed[v]  # (T, L, k)
            dv = dof[v]
            sv = src_lut[dv]
            acc.add(rv, sv, dv)
            T = rv.shape[0]
            topk_out[write_ptr : write_ptr + T] = rv.to(torch.int16).cpu().numpy()
            tok_doc[write_ptr : write_ptr + T] = dv.cpu().numpy()
            pos = torch.arange(args.seq_len, device=device).expand(B, -1)
            tok_pos[write_ptr : write_ptr + T] = pos[v].to(torch.int16).cpu().numpy()
            write_ptr += T
            if (lo // args.batch_size) % args.log_every == 0:
                el = time.time() - t0
                log(
                    f"[pool {pool_label}] rank {args.rank} rows {lo + B}/{R} tokens {write_ptr} "
                    f"{write_ptr / max(el, 1e-6):,.0f} tok/s elapsed {el:.0f}s"
                )
    assert write_ptr == n_valid, (write_ptr, n_valid)
    for h in hooks:
        h.remove()
    topk_out.flush()
    del topk_out
    np.save(rank_dir / "tok_doc.npy", tok_doc)
    np.save(rank_dir / "tok_pos.npy", tok_pos)
    np.savez(
        rank_dir / "counts.npz",
        c_tok=acc.c_tok.cpu().numpy(),
        c_doc=acc.c_doc.cpu().numpy(),
        n_tok=acc.n_tok.cpu().numpy(),
        n_doc=acc.n_doc.cpu().numpy(),
    )
    json.dump(
        dict(
            rank=args.rank,
            world=args.world,
            n_valid=n_valid,
            rows=R,
            num_docs=len(docs),
            elapsed_s=time.time() - t0,
            eval_document_expert_pool=routers[0].eval_document_expert_pool,
        ),
        open(rank_dir / "info.json", "w"),
        indent=2,
    )
    (rank_dir / "DONE").touch()
    log(f"[pool {pool_label}] rank {args.rank} DONE in {time.time() - t0:.0f}s")
    return out


def merge(out: Path, world: int, sources: List[str], meta: dict, log):
    if (out / "MERGED").exists():
        log(f"{out} already merged")
        return
    ranks = [out / f"rank{r}" for r in range(world)]
    while not all((r / "DONE").exists() for r in ranks):
        time.sleep(20)
    c_tok = c_doc = n_tok = n_doc = None
    infos = []
    for r in ranks:
        z = np.load(r / "counts.npz")
        c_tok = z["c_tok"] if c_tok is None else c_tok + z["c_tok"]
        c_doc = z["c_doc"] if c_doc is None else c_doc + z["c_doc"]
        n_tok = z["n_tok"] if n_tok is None else n_tok + z["n_tok"]
        n_doc = z["n_doc"] if n_doc is None else n_doc + z["n_doc"]
        infos.append(json.load(open(r / "info.json")))
    np.savez(out / "counts.npz", c_tok=c_tok, c_doc=c_doc, n_tok=n_tok, n_doc=n_doc)
    N = int(n_tok.sum())
    shp = np.load(ranks[0] / "topk.npy", mmap_mode="r").shape
    topk = np.lib.format.open_memmap(
        out / "topk.npy", mode="w+", dtype=np.int16, shape=(N,) + shp[1:]
    )
    tok_doc = np.empty(N, dtype=np.int32)
    tok_pos = np.empty(N, dtype=np.int16)
    p = 0
    for r in ranks:
        a = np.load(r / "topk.npy", mmap_mode="r")
        n = a.shape[0]
        topk[p : p + n] = a
        tok_doc[p : p + n] = np.load(r / "tok_doc.npy")
        tok_pos[p : p + n] = np.load(r / "tok_pos.npy")
        p += n
    assert p == N
    topk.flush()
    del topk
    np.save(out / "tok_doc.npy", tok_doc)
    np.save(out / "tok_pos.npy", tok_pos)
    json.dump(
        dict(
            sources=sources,
            n_tok=n_tok.tolist(),
            n_doc=n_doc.tolist(),
            num_tokens=N,
            ranks=infos,
            **meta,
        ),
        open(out / "info.json", "w"),
        indent=2,
    )
    (out / "MERGED").touch()
    log(f"merged {out}: {N:,} tokens, per-source tokens {dict(zip(sources, n_tok.tolist()))}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_name", nargs="?", default="coactivation")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--docs", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--eval-pools", default="config", help="comma list: 'config' and/or ints")
    p.add_argument("--max-docs", type=int, default=None, help="use only doc_id < max_docs (pilot)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--model-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--selftest", action="store_true")
    args, _unknown = p.parse_known_args()  # tolerate --data-root from launch_common
    if args.selftest:
        selftest()
        return
    assert args.checkpoint and args.docs and args.out_dir
    args.rank = int(os.environ.get("RANK", 0))
    args.world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    all_docs = []
    with gzip.open(args.docs, "rt") as f:
        for line in f:
            d = json.loads(line)
            if args.max_docs is not None and d["doc_id"] >= args.max_docs:
                continue
            all_docs.append(d)
    sources = sorted({d["source"] for d in all_docs})
    my_docs = [d for d in all_docs if d["doc_id"] % args.world == args.rank]
    log(f"rank {args.rank}/{args.world}: {len(my_docs)}/{len(all_docs)} docs, sources={sources}")

    t0 = time.time()
    model, routers, tcfg = build_model(args.checkpoint, device, getattr(torch, args.model_dtype))
    log(
        f"model loaded in {time.time() - t0:.0f}s; {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B params"
    )

    pools = []
    for s in args.eval_pools.split(","):
        pools.append(("config", None) if s == "config" else (s, int(s)))
    meta = dict(
        checkpoint=str(args.checkpoint),
        docs=str(args.docs),
        max_docs=args.max_docs,
        seq_len=args.seq_len,
        model_dtype=args.model_dtype,
        batch_size=args.batch_size,
    )
    for label, value in pools:
        out = run_pass(args, model, routers, tcfg, label, value, my_docs, sources, device, log)
        if args.rank == 0:
            merge(out, args.world, sources, dict(meta, eval_pool=label), log)


if __name__ == "__main__":
    main()
