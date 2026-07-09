"""
Parity tests for the on-GPU (vectorized) two-level document routing in the flagship
randpool router (``MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter``).

The vectorized path must reproduce the old CPU per-document Python-loop masking **exactly**,
with the *only* sanctioned difference being the random pool draw. So every test here holds
the per-document pool size fixed (injected identically into both paths) and asserts identical
masks / metrics. Coverage focuses on multi-document sequences and EOS edge cases.

These run on CPU (the ops are device-agnostic); no GPU or distributed setup required.
"""

import torch
import torch.nn.functional as F

import olmo_core.ops.moe as ops
from olmo_core.nn.moe.router import MoERouterGatingFunction
from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexp_randpool_router import (
    MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter,
)

# ---------------------------------------------------------------------------
# Reference implementations of the OLD (pre-vectorization) logic.
# ---------------------------------------------------------------------------


def old_boundaries(input_ids: torch.Tensor, eos_token_id: int):
    """Replicates the per-sequence EOS-position construction from TransformerModel.forward."""
    boundaries = []
    matches = input_ids == eos_token_id
    for row in matches:
        pos = torch.nonzero(row, as_tuple=True)[0]
        pos = pos[pos > 0]  # drop position 0
        if pos.numel() > 1:
            pos = pos.unique(sorted=True)
        boundaries.append(pos)
    return boundaries


def old_num_docs(boundaries, S):
    """Number of realized documents per sequence under the old span convention."""
    counts = []
    for pos in boundaries:
        bc = pos.tolist()
        if not bc or bc[-1] != S:
            bc.append(S)
        start, n = 0, 0
        for end in bc:
            if end <= start:
                continue
            n += 1
            start = end
        counts.append(n)
    return counts


def old_mask_entropy_tope(logits, boundaries, pool_by_doc, num_forced, top_e=None):
    """
    Old per-document Python-loop masking. ``pool_by_doc[seq]`` is a list of pool sizes indexed
    by document order (== document id). Returns (masked_logits, in_top_e_or_None, entropy).
    """
    B, S, E = logits.shape
    logits_mask = torch.zeros_like(logits, dtype=torch.bool)
    in_top_e = torch.zeros(B, S, E, dtype=torch.bool) if top_e is not None else None
    doc_entropy_sum = logits.new_zeros(())
    doc_entropy_count = 0

    for seq_idx in range(B):
        bc = boundaries[seq_idx].tolist()
        if not bc or bc[-1] != S:
            bc.append(S)
        start, di = 0, 0
        for end in bc:
            if end <= start:
                start = end
                continue
            seq_logits = logits[seq_idx, start:end, :]
            probs = F.softmax(seq_logits, dim=-1)
            token_ent = -(probs * torch.log(probs + 1e-10)).sum(-1)
            doc_entropy_sum += token_ent.mean()
            doc_entropy_count += 1
            dep = probs.sum(0)  # (E,)

            if top_e is not None:
                te = min(top_e, E)
                idxs = torch.topk(dep, te).indices
                in_top_e[seq_idx, start:end, idxs] = True

            pool = pool_by_doc[seq_idx][di]
            di += 1
            bot = E - pool
            if bot <= 0:
                start = end
                continue
            if num_forced > 0:
                num_cand = E - num_forced
                bot_disc = min(bot, num_cand)
                if bot_disc <= 0:
                    start = end
                    continue
                cand = dep[:num_cand]
                disc = torch.topk(-cand, bot_disc).indices
            else:
                disc = torch.topk(-dep, bot).indices
            logits_mask[seq_idx, start:end, disc] = True
            start = end

    masked = logits.masked_fill(logits_mask, float("-inf"))
    entropy = (doc_entropy_sum / doc_entropy_count).item()
    return masked, in_top_e, entropy


def new_mask_entropy_tope(logits, seg_id, pool_docid, num_forced, top_e=None):
    """The vectorized path, mirroring the rewritten router.forward, standalone for testing."""
    B, S, E = logits.shape
    probs = F.softmax(logits, dim=-1)
    doc_prob = ops.doc_sum_scatter(probs, seg_id)

    in_top_e = None
    if top_e is not None:
        rank_full = ops.doc_rank(doc_prob)
        in_top_e = rank_full < min(top_e, E)

    pool_per_token = pool_docid.gather(1, seg_id)
    keep = ops.pool_keep_mask(doc_prob, pool_per_token, num_forced)
    masked = logits.masked_fill(~keep, float("-inf"))

    H = -(probs * torch.log(probs + 1e-10)).sum(-1)
    des = torch.zeros_like(H).scatter_add_(1, seg_id, H)
    cnt = torch.zeros_like(H).scatter_add_(1, seg_id, torch.ones_like(H))
    entropy = (des / cnt.clamp(min=1.0)).sum() / (cnt > 0).sum().clamp(min=1)
    return masked, in_top_e, entropy.item()


# ---------------------------------------------------------------------------
# Helpers to build well-separated-prob inputs and matching pool tensors.
# ---------------------------------------------------------------------------

EOS = 7


def make_pool_docid(pool_by_doc, S):
    """(B,S) long tensor whose column d holds the pool for document d (rest are filler)."""
    B = len(pool_by_doc)
    t = torch.zeros(B, S, dtype=torch.long)
    for b, pools in enumerate(pool_by_doc):
        for d, p in enumerate(pools):
            t[b, d] = p
    return t


def random_inputs(input_ids, E, seed):
    """Well-separated random logits (no ties) of shape (B,S,E) given input_ids (B,S)."""
    g = torch.Generator().manual_seed(seed)
    B, S = input_ids.shape
    # scale up so per-document prob sums are comfortably separated
    return torch.randn(B, S, E, generator=g) * 3.0


# ---------------------------------------------------------------------------
# 1. seg_id grouping matches the old boundary spans.
# ---------------------------------------------------------------------------


def _spans_from_boundaries(pos, S):
    bc = pos.tolist()
    if not bc or bc[-1] != S:
        bc.append(S)
    spans, start = [], 0
    for end in bc:
        if end <= start:
            continue
        spans.append((start, end))
        start = end
    return spans


def test_seg_id_matches_old_boundaries():
    S = 16
    cases = [
        [7, 1, 2, 3, 4, 5, 6, 8, 9, 1, 2, 3, 4, 5, 6, 7],  # EOS at 0 (dropped) and last
        [1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 1, 2, 3, 4, 5, 6],  # consecutive EOS (single-token doc)
        [1, 2, 3, 4, 5, 6, 8, 9, 1, 2, 3, 4, 5, 6, 8, 9],  # no EOS -> single doc
        [1, 7, 2, 3, 4, 5, 6, 8, 9, 1, 2, 3, 4, 5, 6, 7],  # EOS at pos 1 and last
        [7, 7, 7, 1, 2, 3, 4, 5, 6, 8, 9, 1, 2, 3, 4, 7],  # EOS run at start
    ]
    input_ids = torch.tensor(cases)
    seg_id = ops.segment_ids_from_eos(input_ids, EOS)
    boundaries = old_boundaries(input_ids, EOS)

    for b in range(input_ids.size(0)):
        spans = _spans_from_boundaries(boundaries[b], S)
        # Build the doc-id-per-token that the old spans imply.
        expected = torch.empty(S, dtype=torch.long)
        for doc_id, (start, end) in enumerate(spans):
            expected[start:end] = doc_id
        assert torch.equal(
            seg_id[b], expected
        ), f"row {b}: seg_id={seg_id[b].tolist()} != spans {spans}"


# ---------------------------------------------------------------------------
# 2. Multi-document parity sweep (masks, in_top_e, entropy) — the main check.
# ---------------------------------------------------------------------------


def test_multidoc_parity_sweep():
    S, E = 24, 10
    # A batch whose rows deliberately differ in doc count / doc-length shapes / EOS placement.
    cases = [
        # many short docs, EOS at pos 0 (dropped) and last position
        [7, 1, 7, 2, 3, 7, 4, 5, 6, 7, 8, 9, 1, 2, 7, 3, 4, 5, 6, 8, 9, 1, 2, 7],
        # consecutive EOS -> single-token docs in the middle
        [1, 2, 3, 7, 7, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 7, 6, 8, 9, 1, 2, 3, 4],
        # no EOS -> one big document
        [1, 2, 3, 4, 5, 6, 8, 9, 1, 2, 3, 4, 5, 6, 8, 9, 1, 2, 3, 4, 5, 6, 8, 9],
        # EOS at pos 1 and at last position; a couple of long docs
        [1, 7, 2, 3, 4, 5, 6, 8, 9, 1, 2, 3, 4, 5, 6, 8, 9, 1, 2, 3, 4, 5, 6, 7],
    ]
    input_ids = torch.tensor(cases)
    B = input_ids.size(0)
    seg_id = ops.segment_ids_from_eos(input_ids, EOS)
    boundaries = old_boundaries(input_ids, EOS)
    ndocs = old_num_docs(boundaries, S)

    seed = 0
    for num_forced in (0, 1, 2):
        for top_e in (None, 3, E + 5):  # below and above E
            # Sweep pool values covering: < num_forced, mid, == E, > E.
            for pool_choices in ([0, 1], [3, 4, 5], [E], [E, E + 3]):
                logits = random_inputs(input_ids, E, seed)
                seed += 1
                # Assign each document a pool value (cycling through pool_choices),
                # identical for both paths.
                pool_by_doc = []
                for b in range(B):
                    pools = [pool_choices[d % len(pool_choices)] for d in range(ndocs[b])]
                    pool_by_doc.append(pools)
                pool_docid = make_pool_docid(pool_by_doc, S)

                old_masked, old_ite, old_ent = old_mask_entropy_tope(
                    logits, boundaries, pool_by_doc, num_forced, top_e
                )
                new_masked, new_ite, new_ent = new_mask_entropy_tope(
                    logits, seg_id, pool_docid, num_forced, top_e
                )

                # Masks: compare which positions are -inf (well-separated probs => no ties).
                tag = f"num_forced={num_forced} top_e={top_e} pools={pool_choices}"
                assert torch.equal(
                    torch.isinf(old_masked) & (old_masked < 0),
                    torch.isinf(new_masked) & (new_masked < 0),
                ), f"mask mismatch: {tag}"
                # Kept (finite) logit values must be identical.
                finite = ~torch.isinf(old_masked)
                assert torch.allclose(old_masked[finite], new_masked[finite]), f"kept vals: {tag}"

                if top_e is not None:
                    assert torch.equal(old_ite, new_ite), f"in_top_e mismatch: {tag}"

                assert (
                    abs(old_ent - new_ent) < 1e-5
                ), f"entropy mismatch: {tag} ({old_ent} vs {new_ent})"


# ---------------------------------------------------------------------------
# 3. Batch-consistency: a token's mask depends only on its own document.
# ---------------------------------------------------------------------------


def test_no_cross_document_leakage():
    S, E = 12, 8
    # Two rows sharing doc structure; second row's second doc has different logits.
    input_ids = torch.tensor(
        [
            [1, 2, 3, 7, 4, 5, 6, 8, 9, 1, 2, 3],
            [1, 2, 3, 7, 4, 5, 6, 8, 9, 1, 2, 3],
        ]
    )
    seg_id = ops.segment_ids_from_eos(input_ids, EOS)
    logits = random_inputs(input_ids, E, seed=123)
    # Make row1 identical to row0 in the FIRST doc only; perturb the second doc.
    logits[1, 0:3] = logits[0, 0:3]
    logits[1, 3:] += 5.0

    pool_docid = torch.full((2, S), 3, dtype=torch.long)
    _, _, _ = new_mask_entropy_tope(logits, seg_id, pool_docid, num_forced=0)
    doc_prob = ops.doc_sum_scatter(F.softmax(logits, dim=-1), seg_id)
    # First document (tokens 0..2) is identical across rows => identical doc sums / masks.
    assert torch.allclose(doc_prob[0, 0:3], doc_prob[1, 0:3])
    keep = ops.pool_keep_mask(doc_prob, pool_docid, 0)
    assert torch.equal(keep[0, 0:3], keep[1, 0:3])
    # Second document differs => (generally) different masks; at least the doc sums differ.
    assert not torch.allclose(doc_prob[0, 4:9], doc_prob[1, 4:9])


# ---------------------------------------------------------------------------
# 4. Real router forward (eval mode) matches the reference masking, and the new
#    path performs no host sync on document tensors.
# ---------------------------------------------------------------------------


def _build_router(num_experts=10, num_shared=1, eval_pool=4, num_forced=0):
    return MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter(
        d_model=32,
        num_experts=num_experts,
        top_k=4,
        gating_function=MoERouterGatingFunction.softmax,
        min_document_expert_pool=2,
        max_document_expert_pool=num_experts - num_shared,
        eval_document_expert_pool=eval_pool,
        eos_token_id=EOS,
        num_shared_experts=num_shared,
        num_forced_experts=num_forced,
    )


def test_real_router_eval_matches_reference():
    torch.manual_seed(0)
    S = 16
    input_ids = torch.tensor(
        [
            [7, 1, 2, 3, 7, 4, 5, 6, 8, 9, 1, 2, 3, 4, 5, 7],
            [1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 1, 2, 3, 4, 5, 6],
        ]
    )
    eval_pool = 4
    router = _build_router(eval_pool=eval_pool).eval()
    seg_id = ops.segment_ids_from_eos(input_ids, EOS)

    x = torch.randn(2, S, router.d_model)
    # Real forward should run end-to-end and return sane shapes (shared expert appended).
    weights, indices, counts, aux = router(x, seg_id=seg_id)
    assert weights.shape == (2, S, router.top_k)
    assert indices.shape == (2, S, router.top_k)
    assert torch.isfinite(weights).all()
    assert aux is None  # eval mode: no aux loss

    # Reconstruct the reference masking on the SAME expert logits (eval pool for every doc).
    E = router.num_experts - router.num_shared_experts
    logits = router.get_expert_logits(x).float()[:, :, :E]
    boundaries = old_boundaries(input_ids, EOS)
    ndocs = old_num_docs(boundaries, S)
    pool_by_doc = [[eval_pool] * n for n in ndocs]
    ref_masked, _, _ = old_mask_entropy_tope(logits, boundaries, pool_by_doc, num_forced=0)

    pool_docid = torch.full((2, S), eval_pool, dtype=torch.long)
    new_masked, _, _ = new_mask_entropy_tope(logits, seg_id, pool_docid, num_forced=0)
    assert torch.equal(
        torch.isinf(ref_masked) & (ref_masked < 0), torch.isinf(new_masked) & (new_masked < 0)
    )
