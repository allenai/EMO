"""Parameter counts for the sparse_experts arms vs the meta_learning EMO-128e baseline.

Usage: PYTHONPATH=src python scripts/sparse_experts/model_sizes.py
"""
from olmo_core.nn.transformer import TransformerConfig, TransformerBlockConfig
from olmo_core.data import TokenizerConfig
from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexp_randpool_router import (
    MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouterConfig,
)

tok = TokenizerConfig.dolma2()

def cfg(num_experts, hidden):
    c = TransformerConfig.olmoe_1B_7B(vocab_size=tok.padded_vocab_size())
    moe = c.block.feed_forward_moe
    moe.num_experts = num_experts
    moe.hidden_size = hidden
    rk = moe.router.as_dict(exclude_none=True, recurse=False); rk.pop("name")
    rk.update(min_document_expert_pool=8, max_document_expert_pool=num_experts,
              eos_token_id=tok.eos_token_id, num_shared_experts=1, eval_document_expert_pool=num_experts)
    moe.router = MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouterConfig(**rk)
    return c

rows = [("baseline emo128 (meta_learning)", 128, 1024)] + [(f"sparse 8of{n}", n, 256) for n in (128, 256, 512, 1024)]
print(f"{'model':34s} {'experts':>7s} {'hid':>5s} {'top_k':>5s} {'total':>9s} {'active':>9s} {'non-emb':>9s} {'expert(all)':>12s} {'expert/layer':>12s}")
for name, n, h in rows:
    c = cfg(n, h)
    d = c.d_model; L = c.n_layers
    exp_all = 3 * d * h * n * L
    print(f"{name:34s} {n:7d} {h:5d} {c.block.feed_forward_moe.router.top_k:5d} "
          f"{c.num_params/1e9:8.3f}B {c.num_active_params/1e9:8.3f}B {c.num_non_embedding_params/1e9:8.3f}B "
          f"{exp_all/1e9:11.3f}B {3*d*h*n/1e6:11.1f}M")
c = cfg(128, 1024)
print("d_model", c.d_model, "n_layers", c.n_layers, "vocab", tok.padded_vocab_size(), "embed params", 2*tok.padded_vocab_size()*c.d_model/1e6, "M")
