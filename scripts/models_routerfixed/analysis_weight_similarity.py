"""Weight-space similarity between the models_routerfixed noaux run and its baseline.

Hypothesis: the noaux run froze its router at the baseline's *final* (step-11921) router and
started every other weight from the SAME fresh init as the baseline (init_routerfixed_step0 grafted
ONLY the routers onto the baseline's fresh init). So at step 0 every NON-router tensor -- including
each expert -- is byte-identical across the two runs. Both then train on the same data for 11921
steps; the only difference is the routing trajectory (baseline's router evolved; noaux's was fixed
at the baseline's endpoint). Because gradient descent from an identical init never relabels MLP
neurons, expert i in noaux and expert i in baseline stay index-aligned -- so RAW weight cosine
(no permutation search) is a meaningful "did they converge to the same solution?" probe.

This script compares the two step-11921 checkpoints directly, tensor by tensor:
  - experts: per-(layer, expert) cosine + relative-L2 between the concatenated [w1,w2,w3] of expert
    i in each model. Builds the full (E x E) cross-expert cosine matrix per layer; its DIAGONAL is
    the index-aligned same-expert similarity, its OFF-DIAGONAL is the null (different experts). A
    strong diagonal vs near-zero off-diagonal => the frozen router pinned the experts to the
    baseline's solution.
  - non-expert components (attention w_{q,k,v,out}, the two RMSNorm gains, embeddings, lm_head):
    whole-tensor cosine + relative-L2. These had no router constraint, so they're the control --
    are the experts MORE similar than the attention/other weights?
  - router: cosine (must be ~1.0 by construction; sanity check).

Runs single-process on CPU (plain ``python``). ``load_keys`` reads whole checkpoint shards per call,
so we issue exactly ONE call per checkpoint (all keys at once) -- materialising the full model
(~54 GB fp32 each, ~110 GB for both) rather than re-reading shards per layer.

    python scripts/models_routerfixed/analysis_weight_similarity.py \
        --noaux models_routerfixed/emo_1b14b_50bof130b_routerfixed_noaux/step11921 \
        --baseline models_routerfixed/emo_1b14b_50bof130b/step11921 \
        --output-dir claude_outputs/models_routerfixed/weight_similarity
"""

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _mo(d: str) -> str:
    d = d.rstrip("/")
    return d if d.endswith("model_and_optim") else d + "/model_and_optim"


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else float("nan")


def _relL2(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b|| -- relative change of a (noaux) from b (baseline)."""
    a, b = a.flatten().float(), b.flatten().float()
    nb = b.norm()
    return float((a - b).norm() / nb) if nb > 0 else float("nan")


def _normed_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 1e-12, n, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noaux", required=True, help="noaux checkpoint dir (or its model_and_optim/)")
    ap.add_argument("--baseline", required=True, help="baseline checkpoint dir")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    A = _mo(args.noaux)      # noaux
    B = _mo(args.baseline)   # baseline
    os.makedirs(args.output_dir, exist_ok=True)

    # Compare only model params; the baseline keeps optimizer state for its (trainable) router
    # while noaux (frozen router) does not, so the full key sets differ harmlessly.
    keys = {k for k in get_checkpoint_metadata(A).state_dict_metadata.keys() if k.startswith("model.")}
    keys_b = {k for k in get_checkpoint_metadata(B).state_dict_metadata.keys() if k.startswith("model.")}
    assert keys == keys_b, "checkpoints have different MODEL key sets -- not the same architecture"
    n_layers = 1 + max(int(k.split(".")[2]) for k in keys if k.startswith("model.blocks."))

    comp_suffixes = {
        "attn_q": "attention.w_q.weight",
        "attn_k": "attention.w_k.weight",
        "attn_v": "attention.w_v.weight",
        "attn_out": "attention.w_out.weight",
        "attn_norm": "attention_norm.weight",
        "ffn_norm": "feed_forward_norm.weight",
        "router": "feed_forward_moe.router.weight",
    }
    expert_keys = [
        f"model.blocks.{L}.feed_forward_moe.experts.mlp.{w}"
        for L in range(n_layers) for w in ("w1", "w2", "w3")
    ]
    comp_keys = [f"model.blocks.{L}.{s}" for L in range(n_layers) for s in comp_suffixes.values()]
    glob_keys = ["model.embeddings.weight", "model.lm_head.norm.weight", "model.lm_head.w_out.weight"]
    all_keys = expert_keys + comp_keys + glob_keys

    logger.info(f"{n_layers} blocks; loading {len(all_keys)} tensors from noaux={A}")
    da = dict(zip(all_keys, load_keys(A, all_keys)))
    logger.info(f"loading {len(all_keys)} tensors from baseline={B}")
    db = dict(zip(all_keys, load_keys(B, all_keys)))
    logger.info("both checkpoints loaded; computing similarities")

    # Experts are stored FLAT, stacked over experts: w{1,2,3} have shape
    # (num_experts * hidden_size, d_model). Derive num_experts from the router (numel = E * d_model)
    # so w.reshape(E, -1) yields one full [hidden_size x d_model] weight vector per expert.
    d_model = da["model.blocks.0.feed_forward_moe.experts.mlp.w1"].shape[1]
    n_experts = da["model.blocks.0.feed_forward_moe.router.weight"].numel() // d_model
    logger.info(f"d_model={d_model}, num_experts={n_experts} (expert w stored flat as E*hidden x d_model)")

    # ── Experts: per-layer (E x E) cross-expert cosine matrix; diagonal = same-expert ──────────
    expert_diag_cos, expert_offdiag_cos, expert_relL2 = [], [], []
    heatmaps = {}
    sample_layers = sorted(set(np.linspace(0, n_layers - 1, 4).astype(int).tolist()))
    for L in range(n_layers):
        ek = [f"model.blocks.{L}.feed_forward_moe.experts.mlp.{w}" for w in ("w1", "w2", "w3")]
        E = n_experts
        av = torch.cat([da[k].reshape(E, -1).float() for k in ek], dim=1).numpy()
        bv = torch.cat([db[k].reshape(E, -1).float() for k in ek], dim=1).numpy()
        diff = np.linalg.norm(av - bv, axis=1)
        bnorm = np.linalg.norm(bv, axis=1)
        expert_relL2.append(diff / np.where(bnorm > 0, bnorm, 1.0))
        cos = _normed_rows(av) @ _normed_rows(bv).T  # rows=noaux, cols=baseline
        diag = np.diag(cos).copy()
        expert_diag_cos.append(diag)
        off = cos.copy()
        np.fill_diagonal(off, np.nan)
        expert_offdiag_cos.append(float(np.nanmean(off)))
        if L in sample_layers:
            heatmaps[L] = cos.astype(np.float32)
        logger.info(
            f"  L{L:02d} experts: diag cos med={np.median(diag):.3f} "
            f"offdiag cos mean={expert_offdiag_cos[-1]:.3f} relL2 med={np.median(expert_relL2[-1]):.3f}"
        )
        del av, bv, cos
    expert_diag_cos = np.stack(expert_diag_cos)
    expert_relL2 = np.stack(expert_relL2)
    expert_offdiag_cos = np.array(expert_offdiag_cos)

    # ── Non-expert components: whole-tensor cosine + relative L2, per layer ────────────────────
    comp_cos = {c: [] for c in comp_suffixes}
    comp_relL2 = {c: [] for c in comp_suffixes}
    for L in range(n_layers):
        for c, s in comp_suffixes.items():
            key = f"model.blocks.{L}.{s}"
            comp_cos[c].append(_cos(da[key], db[key]))
            comp_relL2[c].append(_relL2(da[key], db[key]))
    global_cos = {k.replace("model.", ""): _cos(da[k], db[k]) for k in glob_keys}
    global_relL2 = {k.replace("model.", ""): _relL2(da[k], db[k]) for k in glob_keys}

    router_cos = np.array(comp_cos["router"])
    logger.info(f"router cosine: min={router_cos.min():.6f} (must be ~1.0 -- frozen-router sanity)")

    # ── Plots ──────────────────────────────────────────────────────────────────────────────────
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(10, 5))
    med = np.median(expert_diag_cos, axis=1)
    q1 = np.percentile(expert_diag_cos, 25, axis=1)
    q3 = np.percentile(expert_diag_cos, 75, axis=1)
    ax.plot(layers, med, marker="o", color="tab:blue", label="same-expert cos (diag, median ± IQR)")
    ax.fill_between(layers, q1, q3, alpha=0.2, color="tab:blue")
    ax.plot(layers, expert_offdiag_cos, marker="s", color="tab:red",
            label="different-expert cos (off-diag mean, null)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert weight cosine (noaux vs baseline)")
    ax.set_title("Index-aligned expert weight similarity")
    ax.set_ylim(-0.2, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    p = os.path.join(args.output_dir, "expert_cos_vs_layer.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {p}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(layers, med, marker="o", lw=2.4, color="tab:blue", label="experts (diag, median)")
    palette = {"attn_q": "tab:orange", "attn_k": "tab:green", "attn_v": "tab:purple",
               "attn_out": "tab:brown", "attn_norm": "tab:olive", "ffn_norm": "tab:cyan"}
    for c, color in palette.items():
        ax.plot(layers, comp_cos[c], marker=".", alpha=0.8, color=color, label=c)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Whole-tensor cosine (noaux vs baseline)")
    ax.set_title("Per-component weight similarity: are experts more aligned than attention?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    p = os.path.join(args.output_dir, "component_cos_vs_layer.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {p}")

    n = len(heatmaps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, L in zip(axes, sorted(heatmaps)):
        im = ax.imshow(heatmaps[L], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"Layer {L}")
        ax.set_xlabel("baseline expert")
        ax.set_ylabel("noaux expert")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Cross-expert weight cosine (diagonal = same expert index)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(args.output_dir, "expert_cos_heatmap_layers.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {p}")

    # ── Save raw + summary ───────────────────────────────────────────────────────────────────
    np.savez_compressed(
        os.path.join(args.output_dir, "weight_similarity.npz"),
        expert_diag_cos=expert_diag_cos,
        expert_offdiag_cos=expert_offdiag_cos,
        expert_relL2=expert_relL2,
        **{f"heatmap_L{L}": h for L, h in heatmaps.items()},
    )
    summary = {
        "noaux": A,
        "baseline": B,
        "num_layers": int(n_layers),
        "num_experts": int(expert_diag_cos.shape[1]),
        "experts": {
            "diag_cos_median_overall": float(np.median(expert_diag_cos)),
            "diag_cos_mean_overall": float(np.mean(expert_diag_cos)),
            "offdiag_cos_mean_overall": float(np.mean(expert_offdiag_cos)),
            "relL2_median_overall": float(np.median(expert_relL2)),
            "per_layer": [
                {
                    "layer": int(L),
                    "diag_cos_median": float(np.median(expert_diag_cos[L])),
                    "diag_cos_min": float(np.min(expert_diag_cos[L])),
                    "offdiag_cos_mean": float(expert_offdiag_cos[L]),
                    "relL2_median": float(np.median(expert_relL2[L])),
                }
                for L in range(n_layers)
            ],
        },
        "components_cos_mean_over_layers": {
            **{c: float(np.mean(comp_cos[c])) for c in comp_suffixes},
            **global_cos,
        },
        "components_relL2_mean_over_layers": {
            **{c: float(np.mean(comp_relL2[c])) for c in comp_suffixes},
            **global_relL2,
        },
        "router_cos_min": float(router_cos.min()),
    }
    p = os.path.join(args.output_dir, "weight_similarity_summary.json")
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  -> {p}")
    logger.info(
        f"OVERALL experts diag cos median={summary['experts']['diag_cos_median_overall']:.3f} "
        f"vs off-diag mean={summary['experts']['offdiag_cos_mean_overall']:.3f}; "
        f"relL2 median={summary['experts']['relL2_median_overall']:.3f}"
    )


if __name__ == "__main__":
    main()
