# PARENT: "scripts/meta_learning/build_report.py" (visual framework imported: CSS/JS/tab
#          structure, card/table/figure helpers)
# DESCRIPTION:
#     Builds the sparse_experts experiment report at claude_outputs/sparse_experts/report.html.
#     Currently: the expert co-activation analysis of sparse_8of512_10b/step2384 on 40k held-out
#     documents (scripts/sparse_experts/coactivation/). Tables and figures come from
#     claude_outputs/sparse_experts/coactivation/<run>/{analysis.json,figs/} produced by
#     scripts/sparse_experts/coactivation/analyze_coactivation.py -- rerun that first to refresh.
#
#   python scripts/sparse_experts/build_report.py
##############################################################
import html
import os
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "claude_outputs/sparse_experts"
COACT = OUT / "coactivation"
RUN = os.environ.get("COACT_RUN", "sparse_8of512_10b_step2384")  # e.g. the _pilot400 run

_spec = importlib.util.spec_from_file_location(
    "ml_report", ROOT / "scripts/meta_learning/build_report.py"
)
_ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ml)
card, table, img_tag, fig_row, CSS, JS = (
    _ml.card,
    _ml.table,
    _ml.img_tag,
    _ml.fig_row,
    _ml.CSS,
    _ml.JS,
)

SIZES = [
    ("emo128_baseline_20b (meta_learning ref)", "8 / 128", 1024, "13.57B", "1.49B", "805M"),
    ("sparse_8of128_10b", "8 / 128", 256, "3.91B", "0.89B", "201M"),
    ("sparse_8of256_10b", "8 / 256", 256, "7.13B", "0.89B", "403M"),
    ("sparse_8of512_10b", "8 / 512", 256, "13.58B", "0.90B", "805M"),
    ("sparse_8of1024_10b", "8 / 1024", 256, "26.48B", "0.91B", "1611M"),
]


def f(x, nd=2):
    return "&ndash;" if x is None else f"{x:.{nd}f}"


def layer_table(pool: dict, agg: str = "sample") -> str:
    rows = []
    L = len(pool["layers"])
    for li in range(L):
        s = pool["layers"][str(li)][agg]
        st = pool["layers"][str(li)]["sample"]["structure"]
        u = pool["unique_experts_per_doc"]
        rows.append(
            [
                li,
                f(s["usage"]["effective_experts"], 1),
                s["usage"]["unused"],
                f(s["usage"]["gini"]),
                f(st["Q_spectral"], 3),
                f(st["Q_louvain"], 3),
                st["n_louvain_communities"],
                f(st["within_cluster_mass"]),
                f(100 * s["pairs"]["never_coactive_frac"], 1) + "%",
                f(s["pairs"]["lift_median"]),
                f(s["pairs"]["lift_p99"], 1),
                f(100 * s["pairs"]["frac_pairs_lift_gt2"], 1) + "%",
                f(pool["layers"][str(li)]["sample"]["token_vs_doc_spearman"]),
                f(u["mean_by_layer"][li], 1),
                f(u["p90_by_layer"][li], 0),
            ]
        )
    return table(
        [
            "layer",
            "eff. experts",
            "unused",
            "usage Gini",
            "Q spectral (k=8)",
            "Q Louvain",
            "#Louvain comm.",
            "within-cluster mass",
            "pairs never co-active",
            "lift median",
            "lift p99",
            "pairs lift&gt;2",
            "&rho; token vs doc lift",
            "experts/doc mean",
            "experts/doc p90",
        ],
        rows,
    )


def top_pairs_table(pool: dict, layers=(0, 4, 8, 12, 15)) -> str:
    rows = []
    for li in layers:
        tp = pool["layers"][str(li)]["sample"]["top_pairs"][:5]
        rows.append(
            [li, ", ".join(f"({p['i']},{p['j']}) lift {p['lift']:.0f} n={p['count']}" for p in tp)]
        )
    return table(["layer", "top-5 pairs by lift (support &ge; 50 tokens; expert ids)"], rows)


def source_table(pool: dict) -> str:
    ps = pool["per_source"]
    L = len(pool["layers"])
    rows = []
    for s, v in ps.items():
        q = [v["layers"][str(li)]["Q_spectral"] for li in range(L)]
        eff = [v["layers"][str(li)]["usage"]["effective_experts"] for li in range(L)]
        rows.append(
            [
                s,
                f"{pool['n_tok_per_source'][s]:,}",
                pool["n_doc_per_source"][s],
                f(sum(q) / L, 3),
                f(min(q), 3),
                f(max(q), 3),
                f(sum(eff) / L, 1),
            ]
        )
    return table(
        ["source", "tokens", "docs", "mean Q", "min Q", "max Q", "mean eff. experts"], rows
    )


def cross_source_table(pool: dict) -> str:
    xs = pool["cross_source"]
    S = xs["sources"]
    L = len(xs["layers"])
    J = [
        [
            sum(xs["layers"][str(li)]["top1pct_pair_jaccard"][a][b] for li in range(L)) / L
            for b in range(len(S))
        ]
        for a in range(len(S))
    ]
    C = [
        [
            sum(xs["layers"][str(li)]["usage_cosine"][a][b] for li in range(L)) / L
            for b in range(len(S))
        ]
        for a in range(len(S))
    ]
    rows = [
        [S[a]] + [f"{J[a][b]:.2f} / {C[a][b]:.2f}" for b in range(len(S))] for a in range(len(S))
    ]
    return table(["Jaccard(top-1% pairs) / cosine(usage)"] + S, rows)


def build_overview(an: dict) -> str:
    pc, p64 = an["pools"].get("pool_config"), an["pools"].get("pool_64")
    L = len(pc["layers"])
    qmean = sum(pc["layers"][str(li)]["sample"]["structure"]["Q_spectral"] for li in range(L)) / L
    qmean64 = (
        sum(p64["layers"][str(li)]["sample"]["structure"]["Q_spectral"] for li in range(L)) / L
        if p64
        else None
    )
    eff = [pc["layers"][str(li)]["sample"]["usage"]["effective_experts"] for li in range(L)]
    never = [pc["layers"][str(li)]["sample"]["pairs"]["never_coactive_frac"] for li in range(L)]
    rho = [pc["layers"][str(li)]["sample"]["token_vs_doc_spearman"] for li in range(L)]
    upd = pc["unique_experts_per_doc"]["mean_by_layer"]
    goal = (
        "<p><b>sparse_experts</b> asks how EMO's routing structure changes when each expert is a quarter "
        "of the size (expert hidden 256 instead of 1024) and the total expert count grows 128 &rarr; 1024 "
        "at a fixed 8 active (7 routed + 1 shared). First analysis: for the 8-of-512 model after 10B "
        "tokens, how often is every pair of experts active together, per layer, on documents the model "
        "never saw?</p>"
        + table(
            [
                "model",
                "active / total",
                "expert hidden",
                "total params",
                "active params",
                "expert params / layer",
            ],
            [list(r) for r in SIZES],
        )
        + "<p class='note'>Checkpoints: sparse_8of512_10b done (2385 steps, final train CE 2.98); "
        "sparse_8of1024_10b launched after a grouped_gemm &gt;512-expert kernel limit was fixed "
        "(host-resident batch sizes; numerically verified on Beaker); 128/256 arms not yet launched.</p>"
    )
    setup = (
        "<ul>"
        f"<li><b>Model</b>: sparse_8of512_10b/step2384 (10B tokens), native OLMo-core forward in bf16, "
        "routing exactly as at inference: per token top-7 of the 511 standard experts + shared expert 511 "
        "(eval pool = 512 = full routing). Second pass with the randpool eval pool pinned to 64 "
        "(document-level top-64 by doc-summed router probability, then per-token top-7 inside it).</li>"
        f"<li><b>Documents</b>: {pc['num_docs']:,} whole docs from the 20B&ndash;40B training-stream window "
        "(same seed-0 data order the run used; the model trained on 10B tokens, so all unseen), "
        "stratified by source (24k dclm, 4k each starcoder / pes2o / proof-pile-2 / olmo-mix), each "
        f"truncated to 4,095 tokens + EOS and packed whole into 4096-token rows with intra-document "
        f"attention. {pc['num_tokens']:,} scored tokens.</li>"
        "<li><b>Recorded</b>: per layer, 512&times;512 token-level co-activation counts (diagonal = usage), "
        "document-level co-occurrence counts, both per source; and the raw per-token top-7 indices "
        "(int16, 11 GB per pass) for later analyses.</li>"
        "<li><b>Statistics</b>: lift = observed / expected-under-independent-routing "
        "(C<sub>ij</sub>&middot;N / C<sub>i</sub>C<sub>j</sub>); spectral clustering (k=8) of the lift graph and "
        "Louvain communities; Newman modularity Q of the co-activation graph (a shuffled partition gives "
        "Q &asymp; 0); Spearman agreement between token- and document-level lift; per-source structure and "
        "cross-source overlap of the strongest pairs.</li>"
        "</ul>"
    )
    headline = (
        "<ul>"
        f"<li><b>Clear block structure in every layer.</b> Modularity of the token co-activation graph "
        f"averages Q = {qmean:.2f} over layers under full routing (shuffled null &asymp; 0); Louvain finds "
        f"{min(pc['layers'][str(li)]['sample']['structure']['n_louvain_communities'] for li in range(L))}&ndash;"
        f"{max(pc['layers'][str(li)]['sample']['structure']['n_louvain_communities'] for li in range(L))} "
        "communities per layer.</li>"
        f"<li><b>Usage is broad but uneven.</b> Effective number of experts (exp of usage entropy) is "
        f"{min(eff):.0f}&ndash;{max(eff):.0f} of 511 across layers; no expert is unused.</li>"
        f"<li><b>Most pairs do co-activate.</b> Only {100 * min(never):.1f}&ndash;{100 * max(never):.1f}% of pairs "
        "never co-occur on 51M tokens; the structure is in the lift distribution (heavy right tail), not in "
        "hard exclusion.</li>"
        f"<li><b>Token- and document-level structure agree moderately</b> (Spearman "
        f"{min(rho):.2f}&ndash;{max(rho):.2f}); a document touches on average "
        f"{min(upd):.0f}&ndash;{max(upd):.0f} distinct experts per layer under full routing"
        + (
            f", 64 by construction under the pool-64 pass, where Q rises to {qmean64:.2f}."
            if qmean64
            else "."
        )
        + "</li></ul>"
    )
    figs = COACT / RUN / "figs"
    return (
        card("goal", "Goal and the sparse_experts arms", goal)
        + card("method", "What was measured", setup)
        + card(
            "results",
            "Headline",
            headline
            + fig_row(
                img_tag(
                    figs / "summary_by_layer.png",
                    "Per-layer summary: modularity (spectral k=8, Louvain, shuffled null), effective #experts, token-vs-doc lift agreement; full routing vs pool 64.",
                )
            ),
        )
    )


def build_pool_tab(an: dict, key: str, title: str) -> str:
    pool = an["pools"].get(key)
    if pool is None:
        return "<p>not available</p>"
    figs = COACT / RUN / "figs"
    return (
        card(
            "results",
            f"{title}: per-layer statistics",
            "<p>Sample aggregate (all 40k docs as sampled). Q spectral uses the k=8 spectral partition of the "
            "lift graph; within-cluster mass = share of co-activation counts inside those clusters "
            "(8 equal random clusters would give 0.125).</p>" + layer_table(pool),
        )
        + card(
            "results",
            f"{title}: token-level lift, cluster-ordered",
            fig_row(
                img_tag(
                    figs / f"{key}_lift_tok_grid.png",
                    "log2 lift of token-level co-activation per layer; experts ordered by spectral cluster then usage; black lines = cluster boundaries; clipped to ±3 (dark blue = never co-active).",
                )
            ),
        )
        + card(
            "results",
            f"{title}: document-level lift, same ordering",
            fig_row(
                img_tag(
                    figs / f"{key}_lift_doc_grid.png",
                    "log2 lift of document-level co-occurrence (both experts used somewhere in the doc), same expert ordering as the token-level grid.",
                )
            ),
        )
        + card(
            "results",
            f"{title}: usage and lift distributions",
            fig_row(
                img_tag(
                    figs / f"{key}_usage.png", "Sorted per-expert token share by layer (log scale)."
                ),
                img_tag(
                    figs / f"{key}_lift_hist.png",
                    "Distribution of log2 lift over pairs with >0 co-activations.",
                ),
            ),
        )
        + card("results", f"{title}: strongest pairs", top_pairs_table(pool))
    )


def build_sources(an: dict) -> str:
    pool = an["pools"]["pool_config"]
    figs = COACT / RUN / "figs"
    return (
        card(
            "results",
            "Per-source structure (full routing)",
            "<p>Each source analysed on its own tokens (stratified sample; dclm has ~6x the tokens of the others).</p>"
            + source_table(pool)
            + fig_row(
                img_tag(
                    figs / "per_source.png",
                    "Left: modularity per source and layer. Right: Jaccard overlap of the top-1% pairs (by lift, support ≥ 20) between sources, mean over layers.",
                )
            ),
        )
        + card(
            "results",
            "Cross-source agreement",
            "<p>Cell = Jaccard of top-1% lift pairs / cosine of usage vectors, mean over layers.</p>"
            + cross_source_table(pool),
        )
        + card(
            "results",
            "Experts touched per document",
            fig_row(
                img_tag(
                    figs / "unique_experts_per_doc.png",
                    "Distinct routed experts per document per layer (2,000-doc sample; docs truncated to 4,095 tokens).",
                )
            ),
        )
    )


def build_method() -> str:
    return card(
        "method",
        "Pipeline and files",
        """
<ul>
<li><code>scripts/sparse_experts/coactivation/PROPOSAL.md</code> — approved design.</li>
<li><code>sample_docs.py</code> — stratified selection on the window's cluster-label file, one parallel pass over the 16 doc shards; output <code>sparse_experts/coactivation/docs_40k/</code> (docs.jsonl.gz, doc_meta.jsonl, summary.json).</li>
<li><code>extract_coactivation.py</code> — native checkpoint load (TransformerConfig.from_dict + load_model_and_optim_state), LM head replaced by identity, forward hooks on each <code>blocks.*.feed_forward_moe.router</code> capturing expert_indices; per-batch one-hot matmuls into int64 accumulators (exact; self-tested against a naive loop); torchrun sharding by doc_id, rank-0 merge. Beaker: 1 allocated node × 8 GPUs, ~80k tok/s/GPU, both passes in ~5 min.</li>
<li><code>analyze_coactivation.py</code> — metrics + figures into <code>claude_outputs/sparse_experts/coactivation/&lt;run&gt;/</code>.</li>
<li>Outputs on weka: <code>sparse_experts/coactivation/sparse_8of512_10b_step2384/pool_{config,64}/</code>: counts.npz (c_tok, c_doc as (source, layer, 512, 512) int64; n_tok, n_doc), topk.npy (N,16,7) int16, tok_doc.npy, tok_pos.npy, info.json.</li>
<li>Validation: symmetric matrices; diagonal = 7 × tokens exactly; shared expert never in routed slots; pool-64 pass shows ≤ 64 distinct experts per (doc, layer) vs up to 508 under full routing; pilot (400 docs) and full run consistent.</li>
<li>Beaker: pilot 01M1NKCXQ4ETAB0TX1HDV71XHF, full 01M1NS78PJJXY5KJW8GE8ABE88.</li>
</ul>""",
    )


def main():
    an_path = COACT / RUN / "analysis.json"
    an = json.load(open(an_path))
    tabs = [
        ("overview", "Overview", build_overview(an)),
        ("full", "Full routing (pool 512)", build_pool_tab(an, "pool_config", "Full routing")),
        ("pool64", "Pool pinned to 64", build_pool_tab(an, "pool_64", "Pool 64")),
        ("sources", "Per source", build_sources(an)),
        ("method", "Method &amp; files", build_method()),
    ]
    nav = "".join(f'<button data-target="{tid}">{name}</button>' for tid, name, _ in tabs)
    sections = "".join(f'<section class="tab" id="{tid}">{body}</section>' for tid, _, body in tabs)
    title = "EMO sparse_experts: expert co-activation in the 8-of-512 quarter-size-expert model"
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<header>
<a class="home-link" href="/">&larr; all reports</a>
<h1>{html.escape(title)}</h1>
<p>sparse_experts &mdash; quarter-size experts, 8 active of 128/256/512/1024 &middot; pairwise expert
co-activation per layer on 40k unseen documents (51M tokens), full routing and pool-64 routing &middot;
generated by scripts/sparse_experts/build_report.py</p>
</header>
<div class="topbar"><nav>{nav}</nav><div id="subnav"></div></div>
<main>{sections}</main>
<script>{JS}</script>
</body>
</html>
"""
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "report.html"
    out.write_text(page)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
