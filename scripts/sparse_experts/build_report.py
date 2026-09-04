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


def conditional_table(pool: dict) -> str:
    rows = []
    for li in range(len(pool["layers"])):
        c = pool["layers"][str(li)]["sample"].get("conditional")
        if not c:
            continue
        top = ", ".join(
            f"P({t['j']}|{t['i']})={t['p_j_given_i']:.2f} (n_i={t['n_i']:,})" for t in c["top"][:3]
        )
        rows.append(
            [
                li,
                f(c["independence_baseline"], 3),
                f(c["median"], 3),
                f(c["p999"]),
                f(c["max"]),
                f(100 * c["frac_pairs_gt_0p1"]),
                f(100 * c["frac_pairs_gt_0p25"], 3),
                top,
            ]
        )
    return table(
        [
            "layer",
            "independent baseline 7/511",
            "median P(j|i)",
            "p99.9",
            "max",
            "% ordered pairs P&gt;0.1",
            "% ordered pairs P&gt;0.25",
            "strongest (n_i &ge; 1000)",
        ],
        rows,
    )


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


def _stats(an: dict):
    pc, p64 = an["pools"]["pool_config"], an["pools"].get("pool_64")
    L = len(pc["layers"])
    q = [pc["layers"][str(li)]["sample"]["structure"]["Q_spectral"] for li in range(L)]
    ql = [pc["layers"][str(li)]["sample"]["structure"]["Q_louvain"] for li in range(L)]
    nc = [pc["layers"][str(li)]["sample"]["structure"]["n_louvain_communities"] for li in range(L)]
    eff = [pc["layers"][str(li)]["sample"]["usage"]["effective_experts"] for li in range(L)]
    never = [pc["layers"][str(li)]["sample"]["pairs"]["never_coactive_frac"] for li in range(L)]
    rho = [pc["layers"][str(li)]["sample"]["token_vs_doc_spearman"] for li in range(L)]
    med = [pc["layers"][str(li)]["sample"]["pairs"]["lift_median"] for li in range(L)]
    p99 = [pc["layers"][str(li)]["sample"]["pairs"]["lift_p99"] for li in range(L)]
    upd = pc["unique_experts_per_doc"]["mean_by_layer"]
    mid = range(2, L)
    q_mid = sum(q[li] for li in mid) / len(mid)
    psq = {
        s: sum(v["layers"][str(li)]["Q_spectral"] for li in mid) / len(mid)
        for s, v in pc["per_source"].items()
    }
    xs = pc["cross_source"]
    S = len(xs["sources"])
    jv = [
        sum(xs["layers"][str(li)]["top1pct_pair_jaccard"][a][b] for li in range(L)) / L
        for a in range(S)
        for b in range(S)
        if a < b
    ]
    d = dict(
        pc=pc,
        p64=p64,
        L=L,
        q=q,
        ql=ql,
        nc=nc,
        eff=eff,
        never=never,
        rho=rho,
        med=med,
        p99=p99,
        upd=upd,
        q_mid=q_mid,
        psq=psq,
        jmin=min(jv),
        jmax=max(jv),
    )
    if p64:
        d["q64"] = [p64["layers"][str(li)]["sample"]["structure"]["Q_spectral"] for li in range(L)]
        d["rho64"] = [p64["layers"][str(li)]["sample"]["token_vs_doc_spearman"] for li in range(L)]
        d["eff64"] = [
            p64["layers"][str(li)]["sample"]["usage"]["effective_experts"] for li in range(L)
        ]
        d["never64"] = [
            p64["layers"][str(li)]["sample"]["pairs"]["never_coactive_frac"] for li in range(L)
        ]
    return d


def build_overview(an: dict) -> str:
    st = _stats(an)
    pc, L = st["pc"], st["L"]
    figs = COACT / RUN / "figs"
    goal = (
        "<p><b>sparse_experts</b> asks how EMO's emergent routing structure changes when each expert is a "
        "quarter of the size (expert hidden 256 instead of 1024) and the total expert count grows "
        "128 &rarr; 1024 at a fixed 8 active (7 routed + 1 shared). The first question, answered here for "
        "the 8-of-512 model after 10B tokens: <b>which experts fire together?</b> For every layer, how often "
        "is each pair of the 511 standard experts active on the same token, and on the same document, on "
        "documents the model never saw &mdash; and is that co-activation organised into blocks?</p>"
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
    )
    hyp = (
        "<ul>"
        "<li><b>H1 (modularity)</b>: co-activation is block-structured &mdash; experts form groups that fire "
        "together far more than independent routing would predict &mdash; in every layer, not only where "
        "the load-balancing loss leaves room.</li>"
        "<li><b>H2 (depth)</b>: structure is strongest in the first layer (token identity) and weakens with "
        "depth as routing becomes more contextual.</li>"
        "<li><b>H3 (domain)</b>: a large part of the block structure is domain specialisation: within a single "
        "data source the graph is less modular, and the strongest pairs differ between sources.</li>"
        "<li><b>H4 (pools)</b>: pinning the randpool eval pool to 64 experts per document makes token-level "
        "and document-level structure coincide, since routing is then confined per document.</li>"
        "</ul>"
    )
    approach = (
        f"<p>Run the checkpoint natively (no HF conversion) over {pc['num_docs']:,} whole held-out documents "
        f"({pc['num_tokens']:,} tokens) packed like training, hook every layer's router to capture the "
        "top-7 routed experts per token, and accumulate exact 512&times;512 pair counts per layer and per "
        "source at token level (both active on the token) and document level (both active somewhere in the "
        "doc). Compare against independent routing via <i>lift</i> = observed / expected, cluster the lift "
        "graph, and score block structure with Newman modularity Q against a shuffled-partition null. "
        "Repeat with the eval pool pinned to 64.</p>"
    )
    headline = (
        "<ul>"
        f"<li><b>H1 supported.</b> Modularity of the token co-activation graph is Q = "
        f"{min(st['q'][1:]):.2f}&ndash;{max(st['q'][1:]):.2f} in layers 1&ndash;{L - 1} and {st['q'][0]:.2f} in "
        f"layer 0 (shuffled null &asymp; 0; Louvain {min(st['nc'])}&ndash;{max(st['nc'])} communities per "
        "layer). Usage is broad: "
        f"{min(st['eff']):.0f}&ndash;{max(st['eff']):.0f} effective experts of 511, none unused, and beyond "
        f"layer 2 fewer than {100 * max(st['never'][3:]):.1f}% of pairs never co-activate. The structure is a "
        f"heavy right tail of lift (median {min(st['med']):.2f}&ndash;{max(st['med']):.2f}, p99 "
        f"{min(st['p99'][1:]):.0f}&ndash;{max(st['p99']):.0f}), not hard exclusion.</li>"
        f"<li><b>H2 partly.</b> Layer 0 is far more modular ({100 * st['never'][0]:.0f}% of pairs never co-occur) "
        "but beyond it Q is flat with depth and rises slightly in the last two layers; token&ndash;document "
        f"agreement grows with depth (Spearman {st['rho'][0]:.2f} &rarr; {st['rho'][-1]:.2f}) while a document "
        f"touches fewer distinct experts ({st['upd'][0]:.0f} &rarr; {st['upd'][-1]:.0f} per layer).</li>"
        f"<li><b>H3 supported.</b> Within one source Q is lower (layers 2&ndash;15: "
        + ", ".join(f"{s} {q:.2f}" for s, q in st["psq"].items())
        + f") than pooled ({st['q_mid']:.2f}), and the top-1% pairs overlap only "
        f"{st['jmin']:.2f}&ndash;{st['jmax']:.2f} (Jaccard) between sources. Layer 0 is modular within every "
        "source.</li>"
        + (
            f"<li><b>H4 supported.</b> With the pool pinned to 64, token-vs-document agreement is "
            f"{min(st['rho64']):.2f}&ndash;{max(st['rho64']):.2f}, Q {min(st['q64']):.2f}&ndash;{max(st['q64']):.2f} "
            f"and {min(st['eff64']):.0f}&ndash;{max(st['eff64']):.0f} effective experts.</li>"
            if st["p64"]
            else ""
        )
        + "</ul>"
        + fig_row(
            img_tag(
                figs / "summary_by_layer.png",
                "Per-layer summary: modularity (spectral k=8, Louvain, shuffled null), effective #experts, "
                "token-vs-document lift agreement; full routing vs pool 64.",
            )
        )
    )
    status = (
        "<ul>"
        "<li><b>sparse_8of512_10b</b>: trained to 10B tokens (2385 steps, final train CE 2.98); this analysis "
        "on step2384. Beaker: pilot 01M1NKCXQ4ETAB0TX1HDV71XHF, full 01M1NS78PJJXY5KJW8GE8ABE88.</li>"
        "<li><b>sparse_8of1024_10b</b>: training (01M1NG1HKZCH03MHEN8CDXND6A) after a grouped_gemm "
        "&gt;512-expert kernel limit was fixed (host-resident batch sizes; numerically verified on Beaker).</li>"
        "<li><b>sparse_8of128_10b / sparse_8of256_10b</b>: scripts ready, not launched.</li>"
        "<li>Data: <code>sparse_experts/coactivation/sparse_8of512_10b_step2384/pool_{config,64}/</code> on weka "
        "(counts + raw per-token top-7, 50 GB).</li>"
        "</ul>"
    )
    return (
        card("goal", "Goal", goal)
        + card("goal", "Hypotheses", hyp)
        + card("method", "Approach in one paragraph", approach)
        + card("results", "Headline findings", headline)
        + card("results", "Status", status)
    )


def build_method(an: dict) -> str:
    pc = an["pools"]["pool_config"]
    data = (
        "<ul>"
        "<li><b>Source</b>: the 20B&ndash;40B training-stream document window "
        "(<code>meta_learning/data/meta128_20B-40B/</code>, 18.3M whole docs, 36.9B tokens), extracted with "
        "the same seed-0 data order this run used. The model trained on the first 10B tokens, so every "
        "document here is unseen. Window mix by tokens: dclm 93.4%, proof-pile-2 3.2%, starcoder 1.8%, "
        "pes2o 1.6%, olmo-mix 0.1%.</li>"
        f"<li><b>Sample</b>: {pc['num_docs']:,} docs stratified by source &mdash; "
        + ", ".join(f"{s} {n:,}" for s, n in pc["n_doc_per_source"].items())
        + " &mdash; chosen on the compact per-doc label file, token ids pulled in one parallel pass over "
        "the 16 shards (<code>sample_docs.py</code>). Each doc truncated to its first 4,095 tokens + EOS "
        "(9% of docs are longer but hold 59% of tokens; truncation bounds per-doc influence). "
        f"{pc['num_tokens']:,} scored tokens: "
        + ", ".join(f"{s} {n:,}" for s, n in pc["n_tok_per_source"].items())
        + ".</li>"
        "<li><b>Aggregates</b>: <i>sample</i> (as sampled) and <i>natural</i> (per-source counts reweighted "
        "to the window's token shares). Tables below use the sample aggregate; the per-source tab separates "
        "the domains.</li>"
        "</ul>"
    )
    fwd = (
        "<ul>"
        "<li><b>Model</b>: sparse_8of512_10b/step2384 built from the checkpoint's config.json and loaded with "
        "the native DCP loader (same recipe as the k=32-CPT and meta_learning CE evals), bf16, LM head "
        "replaced by identity (routing only). No HF conversion: the HF port routes over all experts and "
        "cannot pin the document pool.</li>"
        "<li><b>Routing</b>: exactly the eval-time routing of the randpool router. Full pass: eval pool = 512 = "
        "all experts, so each token takes the top-7 of the 511 standard experts plus shared expert 511. "
        "Pool-64 pass: per document, the top-64 experts by document-summed router probability form the pool; "
        "each token takes its top-7 inside it.</li>"
        "<li><b>Packing</b>: docs EOS-terminated, first-fit-decreasing into 4096-token rows, "
        "<code>doc_lens</code> passed so attention is intra-document and the router sees per-document "
        "segments, as in training. EOS and pad tokens excluded from all counts.</li>"
        "<li><b>Capture</b>: forward hook on each <code>blocks.*.feed_forward_moe.router</code>; the router "
        "returns <code>expert_indices (B, S, 8)</code> with the shared expert in the last slot (asserted). "
        "Counts accumulated as one-hot matmuls per batch into int64 (exact; self-tested against a naive "
        "loop). 8 GPUs (torchrun, docs sharded by id, rank-0 merge), ~80k tokens/s/GPU, ~5 min for both "
        "passes.</li>"
        "</ul>"
    )
    stats = (
        "<ul>"
        "<li><b>Usage</b>: p<sub>i</sub> = C<sub>ii</sub>/N; effective #experts = exp(entropy); Gini.</li>"
        "<li><b>Lift</b>: C<sub>ij</sub>&middot;N / (C<sub>i</sub>C<sub>j</sub>) = observed / expected under "
        "independent routing with the same marginals. Lift 1 = independent; the 0/1 diagonal of the "
        "heatmaps is log<sub>2</sub> lift clipped to &plusmn;3.</li>"
        "<li><b>Conditional co-activation</b>: P(E<sub>j</sub> | E<sub>i</sub>) = C<sub>ij</sub> / C<sub>ii</sub>, "
        "the share of expert i's tokens (or documents) on which j was also active; asymmetric, rows = i. "
        "Independent routing gives 7/511 &asymp; 0.014 at token level.</li>"
        "<li><b>Block structure</b>: normalised spectral clustering (k = 8) of the token-level lift graph; "
        "Newman modularity Q of that partition on the co-activation count graph; Q of a shuffled partition "
        "as the null (&asymp; 0); Louvain communities and their Q as a resolution-free check; within-cluster "
        "share of co-activation mass (8 equal random clusters give 0.125).</li>"
        "<li><b>Token vs document</b>: Spearman &rho; between token-level and document-level lift over pairs "
        "with support.</li>"
        "<li><b>Per source</b>: usage, Q per source; Jaccard of the top-1% pairs (by lift, support &ge; 20) "
        "between sources; cosine of usage vectors.</li>"
        "<li><b>Footprint</b>: distinct routed experts per (document, layer) from the raw top-7 "
        "(2,000-doc sample).</li>"
        "</ul>"
    )
    valid = (
        "<ul>"
        "<li>Every matrix symmetric; diagonal = 7 &times; tokens exactly; shared expert never in a routed slot; "
        "indices within [0, 510].</li>"
        "<li>Pool-64 pass: no (document, layer) touches more than 64 distinct experts; full routing reaches "
        "up to 508.</li>"
        "<li>400-doc pilot and the full run agree on every per-layer statistic to within sampling noise.</li>"
        "<li>Kernel path: the grouped-GEMM fix used by the 1024-expert sibling is not exercised here "
        "(512 groups take the unchanged device path).</li>"
        "</ul>"
    )
    files = (
        "<ul>"
        "<li><code>scripts/sparse_experts/coactivation/PROPOSAL.md</code> &mdash; approved design.</li>"
        "<li><code>sample_docs.py</code>, <code>extract_coactivation.py</code>, <code>launch_beaker.sh</code>, "
        "<code>analyze_coactivation.py</code>; <code>scripts/sparse_experts/build_report.py</code> (this page).</li>"
        "<li>Outputs: <code>sparse_experts/coactivation/sparse_8of512_10b_step2384/pool_{config,64}/</code>: "
        "counts.npz (c_tok, c_doc as (source, layer, 512, 512) int64; n_tok, n_doc), topk.npy (N, 16, 7) "
        "int16, tok_doc.npy, tok_pos.npy, info.json. Analysis JSON + figures under "
        "<code>claude_outputs/sparse_experts/coactivation/</code>.</li>"
        "</ul>"
    )
    return (
        card("method", "Data", data)
        + card("method", "Forward pass and capture", fwd)
        + card("method", "Statistics", stats)
        + card("method", "Validation", valid)
        + card("method", "Files", files)
    )


def build_full(an: dict) -> str:
    st = _stats(an)
    pc = st["pc"]
    figs = COACT / RUN / "figs"
    q = st["q"]
    return (
        card(
            "goal",
            "Question",
            "<p>Under the model's own inference routing (all 512 experts available, "
            "top-7 + shared), is pairwise co-activation organised into blocks, how strong is the "
            "organisation, and how does it change with depth?</p>",
        )
        + card(
            "hypothesis",
            "Hypothesis",
            "<p>H1: block structure in every layer, well above the shuffled null. "
            "H2: strongest in layer 0, weakening with depth.</p>",
        )
        + card(
            "metrics",
            "Metrics",
            "<p>Per layer: effective #experts, unused experts, usage Gini; Q of the "
            "k=8 spectral partition and of Louvain communities (null: shuffled partition); within-cluster "
            "mass; share of pairs never co-active; lift median / p99 / share of pairs with lift &gt; 2; "
            "Spearman between token- and document-level lift; distinct experts per document; the "
            "conditional rate P(E<sub>j</sub> | E<sub>i</sub>) and its tail.</p>",
        )
        + card("results", "Results &mdash; per-layer table", layer_table(pc))
        + card(
            "results",
            "Results &mdash; token-level lift, cluster-ordered",
            fig_row(
                img_tag(
                    figs / "pool_config_lift_tok_grid.png",
                    "log2 lift of token-level co-activation per layer; experts ordered by spectral cluster then usage; black lines = cluster boundaries; clipped to ±3 (dark blue = never co-active).",
                )
            ),
        )
        + card(
            "results",
            "Results &mdash; document-level lift, same ordering",
            fig_row(
                img_tag(
                    figs / "pool_config_lift_doc_grid.png",
                    "log2 lift of document-level co-occurrence (both experts used somewhere in the document), same expert ordering as the token-level grid.",
                )
            ),
        )
        + card(
            "results",
            "Results &mdash; conditional co-activation P(E<sub>j</sub> | E<sub>i</sub>)",
            "<p>Directed view of the same counts: cell (i, j) = N(E<sub>i</sub>, E<sub>j</sub>) / N(E<sub>i</sub>), "
            "the fraction of the tokens routed to expert i that were also routed to expert j (rows = "
            "conditioning expert; asymmetric). Independent routing gives 7/511 &asymp; 0.014 everywhere. "
            "Experts keep the cluster ordering of the lift grids so the blocks line up.</p>"
            + conditional_table(pc)
            + fig_row(
                img_tag(
                    figs / "pool_config_cond_tok_grid.png",
                    "Token-level P(E_j | E_i) per layer, colour scale 0..0.2 (values above 0.2 saturate); rows = conditioning expert i.",
                )
            )
            + fig_row(
                img_tag(
                    figs / "pool_config_cond_doc_grid.png",
                    "Document-level analogue: of the documents that use expert i anywhere, the fraction that also use expert j; scale 0..1.",
                )
            ),
        )
        + card(
            "results",
            "Results &mdash; usage and lift distributions",
            fig_row(
                img_tag(
                    figs / "pool_config_usage.png",
                    "Sorted per-expert token share by layer (log scale); dashed = uniform 7/511.",
                ),
                img_tag(
                    figs / "pool_config_lift_hist.png",
                    "Distribution of log2 lift over pairs with >0 co-activations, selected layers.",
                ),
            ),
        )
        + card("results", "Results &mdash; strongest pairs", top_pairs_table(pc))
        + card(
            "results",
            "Conclusion",
            f"<p>Block structure is present in every layer (Q {min(q[1:]):.2f}&ndash;{max(q[1:]):.2f} vs null 0, "
            f"Louvain agrees at Q {min(st['ql']):.2f}&ndash;{max(st['ql']):.2f}), so H1 holds. Layer 0 is a "
            f"different regime: Q = {q[0]:.2f}, {100 * st['never'][0]:.0f}% of pairs never co-occur, and "
            f"{st['nc'][0]} Louvain communities &mdash; consistent with near-lexical routing. Beyond layer 0, "
            "modularity does not decay with depth; it is flat through the middle and slightly higher in the "
            "last two layers, so H2 holds only for the first layer. Usage stays broad throughout "
            f"({min(st['eff']):.0f}&ndash;{max(st['eff']):.0f} effective experts): the load-balancing loss "
            "keeps every expert busy, and the organisation lives in which experts are busy <i>together</i> "
            "(the heavy lift tail), not in exclusion. Token-level and document-level lift agree only "
            f"moderately (&rho; {st['rho'][0]:.2f} in layer 0 rising to {st['rho'][-1]:.2f}), i.e. many pairs "
            "co-occur within a document without co-firing on the same tokens.</p>",
        )
    )


def build_pool64(an: dict) -> str:
    st = _stats(an)
    p64 = st["p64"]
    if p64 is None:
        return "<p>not available</p>"
    figs = COACT / RUN / "figs"
    return (
        card(
            "goal",
            "Question",
            "<p>EMO's selective-expert use restricts each document to a pool of experts. "
            "If the eval pool is pinned to 64 (of 511), how do usage, block structure and the token/document "
            "relationship change?</p>",
        )
        + card(
            "hypothesis",
            "Hypothesis",
            "<p>H4: token-level and document-level structure coincide once "
            "routing is confined per document; the pool selection itself (top-64 by document-summed "
            "probability) concentrates usage.</p>",
        )
        + card(
            "metrics",
            "Metrics",
            "<p>Same per-layer metrics as the full-routing tab, on the same tokens, "
            "with the router's <code>eval_document_expert_pool</code> set to 64; distinct experts per "
            "document as the sanity bound (must be &le; 64).</p>",
        )
        + card("results", "Results &mdash; per-layer table", layer_table(p64))
        + card(
            "results",
            "Results &mdash; token-level lift, cluster-ordered",
            fig_row(
                img_tag(
                    figs / "pool_64_lift_tok_grid.png",
                    "log2 lift of token-level co-activation per layer under the 64-expert document pool; same construction as the full-routing grid.",
                )
            ),
        )
        + card(
            "results",
            "Results &mdash; document-level lift",
            fig_row(
                img_tag(
                    figs / "pool_64_lift_doc_grid.png",
                    "log2 lift of document-level co-occurrence under the 64-expert pool.",
                )
            ),
        )
        + card(
            "results",
            "Results &mdash; conditional co-activation P(E<sub>j</sub> | E<sub>i</sub>)",
            "<p>Same directed view under the 64-expert document pool (rows = conditioning expert).</p>"
            + conditional_table(p64)
            + fig_row(
                img_tag(
                    figs / "pool_64_cond_tok_grid.png",
                    "Token-level P(E_j | E_i) per layer under the pool, scale 0..0.2.",
                )
            )
            + fig_row(
                img_tag(
                    figs / "pool_64_cond_doc_grid.png",
                    "Document-level P(doc uses j | doc uses i) under the pool, scale 0..1.",
                )
            ),
        )
        + card(
            "results",
            "Results &mdash; usage and lift distributions",
            fig_row(
                img_tag(
                    figs / "pool_64_usage.png",
                    "Sorted per-expert token share by layer under the pool.",
                ),
                img_tag(
                    figs / "pool_64_lift_hist.png", "Distribution of log2 lift under the pool."
                ),
            ),
        )
        + card(
            "results",
            "Results &mdash; experts touched per document",
            fig_row(
                img_tag(
                    figs / "unique_experts_per_doc.png",
                    "Distinct routed experts per document per layer, full routing vs pool 64 (2,000-doc sample; docs truncated to 4,095 tokens).",
                )
            ),
        )
        + card(
            "results",
            "Conclusion",
            f"<p>Pinning the pool makes token- and document-level structure nearly the same thing "
            f"(&rho; {min(st['rho64']):.2f}&ndash;{max(st['rho64']):.2f} vs "
            f"{min(st['rho']):.2f}&ndash;{max(st['rho']):.2f} under full routing), so H4 holds. Modularity is "
            f"steady at Q {min(st['q64']):.2f}&ndash;{max(st['q64']):.2f} and now slightly <i>higher</i> than "
            "full routing in the middle layers: the pools are drawn from the same expert groups, so "
            f"confinement sharpens the blocks. Usage concentrates ({min(st['eff64']):.0f}&ndash;"
            f"{max(st['eff64']):.0f} effective experts, lowest in layer 0) and the share of never-co-active "
            f"pairs rises to {100 * max(st['never64']):.0f}% in layer 0.</p>",
        )
    )


def build_sources(an: dict) -> str:
    st = _stats(an)
    pc = st["pc"]
    figs = COACT / RUN / "figs"
    return (
        card(
            "goal",
            "Question",
            "<p>How much of the pooled block structure is domain specialisation? "
            "Is the co-activation graph modular <i>within</i> a single data source, and do the strongest "
            "pairs agree across sources?</p>",
        )
        + card(
            "hypothesis",
            "Hypothesis",
            "<p>H3: within-source modularity is markedly lower than pooled, "
            "and the top pairs are largely source-specific.</p>",
        )
        + card(
            "metrics",
            "Metrics",
            "<p>Per source and layer: Q of the k=8 spectral partition on that source's "
            "own tokens, effective #experts; between sources: Jaccard of the top-1% pairs by lift "
            "(support &ge; 20) and cosine of usage vectors, averaged over layers.</p>",
        )
        + card("results", "Results &mdash; per-source table (full routing)", source_table(pc))
        + card(
            "results",
            "Results &mdash; modularity per source and cross-source overlap",
            fig_row(
                img_tag(
                    figs / "per_source.png",
                    "Left: Q per source and layer. Right: Jaccard overlap of the top-1% lift pairs between sources, mean over layers.",
                )
            ),
        )
        + card(
            "results",
            "Results &mdash; cross-source agreement table",
            "<p>Cell = Jaccard of top-1% lift pairs / cosine of usage vectors, mean over layers.</p>"
            + cross_source_table(pc),
        )
        + card(
            "results",
            "Conclusion",
            "<p>H3 holds. Within a single source the graph is far less modular (layers 2&ndash;15 mean Q: "
            + ", ".join(f"{s} {q:.2f}" for s, q in st["psq"].items())
            + f") than the pooled {st['q_mid']:.2f}, and the strongest pairs overlap only "
            f"{st['jmin']:.2f}&ndash;{st['jmax']:.2f} between sources: the largest blocks are sets of "
            "experts that serve the same domain, and web text (dclm) retains the most within-domain "
            "structure. Layer 0 is the exception, modular within every source. Caveat: the minority "
            "sources have 2.6&ndash;9.7M tokens each versus 29M for dclm; their lift graphs are sparser, "
            "which can only lower the spectral Q modestly, not by the factor observed.</p>",
        )
    )


def build_next() -> str:
    return card(
        "method",
        "Next steps",
        """
<ul>
<li><b>Across expert counts</b>: run the identical pass on sparse_8of1024_10b (training) and on the
128-expert baseline (meta128_vanilla/step2384 &mdash; same tokens, 10B) to see whether modularity and
domain specialisation grow, shrink or stay put as experts get smaller and more numerous. The extractor
is parametric in checkpoint.</li>
<li><b>Cross-layer</b>: adjacent-layer co-activation (expert i at layer l with j at l+1) from the saved
per-token indices, to test whether the blocks form consistent pathways through depth.</li>
<li><b>Weighted variant</b>: router-weight-weighted co-activation instead of 0/1 membership.</li>
<li><b>Pool sweep</b>: pools 16 / 32 / 128 to map how confinement sharpens the blocks.</li>
<li><b>Semantic labels</b>: attach the k=32 document clusters (already available for this window) to the
Louvain communities to name the blocks.</li>
</ul>""",
    )


def main():
    an_path = COACT / RUN / "analysis.json"
    an = json.load(open(an_path))
    tabs = [
        ("overview", "Overview", build_overview(an)),
        ("method", "Method", build_method(an)),
        ("full", "1 · Full routing", build_full(an)),
        ("pool64", "2 · Pool pinned to 64", build_pool64(an)),
        ("sources", "3 · Per source", build_sources(an)),
        ("next", "Next steps", build_next()),
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
co-activation per layer on 40k unseen documents (51M tokens), full routing and pool-64 routing, token-
and document-level, per source &middot; generated by scripts/sparse_experts/build_report.py</p>
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
