# PARENT: "scripts/meta_learning/build_report.py" (visual framework imported: CSS/JS/tab
#          structure, card/table/figure helpers)
# DESCRIPTION:
#     Builds the sparse_experts experiment report at claude_outputs/sparse_experts/report.html:
#     the expert co-activation analysis on 40k held-out documents, side by side for the 8-of-512
#     and 8-of-1024 quarter-size-expert models. Tables and figures come from
#     claude_outputs/sparse_experts/coactivation/<run>/{analysis.json,figs/} (produced by
#     scripts/sparse_experts/coactivation/analyze_coactivation.py) and the cross-model figures in
#     claude_outputs/sparse_experts/coactivation/compare/ (compare_runs.py) -- rerun those first.
#     A model whose analysis.json is missing is simply left out of the comparison.
#
#   python scripts/sparse_experts/build_report.py
##############################################################
import html
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "claude_outputs/sparse_experts"
COACT = OUT / "coactivation"
COMPARE = COACT / "compare"
# label -> run dir name under COACT; order = column order in every table / figure pair
RUNS = [("512", "sparse_8of512_10b_step2384"), ("1024", "sparse_8of1024_10b_step2384")]

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
TRAIN = {"512": "2385 steps, final train CE 2.983", "1024": "2385 steps, final train CE 2.965"}
BEAKER = {
    "512": "pilot 01M1NKCXQ4ETAB0TX1HDV71XHF, full 01M1NS78PJJXY5KJW8GE8ABE88",
    "1024": "full 01M1Q2X9PMBW951S6X9540ASAT",
}


def f(x, nd=2):
    return "&ndash;" if x is None else f"{x:.{nd}f}"


# --------------------------------------------------------------------------
# per-model accessors
# --------------------------------------------------------------------------
class Model:
    def __init__(self, label: str, run: str):
        self.label, self.run = label, run
        self.an = json.load(open(COACT / run / "analysis.json"))
        self.figs = COACT / run / "figs"
        self.pc = self.an["pools"]["pool_config"]
        self.p64 = self.an["pools"].get("pool_64")
        self.L = len(self.pc["layers"])
        c0 = self.pc["layers"]["0"]["sample"]["conditional"]
        self.Es = int(round(7.0 / c0["independence_baseline"]))  # standard experts
        self.E = self.Es + 1

    def pool(self, key):
        return self.pc if key == "pool_config" else self.p64

    def series(self, key, fn):
        p = self.pool(key)
        return [fn(p["layers"][str(li)]["sample"]) for li in range(self.L)]

    def fig(self, name, caption):
        return img_tag(self.figs / name, f"{self.label} experts: {caption}")


def load_models():
    ms = []
    for label, run in RUNS:
        if (COACT / run / "analysis.json").exists():
            ms.append(Model(label, run))
    assert ms, "no analysis.json found"
    return ms


def pair_row(models, fn):
    """fig_row of the same figure for each model (side by side)."""
    return fig_row(*(fn(m) for m in models))


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
LAYER_METRICS = [
    (
        "eff. experts (% of std.)",
        lambda s, m: f(100 * s["usage"]["effective_experts"] / m.Es, 0) + "%",
    ),
    ("unused", lambda s, m: s["usage"]["unused"]),
    ("Q spectral k=8", lambda s, m: f(s["structure"]["Q_spectral"], 3)),
    ("Q Louvain", lambda s, m: f(s["structure"]["Q_louvain"], 3)),
    ("#Louvain comm.", lambda s, m: s["structure"]["n_louvain_communities"]),
    ("within-cluster mass", lambda s, m: f(s["structure"]["within_cluster_mass"])),
    ("pairs never co-active", lambda s, m: f(100 * s["pairs"]["never_coactive_frac"], 1) + "%"),
    ("lift median", lambda s, m: f(s["pairs"]["lift_median"])),
    ("lift p99", lambda s, m: f(s["pairs"]["lift_p99"], 1)),
    ("&rho; token vs doc", lambda s, m: f(s["token_vs_doc_spearman"])),
]


def paired_layer_table(models, key) -> str:
    heads = ["layer"]
    for name, _ in LAYER_METRICS + [("experts/doc mean", None)]:
        for m in models:
            heads.append(f"{name} <span class='muted'>({m.label})</span>")
    rows = []
    L = models[0].L
    for li in range(L):
        row = [li]
        for name, fn in LAYER_METRICS:
            for m in models:
                row.append(fn(m.pool(key)["layers"][str(li)]["sample"], m))
        for m in models:
            row.append(f(m.pool(key)["unique_experts_per_doc"]["mean_by_layer"][li], 0))
        rows.append(row)
    return table(heads, rows)


def paired_conditional_table(models, key) -> str:
    heads = ["layer"]
    cols = [
        ("baseline 7/E", lambda c: f(c["independence_baseline"], 4)),
        ("median", lambda c: f(c["median"], 4)),
        ("p99.9", lambda c: f(c["p999"])),
        ("max", lambda c: f(c["max"])),
        ("% pairs &gt;0.1", lambda c: f(100 * c["frac_pairs_gt_0p1"])),
        ("% pairs &gt;0.25", lambda c: f(100 * c["frac_pairs_gt_0p25"], 3)),
    ]
    for name, _ in cols:
        for m in models:
            heads.append(f"{name} <span class='muted'>({m.label})</span>")
    rows = []
    for li in range(models[0].L):
        row = [li]
        for name, fn in cols:
            for m in models:
                row.append(fn(m.pool(key)["layers"][str(li)]["sample"]["conditional"]))
        rows.append(row)
    return table(heads, rows)


def top_pairs_table(models, key, layers=(0, 4, 8, 12, 15)) -> str:
    heads = ["layer"] + [
        f"top-5 pairs by lift, support &ge; 50 ({m.label} experts; ids)" for m in models
    ]
    rows = []
    for li in layers:
        row = [li]
        for m in models:
            tp = m.pool(key)["layers"][str(li)]["sample"]["top_pairs"][:5]
            row.append(
                ", ".join(f"({p['i']},{p['j']}) lift {p['lift']:.0f} n={p['count']}" for p in tp)
            )
        rows.append(row)
    return table(heads, rows)


def paired_source_table(models) -> str:
    srcs = list(models[0].pc["per_source"])
    heads = ["source", "tokens", "docs"]
    for name in ("mean Q (layers 2+)", "min Q", "max Q", "mean eff. experts (% of std.)"):
        for m in models:
            heads.append(f"{name} <span class='muted'>({m.label})</span>")
    rows = []
    for s in srcs:
        row = [s, f"{models[0].pc['n_tok_per_source'][s]:,}", models[0].pc["n_doc_per_source"][s]]
        vals = {}
        for m in models:
            v = m.pc["per_source"][s]
            q = [v["layers"][str(li)]["Q_spectral"] for li in range(m.L)]
            eff = [v["layers"][str(li)]["usage"]["effective_experts"] for li in range(m.L)]
            vals[m.label] = (sum(q[2:]) / (m.L - 2), min(q), max(q), 100 * sum(eff) / m.L / m.Es)
        for k in range(4):
            for m in models:
                row.append(f(vals[m.label][k], 3 if k < 3 else 0) + ("%" if k == 3 else ""))
        rows.append(row)
    return table(heads, rows)


def cross_source_table(m: Model) -> str:
    xs = m.pc["cross_source"]
    S = xs["sources"]
    L = m.L
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
    return (
        f"<p><b>{m.label} experts</b> &mdash; cell = Jaccard(top-1% pairs) / cosine(usage), "
        "mean over layers.</p>" + table([""] + S, rows)
    )


# --------------------------------------------------------------------------
# numbers for the prose
# --------------------------------------------------------------------------
def stats(m: Model, key="pool_config"):
    p = m.pool(key)
    L = m.L
    q = m.series(key, lambda s: s["structure"]["Q_spectral"])
    d = dict(
        q=q,
        q_mid=sum(q[2:]) / (L - 2),
        q_1plus=sum(q[1:]) / (L - 1),
        nc=m.series(key, lambda s: s["structure"]["n_louvain_communities"]),
        eff=m.series(key, lambda s: s["usage"]["effective_experts"]),
        never=m.series(key, lambda s: s["pairs"]["never_coactive_frac"]),
        rho=m.series(key, lambda s: s["token_vs_doc_spearman"]),
        med=m.series(key, lambda s: s["pairs"]["lift_median"]),
        p99=m.series(key, lambda s: s["pairs"]["lift_p99"]),
        c999=m.series(key, lambda s: s["conditional"]["p999"]),
        cgt=m.series(key, lambda s: 100 * s["conditional"]["frac_pairs_gt_0p1"]),
        upd=p["unique_experts_per_doc"]["mean_by_layer"],
    )
    d["eff_frac"] = [e / m.Es for e in d["eff"]]
    d["upd_frac"] = [u / m.Es for u in d["upd"]]
    d["psq"] = {
        s: sum(v["layers"][str(li)]["Q_spectral"] for li in range(2, L)) / (L - 2)
        for s, v in p["per_source"].items()
    }
    xs = p["cross_source"]
    S = len(xs["sources"])
    jv = [
        sum(xs["layers"][str(li)]["top1pct_pair_jaccard"][a][b] for li in range(L)) / L
        for a in range(S)
        for b in range(S)
        if a < b
    ]
    d["jmin"], d["jmax"], d["jmean"] = min(jv), max(jv), sum(jv) / len(jv)
    return d


def rng(v, nd=2, lo=None, hi=None):
    v = v[lo:hi]
    return f"{min(v):.{nd}f}&ndash;{max(v):.{nd}f}"


def mean(v):
    return sum(v) / len(v)


def both(models, fn):
    """'512: x; 1024: y' string."""
    return "; ".join(f"{m.label}: {fn(m)}" for m in models)


def compare_fig(name, caption):
    return (
        card(
            "results",
            "Results &mdash; both models on one axis",
            fig_row(img_tag(COMPARE / name, caption)),
        )
        if (COMPARE / name).exists()
        else ""
    )


# --------------------------------------------------------------------------
# tabs
# --------------------------------------------------------------------------
def build_overview(models) -> str:
    st = {m.label: stats(m) for m in models}
    st64 = {m.label: stats(m, "pool_64") for m in models if m.p64}
    goal = (
        "<p><b>sparse_experts</b> asks how EMO's emergent routing structure changes when each expert is a "
        "quarter of the size (expert hidden 256 instead of 1024) and the total expert count grows "
        "128 &rarr; 1024 at a fixed 8 active (7 routed + 1 shared). This page: <b>which experts fire "
        "together?</b> For every layer, how often is each pair of standard experts active on the same "
        "token, and on the same document, on documents the models never saw &mdash; is that co-activation "
        "organised into blocks &mdash; and <b>does doubling the expert count (512 &rarr; 1024, same expert "
        "size, same 10B tokens, same documents) change the organisation?</b></p>"
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
        "together far more than independent routing would predict &mdash; in every layer.</li>"
        "<li><b>H2 (depth)</b>: structure is strongest in the first layer (token identity) and weakens with "
        "depth as routing becomes more contextual.</li>"
        "<li><b>H3 (domain)</b>: a large part of the block structure is domain specialisation: within a single "
        "data source the graph is less modular, and the strongest pairs differ between sources.</li>"
        "<li><b>H4 (pools)</b>: pinning the randpool eval pool to 64 experts per document makes token-level "
        "and document-level structure coincide, since routing is then confined per document.</li>"
        "<li><b>H5 (expert count)</b>: with twice as many experts of the same size, each expert is used on a "
        "smaller slice of tokens, so the co-activation graph becomes sparser and <i>more</i> modular, and "
        "domain specialisation sharpens (lower within-source Q relative to pooled, lower cross-source "
        "overlap of the strongest pairs).</li>"
        "</ul>"
    )
    approach = (
        f"<p>Run each checkpoint natively (no HF conversion) over the same {models[0].pc['num_docs']:,} whole "
        f"held-out documents ({models[0].pc['num_tokens']:,} tokens) packed like training, hook every layer's "
        "router to capture the top-7 routed experts per token, and accumulate exact E&times;E pair counts per "
        "layer and per source at token level (both active on the token) and document level (both active "
        "somewhere in the doc). Compare against independent routing via <i>lift</i> = observed / expected "
        "and the conditional rate P(E<sub>j</sub> | E<sub>i</sub>), cluster the lift graph, and score block "
        "structure with Newman modularity Q against a shuffled-partition null. Repeat with the eval pool "
        "pinned to 64. Every table and figure below shows the models side by side, "
        + " / ".join(f"{m.label} experts ({m.Es} standard + 1 shared)" for m in models)
        + ".</p>"
    )
    items = [
        "<li><b>H1 supported in both models.</b> Modularity of the token co-activation graph: "
        + both(
            models,
            lambda m: f"Q = {rng(st[m.label]['q'], 2, 1)} in layers 1&ndash;15, {st[m.label]['q'][0]:.2f} in layer 0",
        )
        + " (shuffled null &asymp; 0; Louvain "
        + both(
            models, lambda m: f"{min(st[m.label]['nc'])}&ndash;{max(st[m.label]['nc'])} communities"
        )
        + "). Usage is broad in both: effective experts "
        + both(
            models,
            lambda m: f"{100 * min(st[m.label]['eff_frac']):.0f}&ndash;{100 * max(st[m.label]['eff_frac']):.0f}% of {m.Es}",
        )
        + ".</li>"
    ]
    if len(models) > 1:
        a, b = models[0], models[1]
        sa, sb = st[a.label], st[b.label]
        items.append(
            f"<li><b>H5 (expert count).</b> Doubling the experts moves mean Q (layers 1+) from "
            f"{sa['q_1plus']:.3f} to {sb['q_1plus']:.3f} ({sb['q_1plus'] - sa['q_1plus']:+.3f}); Louvain communities "
            f"{min(sa['nc'])}&ndash;{max(sa['nc'])} vs {min(sb['nc'])}&ndash;{max(sb['nc'])}; the share of pairs "
            f"that never co-activate beyond layer 2 goes from {100 * max(sa['never'][3:]):.1f}% to "
            f"{100 * max(sb['never'][3:]):.1f}% (max over layers); lift p99 from {rng(sa['p99'], 0, 1)} to "
            f"{rng(sb['p99'], 0, 1)}; a document touches {100 * mean(sa['upd_frac']):.0f}% of the standard experts "
            f"per layer with 512 and {100 * mean(sb['upd_frac']):.0f}% with 1024. Within-source modularity "
            f"(dclm, layers 2+) {sa['psq']['dclm']:.2f} &rarr; {sb['psq']['dclm']:.2f} against pooled "
            f"{sa['q_mid']:.2f} &rarr; {sb['q_mid']:.2f}; cross-source top-pair overlap (mean Jaccard) "
            f"{sa['jmean']:.2f} &rarr; {sb['jmean']:.2f}. See the per-tab conclusions for the reading.</li>"
        )
    items.append(
        "<li><b>H2 partly.</b> Layer 0 is far more modular ("
        + both(models, lambda m: f"{100 * st[m.label]['never'][0]:.0f}% of pairs never co-occur")
        + ") but beyond it Q is flat with depth and rises slightly in the last layers; token&ndash;document "
        "agreement grows with depth ("
        + both(
            models,
            lambda m: f"&rho; {st[m.label]['rho'][0]:.2f} &rarr; {st[m.label]['rho'][-1]:.2f}",
        )
        + ").</li>"
    )
    items.append(
        "<li><b>H3 supported.</b> Within one source Q is lower than pooled ("
        + both(
            models,
            lambda m: f"pooled {st[m.label]['q_mid']:.2f} vs dclm {st[m.label]['psq']['dclm']:.2f}, code {st[m.label]['psq']['starcoder']:.2f}",
        )
        + "), and the top-1% pairs overlap only "
        + both(models, lambda m: f"{st[m.label]['jmin']:.2f}&ndash;{st[m.label]['jmax']:.2f}")
        + " (Jaccard) between sources.</li>"
    )
    if st64:
        items.append(
            "<li><b>H4 supported.</b> With the pool pinned to 64, token-vs-document agreement is "
            + both(
                [m for m in models if m.p64],
                lambda m: f"&rho; {rng(st64[m.label]['rho'])}, Q {rng(st64[m.label]['q'])}",
            )
            + ".</li>"
        )
    headline = "<ul>" + "".join(items) + "</ul>"
    if (COMPARE / "compare_summary.png").exists():
        headline += fig_row(
            img_tag(
                COMPARE / "compare_summary.png",
                "Per-layer summary, both models: modularity (spectral k=8 solid, Louvain dashed), effective "
                "experts as a fraction of the standard experts, token-vs-document lift agreement; top row "
                "full routing, bottom row pool 64.",
            )
        )
    status = (
        "<ul>"
        + "".join(
            f"<li><b>{m.run.split('_step')[0]}</b>: {TRAIN.get(m.label, '')}; analysis on step2384. "
            f"Beaker: {BEAKER.get(m.label, '')}. Data: "
            f"<code>sparse_experts/coactivation/{m.run}/pool_{{config,64}}/</code>.</li>"
            for m in models
        )
        + "<li><b>sparse_8of128_10b / sparse_8of256_10b</b>: scripts ready, not launched.</li>"
        "<li>The 1024-expert training run needed a grouped_gemm &gt;512-expert kernel fix (host-resident "
        "batch sizes; numerically verified on Beaker) &mdash; see the sparse_experts README.</li></ul>"
    )
    return (
        card("goal", "Goal", goal)
        + card("goal", "Hypotheses", hyp)
        + card("method", "Approach in one paragraph", approach)
        + card("results", "Headline findings", headline)
        + card("results", "Status", status)
    )


def build_method(models) -> str:
    pc = models[0].pc
    data = (
        "<ul>"
        "<li><b>Source</b>: the 20B&ndash;40B training-stream document window "
        "(<code>meta_learning/data/meta128_20B-40B/</code>, 18.3M whole docs, 36.9B tokens), extracted with "
        "the same seed-0 data order these runs used. Both models trained on the first 10B tokens, so every "
        "document here is unseen by both. Window mix by tokens: dclm 93.4%, proof-pile-2 3.2%, starcoder 1.8%, "
        "pes2o 1.6%, olmo-mix 0.1%.</li>"
        f"<li><b>Sample</b>: {pc['num_docs']:,} docs stratified by source &mdash; "
        + ", ".join(f"{s} {n:,}" for s, n in pc["n_doc_per_source"].items())
        + " &mdash; chosen on the compact per-doc label file, token ids pulled in one parallel pass over "
        "the 16 shards (<code>sample_docs.py</code>). Each doc truncated to its first 4,095 tokens + EOS "
        "(9% of docs are longer but hold 59% of tokens; truncation bounds per-doc influence). "
        f"{pc['num_tokens']:,} scored tokens: "
        + ", ".join(f"{s} {n:,}" for s, n in pc["n_tok_per_source"].items())
        + ". <b>The same packed sequences are fed to both models.</b></li>"
        "<li><b>Aggregates</b>: <i>sample</i> (as sampled) and <i>natural</i> (per-source counts reweighted "
        "to the window's token shares). Tables use the sample aggregate; the per-source tab separates "
        "the domains.</li>"
        "</ul>"
    )
    fwd = (
        "<ul>"
        "<li><b>Models</b>: "
        + " and ".join(f"{m.run.split('_step')[0]}/step2384" for m in models)
        + " built from the checkpoint's config.json and loaded with the native DCP loader (same recipe as "
        "the k=32-CPT and meta_learning CE evals), bf16, LM head replaced by identity (routing only). No HF "
        "conversion: the HF port routes over all experts and cannot pin the document pool.</li>"
        "<li><b>Routing</b>: exactly the eval-time routing of the randpool router. Full pass: eval pool = all "
        "experts, so each token takes the top-7 of the standard experts (511 or 1023) plus the shared expert "
        "(last index). Pool-64 pass: per document, the top-64 experts by document-summed router probability "
        "form the pool; each token takes its top-7 inside it.</li>"
        "<li><b>Packing</b>: docs EOS-terminated, first-fit-decreasing into 4096-token rows, "
        "<code>doc_lens</code> passed so attention is intra-document and the router sees per-document "
        "segments, as in training. EOS and pad tokens excluded from all counts.</li>"
        "<li><b>Capture</b>: forward hook on each <code>blocks.*.feed_forward_moe.router</code>; the router "
        "returns <code>expert_indices (B, S, 8)</code> with the shared expert in the last slot (asserted). "
        "Counts accumulated as one-hot matmuls per batch into int64 (exact; self-tested against a naive "
        "loop). 8 GPUs (torchrun, docs sharded by id, rank-0 merge); ~80k tokens/s/GPU for 512, both "
        "passes in ~5 min.</li>"
        "</ul>"
    )
    statsc = (
        "<p><b>Running example used below.</b> One layer, N = 1,000,000 scored tokens; each token "
        "activates 7 of the E standard experts (E = 511 or 1023). Expert A is active on 20,000 tokens, "
        "expert B on 10,000, and both together on 1,000.</p>"
        "<ul>"
        "<li><b>Usage</b> p<sub>i</sub> = C<sub>ii</sub>/N: the share of tokens an expert is active on. "
        "A: 0.02, B: 0.01; perfectly even usage would put every expert at 7/E (0.0137 for 511, 0.0068 for "
        "1023). <b>Effective #experts</b> = exp(entropy of the usage distribution): E if all experts were "
        "used equally, 100 if only 100 were used equally. Tables report it as a % of E so the two models are "
        "comparable. <b>Gini</b>: 0 = perfectly even, 1 = one expert takes everything.</li>"
        "<li><b>Expected co-activation under independence</b> = N &middot; p<sub>A</sub> &middot; p<sub>B</sub> "
        "= 1,000,000 &times; 0.02 &times; 0.01 = 200 tokens: how often A and B would share a token if the "
        "router picked them without regard to each other.</li>"
        "<li><b>Lift</b> = observed / expected = 1,000 / 200 = 5. Lift 1 = independent; 5 = five times "
        "more often than chance; 0.2 = they avoid each other. Heatmaps show log<sub>2</sub> lift (0 = "
        'independent, +2.3 = lift 5; dark blue = never co-fired). "lift median" / "lift p99" '
        "describe this distribution over all pairs (130k for 511 experts, 523k for 1023): a median below 1 "
        "with p99 near 10 means most pairs are slightly below chance while a small tail is strongly bound. "
        "Lift is scale-free in E, so it compares directly across the two models.</li>"
        "<li><b>Conditional rate</b> P(E<sub>j</sub> | E<sub>i</sub>) = C<sub>ij</sub> / C<sub>ii</sub>: "
        'P(B | A) = 1,000 / 20,000 = 0.05 and P(A | B) = 1,000 / 10,000 = 0.10 &mdash; "when A fires, how '
        'often does B fire too", hence asymmetric (rows = conditioning expert). Independent routing gives '
        "&asymp; 7/E for every pair (0.014 vs 0.007), so absolute values are naturally lower for 1024.</li>"
        "<li><b>Pairs never co-active</b>: the share of pairs with C<sub>ij</sub> = 0. With twice as many "
        "experts and the same 51M tokens each pair has ~4&times; fewer expected co-activations, so this "
        "share rises mechanically for 1024; the lift tail is the fairer comparison.</li>"
        "<li><b>Modularity Q</b>: experts are nodes, co-activation counts are edge weights. Given a split "
        "into groups, Q = (fraction of total edge weight inside groups) &minus; (fraction expected if edges "
        "were rewired at random keeping each expert's total). If 45% of co-activation mass is inside our 8 "
        "groups but random wiring would put 15% inside, Q = 0.30. Q &asymp; 0 = no block structure; "
        "&ge; 0.3 = strong community structure. <b>Within-cluster mass</b> is that 45% on its own "
        "(8 equal random groups give 0.125).</li>"
        "<li><b>Spectral (k = 8) vs Louvain</b>: spectral clustering is told to find 8 groups (normalised "
        "spectral clustering of the lift graph) and we report Q for that split; Louvain chooses the number "
        "of groups itself, so its community count and Q are an independent check that the structure is "
        "not an artifact of picking 8. <b>Shuffled null</b>: same group sizes, experts assigned at random "
        "&rarr; Q &asymp; 0, confirming the reported Q is not something any partition would give.</li>"
        "<li><b>Token vs document (Spearman &rho;)</b>: lift computed twice, from token co-activation and "
        "from document co-occurrence (both experts used anywhere in the same document), then "
        "rank-correlated over pairs. 0.5 = pairs strong at token level tend to be strong at document level, "
        "far from perfectly.</li>"
        "<li><b>Experts per doc</b>: distinct experts a document touches in a layer (capped at 64 by "
        "construction under the pool-64 pass); also shown as a fraction of E.</li>"
        "<li><b>Per source</b>: usage and Q recomputed on each source's own tokens. <b>Jaccard of top-1% "
        "pairs</b>: take each source's 1% highest-lift pairs (support &ge; 20) and measure "
        "intersection / union between two sources; 0.2 = only one in five of the strongest pairs is shared "
        "between, e.g., code and web text. <b>Usage cosine</b>: cosine similarity of the two sources' "
        "usage vectors.</li>"
        "</ul>"
    )
    valid = (
        "<ul>"
        "<li>Every matrix symmetric; diagonal = 7 &times; tokens exactly; shared expert never in a routed slot; "
        "indices within [0, E&minus;1].</li>"
        "<li>Pool-64 pass: no (document, layer) touches more than 64 distinct experts; full routing reaches "
        "up to 508 (512 model).</li>"
        "<li>400-doc pilot and the full run agree on every per-layer statistic to within sampling noise "
        "(512 model).</li>"
        "<li>The 1024-expert forward exercises the &gt;512-group grouped-GEMM host path, verified "
        "bit-exact on forward / input-gradient against a per-expert matmul loop before the training run.</li>"
        "</ul>"
    )
    files = (
        "<ul>"
        "<li><code>scripts/sparse_experts/coactivation/PROPOSAL.md</code> &mdash; approved design.</li>"
        "<li><code>sample_docs.py</code>, <code>extract_coactivation.py</code>, <code>launch_beaker.sh</code> "
        "(<code>MODEL=</code> selects the arm), <code>analyze_coactivation.py</code>, "
        "<code>compare_runs.py</code>; <code>scripts/sparse_experts/build_report.py</code> (this page).</li>"
        "<li>Outputs: <code>sparse_experts/coactivation/&lt;run&gt;/pool_{config,64}/</code>: counts.npz "
        "(c_tok, c_doc as (source, layer, E, E) int64; n_tok, n_doc), topk.npy (N, 16, 7) int16, "
        "tok_doc.npy, tok_pos.npy, info.json. Analysis JSON + figures under "
        "<code>claude_outputs/sparse_experts/coactivation/</code>.</li>"
        "</ul>"
    )
    return (
        card("method", "Data", data)
        + card("method", "Forward pass and capture", fwd)
        + card("method", "Statistics", statsc)
        + card("method", "Validation", valid)
        + card("method", "Files", files)
    )


def results_block(models, key, tag):
    """Side-by-side results cards shared by the full-routing and pool-64 tabs."""
    return (
        card(
            "results",
            "Results &mdash; per-layer table (side by side)",
            "<p>Each metric is shown for the 512-expert model and the 1024-expert model in adjacent "
            "columns. Q spectral uses the k=8 spectral partition of the lift graph; within-cluster mass = "
            "share of co-activation counts inside those clusters (8 equal random clusters give 0.125).</p>"
            + paired_layer_table(models, key),
        )
        + card(
            "results",
            "Results &mdash; token-level lift, cluster-ordered",
            "<p>log2 lift of token-level co-activation per layer; experts ordered by spectral cluster then "
            "usage; black lines = cluster boundaries; clipped to &plusmn;3 (dark blue = never co-active). "
            "Left 512, right 1024.</p>"
            + pair_row(
                models,
                lambda m: m.fig(f"{tag}_lift_tok_grid.png", "token-level log2 lift per layer"),
            ),
        )
        + card(
            "results",
            "Results &mdash; document-level lift, same ordering",
            pair_row(
                models,
                lambda m: m.fig(
                    f"{tag}_lift_doc_grid.png",
                    "document-level log2 lift per layer (both experts used somewhere in the document), same ordering",
                ),
            ),
        )
        + card(
            "results",
            "Results &mdash; conditional co-activation P(E<sub>j</sub> | E<sub>i</sub>)",
            "<p>Directed view of the same counts: cell (i, j) = N(E<sub>i</sub>, E<sub>j</sub>) / N(E<sub>i</sub>), "
            "the fraction of expert i's tokens on which j was also active (rows = conditioning expert; "
            "asymmetric). Independent routing gives 7/E: 0.014 (512) vs 0.007 (1024), so compare the "
            "tails relative to the baseline.</p>"
            + paired_conditional_table(models, key)
            + pair_row(
                models,
                lambda m: m.fig(
                    f"{tag}_cond_tok_grid.png", "token-level P(E_j | E_i) per layer, scale 0..0.2"
                ),
            )
            + pair_row(
                models,
                lambda m: m.fig(
                    f"{tag}_cond_doc_grid.png",
                    "document-level P(doc uses j | doc uses i), scale 0..1",
                ),
            ),
        )
        + card(
            "results",
            "Results &mdash; usage and lift distributions",
            pair_row(
                models,
                lambda m: m.fig(
                    f"{tag}_usage.png", "sorted per-expert token share by layer (log scale)"
                ),
            )
            + pair_row(
                models,
                lambda m: m.fig(
                    f"{tag}_lift_hist.png",
                    "distribution of log2 lift over pairs with >0 co-activations",
                ),
            ),
        )
        + card("results", "Results &mdash; strongest pairs", top_pairs_table(models, key))
    )


def build_full(models) -> str:
    st = {m.label: stats(m) for m in models}
    concl = ""
    for m in models:
        s = st[m.label]
        concl += (
            f"<p><b>{m.label} experts.</b> Block structure in every layer (Q {rng(s['q'], 2, 1)} vs null 0), "
            f"layer 0 a different regime (Q = {s['q'][0]:.2f}, {100 * s['never'][0]:.0f}% of pairs never co-occur, "
            f"{s['nc'][0]} Louvain communities). Beyond layer 0 modularity does not decay with depth. Usage stays "
            f"broad ({100 * min(s['eff_frac']):.0f}&ndash;{100 * max(s['eff_frac']):.0f}% effective experts); the "
            f"organisation is the heavy lift tail (p99 {rng(s['p99'], 0, 1)}). Token- and document-level lift "
            f"agree moderately (&rho; {s['rho'][0]:.2f} &rarr; {s['rho'][-1]:.2f}).</p>"
        )
    if len(models) > 1:
        a, b = st[models[0].label], st[models[1].label]
        concl += (
            "<p><b>512 vs 1024.</b> Mean Q over layers 1+: "
            f"{a['q_1plus']:.3f} vs {b['q_1plus']:.3f}; Louvain communities "
            f"{min(a['nc'])}&ndash;{max(a['nc'])} vs {min(b['nc'])}&ndash;{max(b['nc'])}; "
            f"effective experts {100 * mean(a['eff_frac']):.0f}% vs {100 * mean(b['eff_frac']):.0f}% of the "
            f"standard experts; a document touches {100 * mean(a['upd_frac']):.0f}% vs "
            f"{100 * mean(b['upd_frac']):.0f}% of them per layer; lift p99 {rng(a['p99'], 0, 1)} vs "
            f"{rng(b['p99'], 0, 1)}; conditional p99.9 {rng(a['c999'])} vs {rng(b['c999'])} against baselines "
            "0.014 vs 0.007. Read with the caveat in Method: with 4&times; fewer expected co-activations per "
            'pair, the 1024 model\'s raw pair counts are sparser, which inflates "never co-active" and can '
            "only lower a spectral Q; a higher Q for 1024 is therefore a conservative signal, a lower one is "
            "ambiguous.</p>"
        )
    return (
        card(
            "goal",
            "Question",
            "<p>Under each model's own inference routing (all experts available, top-7 + shared), is pairwise "
            "co-activation organised into blocks, how strong is the organisation, how does it change with "
            "depth &mdash; and does it differ between 512 and 1024 experts?</p>",
        )
        + card(
            "hypothesis",
            "Hypothesis",
            "<p>H1: block structure in every layer, well above the shuffled null. H2: strongest in layer 0, "
            "weakening with depth. H5: sparser, more modular with 1024 experts.</p>",
        )
        + card(
            "metrics",
            "Metrics",
            "<p>Per layer and per model: effective experts (% of E), unused experts; Q of the k=8 spectral "
            "partition and of Louvain communities (null: shuffled partition); within-cluster mass; share of "
            "pairs never co-active; lift median / p99; Spearman between token- and document-level lift; the "
            "conditional rate P(E<sub>j</sub> | E<sub>i</sub>) and its tail; distinct experts per document.</p>",
        )
        + compare_fig(
            "compare_pairs.png",
            "Full routing, both models: % pairs never co-active (symlog), lift median, lift p99, conditional "
            "P(j|i) p99.9, by layer.",
        )
        + results_block(models, "pool_config", "pool_config")
        + card("results", "Conclusion", concl)
    )


def build_pool64(models) -> str:
    ms = [m for m in models if m.p64]
    if not ms:
        return "<p>not available</p>"
    st = {m.label: stats(m, "pool_64") for m in ms}
    stf = {m.label: stats(m) for m in ms}
    concl = ""
    for m in ms:
        s, sf = st[m.label], stf[m.label]
        concl += (
            f"<p><b>{m.label} experts.</b> Pinning the pool makes token- and document-level structure nearly the "
            f"same thing (&rho; {rng(s['rho'])} vs {rng(sf['rho'])} under full routing). Q is steady at "
            f"{rng(s['q'])}; usage concentrates to {100 * min(s['eff_frac']):.0f}&ndash;{100 * max(s['eff_frac']):.0f}% "
            f"effective experts (lowest in layer 0).</p>"
        )
    if len(ms) > 1:
        a, b = st[ms[0].label], st[ms[1].label]
        concl += (
            f"<p><b>512 vs 1024 under the same 64-expert pool.</b> Mean Q {mean(a['q']):.3f} vs {mean(b['q']):.3f}; "
            f"&rho; {rng(a['rho'])} vs {rng(b['rho'])}; effective experts {100 * mean(a['eff_frac']):.0f}% vs "
            f"{100 * mean(b['eff_frac']):.0f}% of E. The pool size is the same absolute 64 for both, i.e. 12.5% of "
            "the 512 model's experts but 6.3% of the 1024 model's, so the 1024 model is the more selective "
            "setting here.</p>"
        )
    return (
        card(
            "goal",
            "Question",
            "<p>EMO's selective-expert use restricts each document to a pool of experts. If the eval pool is "
            "pinned to 64, how do usage, block structure and the token/document relationship change &mdash; "
            "and does the 1024-expert model behave differently under the same absolute pool?</p>",
        )
        + card(
            "hypothesis",
            "Hypothesis",
            "<p>H4: token-level and document-level structure coincide once routing is confined per document; "
            "the pool selection itself (top-64 by document-summed probability) concentrates usage.</p>",
        )
        + card(
            "metrics",
            "Metrics",
            "<p>Same per-layer metrics as the full-routing tab, on the same tokens, with the router's "
            "<code>eval_document_expert_pool</code> set to 64; distinct experts per document as the sanity "
            "bound (must be &le; 64).</p>",
        )
        + compare_fig(
            "compare_unique_per_doc.png",
            "Distinct routed experts per document per layer, absolute and as a fraction of the standard "
            "experts; solid = full routing, dashed = pool 64.",
        )
        + results_block(ms, "pool_64", "pool_64")
        + card("results", "Conclusion", concl)
    )


def build_sources(models) -> str:
    st = {m.label: stats(m) for m in models}
    concl = ""
    for m in models:
        s = st[m.label]
        concl += (
            f"<p><b>{m.label} experts.</b> Within a single source the graph is far less modular (layers 2&ndash;15 mean Q: "
            + ", ".join(f"{k} {v:.2f}" for k, v in s["psq"].items())
            + f") than pooled ({s['q_mid']:.2f}); the strongest pairs overlap only {s['jmin']:.2f}&ndash;{s['jmax']:.2f} "
            "between sources. The largest blocks are sets of experts that serve the same domain; layer 0 is the "
            "exception, modular within every source.</p>"
        )
    if len(models) > 1:
        a, b = st[models[0].label], st[models[1].label]
        concl += (
            "<p><b>512 vs 1024.</b> Ratio of within-source to pooled Q (dclm, layers 2+): "
            f"{a['psq']['dclm'] / a['q_mid']:.2f} vs {b['psq']['dclm'] / b['q_mid']:.2f}; mean cross-source "
            f"top-pair Jaccard {a['jmean']:.2f} vs {b['jmean']:.2f}. A lower ratio and lower overlap for 1024 "
            "would mean the extra experts went into sharper domain specialisation (H5); similar values mean "
            "the organisation is the same, just spread over more experts.</p>"
            "<p>Caveat: the minority sources have 2.6&ndash;9.7M tokens each versus 29M for dclm; their lift "
            "graphs are sparser (more so for 1024), which lowers the spectral Q modestly.</p>"
        )
    return (
        card(
            "goal",
            "Question",
            "<p>How much of the pooled block structure is domain specialisation? Is the co-activation graph "
            "modular <i>within</i> a single data source, do the strongest pairs agree across sources &mdash; "
            "and does doubling the experts sharpen the specialisation?</p>",
        )
        + card(
            "hypothesis",
            "Hypothesis",
            "<p>H3: within-source modularity is markedly lower than pooled, and the top pairs are largely "
            "source-specific. H5: more so with 1024 experts.</p>",
        )
        + card(
            "metrics",
            "Metrics",
            "<p>Per source and layer: Q of the k=8 spectral partition on that source's own tokens, effective "
            "experts (% of E); between sources: Jaccard of the top-1% pairs by lift (support &ge; 20) and "
            "cosine of usage vectors, averaged over layers.</p>",
        )
        + card(
            "results",
            "Results &mdash; per-source table (full routing, side by side)",
            paired_source_table(models),
        )
        + compare_fig(
            "compare_per_source.png",
            "Q per source and layer (full routing), pooled in black; one panel per model.",
        )
        + card(
            "results",
            "Results &mdash; cross-source overlap of the strongest pairs",
            pair_row(
                models,
                lambda m: m.fig(
                    "per_source.png",
                    "left: Q per source and layer; right: Jaccard overlap of the top-1% lift pairs between sources, mean over layers",
                ),
            ),
        )
        + card(
            "results",
            "Results &mdash; cross-source agreement tables",
            "".join(cross_source_table(m) for m in models),
        )
        + card("results", "Conclusion", concl)
    )


def build_next() -> str:
    return card(
        "method",
        "Next steps",
        """
<ul>
<li><b>Complete the sweep</b>: launch sparse_8of128_10b and sparse_8of256_10b and run the identical pass, so
modularity and domain specialisation can be read as a function of expert count at fixed expert size;
add the 128-expert full-size baseline (meta128_vanilla/step2384) for the size axis.</li>
<li><b>Matched-sparsity comparison</b>: for the 1024 model, a pool of 128 (12.5%, the 512 model's pool-64 ratio)
next to the absolute pool of 64.</li>
<li><b>Expert matching across models</b>: align 512 and 1024 experts by usage profile over the shared documents
to test whether the 1024 blocks are refinements (splits) of the 512 blocks.</li>
<li><b>Cross-layer</b>: adjacent-layer co-activation from the saved per-token indices, to test whether the blocks
form consistent pathways through depth.</li>
<li><b>Semantic labels</b>: attach the k=32 document clusters (available for this window) to the Louvain
communities to name the blocks.</li>
</ul>""",
    )


def main():
    models = load_models()
    tabs = [
        ("overview", "Overview", build_overview(models)),
        ("method", "Method", build_method(models)),
        ("full", "1 · Full routing", build_full(models)),
        ("pool64", "2 · Pool pinned to 64", build_pool64(models)),
        ("sources", "3 · Per source", build_sources(models)),
        ("next", "Next steps", build_next()),
    ]
    nav = "".join(f'<button data-target="{tid}">{name}</button>' for tid, name, _ in tabs)
    sections = "".join(f'<section class="tab" id="{tid}">{body}</section>' for tid, _, body in tabs)
    labels = " vs ".join(m.label for m in models)
    title = f"EMO sparse_experts: expert co-activation, {labels} quarter-size experts"
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
<style>.muted{{color:var(--muted);font-weight:normal;font-size:0.85em}}</style>
</head>
<body>
<header>
<a class="home-link" href="/">&larr; all reports</a>
<h1>{html.escape(title)}</h1>
<p>sparse_experts &mdash; quarter-size experts, 8 active of 128/256/512/1024 &middot; pairwise expert
co-activation per layer on 40k unseen documents (51M tokens), full routing and pool-64 routing, token-
and document-level, per source &middot; {labels} experts side by side &middot; generated by
scripts/sparse_experts/build_report.py</p>
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
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB) — models: {labels}")


if __name__ == "__main__":
    main()
