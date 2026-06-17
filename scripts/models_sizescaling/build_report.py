"""Build a self-contained HTML report for the size-scaling specialization analyses.

Reads the figures and summary JSONs produced by analyses 1-4 under
``claude_outputs/models_sizescaling/`` and emits a single tabbed HTML page
(one tab per analysis, plus an overview) with all images base64-embedded, so
the file can be shared or synced to S3 on its own.

Usage (after analyses 1-4 have run):

    python scripts/models_sizescaling/build_report.py \\
        [--base-dir claude_outputs/models_sizescaling] \\
        [--output claude_outputs/models_sizescaling/report.html]
"""

import argparse
import base64
import json
import statistics
from pathlib import Path

MODELS = [
    ("emo_1b4b_130b", "32e", "~4B"),
    ("emo_1b7b_130b", "64e", "~7B"),
    ("emo_1b11b_130b", "96e", "~11B"),
    ("emo_1b14b_130b", "128e", "~14B"),
]
PAIRS = ["32e_vs_64e", "64e_vs_96e", "96e_vs_128e", "32e_vs_128e"]
EMB = "probs"  # primary embedding type; topk_freq variants go in collapsibles


# --------------------------------------------------------------------------
# HTML helpers
# --------------------------------------------------------------------------


def img_tag(path: Path, caption: str) -> str:
    if not path.is_file():
        return f'<p class="missing">[missing figure: {path}]</p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        '<figure>'
        f'<img src="data:image/png;base64,{data}" alt="{caption}" '
        'onclick="this.classList.toggle(\'zoom\')" title="click to zoom">'
        f'<figcaption>{caption}</figcaption></figure>'
    )


def fig_row(*figs: str) -> str:
    return '<div class="figrow">' + "".join(figs) + "</div>"


def card(kind: str, title: str, body: str) -> str:
    return f'<div class="card {kind}"><h3>{title}</h3>{body}</div>'


def details(summary: str, body: str) -> str:
    return f"<details><summary>{summary}</summary>{body}</details>"


def table(headers: list, rows: list) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# --------------------------------------------------------------------------
# Per-tab builders
# --------------------------------------------------------------------------


def build_overview(base: Path) -> str:
    info_path = base / "weborganizer" / MODELS[0][0] / "info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    docs = f"{info.get('total_docs', '?'):,}" if info else "?"
    toks = f"{info.get('total_tokens', 0) / 1e6:.0f}M" if info else "?"

    model_rows = [
        (run, lbl, lbl[:-1], f"{int(lbl[:-1]) - 1}", params)
        for run, lbl, params in MODELS
    ]
    body = f"""
<p><strong>Question:</strong> when an EMO model's expert pool grows
(32 &rarr; 64 &rarr; 96 &rarr; 128) with active parameters held fixed, how does the
emergent modular structure change? Do experts specialize harder, do new
specializations appear, or does the same organization just get carved finer?</p>

{card("goal", "Setup", table(
    ["Run", "Label", "Experts", "Standard (excl. shared)", "Total params"],
    model_rows,
) + '''
<p>All four models share everything except <code>num_experts</code>: 1B active
parameters, top-k&nbsp;=&nbsp;8 with 1 shared expert (7 routed), and the identical
130B-token OLMoE-mix-0824 recipe with the same data order. Checkpoints:
<code>models_sizescaling/&lt;run&gt;/step30995-hf</code>.</p>
<p class="note"><strong>Caveat:</strong> the models do <em>not</em> share expert
initialization &mdash; a single sequential RNG stream initializes each model, and the
per-block router weight (whose size depends on the expert count) is drawn before
the expert weights, so the streams diverge at block 0. Expert <em>i</em> in one
model has no relation to expert <em>i</em> in another; all cross-model
correspondence below is discovered functionally (Analysis 4).</p>''')}

{card("method", "Pipeline", f'''
<ol>
<li><strong>Extraction</strong> &mdash; run all four models over the identical
{docs} documents ({toks} tokens, 24 web topics) and record per-document,
per-layer, per-expert routing usage.</li>
<li><strong>Usage trends</strong> &mdash; how many experts does a typical document
effectively use, as a number and as a fraction of the pool?</li>
<li><strong>Expert profiles</strong> &mdash; does each individual expert get more
topic-specialized as the pool grows?</li>
<li><strong>Expert matching</strong> &mdash; match experts across models by their
usage fingerprints: splitting vs redundancy vs novelty.</li>
<li><strong>Cluster attribution</strong> &mdash; are the semantically meaningful
document clusters (k-means on router probs) driven by a few individual experts
or by the broad activation pattern?</li>
</ol>''')}

{card("results", "Headline findings", '''
<ul>
<li><strong>Roughly constant fraction, slowly compressing</strong> &mdash; a typical
document's effective expert count grows 19.9 &rarr; 59.8 as the pool grows
32 &rarr; 128 (3&times; for a 4&times; pool): utilization stays high but drifts from
64% to 47% of the pool.</li>
<li><strong>Experts stay topic-level generalists</strong> &mdash; per-expert topic
entropy sharpens only mildly (0.944 &rarr; 0.903 normalized) and essentially no
expert becomes a single-topic specialist, even at 128 experts.</li>
<li><strong>Scaling = splitting, not novelty</strong> &mdash; despite independent
initializations, Hungarian-matched cross-model correlation holds at ~0.73 for
every pair (even 32e&harr;128e). Each 32e expert is covered by ~9.5 weakly
inter-correlated 128e experts (they divide its documents rather than copy it),
and only 0.4&ndash;3.5% of larger-model experts are novel.</li>
<li><strong>Mildly rising redundancy</strong> &mdash; within-model nearest-neighbor
similarity creeps up (0.56 &rarr; 0.61) with pool size, consistent with EMO's
prunability.</li>
<li><strong>Clusters are pattern-driven, with experts as redundant markers</strong>
&mdash; the document clusters are not caused by particular experts: each cluster's
signature is spread over hundreds of (layer, expert) dims, and deleting its
strongest dims still re-forms an equally topic-aligned clustering. Yet single
near-perfect marker experts exist for every cluster (AUC ~0.99), and ~32 dims
suffice to nearly recover the structure &mdash; the information is massively
redundant (Analysis 5).</li>
</ul>'''
)}
"""
    return body


def build_extraction(base: Path) -> str:
    wb = base / "weborganizer"
    info_path = wb / MODELS[0][0] / "info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}

    figs = [
        img_tag(
            wb / run / f"doc_{EMB}_coverage_above_uniform_heatmap.png",
            f"{lbl}: topic &times; layer expert coverage (fraction of experts used above uniform)",
        )
        for run, lbl, _ in MODELS
    ]
    entropy_figs = [
        img_tag(wb / run / f"doc_{EMB}_entropy_heatmap.png", f"{lbl}: routing entropy")
        for run, lbl, _ in MODELS
    ]

    stats_rows = []
    for run, lbl, _ in MODELS:
        p = wb / run / "info.json"
        if p.is_file():
            i = json.loads(p.read_text())
            stats_rows.append(
                (lbl, i["num_standard_experts"], f"{i['total_docs']:,}",
                 f"{i['total_tokens']/1e6:.1f}M", i["num_layers"])
            )

    return f"""
{card("goal", "Goal", '''<p>Produce <em>comparable</em> per-document expert-usage
fingerprints for all four models: every downstream analysis (trends, profiles,
matching) is a cheap replay of these arrays.</p>''')}

{card("method", "Method", f'''
<p>{info.get('total_docs', '?'):,} documents (~{info.get('total_tokens', 0)/1e6:.0f}M
tokens) sampled uniformly across {info.get('num_topics', 24)} cc_all_dressed
weborganizer topics, with a shared <code>mix_composition.json</code> (shuffle seed
{info.get('shuffle_seed', 42)}) so <strong>all four models see the identical
document set</strong> &mdash; the basis for per-document matching in Analysis 4.</p>
<p>One forward pass per model records, per document and per layer, (a) mean router
probabilities (<code>probs</code>, used throughout this report) and (b) top-k
selection frequencies (<code>topk_freq</code>). The shared expert is excluded, so a
model with E experts contributes E&minus;1 standard experts. Arrays have shape
(docs, layers &times; standard experts). Cost: ~40&nbsp;min/model on one A100; everything
downstream is CPU-seconds.</p>''' + table(
        ["Model", "Standard experts", "Docs", "Tokens", "Layers"], stats_rows))}

{card("results", "Results", f'''
<p>Topic-blocked structure in expert coverage is visible at every scale &mdash; the
same qualitative organization, drawn over more experts. Rows (topics) are aligned
across models via a shared <code>topic_order.json</code>.</p>
{fig_row(*figs[:2])}
{fig_row(*figs[2:])}'''
+ details("Routing-entropy heatmaps (per topic &times; layer)", fig_row(*entropy_figs[:2]) + fig_row(*entropy_figs[2:])))}
"""


def _trend_means(base: Path):
    path = base / "trends" / f"{EMB}_trends.json"
    if not path.is_file():
        return []
    t = json.loads(path.read_text())
    rows = []
    for _, lbl, _ in MODELS:
        if lbl not in t:
            continue
        m = t[lbl]
        eff = statistics.mean(v for row in m["eff_topic_layer"] for v in row)
        # cov_topic_layer stores a *count* of experts above uniform, not a fraction
        cov = statistics.mean(v for row in m["cov_topic_layer"] for v in row)
        e = m["num_standard_experts"]
        rows.append((lbl, e, f"{eff:.1f}", f"{100*eff/e:.1f}%",
                     f"{cov:.1f} ({100*cov/e:.1f}%)"))
    return rows


def build_trends(base: Path) -> str:
    tr = base / "trends"
    rows = _trend_means(base)
    return f"""
{card("goal", "Goal", '''<p>As the pool grows, does a document keep using a
roughly <strong>constant number</strong> of experts (extra experts idle), a
<strong>constant fraction</strong> (usage scales with the pool), or something in
between?</p>''')}

{card("method", "Method", '''<p>For each document and layer, the routing
distribution over standard experts gives an <strong>effective expert count</strong>
2<sup>H</sup> (H = entropy in bits) and a <strong>coverage</strong> = fraction of
experts used above the uniform level 1/E. Both are averaged per topic and layer,
then summarized across the 24 topics and 16 layers.</p>''')}

{card("results", "Results", table(
        ["Model", "Standard experts E", "Effective experts (mean)",
         "Fraction of pool", "Experts above uniform (share of pool)"], rows) + '''
<p>Neither extreme holds, but the data sit much closer to <strong>constant
fraction</strong>: a 4&times; bigger pool yields a 3&times; higher effective expert
count. Added experts genuinely get used &mdash; there is no idle capacity &mdash;
while utilization slowly concentrates (64% &rarr; 47% of the pool).</p>'''
+ fig_row(
    img_tag(tr / f"{EMB}_eff_experts_vs_E.png",
            "Effective experts vs pool size, per layer (left: absolute, right: fraction of pool)"),
    img_tag(tr / f"{EMB}_coverage_vs_E.png",
            "Coverage above uniform vs pool size"))
+ details("Per-topic breakdown",
          fig_row(img_tag(tr / f"{EMB}_per_topic_eff.png",
                          "Effective experts per topic (absolute + fraction)")))
+ details("topk_freq variant (top-k selection frequencies instead of router probabilities)",
          fig_row(img_tag(tr / "topk_freq_eff_experts_vs_E.png", "Effective experts (topk_freq)"),
                  img_tag(tr / "topk_freq_coverage_vs_E.png", "Coverage (topk_freq)"))))}
"""


def build_profiles(base: Path) -> str:
    pr = base / "profiles"
    path = pr / f"{EMB}_profiles_summary.json"
    rows = []
    if path.is_file():
        s = json.loads(path.read_text())
        for _, lbl, _ in MODELS:
            if lbl not in s:
                continue
            m = s[lbl]
            rows.append((
                lbl,
                f"{m['mean_entropy']:.3f}",
                f"{min(m['median_entropy_per_layer']):.3f}",
                f"{100*m['frac_specialists_lt_0.5']:.1f}%",
            ))
    return f"""
{card("goal", "Goal", '''<p>Does each <em>individual</em> expert become more
topic-specialized when there are more experts to share the work?</p>''')}

{card("method", "Method", '''<p>Each expert gets a 24-dim <strong>topic
profile</strong>: its mean usage per topic, normalized over topics. Specialization
is measured as normalized topic entropy H / log<sub>2</sub>(24) &mdash; 1.0 =
perfectly uniform generalist, 0.0 = single-topic specialist. The 24-topic
granularity is fixed across models, so scores are directly comparable.</p>''')}

{card("results", "Results", table(
        ["Model", "Mean entropy", "Most-specialized layer (median)",
         "Specialists (entropy &lt; 0.5)"], rows) + '''
<p>Specialization sharpens only <strong>mildly</strong> with pool size, and
essentially no expert becomes a topic specialist even at 128 experts. Later
layers are consistently more specialized than early ones in every model.
Combined with Analysis 4, this says the extra experts subdivide work at a
<em>finer-than-topic</em> granularity (sub-domains within topics), not by
claiming whole topics.</p>'''
+ fig_row(
    img_tag(pr / f"{EMB}_entropy_vs_layer.png",
            "Per-expert topic entropy by layer (median &pm; IQR per model)"),
    img_tag(pr / f"{EMB}_entropy_cdf.png",
            "Entropy CDFs at four representative layers"))
+ details("Max topic share CDFs",
          fig_row(img_tag(pr / f"{EMB}_max_share_cdf.png",
                          "CDF of each expert's largest single-topic share"))))}
"""


def _pair_medians(base: Path, sub: str = "matching") -> list:
    """Per-pair medians (across the 16 layers) of every match statistic."""
    out = []
    for pair in PAIRS:
        path = base / sub / pair / "match_summary.json"
        if not path.is_file():
            continue
        s = json.loads(path.read_text())
        med = lambda k: statistics.median(l[k] for l in s["per_layer"])  # noqa: E731
        out.append({
            "dir": pair,
            "name": f"{s['label_a']} &harr; {s['label_b']}",
            "matched": med("matched_sim_median"),
            "splits": med("mean_splits_per_a_expert"),
            "coherence": med("split_coherence"),
            "novel": med("frac_novel_b"),
            "nn_a": med("nn_a_median"),
            "nn_b": med("nn_b_median"),
        })
    return out


def build_matching(base: Path) -> str:
    ma = base / "matching"
    pairs = _pair_medians(base)
    featured = ma / "32e_vs_128e"

    def pair_figs(fig_name: str, caption: str) -> str:
        return details(
            "Same figure for the other pairs (32e&harr;64e, 64e&harr;96e, 96e&harr;128e)",
            fig_row(*[img_tag(ma / p / fig_name,
                              f"{p.replace('_vs_', ' &harr; ')}: {caption}")
                      for p in PAIRS[:-1]]))

    q1_rows = [(p["name"], f"{p['matched']:.2f}") for p in pairs]
    q2_rows = [(p["name"], f"{p['splits']:.1f}", f"{p['coherence']:.2f}",
                f"{100*p['novel']:.1f}%") for p in pairs]
    q3_rows = [(p["name"], f"{p['nn_a']:.2f}", f"{p['nn_b']:.2f}") for p in pairs]

    return f"""
{card("goal", "Goal", '''<p>When the pool grows (32 &rarr; 128 experts), what does
the larger model <em>do</em> with its extra experts? Three competing pictures:</p>
<ul>
<li><strong>H-split (refinement)</strong> &mdash; the large model keeps the small
model's organization but carves it finer: each small-model expert's role is
<em>divided</em> among several large-model experts.</li>
<li><strong>H-copy (redundancy)</strong> &mdash; the extra experts are near-duplicates
of existing ones; the organization stays the same granularity, just oversampled.</li>
<li><strong>H-novel (new territory)</strong> &mdash; the extra experts pick up
specializations the small model simply does not have.</li>
</ul>
<p>A fourth possibility frames all of them: maybe each scale finds an
<em>unrelated</em> organization altogether. The models share <strong>no expert
initialization</strong> (see Overview), so any correspondence found below emerged
from the data alone &mdash; expert indices are meaningless across models and all
matching is functional.</p>''')}

{card("method", "Setup (shared by all three questions)", '''<p>All four models were
run over the <strong>identical</strong> 29,042 documents (Analysis 1), so each
expert has a comparable <strong>fingerprint</strong>: its mean routing probability
on every document &mdash; a 29,042-dim vector describing <em>which documents it
works on</em>. Two experts with correlated fingerprints play the same functional
role, regardless of their indices or which model they live in.</p>
<p>Per layer, Pearson correlation (over documents) between every
(A-expert,&nbsp;B-expert) pair gives a similarity matrix &mdash; A is the smaller
model, B the larger. Every statistic below is read off these matrices; all table
values are medians across the 16 layers.</p>''')}

<h4>Question 1 &middot; Does the small model's organization survive in the large one?</h4>

{card("hypothesis", "Q1 &middot; Hypothesis", '''<p>If each scale settled into an
unrelated organization, the best one-to-one pairing of A-experts to B-experts
would show correlation near 0 (the null for independently initialized models). If
the large model contains the small model's structure, matched correlations should
be high &mdash; at every layer.</p>''')}

{card("method", "Q1 &middot; Method", '''<p>Hungarian (optimal one-to-one)
assignment on each layer's correlation matrix: every A-expert is paired with a
distinct B-expert so that total correlation is maximized. This is the strictest
form of correspondence &mdash; no B-expert can be reused.</p>''')}

{card("metrics", "Q1 &middot; Metrics", '''<ul>
<li><strong>Matched similarity</strong> &mdash; median Pearson correlation of the
Hungarian-matched pairs. Reading: ~0 = unrelated organizations; 1.0 = the small
model is literally embedded in the large one. Note a structural ceiling: if a
parent expert's documents are <em>divided</em> among several children (H-split),
no single child can correlate 1.0 with the parent &mdash; so even perfect
refinement caps this metric well below 1.</li>
</ul>''')}

{card("results", "Q1 &middot; Results", table(["Pair", "Matched similarity (median)"], q1_rows) + '''
<p><strong>Verdict: the same organization at every scale.</strong> Matched
correlation holds at ~0.71&ndash;0.73 for every pair &mdash; including 32e&harr;128e
directly, with no degradation over the 4&times; gap. Given independent
initializations and the splitting ceiling above, this is strong correspondence:
the 32-expert model's division of labor survives recognizably inside the
128-expert pool. The heatmaps make it visible as a bright diagonal band.</p>'''
+ fig_row(
    img_tag(featured / "corr_heatmap_layers.png",
            "32e &harr; 128e: correlation heatmaps at 4 layers, B columns sorted by Hungarian match. The bright diagonal band in the first 31 columns = matched structure; the rest are each B-expert's best remaining alignment"),
    img_tag(featured / "matched_sim_vs_layer.png",
            "32e &harr; 128e: matched similarity by layer (median &pm; IQR)"))
+ pair_figs("matched_sim_vs_layer.png", "matched similarity by layer"))}

<h4>Question 2 &middot; What do the extra experts do &mdash; split, copy, or new territory?</h4>

{card("hypothesis", "Q2 &middot; Hypothesis", '''<p>The three pictures make distinct
predictions about the B-experts related to one A-expert:</p>
<ul>
<li><strong>H-split</strong> &mdash; several B-experts match each A-expert, and those
B-experts are <em>dissimilar to each other</em> (they partition the parent's
documents); few B-experts are unmatched.</li>
<li><strong>H-copy</strong> &mdash; several B-experts match each A-expert and are
<em>highly similar to each other</em> (duplicates).</li>
<li><strong>H-novel</strong> &mdash; a sizable fraction of B-experts match
<em>no</em> A-expert at all.</li>
</ul>''')}

{card("method", "Q2 &middot; Method", '''<p>From each layer's correlation matrix:
for every A-expert, collect the B-experts correlating with it above a match
threshold (0.4), then measure how those B-experts correlate <em>with each
other</em> (using the B-model's within-model correlation matrix). Independently,
for every B-expert take its best correlation to <em>any</em> A-expert; below a
novelty threshold (0.3) it counts as having no counterpart.</p>''')}

{card("metrics", "Q2 &middot; Metrics", '''<ul>
<li><strong>Splits per A-expert</strong> &mdash; how many B-experts match one
A-expert (corr &gt; 0.4). &gt;1 means the parent's role is shared by several
children.</li>
<li><strong>Split coherence</strong> &mdash; mean pairwise correlation among those
children. Reading: high (&rarr;1) = redundant copies (H-copy); low = they divide
the parent's documents between them (H-split).</li>
<li><strong>Novel B-experts</strong> &mdash; fraction of B-experts with max corr to
any A-expert &lt; 0.3. Reading: large = new specializations (H-novel).</li>
</ul>''')}

{card("results", "Q2 &middot; Results", table(
        ["Pair", "Splits per A-expert", "Split coherence", "Novel B-experts"],
        q2_rows) + '''
<p><strong>Verdict: splitting, not copying, and almost no novelty.</strong> Each
32e expert is covered by ~6 64e experts, and ~9.5 128e experts &mdash; but those
children correlate only ~0.34&ndash;0.40 with <em>each other</em>: they divide the
parent's documents rather than duplicate it. Meanwhile only 0.4&ndash;3.5% of
larger-model experts lack any counterpart. So a 4&times; bigger pool buys a
<em>finer partition of the same space</em>, not new specializations &mdash;
consistent with Analysis 3, where individual experts barely sharpen at the
24-topic granularity (the splitting happens at finer-than-topic level).</p>'''
+ fig_row(
    img_tag(featured / "splitting_novelty_vs_layer.png",
            "32e &harr; 128e: splits per A-expert (green, left axis) and fraction of novel B-experts (red, right axis) by layer"))
+ pair_figs("splitting_novelty_vs_layer.png", "splitting &amp; novelty by layer"))}

<h4>Question 3 &middot; Does within-model redundancy grow with the pool?</h4>

{card("hypothesis", "Q3 &middot; Hypothesis", '''<p>If splitting eventually
saturates &mdash; more experts than the data has distinctions to assign them &mdash;
extra experts should start crowding: each expert acquires closer near-duplicates
within its own model as the pool grows. This is also the property that would
explain EMO's tolerance to expert pruning.</p>''')}

{card("method", "Q3 &middot; Method", '''<p>For each model separately: correlate
every expert's fingerprint with every other expert's <em>in the same model and
layer</em>, and take each expert's nearest neighbor (highest correlation,
excluding itself).</p>''')}

{card("metrics", "Q3 &middot; Metrics", '''<ul>
<li><strong>Within-model NN correlation</strong> &mdash; median over experts of the
nearest-neighbor correlation. Reading: how close is each expert's closest
functional sibling? Rising with pool size = growing redundancy.</li>
</ul>''')}

{card("results", "Q3 &middot; Results", table(
        ["Pair", "NN corr: smaller model", "NN corr: larger model"], q3_rows) + '''
<p><strong>Verdict: mildly rising redundancy.</strong> The median nearest-neighbor
correlation creeps up from 0.56 (32e) to 0.61 (128e) &mdash; experts do get closer
functional siblings as the pool grows, but only gradually; the pool is far from
saturated duplicates. This mild redundancy is consistent with EMO&rsquo;s
prunability and with Analysis 5&rsquo;s finding that cluster information is
massively redundant across experts.</p>'''
+ fig_row(
    img_tag(featured / "nn_redundancy_vs_layer.png",
            "Within-model nearest-neighbor redundancy by layer (32e vs 128e, median &pm; IQR)"))
+ pair_figs("nn_redundancy_vs_layer.png", "within-model NN redundancy by layer"))}

{card("results", "Conclusion", '''<p><strong>Scaling the pool refines the existing
organization rather than replacing or duplicating it.</strong> Q1: the small
model's structure survives recognizably at every scale (matched corr ~0.73, even
32e&harr;128e, despite independent initializations). Q2: extra experts subdivide
existing roles &mdash; ~9.5 weakly inter-correlated 128e children per 32e parent
&mdash; with near-zero novelty. Q3: redundancy rises only mildly (NN corr 0.56
&rarr; 0.61). Together with Analysis 2 (effective expert count grows as a roughly
constant fraction of the pool) and Analysis 3 (per-expert topic profiles barely
sharpen), the picture is consistent: EMO discovers one stable organization of the
data and spends a larger pool on carving the same map at finer granularity.</p>''')}
"""


def _token_setup(base: Path) -> dict:
    p = base / "weborganizer_tokens" / MODELS[0][0] / "info.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def build_token_matching(base: Path) -> str:
    """Token-level twin of build_matching: the same Hungarian/splitting/redundancy
    analysis, but each expert's fingerprint is its routing over individual TOKENS
    rather than its mean over DOCUMENTS."""
    ma = base / "matching_tokens"
    info = _token_setup(base)
    doc_pairs = {p["dir"]: p for p in _pair_medians(base, "matching")}
    tok_pairs = _pair_medians(base, "matching_tokens")
    featured = ma / "32e_vs_128e"

    n_tok = f"{info.get('total_tokens', 0):,}" if info else "?"
    n_doc = f"{info.get('total_docs', 0):,}" if info else "?"
    tpd = info.get("tokens_per_doc", 100)

    def pair_figs(fig_name: str, caption: str) -> str:
        return details(
            "Same figure for the other pairs (32e&harr;64e, 64e&harr;96e, 96e&harr;128e)",
            fig_row(*[img_tag(ma / p / fig_name,
                              f"{p.replace('_vs_', ' &harr; ')}: {caption}")
                      for p in PAIRS[:-1]]))

    # Headline: doc-level vs token-level side by side, per pair.
    cmp_rows = []
    for p in tok_pairs:
        d = doc_pairs.get(p["dir"], {})
        dv = lambda k: f"{d[k]:.2f}" if k in d else "&ndash;"  # noqa: E731
        cmp_rows.append((
            p["name"],
            f"{dv('matched')} / <strong>{p['matched']:.2f}</strong>",
            f"{dv('splits')} / <strong>{p['splits']:.1f}</strong>",
            f"{100*d['novel']:.1f}% / <strong>{100*p['novel']:.1f}%</strong>" if "novel" in d
            else f"&ndash; / <strong>{100*p['novel']:.1f}%</strong>",
            f"{d.get('nn_b', float('nan')):.2f} / <strong>{p['nn_b']:.2f}</strong>",
        ))

    q1_rows = [(p["name"], f"{p['matched']:.2f}") for p in tok_pairs]
    q2_rows = [(p["name"], f"{p['splits']:.1f}", f"{p['coherence']:.2f}",
                f"{100*p['novel']:.1f}%") for p in tok_pairs]
    q3_rows = [(p["name"], f"{p['nn_a']:.2f}", f"{p['nn_b']:.2f}") for p in tok_pairs]

    return f"""
{card("goal", "Goal", '''<p>All four models have the same active parameters and the
same training; they differ only in how many experts the router can choose from
(32, 64, 96, 128). When that pool grows, what does the larger model <em>do</em>
with its extra experts? Three competing pictures:</p>
<ul>
<li><strong>H-split (refinement)</strong> &mdash; the large model keeps the small
model's organization but carves it finer: each small-model expert's role is
<em>divided</em> among several large-model experts.</li>
<li><strong>H-copy (redundancy)</strong> &mdash; the extra experts are
near-duplicates of existing ones; the granularity stays the same, just
oversampled.</li>
<li><strong>H-novel (new territory)</strong> &mdash; the extra experts pick up
specializations the small model simply does not have.</li>
</ul>
<p>A fourth possibility frames all three: perhaps each scale finds an
<em>unrelated</em> organization altogether. This matters because the models share
<strong>no expert initialization</strong> &mdash; each is seeded by its own RNG
stream, and the per-block router weight (whose size depends on the expert count) is
drawn before the expert weights, so the streams diverge at the very first block.
Expert <em>i</em> in one model therefore has nothing to do with expert <em>i</em> in
another: every correspondence below is discovered <strong>functionally</strong>,
from behaviour on shared data, never by index.</p>
<p>This tab answers those questions using a <strong>token-level</strong> fingerprint
for each expert &mdash; how it behaves on individual tokens. A companion tab
(&ldquo;4 &middot; Expert matching&rdquo;) runs the identical analysis on a coarser
<em>document-level</em> fingerprint; where the two disagree is itself informative
and is flagged below, but this page stands on its own.</p>''')}

{card("method", "Setup (shared by all three questions)", f'''<p>Each expert's
<strong>fingerprint</strong> is the vector of its routing activity across a fixed
set of inputs; two experts with correlated fingerprints play the same functional
role, whatever their indices or which model they live in. Here the inputs are
individual <strong>tokens</strong>: all four models are run over the
<strong>identical</strong> {n_tok} tokens (the first {tpd} tokens of {n_doc}
documents, spread evenly across 24 web topics), recording, per token and per layer,
each expert's softmax routing probability. The same tokens in the same order are
fed to every model, so a given fingerprint coordinate is literally the same token in
all four &mdash; the prerequisite for comparing experts across independently
initialized models.</p>
<p>Per layer, we take the Pearson correlation (over the {n_tok} tokens) between
every (A-expert, B-expert) pair, giving a similarity matrix from which every
statistic below is read. <strong>A is the smaller model, B the larger</strong>, and
all table values are <strong>medians across the 16 layers</strong>.</p>
<p class="note"><strong>Token-level vs document-level &mdash; what to expect.</strong>
Routing happens one token at a time, so a token fingerprint is the native object; a
document fingerprint (the companion tab) instead averages each expert's routing over
a whole document. Averaging cancels per-token noise and therefore <em>inflates</em>
correlations across the board, so token-level numbers are uniformly lower. The
question is which conclusions are robust to that shift &mdash; the comparison table
near the end makes the gap explicit.</p>''')}

<h4>Question 1 &middot; Does the small model's organization survive in the large one?</h4>

{card("hypothesis", "Q1 &middot; Hypothesis", '''<p>If each scale settled into an
unrelated organization, then even the <em>best possible</em> one-to-one pairing of
A-experts to B-experts would show correlation near 0 (the null expectation for
independently initialized models). If instead the large model contains the small
model's structure, those matched correlations should be high &mdash; and at
<em>every</em> layer.</p>''')}

{card("method", "Q1 &middot; Method", '''<p>On each layer's correlation matrix we run
<strong>Hungarian (optimal one-to-one) assignment</strong>: every A-expert is paired
with a distinct B-expert so that the total matched correlation is maximized, with no
B-expert reused. This is the strictest possible test of correspondence &mdash; it
cannot reuse a good B-expert to flatter several A-experts.</p>''')}

{card("metrics", "Q1 &middot; Metric", '''<ul>
<li><strong>Matched similarity</strong> &mdash; median Pearson correlation of the
Hungarian-matched pairs. Reading: ~0 = unrelated organizations; 1.0 = the small
model is literally embedded in the large one. Note a structural ceiling: if a parent
expert's tokens are <em>divided</em> among several children (the H-split picture), no
single child can correlate 1.0 with the parent &mdash; so even perfect refinement
caps this metric below 1, and more so for single tokens than for document
averages.</li>
</ul>''')}

{card("results", "Q1 &middot; Results", table(
        ["Pair", "Matched similarity (median)"], q1_rows) + '''
<p><strong>Verdict: the organization survives at token grain.</strong> Matched
correlation is ~0.48&ndash;0.52 for every pair &mdash; far above the ~0 null for
independently-initialized models &mdash; and, decisively, shows <em>no
degradation across the 4&times; gap</em>: 32e&harr;128e = 0.51, indistinguishable
from the adjacent pairs. So &ldquo;the same organization at every scale&rdquo; is
not a product of document averaging; it holds token-by-token. The absolute level is
lower than the document-level ~0.73 (see the comparison table) simply because a
single-token fingerprint is far noisier than a document mean, and the splitting
ceiling bites harder per token &mdash; not because the cross-scale correspondence is
any weaker. The heatmaps show this as a bright diagonal band.</p>'''
+ fig_row(
    img_tag(featured / "corr_heatmap_layers.png",
            "32e &harr; 128e (token-level): correlation heatmaps at 4 layers, B columns sorted by Hungarian match. The bright diagonal band in the first 31 columns is the matched structure"),
    img_tag(featured / "matched_sim_vs_layer.png",
            "32e &harr; 128e (token-level): matched similarity by layer (median &pm; IQR)"))
+ pair_figs("matched_sim_vs_layer.png", "matched similarity by layer"))}

<h4>Question 2 &middot; What do the extra experts do &mdash; split, copy, or new territory?</h4>

{card("hypothesis", "Q2 &middot; Hypothesis", '''<p>The three pictures make distinct
predictions about the B-experts (large model) related to one A-expert (small
model):</p>
<ul>
<li><strong>H-split</strong> &mdash; several B-experts match each A-expert, and those
B-experts are <em>dissimilar to each other</em> (they partition the parent's
tokens); few B-experts are left unmatched.</li>
<li><strong>H-copy</strong> &mdash; several B-experts match each A-expert and are
<em>highly similar to each other</em> (duplicates).</li>
<li><strong>H-novel</strong> &mdash; a sizable fraction of B-experts match
<em>no</em> A-expert at all.</li>
</ul>''')}

{card("method", "Q2 &middot; Method", '''<p>From each layer's correlation matrix: for
every A-expert, collect the B-experts that correlate with it above a match threshold
(0.4), then measure how those B-experts correlate <em>with each other</em> (using the
large model's own within-model correlation matrix). Separately, for every B-expert
take its single best correlation to <em>any</em> A-expert; if that falls below a
novelty threshold (0.3) the B-expert counts as having no counterpart.</p>''')}

{card("metrics", "Q2 &middot; Metrics", '''<ul>
<li><strong>Splits per A-expert</strong> &mdash; how many B-experts match one
A-expert (corr&nbsp;&gt;&nbsp;0.4). &gt;1 means the parent's role is shared among
several children.</li>
<li><strong>Split coherence</strong> &mdash; mean pairwise correlation <em>among
those children</em>. Reading: high (&rarr;1) = redundant copies (H-copy); low = they
divide the parent's tokens between them (H-split).</li>
<li><strong>Novel B-experts</strong> &mdash; fraction of B-experts whose best
correlation to any A-expert is &lt; 0.3. Reading: large = new specializations
(H-novel).</li>
</ul>
<p class="note">Splits and novelty are <em>threshold-based</em> (counted at fixed
0.4 / 0.3 cutoffs), so they shift whenever the overall correlation scale moves &mdash;
keep that in mind when comparing to the document-level tab. Split coherence is
threshold-free among the matched set and is the more robust signal.</p>''')}

{card("results", "Q2 &middot; Results", table(
        ["Pair", "Splits per A-expert", "Split coherence", "Novel B-experts"],
        q2_rows) + '''
<p><strong>Verdict: splitting, not copying; the apparent novelty is mostly a
threshold effect.</strong> The trustworthy, threshold-free signal is <strong>split
coherence</strong>, and it is low &mdash; ~0.28&ndash;0.32: the larger-model experts
that match a given parent are only weakly correlated <em>with each other</em>, i.e.
they partition the parent's tokens rather than duplicate it. <strong>H-copy is
rejected.</strong> The threshold-based counts read as ~1.3&ndash;1.7 splits per
A-expert and ~12&ndash;33% novel B-experts; but because every correlation sits
lower at token grain, fewer children clear the 0.4 bar and more experts fall under
the 0.3 bar, so these numbers overstate true novelty. The honest reading is
&ldquo;weakly-matched experts under a downshifted scale,&rdquo; not genuine new
territory &mdash; and if the larger pool really held substantial novel
specializations, Q1's Hungarian matched similarity would fall across the 4&times;
gap, which it does not. So <strong>H-split</strong>, with <strong>H-novel</strong>
playing at most a minor role.</p>'''
+ fig_row(
    img_tag(featured / "splitting_novelty_vs_layer.png",
            "32e &harr; 128e (token-level): splits per A-expert (left axis) and novel-B fraction (right axis) by layer"))
+ pair_figs("splitting_novelty_vs_layer.png", "splitting &amp; novelty by layer"))}

<h4>Question 3 &middot; Does within-model redundancy grow with the pool?</h4>

{card("hypothesis", "Q3 &middot; Hypothesis", '''<p>If splitting eventually saturates
&mdash; more experts than the data has distinctions to hand them &mdash; then extra
experts should start crowding: each expert would acquire closer near-duplicates
<em>within its own model</em> as the pool grows. This is also the property that
would explain EMO's tolerance to expert pruning.</p>''')}

{card("method", "Q3 &middot; Method", '''<p>For each model separately: correlate every
expert's fingerprint with every other expert's <em>in the same model and layer</em>,
and take each expert's nearest neighbour (its highest correlation, excluding
itself).</p>''')}

{card("metrics", "Q3 &middot; Metric", '''<ul>
<li><strong>Within-model NN correlation</strong> &mdash; median over experts of that
nearest-neighbour correlation: how close is each expert's most similar functional
sibling? Rising with pool size = growing redundancy.</li>
</ul>''')}

{card("results", "Q3 &middot; Results", table(
        ["Pair", "NN corr: smaller model", "NN corr: larger model"], q3_rows) + '''
<p><strong>Verdict: no redundancy growth at token grain.</strong> Each expert's
nearest within-model sibling correlates only ~0.31 &mdash; and this is essentially
<em>flat</em> across all four pool sizes (~0.31 at 32e, 64e, 96e and 128e alike).
Experts are markedly distinct per token, and growing the pool does not make them
more token-level redundant. (The document-level tab reports a mild rise,
0.56&nbsp;&rarr;&nbsp;0.61; that rise is an averaging artifact &mdash; it does not
appear here.) Whatever underlies EMO's tolerance to expert pruning, it is not that
experts become near-duplicates at the token level as the pool grows.</p>'''
+ fig_row(
    img_tag(featured / "nn_redundancy_vs_layer.png",
            "Within-model nearest-neighbor redundancy by layer (32e vs 128e, token-level, median &pm; IQR)"))
+ pair_figs("nn_redundancy_vs_layer.png", "within-model NN redundancy by layer"))}

<h4>Token-level vs document-level, side by side</h4>

{card("results", "Comparison with the document-level fingerprint", table(
        ["Pair", "Matched sim (doc / <strong>tok</strong>)",
         "Splits per A-expert (doc / <strong>tok</strong>)",
         "Novel B (doc / <strong>tok</strong>)",
         "Within-model NN, larger (doc / <strong>tok</strong>)"], cmp_rows) + '''
<p class="note">Plain = document-level (tab 4); <strong>bold = token-level</strong>
(this tab). Medians across 16 layers.</p>
<p><strong>Document averaging inflates every similarity.</strong> Going from
document to token fingerprints, matched similarity drops ~0.73&nbsp;&rarr;&nbsp;~0.50,
splits-per-expert ~6&ndash;9.5&nbsp;&rarr;&nbsp;~1.3&ndash;1.7, and within-model
redundancy ~0.60&nbsp;&rarr;&nbsp;~0.31, while apparent novelty <em>rises</em>
~1%&nbsp;&rarr;&nbsp;12&ndash;33%. This is exactly what a coarser fingerprint
predicts: averaging a whole document cancels per-token noise and mechanically lifts
all correlations, and the two threshold-based metrics swing once the whole scale
shifts down ~0.2. The value of the token view is separating the conclusions that
are real from the ones that were averaging artifacts.</p>''')}

{card("results", "Conclusion", '''<p><strong>Matching experts one token at a time
&mdash; the level routing actually happens &mdash; keeps the backbone of the scaling
story and trims its margins.</strong> What survives the finer view: (Q1) the small
model's division of labour is recognizable at every scale, with zero degradation
across the 4&times; gap (matched sim ~0.50, 32e&harr;128e = adjacent pairs); (Q2) the
experts that match a parent <em>divide</em> its work rather than duplicate it (split
coherence stays low). What turns out to have been inflated by document averaging: the
absolute split counts and the near-zero novelty (both threshold artifacts of a
downshifted correlation scale), and the mild rise in within-model redundancy, which
disappears at token grain (flat ~0.31).</p>
<p><strong>Methodological takeaway:</strong> document fingerprints inflate every
correlation, so threshold-based metrics (split / novelty counts) must be read
cautiously across grains. The threshold-free Hungarian matched similarity &mdash;
flat across the 4&times; gap in <em>both</em> regimes &mdash; is the trustworthy
anchor, and it tells the same story as the rest of the report: EMO discovers one
stable organization of the data and spends a larger pool on carving that same map at
finer granularity.</p>''')}
"""


def _attr_summary(base: Path, run: str) -> dict:
    path = base / "expert_attribution" / run / "attribution_summary.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _attr_ablation(s: dict, kind: str, m: int, key: str):
    for r in s.get("ablations", []):
        if r["kind"] == kind and r["m"] == m:
            return r[key]
    return None


def build_attribution(base: Path) -> str:
    at = base / "expert_attribution"
    featured_run, featured_lbl = MODELS[-1][0], MODELS[-1][1]

    conc_rows = []
    auc_rows = []
    ablation_rows = []
    for run, lbl, _ in MODELS:
        s = _attr_summary(base, run)
        if not s:
            continue
        c = s["concentration_mean"]
        conc_rows.append((
            lbl,
            s["n_dims"],
            f"{100*c['top1_mass']:.1f}%",
            f"{100*c['top5_mass']:.1f}%",
            f"{100*c['top32_mass']:.1f}%",
            f"{100*c['top100_mass']:.1f}%",
            f"{s['effective_dims_median']:.0f} ({100*s['effective_dims_median']/s['n_dims']:.0f}%)",
        ))
        a = s["best_single_dim_auc"]
        auc_rows.append((
            lbl,
            f"{a['median']:.3f}",
            f"{a['min']:.3f} &ndash; {a['max']:.3f}",
            f"{s['full_pattern_auc_median']:.3f}",
        ))
        ablation_rows.append((
            lbl,
            f"{_attr_ablation(s, 'baseline', 0, 'ari_vs_original'):.2f}",
            f"{_attr_ablation(s, 'drop_top', 16, 'ari_vs_original'):.2f} / "
            f"{_attr_ablation(s, 'drop_random', 16, 'ari_vs_original'):.2f}",
            f"{_attr_ablation(s, 'baseline', 0, 'nmi_vs_topics'):.2f}",
            f"{_attr_ablation(s, 'drop_top', 16, 'nmi_vs_topics'):.2f} / "
            f"{_attr_ablation(s, 'drop_random', 16, 'nmi_vs_topics'):.2f}",
            f"{_attr_ablation(s, 'keep_top', 1, 'nmi_vs_topics'):.2f}",
        ))

    # Per-cluster marker table for the featured model (test 2 evidence).
    fs = _attr_summary(base, featured_run)
    marker_rows = []
    for r in sorted(fs.get("per_cluster", []), key=lambda r: -r["best_dim"]["auc"]):
        bd = r["best_dim"]
        topics = ", ".join(f"{t} ({n})" for t, n in r["top_topics"][:2])
        marker_rows.append((
            r["cluster"], r["size"], topics,
            f"L{bd['layer']} / E{bd['expert']} ({bd['direction']}-activated)",
            f"{bd['auc']:.3f}", f"{r['full_pattern_auc']:.3f}",
        ))

    def per_model_figs(fig_name: str, caption: str) -> str:
        return details(
            "Same figure for the other models (32e / 64e / 96e)",
            fig_row(*[img_tag(at / run / fig_name, f"{lbl}: {caption}")
                      for run, lbl, _ in MODELS[:-1]]))

    return f"""
{card("goal", "Goal", '''<p>The published clustering result shows that k-means on
router-probability embeddings yields semantically meaningful document clusters.
This analysis asks <strong>why</strong>: did each cluster form because of a few
individual, particular experts, or because of the broad expert-activation
pattern?</p>
<p>Two competing pictures of how the clusters could arise:</p>
<ul>
<li><strong>H-experts</strong> &mdash; each cluster is <em>caused</em> by a handful of
dedicated experts: those experts fire for the cluster's documents, k-means picks up
on those few dimensions, and without them the cluster would not exist.</li>
<li><strong>H-pattern</strong> &mdash; each cluster reflects a <em>broad, correlated
shift</em> in how the whole pool routes; many experts each carry a little of the
signal, so no small set of them is load-bearing.</li>
</ul>
<p>Three tests, ordered from descriptive to causal, separate these. All run in the
raw (layer&nbsp;&times;&nbsp;expert) probability space &mdash; every dimension is one
expert at one layer (e.g. 16&nbsp;&times;&nbsp;127&nbsp;=&nbsp;2,032 dims at 128e) &mdash;
on the same k=32 spherical k-means recipe as the published runs
(<code>doc_probs</code>, mean-center &rarr; PCA 95% &rarr; L2).</p>''')}

<h4>Test 1 &middot; Signature concentration (descriptive)</h4>

{card("hypothesis", "Test 1 &middot; Hypothesis", '''<p>If H-experts is right, a
cluster's <em>signature</em> &mdash; the way its documents' average routing differs
from a typical document's &mdash; should be concentrated in a few (layer, expert)
dimensions. If H-pattern is right, the signature should be smeared across hundreds
of dimensions, each contributing a sliver.</p>''')}

{card("method", "Test 1 &middot; Method", '''<p>For each cluster c, compute
&delta;<sub>c</sub> = (mean embedding of c's documents) &minus; (global mean over
all documents), in the raw space. |&delta;<sub>c</sub>| says how strongly each
individual expert-at-a-layer is over- or under-used by that cluster. Sort the
dimensions by |&delta;<sub>c</sub>| and ask how the total mass is distributed. This
is purely descriptive &mdash; no re-clustering, no model re-runs.</p>''')}

{card("metrics", "Test 1 &middot; Metrics", '''<ul>
<li><strong>Top-m mass</strong> &mdash; fraction of the signature's total
|&delta;| carried by its m largest dimensions. Reading: top-5 mass near 100% means
five experts essentially <em>are</em> the signature (H-experts); a few percent means
they are a drop in the bucket (H-pattern).</li>
<li><strong>Effective dims</strong> &mdash; exp(entropy) of the normalized |&delta;|
distribution: the number of dimensions that would carry the mass if it were spread
evenly. Reading: small (tens) &rArr; concentrated; comparable to the full
dimensionality &rArr; distributed.</li>
</ul>''')}

{card("results", "Test 1 &middot; Results", table(
        ["Model", "Dims (layers &times; experts)", "Top-1 dim mass", "Top-5",
         "Top-32", "Top-100", "Effective dims (median)"], conc_rows) + '''
<p class="note">Top-m masses are means over the 32 clusters.</p>
<p><strong>Verdict: distributed &mdash; supports H-pattern.</strong> A cluster's
single strongest dimension carries only ~1.4&ndash;2.1% of its signature, the top 32
dims only ~20&ndash;29%, and the effective dimensionality is roughly <em>half the
entire space</em> (~1,038 of 2,032 dims at 128e). No cluster's identity is written
in a handful of experts. The concentration also <em>decreases</em> as the pool
grows (top-5 mass 8.4% &rarr; 5.5% from 32e to 128e): bigger pools spread the
signature wider.</p>'''
+ fig_row(img_tag(at / featured_run / "signature_concentration.png",
                  f"{featured_lbl}: per cluster, the fraction of |signature| mass in its top-1/5/10/32 dims. Even the best case stays below ~35%"))
+ per_model_figs("signature_concentration.png", "signature concentration"))}

<h4>Test 2 &middot; Single-dim separability (are there marker experts?)</h4>

{card("hypothesis", "Test 2 &middot; Hypothesis", '''<p>Test 1 says no few dims carry
the signature's <em>mass</em> &mdash; but mass is not discriminative power. A single
expert could still be a near-perfect <em>marker</em>: fire if-and-only-if the
document belongs to the cluster, even while contributing little |&delta;|. If
H-experts is right in its weaker, diagnostic form, each cluster should have such a
marker; if even the best single dimension separates poorly, the cluster boundary
must be encoded jointly across many experts.</p>''')}

{card("method", "Test 2 &middot; Method", '''<p>For every cluster, score every raw
dimension on the one-vs-rest task &ldquo;does this dim's value separate the
cluster's documents from everyone else's?&rdquo; using rank-based (Mann-Whitney)
AUC, computed for all dims &times; all clusters at once. Under-activation counts
too: an expert the cluster reliably <em>avoids</em> is also a marker, so each dim's
score is max(AUC, 1&minus;AUC). The baseline is the <strong>full-pattern AUC</strong>:
how well cosine similarity to the cluster's centroid (using <em>all</em> dims, in
the preprocessed space) separates the same cluster.</p>''')}

{card("metrics", "Test 2 &middot; Metrics", '''<ul>
<li><strong>Best single-dim AUC</strong> (per cluster) &mdash; probability that a
random in-cluster document scores higher on that dim than a random out-of-cluster
one. 0.5 = chance, 1.0 = perfect separation.</li>
<li><strong>Full-pattern AUC</strong> &mdash; the same number when the whole
activation pattern is used. The gap between the two is what the rest of the
pattern adds over the best single expert.</li>
</ul>''')}

{card("results", "Test 2 &middot; Results", table(
        ["Model", "Best single-dim AUC (median over clusters)", "Range (min &ndash; max)",
         "Full-pattern AUC (median)"], auc_rows) + '''
<p><strong>Verdict: near-perfect markers exist for every cluster.</strong> The
median best single-dim AUC is ~0.98&ndash;0.99 (worst cluster ~0.96), barely below
the full-pattern ~0.996. So individual experts are excellent <em>identifiers</em>
of each cluster &mdash; H-experts survives in its diagnostic form. Combined with
Test 1, this already hints at heavy redundancy: a dim carrying 2% of the signature
can nonetheless identify the cluster almost perfectly, which is only possible if
many dims carry near-duplicate information. Whether the markers are
<em>load-bearing</em> is exactly what Test 3 settles.</p>'''
+ fig_row(img_tag(at / featured_run / "single_dim_vs_full_auc.png",
                  f"{featured_lbl}: per cluster, best single-dim AUC (red) vs full-pattern AUC (blue). The red curve hugs the blue one"))
+ details("Per-cluster markers for " + featured_lbl + " (which expert, which layer, dominant topics)",
          table(["Cluster", "Docs", "Dominant topics (count)", "Best marker dim",
                 "Marker AUC", "Full-pattern AUC"], marker_rows))
+ per_model_figs("single_dim_vs_full_auc.png", "single-dim vs full-pattern AUC"))}

<h4>Test 3 &middot; Drop / keep ablations (necessity and sufficiency)</h4>

{card("hypothesis", "Test 3 &middot; Hypothesis", '''<p>The causal question, in two
halves:</p>
<ul>
<li><strong>Necessity</strong> &mdash; if the top expert dims <em>cause</em> the
clusters (H-experts), deleting them should destroy the clustering: re-running the
pipeline without them should produce clusters that no longer correspond to the
originals <em>and</em> are no longer topic-aligned. Under H-pattern, an equivalent
organization should re-form from the remaining dims.</li>
<li><strong>Sufficiency</strong> &mdash; conversely, if a few marker dims carry the
information (as Test 2 suggests), keeping <em>only</em> them should reproduce
roughly the same organization.</li>
</ul>''')}

{card("method", "Test 3 &middot; Method", '''<p>For each ablation size m: take the
union over clusters of each cluster's top-m |&delta;| dims (e.g. m=16 &rarr; ~460
distinct dims at 128e), delete those columns from the raw embedding, and re-run the
<em>entire</em> published pipeline from scratch (mean-center &rarr; PCA 95% &rarr;
L2 &rarr; spherical k-means, k=32). Compare the resulting clustering to the original
assignments and to the 24 weborganizer topic labels.</p>
<p>Two controls: (a) <strong>matched random drops</strong> &mdash; delete the same
<em>number</em> of randomly chosen dims, isolating &ldquo;we removed the important
dims&rdquo; from &ldquo;we removed dims at all&rdquo;; (b) a <strong>no-ablation
rerun baseline</strong> &mdash; the pipeline re-run on all dims, which captures how
unstable k-means itself is.</p>
<p>The converse <strong>keep-only</strong> runs use just the union of top-m dims
(m=1 &rarr; ~32 dims of 2,032) and re-run the same pipeline.</p>''')}

{card("metrics", "Test 3 &middot; Metrics", '''<ul>
<li><strong>ARI vs original</strong> &mdash; do the re-formed clusters match the
original assignments? (1 = identical partition, 0 = chance agreement.)</li>
<li><strong>NMI vs topics</strong> &mdash; is the re-formed clustering still aligned
with the external weborganizer topic labels, regardless of whether the specific
cluster boundaries moved? This is the cleaner measure of &ldquo;is an equivalent
semantic organization still there?&rdquo;</li>
</ul>
<p class="note"><strong>How to read ARI here:</strong> re-running k-means after
<em>any</em> perturbation lands at ARI ~0.6&ndash;0.75 (see the random-drop control)
because the k=32 solution is only marginally stable &mdash; for the 32e model even
float-level noise flips the rerun to ARI 0.62. So ARI must be read against the
random-control floor, not against 1.0; a drop below the control means the deleted
dims mattered <em>for the specific cluster boundaries</em>. NMI vs topics is the
external anchor that says whether the semantics survived.</p>''')}

{card("results", "Test 3 &middot; Results", table(
        ["Model", "Rerun baseline ARI", "ARI: drop top-16/cluster (vs random)",
         "NMI topics: baseline", "NMI topics: drop top-16 (vs random)",
         "NMI topics: keep ONLY top-1/cluster (~32 dims)"], ablation_rows) + '''
<p><strong>Verdict: not necessary, nearly sufficient &mdash; the pattern carries the
structure.</strong></p>
<ul>
<li><strong>Necessity fails.</strong> Deleting every cluster's top-16 dims (~460
dims at 128e) does hurt the specific cluster boundaries more than random deletions
(ARI 0.39 vs 0.67) &mdash; the markers do anchor <em>which exact</em> partition
k-means picks. But the re-formed clustering is almost as topic-aligned as the
original (NMI 0.40 vs 0.45 baseline): remove the markers and an equivalent semantic
organization re-emerges from the remaining pattern. Under H-experts the topic
alignment should have collapsed.</li>
<li><strong>Sufficiency nearly holds.</strong> Keeping ONLY each cluster's single
top dim (~32 of 2,032 dims, 1.6%) already recovers NMI-vs-topics 0.41 of the 0.45
baseline; ~240 dims (keep top-8/cluster) fully match it (0.46).</li>
<li><strong>Stable across pool sizes</strong> &mdash; the same drop/keep picture
holds at 32e, 64e, 96e and 128e.</li>
</ul>'''
+ fig_row(img_tag(at / featured_run / "ablation_curves.png",
                  f"{featured_lbl}: drop/keep ablations vs m. Left (ARI): red (drop top-m) falls below gray (drop random) — markers anchor the exact boundaries. Right (NMI vs topics): red barely moves — semantics survive; green (keep only top-m) reaches baseline by ~240 dims"))
+ per_model_figs("ablation_curves.png", "drop/keep ablation curves"))}

{card("results", "Conclusion", '''<p><strong>The clusters are carried by the broad
activation pattern; individual experts are redundant markers, not the cause.</strong>
Test 1: no cluster's signature lives in a few dims (effective dimensionality ~half
the space). Test 2: yet every cluster has a single near-perfect marker expert
(AUC ~0.99) &mdash; possible only because many dims duplicate the same signal.
Test 3: deleting the markers leaves an equally topic-aligned organization to re-form
(necessity fails), while ~32 dims almost suffice to reconstruct it (sufficiency
nearly holds).</p>
<p>In short: cluster information in the router space is <em>massively redundant</em>.
A handful of expert dims would suffice to reconstruct the clusters, and no handful
is load-bearing &mdash; the semantic structure lives in the correlated activation
pattern of hundreds of experts simultaneously. This is the same redundancy that
shows up as EMO's prunability elsewhere in the report.</p>''')}
"""


# --------------------------------------------------------------------------
# Page assembly
# --------------------------------------------------------------------------

CSS = """
:root { --fg:#1e293b; --muted:#64748b; --bg:#f8fafc; --card:#ffffff; --line:#e2e8f0; }
* { box-sizing:border-box; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
       color:var(--fg); background:var(--bg); line-height:1.55; }
header { background:#0f172a; color:#f1f5f9; padding:18px 28px; }
header h1 { margin:0 0 2px; font-size:20px; }
header p { margin:0; color:#94a3b8; font-size:13px; }
.home-link { display:inline-block; margin-bottom:8px; color:#94a3b8; font-size:13px; text-decoration:none; }
.home-link:hover { color:#f1f5f9; }
.topbar { position:sticky; top:0; z-index:10; }
nav { display:flex; gap:6px; flex-wrap:wrap; padding:10px 28px; background:#1e293b; }
nav button { border:0; border-radius:6px; padding:7px 14px; font-size:14px; cursor:pointer;
             background:transparent; color:#cbd5e1; }
nav button:hover { background:#334155; }
nav button.active { background:#3b82f6; color:#fff; }
#subnav { display:flex; gap:6px 16px; flex-wrap:wrap; padding:8px 28px; background:#eef2f7;
          border-bottom:1px solid var(--line); font-size:13px; }
#subnav a { color:#475569; text-decoration:none; white-space:nowrap; }
#subnav a:hover { color:#2563eb; text-decoration:underline; }
#subnav:empty { display:none; }
main { max-width:1180px; margin:0 auto; padding:24px 28px 80px; }
section.tab { display:none; }
section.tab.active { display:block; }
.card { background:var(--card); border:1px solid var(--line); border-left:4px solid var(--line);
        border-radius:8px; padding:16px 20px; margin:16px 0; }
.card h3 { margin:0 0 8px; font-size:15px; text-transform:uppercase; letter-spacing:0.05em;
           scroll-margin-top:96px; }
.card.goal { border-left-color:#2563eb; } .card.goal h3 { color:#2563eb; }
.card.hypothesis { border-left-color:#d97706; } .card.hypothesis h3 { color:#d97706; }
.card.method { border-left-color:#7c3aed; } .card.method h3 { color:#7c3aed; }
.card.metrics { border-left-color:#0e7490; } .card.metrics h3 { color:#0e7490; }
.card.results { border-left-color:#059669; } .card.results h3 { color:#059669; }
table { border-collapse:collapse; margin:12px 0; font-size:14px; width:auto; }
th, td { border:1px solid var(--line); padding:6px 12px; text-align:left; }
th { background:#f1f5f9; }
tbody tr:nth-child(even) { background:#f8fafc; }
.figrow { display:flex; gap:16px; flex-wrap:wrap; margin:14px 0; }
figure { flex:1 1 380px; min-width:300px; max-width:560px; margin:0; }
figure img { width:100%; border:1px solid var(--line); border-radius:6px; cursor:zoom-in; background:#fff; }
figure img.zoom { max-width:none; width:auto; max-height:90vh; position:fixed; inset:0; margin:auto;
                  z-index:100; box-shadow:0 0 0 100vmax rgba(15,23,42,.75); cursor:zoom-out; }
figcaption { font-size:12.5px; color:var(--muted); margin-top:4px; }
details { margin:12px 0; }
summary { cursor:pointer; color:#2563eb; font-size:14px; }
.note { font-size:13px; color:var(--muted); }
.missing { color:#dc2626; font-size:13px; }
code { background:#eef2f7; padding:1px 5px; border-radius:4px; font-size:0.9em; }
h4 { margin:18px 0 4px; }
"""

JS = """
function slug(t){ return t.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/(^-|-$)/g,''); }
function buildSubnav(id) {
  const sec = document.getElementById(id);
  const sub = document.getElementById('subnav');
  sub.innerHTML = '';
  if (!sec) return;
  sec.querySelectorAll('.card > h3').forEach(h => {
    if (!h.id) h.id = id + '--' + slug(h.textContent);
    const a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    a.addEventListener('click', e => { e.preventDefault(); h.scrollIntoView({behavior:'smooth', block:'start'}); });
    sub.appendChild(a);
  });
}
function show(id) {
  document.querySelectorAll('section.tab').forEach(s => s.classList.toggle('active', s.id === id));
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('active', b.dataset.target === id));
  history.replaceState(null, '', '#' + id);
  buildSubnav(id);
}
document.querySelectorAll('nav button').forEach(b => b.addEventListener('click', () => show(b.dataset.target)));
show(location.hash && document.getElementById(location.hash.slice(1)) ? location.hash.slice(1) : 'overview');
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path,
                        default=Path("claude_outputs/models_sizescaling"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    base = args.base_dir
    out = args.output or base / "report.html"

    tabs = [
        ("overview", "Overview", build_overview(base)),
        ("extraction", "1 · Extraction", build_extraction(base)),
        ("trends", "2 · Usage trends", build_trends(base)),
        ("profiles", "3 · Expert profiles", build_profiles(base)),
        ("matching", "4 · Expert matching", build_matching(base)),
        ("matching_tokens", "4.1 · Matching (token-level)", build_token_matching(base)),
        ("attribution", "5 · Cluster attribution", build_attribution(base)),
    ]
    nav = "".join(f'<button data-target="{tid}">{name}</button>' for tid, name, _ in tabs)
    sections = "".join(f'<section class="tab" id="{tid}">{body}</section>'
                       for tid, _, body in tabs)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EMO size-scaling: expert specialization report</title>
<style>{CSS}</style>
</head>
<body>
<header>
<a class="home-link" href="/">&larr; all reports</a>
<h1>EMO size-scaling: how does expert specialization change with pool size?</h1>
<p>models_sizescaling &mdash; 32 / 64 / 96 / 128 experts, 1B active params, identical 130B-token recipe
&middot; generated by scripts/models_sizescaling/build_report.py</p>
</header>
<div class="topbar"><nav>{nav}</nav><div id="subnav"></div></div>
<main>{sections}</main>
<script>{JS}</script>
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
