#!/usr/bin/env python3
"""Build the modular_extension experiment report (self-contained HTML).

A goal-driven writeup (structured like scripts/models_routerfixed/build_report.py) of the
modular_extension program: grow an EMO MoE by adding experts during continued pretraining
under a fixed 64-expert memory budget, using EMO's emergent data clusters to keep added
experts non-redundant. The report leads with the goal + a staged roadmap (Overview), then
documents Stage 1 -- characterizing the emergent clusters (Method/Findings/Per-cluster,
from modular_extension/cluster/emo100b_step23842/expert_attribution/) -- and sketches
Stage 2, a cheap cluster router (Next steps).

CSS/JS/tab structure kept in sync with scripts/models_routerfixed/build_report.py so this
report matches the rest of the reports on https://emo-reports.pages.dev/.

Run:  python scripts/modular_extension/build_report.py
Registered in scripts/publish_reports.sh (deploys to https://emo-reports.pages.dev/).
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATTR = ROOT / "modular_extension/cluster/emo100b_step23842/expert_attribution"
PART = ROOT / "modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-110B/doc_clusters_k64_summary.json"
JOINT = ROOT / "modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-130B/doc_clusters_k64_summary.json"
SUBSTAB = ROOT / "modular_extension/cluster/emo100b_step23842_100B-130B/subsample_stability"
KSEL = ROOT / "modular_extension/cluster/emo100b_step23842_100B-130B/k_selection"
K32CPT = ROOT / "modular_extension/cluster/emo100b_step23842_100B-130B/k32_cpt"
ORACLE_SPLIT = (ROOT / "modular_extension/cluster/emo100b_step23842/doc_classifier/oracle"
                / "cluster_agreement.json")

# Spherical-k-means objective of the REFERENCE k=64 partition (seed 42), computed
# post-hoc from its saved assignments with the same estimator k_selection.py uses
# (assign_all_with_objective). The comparison seeds' objectives are read from the
# k_selection run.jsons at build time.
K64_REF_OBJECTIVE = 0.776350


# --------------------------------------------------------------------------
# HTML helpers (kept in sync with scripts/models_routerfixed/build_report.py)
# --------------------------------------------------------------------------


def card(kind: str, title: str, body: str) -> str:
    return f'<div class="card {kind}"><h3>{title}</h3>{body}</div>'


def table(headers: list, rows: list) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    # wrapped so a wide table scrolls inside its own box on mobile instead of
    # pushing the whole page out of bounds.
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def img_tag(path: Path, caption: str) -> str:
    if not path.is_file():
        return f'<p class="missing">[missing figure: {path.name}]</p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        "<figure>"
        f'<img loading="lazy" src="data:image/png;base64,{data}" alt="{html.escape(caption)}" '
        "onclick=\"this.classList.toggle('zoom')\" title=\"click to zoom\">"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def fig(emb: str, name: str, caption: str) -> str:
    return img_tag(ATTR / emb / name, caption)


def fig_row(*figs: str) -> str:
    return '<div class="figrow">' + "".join(figs) + "</div>"


def details(summary: str, body: str) -> str:
    return f"<details><summary>{summary}</summary>{body}</details>"


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------


def build_overview(soft, hard, n_docs) -> str:
    return f"""
<p><strong>Goal:</strong> grow an EMO mixture-of-experts model <em>beyond the expert count it was trained
with</em> (e.g. 64&nbsp;&rarr;&nbsp;128 experts) by adding experts <em>during</em> continued pretraining
&mdash; while never holding more than 64 experts in memory at once. If this works, model capacity grows
but the per-step training footprint stays flat: at any moment we train some 64-expert subset, and over
time those subsets compose into a larger model.</p>

{card("goal", "Why it's hard &mdash; added experts go redundant", '''
<p>The naive version &mdash; spawn fresh experts, train random 64-expert subsets on the usual data stream
&mdash; wastes the new capacity. With nothing steering them apart, added experts relearn what existing
experts already do; the extra parameters buy little. To make added capacity pay off, training has to push
new experts toward <em>distinct</em> regions of the data.</p>''')}

{card("goal", "The idea &mdash; let EMO's emergent clusters route the capacity", '''
<p>EMO already sorts data into behaviourally distinct groups <em>on its own</em> during pretraining &mdash;
no human-defined priors. That gives us a free partition of the data, which the plan uses directly:</p>
<ol>
<li><strong>Start</strong> from a 64-expert EMO checkpoint that has developed these emergent clusters.</li>
<li><strong>Partition</strong> the upcoming pretraining stream by EMO cluster.</li>
<li><strong>Grow &amp; train</strong> new experts against specific cluster-partitions, so each added expert
gets a distinct slice of data &mdash; minimizing redundancy with what's already there.</li>
</ol>
<p>Everything shipped so far is <strong>Stage&nbsp;1</strong> below: before relying on the clusters we had
to confirm they are real and understand what defines them, because the whole plan rests on them being a
meaningful partition.</p>''')}

{card("method", "Roadmap", _roadmap_table() + '''
<p class="note">Status reflects what's in this report. Stage&nbsp;1 (cluster structure) is complete and
written up in <strong>Method</strong>/<strong>Findings</strong>/<strong>Per-cluster</strong>;
Stage&nbsp;2 (a cheap cluster router) is the next investigation, sketched in <strong>Next steps</strong>.</p>''')}

{card("results", "Stage 1 result &mdash; the clusters are real, and broad-redundant", f'''
<p>We fingerprinted {n_docs:,} documents from the 100B&ndash;110B window by their router behaviour and
clustered them into <strong>{soft['k']} groups</strong>, then asked <em>what makes one cluster different
from another</em>. Answer: each cluster is a <strong>broad, redundant expert-usage signature</strong>. The
distinguishing signal is spread across most of the {soft['n_dims']} experts (the top-5 hold only
~{pct(soft['median_top5_mass'])} of a cluster's deviation), yet any one of hundreds of experts separates
the cluster near-perfectly (best single-expert AUC <strong>{soft['median_best_single_dim_auc']:.3f}</strong>,
~{pct(soft['median_single_vs_full_ratio'])} of the full-pattern {soft['median_full_pattern_auc']:.3f}). So
the clusters are genuine and strongly identifiable &mdash; but they are <em>not</em> owned by a nameable
handful of experts. Full detail in <strong>Findings</strong>.</p>
<p><strong>Why that's encouraging for Stage&nbsp;2.</strong> If a cluster's identity is written redundantly
across hundreds of experts and even a single expert flags it at AUC&nbsp;&asymp;&nbsp;0.99, then a much
<em>cheaper</em> view of a document (e.g. only the first block's activations) plausibly keeps enough of that
signal to classify the cluster <em>without a full forward pass</em> &mdash; which is exactly what
partitioning the data stream at scale needs. That is the hypothesis <strong>Next steps</strong> lays out.</p>''')}

{card("results", "Stage 1 headline numbers", _headline_table(soft, hard) + '''
<p class="note">Both routing views (soft affinity and hard selection) give the same verdict: all 64 clusters
broad-redundant &mdash; broad in <em>which</em> experts, strong in <em>how much</em> each one moves.</p>''')}
"""


def _headline_table(soft, hard) -> str:
    # Transposed: metrics as rows, the two signals as the only two value columns
    # (an 8-wide row-per-signal table overflowed on mobile).
    def split(s):
        vc = s["verdict_counts"]
        return (f'<span class="v-few-expert">{vc["few-expert"]}</span> / '
                f'<span class="v-broad-redundant">{vc["broad-redundant"]}</span> / '
                f'<span class="v-subtle">{vc["subtle"]}</span>')

    metrics = [
        ("verdict split &mdash; few-expert / broad-redundant / subtle",
         split(soft), split(hard)),
        (f"median effective deviating experts (of {soft['n_dims']})",
         f"{soft['median_effective_dims']:.1f}", f"{hard['median_effective_dims']:.1f}"),
        ("median top-5 experts' share of the deviation",
         pct(soft["median_top5_mass"]), pct(hard["median_top5_mass"])),
        ("median best single-expert AUC",
         f"{soft['median_best_single_dim_auc']:.3f}", f"{hard['median_best_single_dim_auc']:.3f}"),
        ("median full-pattern AUC",
         f"{soft['median_full_pattern_auc']:.3f}", f"{hard['median_full_pattern_auc']:.3f}"),
        (f"median experts used / layer (of {soft['n_experts_per_layer']}; corpus "
         f"~{soft['global_effective_experts']:.0f})",
         f"{soft['median_effective_experts_used']:.1f}", f"{hard['median_effective_experts_used']:.1f}"),
    ]
    return table(
        ["metric", "doc_probs<br>(soft affinity)", "doc_topk_freq<br>(hard selection)"],
        metrics,
    )


def build_method(soft, ill) -> str:
    n_layers, n_exp = soft["n_layers"], soft["n_experts_per_layer"]
    thr = int(0.1 * soft["n_dims"])
    return f"""
<p class="note">This tab and <strong>Findings</strong> document <strong>Stage&nbsp;1</strong> of the roadmap
(see <strong>Overview</strong>): confirming the emergent clusters are real and characterizing what defines
them, so the extension plan can rely on them as a data partition.</p>

{card("goal", "A routing fingerprint per document", f'''
<p>The published EMO clustering pipeline fingerprints individual <em>tokens</em>. To reason about
<em>documents</em>, we pool: run the EMO 100B checkpoint (step 23,842) with router logits exposed and
<strong>average each document's per-token routing</strong>, per layer and per expert. That gives every
document a {n_layers}&times;{n_exp} = {soft['n_dims']}-dimensional vector
(<strong>layer-major</strong>: dimension <code>d</code> is layer <code>d//{n_exp}</code>, expert
<code>d%{n_exp}</code>). Two views are recorded, and the whole analysis is run on each:</p>
<ul>
<li><strong><code>doc_probs</code> &mdash; soft affinity.</strong> Mean router softmax per expert; each
layer sums to 1. Includes experts the tokens leaned toward but never actually selected.</li>
<li><strong><code>doc_topk_freq</code> &mdash; hard selection.</strong> How often tokens actually routed
to each expert; each layer sums to the routed top-k = 7. This is what the model really <em>did</em>.</li>
</ul>
<p>Reporting both matters: a cluster that is only separable in <code>doc_probs</code> but not
<code>doc_topk_freq</code> would be &ldquo;subtle&rdquo; in the strict sense &mdash; a lean that never
became a decision.</p>''')}

{card("method", "Clustering", f'''
<p>Documents are clustered on the soft-affinity view with the same recipe as the token pipeline:
PCA to 95% variance (&rarr; 150 dims) + L2-normalize, then spherical k-means, <strong>k={soft['k']}</strong>,
seed 42. The characterization below re-uses that single assignment for <em>both</em> routing views, so
soft and hard are measured on exactly the same clusters. (Per-document embeddings are saved, so other
<em>k</em> can be explored later without re-running the model.)</p>''')}

{card("method", "Metric 1 &mdash; deviation from the mean (breadth, tests H1)", f'''
<p>Take one cluster and one expert <code>j</code>. Average that expert's routing probability over the
cluster's documents (<code>mean_in[j]</code>) and over the whole corpus (<code>mean_global[j]</code>).
Their difference is the cluster's per-expert <strong>deviation</strong>:</p>
<p style="text-align:center"><code>&delta;[j] = mean_in[j] &minus; mean_global[j]</code>
&mdash; how much more (or less) this cluster routes to expert <code>j</code> than a typical document.</p>
<p>&delta; is a {soft['n_dims']}-vector (one entry per layer&times;expert). We summarize how its
<em>size</em> is spread with two numbers:</p>
<ul>
<li><strong><code>top5 mass</code></strong> = share of the total shift &Sigma;|&delta;| held by the 5
largest-deviation experts.</li>
<li><strong><code>eff. dims</code></strong> = the effective number of experts carrying the shift,
exp(entropy of |&delta;|) &mdash; near 5 if a handful dominate, near {soft['n_dims']} if it's spread evenly.</li>
</ul>
<p><strong>H1 (a few signature experts)</strong> predicts the shift piles onto a few experts: high
<code>top5 mass</code>, small <code>eff. dims</code>. Note &delta; only compares <em>averages</em> &mdash;
it says nothing yet about whether individual documents are consistent. That's Metric 2.</p>''')}

{card("method", "Metric 2 &mdash; single-expert AUC (strength, separates H1 from H2)", f'''
<p>For expert <code>j</code>, form two piles of numbers: the routing values <code>x[doc,j]</code> for the
cluster's documents (<em>in-pile</em>) and for all other documents (<em>out-pile</em>). The
<strong>AUC</strong> is the probability that a random in-doc has a higher value than a random out-doc,
<code>P(x_in &gt; x_out)</code> &mdash; run every in&times;out pair as a &ldquo;higher value wins&rdquo;
contest and take the in-pile's win rate. 0.5 = indistinguishable (a coin flip); 1.0 = every cluster doc
routes higher than every non-cluster doc (perfect separation). In practice we don't enumerate pairs;
writing <code>n_in</code> and <code>n_out</code> for the number of docs in the in-pile and out-pile, we
pool all <code>n_in+n_out</code> values, rank them from smallest (rank&nbsp;1) to largest, and let
<code>R_in</code> = the sum of the in-pile's ranks. An in-doc's rank is 1 plus the number of values below
it, so <code>R_in &minus; n_in(n_in+1)/2</code> is exactly the number of in&times;out contests the in-pile
wins, giving <code>AUC = (R_in &minus; n_in(n_in+1)/2) / (n_in&middot;n_out)</code> &mdash; the same
win rate, computed by sorting once. We fold both directions into a separability
<code>|AUC&minus;0.5|+0.5</code> so an expert the cluster routes <em>away</em> from counts too. (This
win-rate equals the area under the ROC curve &mdash; same quantity, ranking view.)</p>
<p><strong>Why AUC is signal-to-noise, and why that matters here.</strong> Whether the two piles separate
depends on the gap between their means <em>relative to their spread</em>: AUC rises with
<code>|&delta;| / &sigma;</code> (the gap over the doc-to-doc standard deviation, i.e. Cohen's&nbsp;d).
So an expert with a <em>small</em> shift can still separate a cluster near-perfectly if that shift is
<em>consistent</em> (&sigma; even smaller). Metric&nbsp;1 (<code>top5 mass</code>) ranks experts by the raw
size of &delta;; Metric&nbsp;2 (AUC) ranks them by &delta; relative to noise &mdash; so a broad,
low-magnitude signature (Metric&nbsp;1 says &ldquo;spread out&rdquo;) can still give strong single-expert
classifiers (Metric&nbsp;2 says &ldquo;decisive&rdquo;). That apparent tension is the whole finding; it is
resolved concretely in <strong>Findings</strong>.</p>
<ul>
<li><strong><code>best-AUC</code></strong> = the single most-telling expert's separability &mdash; can one
expert alone identify the cluster?</li>
<li><strong><code>full-AUC</code></strong> = the same AUC, but with the score for each document being not
one expert's routing value but its <em>entire</em> fingerprint compared against the cluster: score =
cosine similarity between the document's full {soft['n_dims']}-dim routing vector and the cluster's mean
vector (centroid). Because this score uses every expert at once, it is the best &ldquo;use everything&rdquo;
separability we measure &mdash; the <em>whole-pattern ceiling</em> that any single expert is judged against.
<strong><code>ratio</code></strong> = (best&minus;0.5)/(full&minus;0.5), the fraction of that ceiling the
single best expert already recovers on its own.</li>
</ul>
<p><strong>H2 (subtle)</strong> predicts weak <code>best-AUC</code> and low <code>ratio</code> &mdash; no
single expert separates the cluster, only the joint pattern does.</p>''')}

{card("method", "Cheat-sheet + verdict rule", table(
    ["Metric", "One-line meaning", "Which hypothesis it bears on"],
    [
        ("<code>eff. dims</code> / <code>top5 mass</code>",
         "how spread vs concentrated the mean shift &delta; is",
         "<strong>breadth</strong> &mdash; H1 wants concentrated"),
        ("<code>best-AUC</code>",
         "can the single best expert separate the cluster?",
         "<strong>strength</strong> &mdash; H2 wants this weak"),
        ("<code>full-AUC</code> / <code>ratio</code>",
         "AUC when scoring docs by cosine to the cluster's mean fingerprint (all experts at once) "
         "= the ceiling; ratio = fraction of it one expert recovers",
         "strength &mdash; low ratio ⇒ only joint pattern (H2)"),
        ("<code>exp. used</code>",
         "effective experts the cluster routes to per layer, vs corpus ~%.0f" % soft["global_effective_experts"],
         "absolute peakedness (is it more focused than average?)"),
        ("<code>cos</code>",
         "mean cosine of a cluster's docs to their centroid",
         "within-cluster consistency (feeds the &sigma; in AUC)"),
    ]) + f'''
<p class="note"><strong>Verdict rule.</strong> A cluster is <em>strong</em> when
<code>best-AUC</code>&nbsp;&ge;&nbsp;0.8 <em>and</em> <code>ratio</code>&nbsp;&ge;&nbsp;0.7; <em>sparse</em>
when its deviation lives in &le;10% of experts (&le;{thr} of {soft['n_dims']}). Then:
strong&nbsp;&amp;&nbsp;sparse&nbsp;&rarr;&nbsp;<span class="v-few-expert">few-expert</span> (H1);
strong&nbsp;&amp;&nbsp;broad&nbsp;&rarr;&nbsp;<span class="v-broad-redundant">broad-redundant</span>;
not&nbsp;strong&nbsp;&rarr;&nbsp;<span class="v-subtle">subtle</span> (H2).</p>''')}
"""


def build_findings(soft, hard, code, ill, share) -> str:
    cc = ill.get("cluster", 5)
    be = ill.get("best_expert", {})
    st = ill.get("strong_experts", {})
    cnt = ill.get("counts", {})
    n_exp = soft["n_dims"]
    ge90 = cnt.get("ge_0.90", "—")
    ge95 = cnt.get("ge_0.95", "—")
    pcs = share.get("per_cluster_strong_dims", {})
    sh = share.get("sharing", {})
    ex = share.get("shared_expert_example", {})
    us = share.get("best_expert_usage", {})
    return f"""
{card("results", "Finding 1 &mdash; the signal is broad, not a few experts (H1 refuted)", f'''
<p>If a cluster were defined by a few signature experts, its deviation would pile onto those experts and
its effective-dimension count would be small. It does not. The top-5 experts hold only
~{pct(soft['median_top5_mass'])} of the median cluster's deviation (~{pct(hard['median_top5_mass'])}
under hard selection), and a median of <strong>~{soft['median_effective_dims']:.0f} of {soft['n_dims']}</strong>
experts carry meaningful deviation. The concentration curves are shallow, and the deviation heatmap (one
row per cluster, one column per layer&times;expert) shows it visually: under H1 each row would concentrate
into a few saturated cells &mdash; that cluster's signature experts &mdash; with the rest of the row
near-white (and if clusters <em>shared</em> signature experts, those cells would line up into strong
vertical stripes). Neither pattern appears: every row spreads its deviation thinly across hundreds of
columns, and no expert column stands out across clusters.</p>
<p>This is not the same as &ldquo;the cluster uses every expert equally&rdquo;: clusters <em>are</em> more
peaked than the corpus, routing to an effective ~{soft['median_effective_experts_used']:.0f} of
{soft['n_experts_per_layer']} experts per layer vs the corpus's ~{soft['global_effective_experts']:.0f}.
The specialization is real; it is just <em>distributed</em> over dozens of experts, not a handful.</p>'''
+ fig_row(
    fig("doc_probs", "signature_concentration.png",
        "Share of each cluster's deviation carried by its top-m experts. Shallow curves ⇒ no small signature set."),
    fig("doc_probs", "deviation_heatmap.png",
        "Per-cluster expert deviation (z-scored), one row per cluster × 1008 layer·expert columns. "
        "H1 would show a few saturated cells per row (signature experts); instead each row's deviation "
        "is spread thinly across hundreds of columns."),
))}

{card("results", "Finding 2 &mdash; yet hundreds of experts each classify the cluster (H2 refuted)", f'''
<p>Breadth alone could still mean H2: many experts, each nudged imperceptibly, separable only in the
joint pattern. The strength metric says otherwise. The single best expert separates the median cluster
with AUC <strong>{soft['median_best_single_dim_auc']:.3f}</strong> &mdash; almost the full-pattern
{soft['median_full_pattern_auc']:.3f}, a <code>ratio</code> of {soft['median_single_vs_full_ratio']:.2f},
i.e. <strong>one expert alone already recovers {pct(soft['median_single_vs_full_ratio'])} of what the
entire {soft['n_dims']}-dimensional pattern achieves.</strong></p>
<p>And it is not <em>one</em> privileged expert &mdash; it is a crowd. For the code cluster
(#{cc}), <strong>{ge90} of {n_exp} experts individually reach AUC&nbsp;&ge;&nbsp;0.9</strong>
({ge95} reach &ge;&nbsp;0.95); the &ldquo;best&rdquo; expert is merely the top of that pile. The code
cluster is the extreme of that redundancy, not the norm: across all {soft['k']} clusters the count of
&ge;0.9 experts ranges <strong>{pcs.get('min', '—')}&ndash;{pcs.get('max', '—')}</strong> with a median of
~{pcs.get('median', 0):.0f} &mdash; but even the least redundant cluster has over a dozen individually
strong separators, and <em>every</em> cluster's best expert is near-perfect (see the spectrum figure
below). So the cluster's identity is written <strong>redundantly</strong>: no single expert is necessary,
because many carry the same signal. Documents also route <em>consistently</em> (median within-cluster cosine to centroid
{soft['median_cosine_to_centroid']:.2f}), so these shifts are a stable property of the cluster, not noise
averaged over a loose bag of docs. Combined with Finding&nbsp;1, every cluster is
<strong>strong &amp; broad</strong> &rarr; <span class="v-broad-redundant">broad-redundant</span>
({soft['verdict_counts']['broad-redundant']}/{soft['k']}).</p>'''
+ fig_row(
    fig("doc_probs", "single_dim_vs_full_auc.png",
        "Best single expert vs full joint pattern, per cluster. Near the diagonal ⇒ one expert ≈ the whole pattern."),
    fig("doc_probs", "verdict_scatter.png",
        "Breadth (effective deviating experts, x) vs peakedness (y), sized by #docs. All clusters fall in the broad-redundant regime."),
)
+ fig_row(img_tag(ATTR / "best_expert_spectrum.png",
    "No cherry-picking: best-expert in/out routing histograms for 12 clusters spanning the "
    "redundancy spectrum (fewest → most experts ≥0.9). Every cluster's best expert separates it "
    "near-perfectly, always via a large routing lean (|δ| ≈ 0.08–0.30), mostly in late layers."))
)}

{card("results", "How can both be true? Magnitude vs signal-to-noise", f'''
<p>Findings 1 and 2 sound contradictory &mdash; &ldquo;no few experts carry the deviation&rdquo; yet
&ldquo;one expert separates the cluster&rdquo; &mdash; but they measure different things (see
<strong>Method</strong>). Metric&nbsp;1 ranks experts by the <em>size</em> of the mean shift |&delta;|;
AUC ranks them by the shift <em>relative to the doc-to-doc noise</em>, |&delta;|/&sigma;. A small shift that
is highly consistent scores low on the first and high on the second. The panels below make this concrete on
the code cluster (#{cc}, {ill.get("n_in", 0):,} docs):</p>'''
+ fig_row(img_tag(ATTR / "auc_illustration.png",
    f"(A) the single best expert (L{be.get('layer','?')}·E{be.get('expert','?')}): the in-pile and "
    f"out-pile barely overlap → AUC {be.get('auc','?')}. (B) {ge90}/{n_exp} experts individually reach "
    f"AUC≥0.9 — redundant, not one signature expert. (C) magnitude |δ| and separability AUC are only "
    f"weakly correlated (ρ={ill.get('spearman_absdelta_sep','?')}): the biggest-shift expert is AUC-rank "
    f"{ill.get('biggest_shift_expert',{}).get('auc_rank','?')}, the best separator only magnitude-rank "
    f"{be.get('mag_rank','?')}."))
+ f'''
<p>The many redundant classifiers are mostly <em>small</em> shifts: among the {st.get("n","—")} experts
with AUC&nbsp;&ge;&nbsp;0.9, the median shift is only |&delta;|&nbsp;&asymp;&nbsp;{st.get("median_abs_delta","—")}
(a ~1-percentage-point change in a routing probability), but the doc-to-doc spread is even smaller
(&sigma;&nbsp;&asymp;&nbsp;{st.get("median_std","—")}), so |&delta;|/&sigma;&nbsp;&asymp;&nbsp;{st.get("median_cohen_d","—")}
&mdash; the two piles barely overlap and the expert separates near-perfectly. <strong>Why so many experts
at once?</strong> Routing is coupled: within a layer the softmax sums to 1, so leaning toward some experts
pulls weight off others (they move together), and the same document content re-drives routing at all
{soft['n_layers']} layers. One latent &ldquo;this is code&rdquo; factor is re-expressed as small, reliable
shifts on hundreds of expert dimensions &mdash; each an individually-decisive shadow of it.</p>''')}

{card("results", "Is a high-AUC expert cluster-specific? Leaned-on, yes &mdash; owned, no", f'''
<p>A natural reading of &ldquo;one expert separates the cluster at AUC&nbsp;0.99&rdquo; is that the expert
<em>belongs</em> to that cluster. The full {soft['k']}&times;{soft['n_dims']} separability matrix says it's
subtler, in both directions:</p>
<ul>
<li><strong>Strong separators are shared.</strong> {sh.get('n_dims_strong_for_someone', '—')} of
{sh.get('n_dims', soft['n_dims'])} experts are a &ge;0.9 separator for <em>somebody</em>; the median such
expert strongly separates <strong>{sh.get('median_clusters_per_strong_dim', 0):.0f} clusters</strong>
(max {sh.get('max_clusters_per_strong_dim', '—')}), and {pct(sh.get('frac_strong_dims_shared_by_ge2', 0))}
serve &ge;2. This is possible because the AUC is one-vs-<em>rest</em> and clusters are small
(~1&ndash;2% of docs each): in the figure below, the same expert
(L{ex.get('layer', '?')}&middot;E{ex.get('expert', '?')}) separates
{ex.get('n_clusters_ge_strong', '—')} clusters, and for any one of them the <em>other</em> leaning
clusters are only ~{pct(ex.get('max_other_leaners_out_pile_share', 0.14))} of its out-pile &mdash; they cost
a few AUC points, not fifty. The flip side: those co-leaning clusters occupy overlapping bands, so the
shared expert cannot tell them apart, only each-from-the-bulk &mdash; used as a <em>detector</em> for one
cluster it would have poor precision.</li>
<li><strong>But the top expert is strongly leaned on.</strong> The median cluster's best expert receives
{pct(us.get('median_share_of_experts_mass_from_cluster', 0.14))} of its total corpus routing mass from that
cluster, which holds only {pct(us.get('median_cluster_doc_share', 0.015))} of documents &mdash; a
~{us.get('median_amplification', 8.5):.0f}&times; over-use, with typically just
{us.get('median_n_other_clusters_using_ge_50pct', 1):.0f} other cluster using it at even half that level.
Under hard selection the median cluster's tokens pick their best expert
{pct(us.get('median_hard_selection_in_cluster', 0.83))} of the time vs
{pct(us.get('median_hard_selection_global', 0.12))} corpus-wide.</li>
</ul>
<p>So high-AUC experts are <em>relatively</em> cluster-specific &mdash; a cluster leans on its top experts
~8&times; harder than anyone else &mdash; but never exclusively theirs: most of even the best expert's
traffic still comes from other clusters, because the cluster itself is small. This is the concrete sense
behind &ldquo;broad-redundant, not owned&rdquo;, and why Stage&nbsp;3 reuses a cluster's most-relevant
experts softly instead of hard-assigning experts to clusters.</p>'''
+ fig_row(img_tag(ATTR / "one_expert_shared.png",
    "How one expert has high AUC on many clusters: the same expert scored one-vs-rest against each "
    "cluster it separates at ≥0.9. Each cluster sits away from the near-zero bulk (some by routing "
    "toward it, two by avoiding it); the other leaning clusters are a tiny share of any one cluster's "
    "out-pile, so they barely dent the AUC.")))}

{card("results", "Finding 3 &mdash; it's real selection, not a soft-affinity artifact", f'''
<p>Could the signal live only in soft leanings that never become routing decisions? No. Re-running the
whole analysis on hard selection (<code>doc_topk_freq</code> &mdash; which experts tokens actually go to)
gives the same picture: best single-expert AUC {hard['median_best_single_dim_auc']:.3f} (vs
{soft['median_best_single_dim_auc']:.3f} soft), spread over a median ~{hard['median_effective_dims']:.0f}
experts. In the soft-vs-hard scatter the clusters track the diagonal &mdash; a cluster's best expert is
about as identifying in what the model <em>did</em> as in what it <em>leaned</em> toward.</p>'''
+ fig_row(
    img_tag(ATTR / "hard_vs_soft.png",
            "Best single-expert separability per cluster: hard selection (x) vs soft affinity (y). "
            "On the diagonal ⇒ real selection identifies the cluster, not a soft-only signal."),
    fig("doc_topk_freq", "deviation_heatmap.png",
        "Selection-frequency deviation per cluster — diffuse, mirroring the soft view."),
))}

{card("method", "Takeaway &mdash; and what it means for the extension plan", f'''
<p>The router does carve documents into behaviourally distinct groups &mdash; specialization is present
and consistent. But that specialization is written <strong>redundantly across dozens of experts</strong>
rather than concentrated in a nameable few. The code cluster below (#{code['cluster']},
{code['num_docs']:,} docs, dominated by <code>{html.escape(code['top'].split('/')[0])}</code>) reads as
broad-redundant like all the others: its single most telling expert at AUC&nbsp;&asymp;&nbsp;{code['auc']:.3f}
but its deviation spread over hundreds of experts.</p>
<p>Two consequences for the roadmap (see <strong>Overview</strong> / <strong>Next steps</strong>):</p>
<ul>
<li><strong>Good news for Stage&nbsp;2 (cheap cluster router).</strong> Because the cluster is <em>strongly</em>
and <em>redundantly</em> identifiable &mdash; not a subtle joint pattern &mdash; a truncated, much cheaper
view of a document should retain enough signal to recover its cluster without a full forward pass.</li>
<li><strong>It shapes Stage&nbsp;3 (adding experts).</strong> Because a cluster's &ldquo;relevant experts&rdquo;
are a soft, broad set rather than a nameable few, Stage&nbsp;3 doesn't try to hard-assign experts to clusters:
per partition it reuses the <em>n</em> most-relevant pool experts, adds 64&minus;n <em>new</em> experts born
pointed at that cluster, and lets training sort out the specialization (see <strong>Next steps</strong>).</li>
</ul>
<p>Whether the broad-redundant picture holds at the full 9.17M-document scale is the natural next check on
Stage&nbsp;1 itself.</p>''')}
"""


def _roadmap_table() -> str:
    done = '<span class="v-broad-redundant">done</span>'
    nxt = '<span class="v-subtle">next</span>'
    fut = '<span class="note">future</span>'
    rows = [
        ("0", "Base model",
         "A 64-expert EMO checkpoint with emergent, unsupervised data clusters.",
         f"{done} &mdash; released EMO 64-expert model"),
        ("1", "Characterize the clusters",
         "Are the emergent clusters real, and what defines them? (must hold before the plan can rely on them)",
         f"{done} &mdash; this report"),
        ("2", "Cheap cluster router",
         "Label streaming docs with their EMO cluster <em>without a full forward pass</em>, to partition data at scale.",
         '<span class="note">deprioritized</span> &mdash; superseded for the sanity check by the full '
         'forward-pass oracle partition (see <strong>Oracle partition</strong>)'),
        ("3", "Add experts + partitioned training",
         "Per cluster partition: reuse the <em>n</em> most-relevant pool experts, initialize 64&minus;n new "
         "ones, train the 64-expert working set, then grow the pool with the new experts and write the reused "
         "ones back.",
         f"{nxt} &mdash; oracle labels for the full 100B&ndash;130B window are ready"),
    ]
    return table(["stage", "what", "why", "status"], rows)


def build_nextsteps(soft) -> str:
    return f"""
<p><strong>Next question:</strong> to partition the continued-pretraining stream by EMO cluster at scale
(Stage&nbsp;2), we need to label each incoming document with its cluster <em>cheaply</em>. The fingerprints
in this report came from a full {soft['n_layers']}-layer forward pass per document &mdash; far too expensive
to run over an entire pretraining corpus. So Stage&nbsp;2 asks: <strong>can we classify a document into its
EMO cluster without a full forward pass?</strong></p>

{card("goal", "Why we expect this to be possible", '''
<p>Stage&nbsp;1 found the cluster signal is <em>broad and redundant</em>: hundreds of experts each separate a
cluster near-perfectly, and it is not hidden in a subtle joint pattern. Redundant, strong signal is exactly
what survives aggressive truncation. If the &ldquo;this is code&rdquo; factor is already legible in, say, the
first block's routing, we don't need the other layers to recover the cluster label. The Stage&nbsp;1 finding
turns this from a hope into a testable hypothesis.</p>''')}

{card("method", "Candidate cheap classifiers (in the order we'll try them)", table(
    ["input", "cost", "role"],
    [
        ("mean-pooled token embeddings (the model's own embedding matrix)", "embedding lookup, no block forward",
         "<strong>first to try</strong> &mdash; essentially free, still EMO-native"),
        ("n-gram / bag-of-words features", "no model at all",
         "next &mdash; is the partition recoverable from surface text alone?"),
        ("first-block expert activations / router logits", "one transformer block",
         "later, if the above underperform"),
        ("full doc router embedding (all layers)", "full forward pass",
         "<strong>oracle / upper bound</strong> &mdash; the fingerprint we clustered on; defines the ceiling"),
    ]) + '''
<p>The experiment: freeze the k=64 assignments as labels, hold out documents, train a small classifier on
each input above, and report cluster-recovery accuracy (plus top-k and confusion vs cluster size) against the
full-embedding oracle. We start from the cheapest, EMO-native option (the model's own embedding matrix) and
only climb the cost ladder if it doesn't recover the partition well enough.</p>''')}

{card("results", "Open questions before we start", '''
<ul>
<li><strong>How cheap can we go?</strong> The bet is that the model's own token embeddings (no block forward)
already recover the clusters; n-gram features test whether even the model is unnecessary. First-block
activations are the fallback if surface-level views fall short.</li>
<li><strong>How good is good enough?</strong> Partitioning tolerates some misclassification; the target
cluster-recovery accuracy should be tied to Stage&nbsp;3's tolerance, not to perfection.</li>
</ul>
<p class="note">Stage&nbsp;1's expert-usage detail was exploratory &mdash; useful for intuition, but the only
requirement it needs to have established is that a cheap view of a document recovers its cluster. Whichever
redundancy the added experts do or don't share is left for training to sort out (see Stage&nbsp;3).</p>''')}

{card("goal", "Stage 3 preview &mdash; the expert-addition loop", '''
<p>Once data can be cheaply partitioned by cluster, capacity is grown one partition at a time while the
training working set stays capped at 64 experts. Keeping a growing <em>pool</em> of experts, for each cluster
partition in turn:</p>
<ol>
<li><strong>Select</strong> the <em>n</em> experts already in the pool most relevant to this cluster.</li>
<li><strong>Initialize</strong> 64&minus;n new experts.</li>
<li><strong>Train</strong> the resulting 64-expert working set on this cluster's data partition &mdash; only
64 experts are ever in memory, so the training footprint stays fixed.</li>
<li><strong>Write back:</strong> push the 64&minus;n new experts into the pool (capacity grows) and update the
<em>n</em> reused experts in the pool.</li>
<li>Move to the next partition.</li>
</ol>
<p>Across partitions the pool grows past 64 (toward 128+) while every step touches only 64. New experts are
born pointed at a specific cluster, so they specialize instead of duplicating existing ones &mdash; the
redundancy the whole design is built to avoid.</p>
<p class="note">Two design questions this raises, for later: (a) <strong>routing over a growing pool</strong>
&mdash; at inference the gate must route among all pool experts (e.g. 128) though training only ever exposed
64 at a time; (b) <strong>sequential forgetting</strong> &mdash; training partitions in turn may erode reused
experts' behaviour on earlier clusters, so some interleaving or replay may be needed.</p>''')}
"""


def build_substab(joint, scores, oracle_split) -> str:
    """The 'Oracle partition' tab: full 100B-130B partition + subsample-stability results."""
    n_docs = joint["num_docs"]
    n_tok = joint["total_doc_tokens"]
    sizes = sorted(c["num_docs"] for c in joint["clusters"])
    intro = f"""
<p>Stage&nbsp;3 (partition + grow) needs every training document labeled with its cluster. We did the
expensive, assumption-free version first: <strong>forward-pass the entire 100B&ndash;130B continued-pretraining
window through the EMO 100B checkpoint</strong> (whole documents, first 2048 tokens fingerprinted), then fit
one k=64 spherical k-means over all of it. The result is the <strong>oracle partition</strong>:
{n_docs / 1e6:.1f}M documents / {n_tok / 1e9:.1f}B doc-tokens with cluster sizes
{sizes[0] / 1e3:.0f}K&ndash;{sizes[-1] / 1e3:.0f}K docs (median {sizes[len(sizes) // 2] / 1e3:.0f}K), exported
keyed by <code>(source_path, doc_start_offset)</code> so it joins directly onto the token stream.</p>
<p>That raises the first design question for any production loop: <strong>was fitting on all 27.5M documents
necessary?</strong> If a clustering fit on a small calibration sample recovers (nearly) the same partition,
future windows only need cheap centroid assignment, not a full re-fit. This tab measures exactly that.</p>
"""

    arms_table = table(
        ["arm", "what is re-fit on the subsample", "what it isolates"],
        [
            ("<strong>honest</strong> &mdash; n &isin; {100K, 1M, 5M}",
             "everything: mean, PCA basis <em>and</em> its 95%-variance component count, k-means",
             "the true small-calibration-set setting (nothing borrowed from the full fit)"),
            ("<strong>frozen</strong> &mdash; same n, same document draws",
             "k-means only (the full fit's mean/PCA/L2 transform is reused)",
             "honest&minus;frozen gap = how much the <em>embedding geometry</em> destabilizes at small n"),
            ("<strong>fullseed</strong> &mdash; all 27.5M docs, seeds 1 and 2",
             "k-means on the full data (reference used seed 42)",
             "the ceiling: how much k-means disagrees with <em>itself</em> at full sample size"),
        ],
    )
    method = f"""
<p>Ten arms, one Beaker job each. Every arm ends the same way: assign <em>all</em> {n_docs / 1e6:.1f}M
documents to the arm's fitted centroids, then compare against the reference full-fit partition. The recipe
(doc_probs fingerprint, mean-center + PCA to 95% variance + L2, spherical k-means, k=64) is never varied
&mdash; only which rows the data-dependent pieces are fit on. At equal seed, honest and frozen arms draw the
<em>same documents</em>, so their gap is attributable to the transform alone. 1M runs twice (seeds 0, 1) to
gauge draw-to-draw variance.</p>
<p><strong>Metrics.</strong> Cluster IDs are arbitrary, so we report agreement after the best one-to-one
relabeling (Hungarian matching on the 64&times;64 contingency table): <strong>matched accuracy</strong> over
documents and over doc-tokens (what Stage&nbsp;3 actually partitions), plus <strong>ARI</strong> (chance-corrected
pair agreement, matching-free). Code: <code>src/scripts/clustering/subsample_stability.py</code>;
launcher: <code>scripts/modular_extension/launch_subsample_stability.sh</code>.</p>
{arms_table}
"""

    pilot_acc = oracle_split.get("test", {}).get("accuracy")
    pilot_ari = oracle_split.get("test", {}).get("ari")
    pilot_note = (
        f"a 70%-train-split fit agreed with the full fit on only {pilot_acc:.0%} of held-out docs "
        f"(ARI {pilot_ari:.2f})" if pilot_acc else "a train-split fit agreed with the full fit on only "
        "~70% of held-out docs (ARI ~0.63)")
    hypothesis = card("goal", "Hypothesis", f"""
<p>The Stage-2 gate on the 1.15M-doc pilot already hinted the partition is not perfectly reproducible:
{pilot_note}. But that experiment couldn't tell <em>why</em> &mdash; sample size, transform instability, or
k-means run-to-run noise. The arms above separate the three. Our prior, from the PCA dimension staying
~150 from 1.15M to 27.5M docs, was that the transform is stable (honest &asymp; frozen) and most disagreement
is k-means seed noise &mdash; i.e. the subsample curves should rise toward a fullseed ceiling that itself
sits well below 100%.</p>""")

    # -- results (filled once the Beaker arms have landed) --------------------
    if not scores:
        results = card("results", "Results", "<p class='missing'>[arms still running]</p>")
        interp = ""
    else:
        results, interp = _substab_results(scores)

    ksel_path = KSEL / "k_selection_summary.json"
    ksel = json.load(open(ksel_path)) if ksel_path.exists() else {}
    k4_path = KSEL / "k4_characterization.json"
    k4 = json.load(open(k4_path)) if k4_path.exists() else []
    pointer = ('<p class="note">The frozen k=32 partition, its cluster sizes, the Stage-3 baseline '
               'run, and the first training experiment now live in the '
               '<strong>k=32 CPT (no extension)</strong> tab.</p>')
    return intro + card("method", "Experiment: subsample fit vs full fit", method) + \
        hypothesis + results + interp + build_ksel(ksel, k4) + pointer


def _substab_results(scores) -> tuple:
    """Results + interpretation cards from subsample_stability/scores.json."""
    def sort_key(item):
        name, s = item
        return ({"honest": 0, "frozen": 1, "fullseed": 2}[s["mode"]],
                s.get("n_sample") or 10**9, s["seed"])

    rows = []
    for name, s in sorted(scores.items(), key=sort_key):
        n = s.get("n_sample")
        rows.append((
            s["mode"],
            f"{n / 1e6:.1f}M" if n else "27.5M (all)",
            s["seed"],
            s.get("n_components", "&mdash;"),
            pct(s["acc_docs"]),
            pct(s["acc_tokens"]),
            f"{s['ari']:.2f}",
            f"{s['nmi']:.2f}",
        ))
    res_table = table(
        ["arm", "fit on", "seed", "PCA dims", "doc agreement", "token agreement", "ARI", "NMI"], rows)

    fig_html = img_tag(SUBSTAB / "substab_agreement.png",
                       "Agreement with the full-fit oracle partition vs number of documents the clustering "
                       "was fit on. Orange: honest re-fit (transform + k-means). Blue: frozen transform "
                       "(k-means only). Green band: full-data k-means re-fit with a new seed — the "
                       "reproducibility ceiling. Open circles: individual seeds at n=1M.")

    results = card("results", "Results", f"""
{res_table}
<div class="figrow">{fig_html}</div>
""")
    return results, _substab_interpretation(scores)


def _substab_interpretation(scores) -> str:
    def accs(mode, n=None):
        return sorted(s["acc_docs"] for s in scores.values()
                      if s["mode"] == mode and (n is None or s.get("n_sample") == n))

    ceil = accs("fullseed")
    h1m, f1m = accs("honest", 1_000_000), accs("frozen", 1_000_000)
    h100k = accs("honest", 100_000)
    dims = {s.get("n_components") for s in scores.values() if s["mode"] == "honest"}
    nmis = sorted(s["nmi"] for s in scores.values())

    return card("results", "Reading the result &mdash; the partition is ~75% reproducible, "
                           "and a 1M-doc fit already hits that ceiling", f"""
<ul>
<li><strong>The ceiling is ~75%, not 100%.</strong> Two k-means re-fits on <em>all 27.5M docs</em>, changing
only the seed, agree with the reference partition on {pct(ceil[0])}&ndash;{pct(ceil[-1])} of documents
(ARI 0.67&ndash;0.70). About a quarter of documents flip clusters between equally good k-means optima
&mdash; perfect recovery was never available, from any sample size.</li>
<li><strong>1M documents (3.6% of the data) reach the ceiling.</strong> The four 1M-doc arms score
{pct(min(h1m + f1m))}&ndash;{pct(max(h1m + f1m))} &mdash; indistinguishable from the full-data re-fits.
Even 100K docs only cost a few points ({pct(h100k[0])} honest). The curves are flat rather than rising:
sample size stops mattering somewhere <em>below</em> 1M. (The 5M arms landing slightly below the 1M arms
is draw-to-draw scatter, the same magnitude as the gap between the two full-data re-fits; only n=1M was
run with two seeds.)</li>
<li><strong>All of the instability is k-means, none is the embedding geometry.</strong> Honest and frozen
arms are statistically indistinguishable at every size, and every honest arm &mdash; even at 100K docs
&mdash; independently selected {sorted(dims)[0]} PCA components under the 95%-variance rule. Re-fitting the
transform on a small sample loses nothing; the multi-modality lives entirely in the k=64 k-means
optimization.</li>
<li><strong>Disagreement is fragmentation, not relabeling.</strong> NMI stays high
({nmis[0]:.2f}&ndash;{nmis[-1]:.2f}) while one-to-one matched accuracy sits at ~75%, and inspecting a
full-data re-fit's contingency table confirms why: the documents a reference cluster loses land in just
one or two specific clusters (median 55% of a cluster's leaked mass goes to a single destination), i.e.
re-fits split and merge clusters rather than scrambling documents into unrelated ones. Cluster-level
structure survives; exact boundaries don't.</li>
</ul>
<p><strong>Consequences for the program.</strong> (a)&nbsp;Fitting on all 27.5M documents was unnecessary
&mdash; a ~1M-doc calibration fit plus nearest-centroid assignment recovers as much of the partition as a
full re-fit can. (b)&nbsp;More importantly, the oracle partition should be treated as <em>one frozen
artifact</em>: future data windows get assigned to the existing centroids, never re-clustered, because any
re-fit &mdash; even on all the data &mdash; silently relabels ~25% of documents. (c)&nbsp;Stage-3
conclusions must not hinge on exact cluster boundaries: an effect that vanishes when 25% of boundary
documents are relabeled was noise. Where it matters, grow experiments should be spot-checked against a
second partition seed.</p>"""
    )


def build_ksel(ksel, k4) -> str:
    """The principled-k follow-up inside the Oracle partition tab."""
    if not ksel:
        return ""
    stab = ksel["stability"]
    chosen = ksel["chosen_k"]
    ks = sorted(int(k) for k in stab["global"])

    seed_objs = [ksel["arms"][f"global_k64_seed{s}"]["objective"] for s in (1, 2)]
    objs = sorted([K64_REF_OBJECTIVE] + seed_objs)
    spread = (objs[-1] - objs[0]) / objs[0]

    method = card("method", "Follow-up &mdash; is k=64 the problem? A principled-k sweep", f"""
<p>The ~75% ceiling above invites a suspicion: maybe k=64 (inherited from the published pretraining
clustering) is simply a bad k, and a principled choice would cluster more reproducibly. Two checks.</p>
<p><strong>First, is it under-optimization?</strong> No: the three full-data k=64 fits (reference plus both
new seeds) reach spherical objectives of {objs[0]:.4f}&ndash;{objs[-1]:.4f} &mdash; a {spread:.2%} spread,
with the <em>reference</em> the lowest &mdash; while disagreeing on ~25% of documents. These are different,
equally good optima, not one good solution found by some seeds and missed by others. A better optimizer
cannot close a gap that isn't there; only a different k might.</p>
<p><strong>So, sweep k</strong> &isin; {{4, 8, 16, 32, 48, 64, 96, 128}}, entirely on the frozen transform
(established above as stable): the full 27.5M docs &times; 2 seeds per k (seed-pair stability), plus three
independent 1M-doc draws &times; every k (do subsamples choose the same k, and do they recover the global
fit at that k). Selection criteria per fit: the objective curve (elbow), cosine silhouette and
Davies-Bouldin on a fixed 50K evaluation sample &mdash; and reproducibility itself. Code:
<code>src/scripts/clustering/k_selection.py</code>.</p>""")

    chosen_rows = []
    for key in sorted(chosen):
        c = chosen[key]
        chosen_rows.append((key.replace("_", " "), c["elbow_objective"],
                            c["silhouette_argmax"], c["davies_bouldin_argmin"]))
    chosen_table = table(["fit set", "elbow (objective)", "silhouette argmax", "DB argmin"],
                         chosen_rows)

    stab_rows = []
    for k in ks:
        sk = str(k)
        stab_rows.append((
            k,
            f"{stab['global'][sk]['ari']:.2f} ({stab['global'][sk]['acc'] * 100:.1f}%)",
            f"{stab['sub1M'][sk]['ari']:.2f}",
            f"{stab['recover'][sk]['ari']:.2f}",
        ))
    stab_table = table(
        ["k", "full-data seed pair &mdash; ARI (matched acc)", "1M draw pairs &mdash; ARI",
         "1M draw vs full fit &mdash; ARI"], stab_rows)

    fig_html = img_tag(KSEL / "ksel_curves.png",
                       "Left: the objective rises smoothly with k — no sharp elbow (chord-rule knee at 32). "
                       "Middle: silhouette peaks on a shallow 16–32 plateau. Right: reproducibility (ARI) "
                       "vs k — near-perfect at k=4, then a flat ~0.6–0.73 band for every k ≥ 8; the three "
                       "curve types (full-vs-full, draw-vs-draw, draw-vs-full) lie on top of each other. "
                       "Grey lines: the three 1M-doc draws; green: full-data fits.")

    k4_txt = ""
    if k4:
        tok = sum(c["num_tokens"] for c in k4)
        shares = ", ".join(f"{c['num_tokens'] / tok:.0%}" for c in k4)
        code_c = max(k4, key=lambda c: sum(v for s, v in c["top_sources"].items()
                                           if "starcoder" in s or "proof-pile" in s))
        code_share = sum(v for s, v in code_c["top_sources"].items()
                         if "starcoder" in s or "proof-pile" in s)
        k4_txt = (f"<p>The four stable clusters are broad semantic modes, not source partitions "
                  f"(dclm dominates the whole mix): {shares} of doc-tokens respectively, with one mode "
                  f"leaning technical/code ({pct(code_share)} starcoder+proof-pile vs ~0&ndash;2% in the "
                  f"others).</p>")

    results = card("results", "Results &mdash; the criteria agree on k&asymp;16&ndash;32, "
                              "but no k &ge; 8 is reproducible", f"""
<p><strong>(1) Do different subsamples choose the same k? Yes.</strong> All three 1M draws pick k=32 by
all three criteria; the full-data fits pick 32 by the elbow rule and 16 by silhouette (the silhouette
surface is a shallow 16&ndash;32 plateau &mdash; the 16-vs-32 difference is &lt;0.02). k selection is
about as sample-size-insensitive as the clustering itself.</p>
{chosen_table}
<p><strong>(2) Does forcing a common k align the fits? No &mdash; and it never could.</strong> At every k,
a 1M-draw fit agrees with the full-data fit about as well as two full-data fits agree with <em>each
other</em> (right panel: the three curves track each other; at k=8 the draws actually agree with each other
<em>more</em> than the full fits do). At no k does fit-set size explain the disagreement &mdash; the
variance is k-means degeneracy at that k.</p>
{stab_table}
<div class="figrow">{fig_html}</div>
{k4_txt}""")

    reading = card("goal", "Reading &mdash; k is a design knob, and only k=4 buys reproducibility", f"""
<ul>
<li><strong>A principled k exists and it isn't 64:</strong> internal criteria converge on 16&ndash;32,
and k=32 is also the local stability maximum (ARI {stab['global']['32']['ari']:.2f} vs
{stab['global']['64']['ari']:.2f} at k=64). Switching 64&rarr;32 would buy a few points of
reproducibility and halve the partition count.</li>
<li><strong>But no useful k fixes the seed variance.</strong> Every k &ge; 8 sits in the same
~0.6&ndash;0.73 ARI band &mdash; k=8 is actually the <em>least</em> stable
(ARI {stab['global']['8']['ari']:.2f}). Only k=4 is genuinely reproducible
(ARI {stab['global']['4']['ari']:.2f}, {stab['global']['4']['acc'] * 100:.1f}% matched accuracy). The router
geometry has about four hard modes; every finer k also slices continuous density, and where the slices
fall is decided by the seed.</li>
<li><strong>Consequence:</strong> the instability is intrinsic to fine-grained k-means on this geometry,
not a hyperparameter bug &mdash; so the frozen-artifact discipline from the previous section stays the
answer at any fine k. For Stage&nbsp;3 this makes k a <em>design</em> choice: fewer, larger, more stable
partitions (16&ndash;32) vs more, finer, more arbitrary ones (64+); and the four coarse modes offer
high-confidence units for the first grow experiments, with the finer slices nested inside them.</li>
</ul>""")

    return method + results + reading


def build_k32_freeze(k32) -> str:
    """The designated frozen k=32 partition + the Stage-3 baseline run."""
    if not k32:
        return ""
    clusters = sorted(k32["clusters"], key=lambda c: -c["num_tokens"])
    tot_tok = sum(c["num_tokens"] for c in clusters)
    rows = []
    for c in clusters:
        top = ", ".join(f"{s} ({v / c['num_docs']:.0%})"
                        for s, v in list(c["top_sources"].items())[:2])
        rows.append((c["cluster"], f"{c['num_docs'] / 1e6:.2f}M",
                     f"{c['num_tokens'] / 1e9:.2f}B", pct(c["num_tokens"] / tot_tok), top))
    k32_table = table(["cluster", "docs", "doc-tokens", "token share", "top sources (docs)"], rows)
    n_docs = sum(c["num_docs"] for c in clusters)
    docs_sorted = sorted(c["num_docs"] for c in clusters)
    toks_sorted = sorted(c["num_tokens"] for c in clusters)

    return card("goal", "Decision &mdash; the frozen k=32 partition and the Stage-3 baseline", f"""
<p>Based on the above, Stage&nbsp;3 proceeds at <strong>k=32</strong> (the criteria's choice and the local
stability maximum), using the <strong>full-data fit with seed 1</strong> (<code>global_k32_seed1</code>) as
<em>the</em> frozen partition &mdash; exported to
<code>modular_extension/data/&hellip;_100B-130B/doc_clusters_k32.jsonl.gz</code>, provenance-keyed like the
k=64 export. Per the stability findings it will never be re-fit; future windows get nearest-centroid
assignment.</p>
<p><strong>Cluster sizes.</strong> {n_docs / 1e6:.1f}M documents / {tot_tok / 1e9:.1f}B doc-tokens across 32
clusters; docs per cluster range {docs_sorted[0] / 1e3:.0f}K&ndash;{docs_sorted[-1] / 1e6:.2f}M (median
{docs_sorted[16] / 1e3:.0f}K), doc-tokens {toks_sorted[0] / 1e9:.2f}B&ndash;{toks_sorted[-1] / 1e9:.2f}B
&mdash; every partition is large enough to train on, none dominates.</p>
{details("Per-cluster document and token counts (sorted by token mass)", k32_table)}
<p><strong>Baseline (launched).</strong> The comparison point for partition+grow is the same model trained
<em>normally</em> over the same window: <code>emo64_100b130b_baseline</code> re-trains 100B&rarr;130B from
the trunk's step23842 checkpoint with full trainer/optimizer state (deterministic data cursor &rarr;
exactly the tokens partitioned here), the 1T scheduler cap (LR stays flat at 2e-3, as in the original
run), and a hard stop at 130B where the final checkpoint (~step 30996) is saved &mdash; the original
extension run kept no checkpoint between 100B and 200B. Script:
<code>scripts/modular_extension/emo64_100b130b_baseline.sh</code>.</p>""")


def build_k32cpt(k32, conc, drift) -> str:
    """The 'k=32 CPT (no extension)' tab: sequential per-cluster training of existing
    experts under a 32-expert working-set budget, plus its two pre-flight analyses."""
    intro = """
<p><strong>The experiment:</strong> before growing any new experts, test whether cluster-partitioned
training of the <em>existing</em> experts already helps. Constraint: at most <strong>32 experts</strong>
can be trained at once. For each of the 32 clusters (in sequence): select the 32 experts most relevant to
that cluster as the working set, continue pretraining on that cluster's partition with the usual EMO loss,
then write the updated experts back into the 64-expert pool. No capacity is added &mdash; this isolates the
value of <em>specialization via partitioned training</em> from the value of <em>extra experts</em>.</p>
<p>Two questions had to be answered before launching: (1) does a cluster's traffic actually concentrate
on 32 experts, so a 32-expert working set sees most of the routed mass; (2) what should happen to the
non-expert parameters (attention, embeddings, norms, router) while only 32 experts train.</p>
"""

    conc_html = ""
    if conc:
        cl = sorted(c["top32_mass"] for c in conc["clusters"])
        rnd = [r["top32_mass"] for r in conc["random_sets"]]
        n_lo = sum(1 for x in cl if x < 0.75)
        conc_html = card("results", "Pre-flight 1 &mdash; each cluster's top-32 experts capture "
                                    f"{pct(cl[0])}&ndash;{pct(cl[-1])} of its router mass "
                                    f"(random docs: {pct(min(rnd))})", f"""
<p>For every cluster, we take its mean per-layer router distribution (from the saved doc_probs
fingerprints &mdash; no new forward passes), pick the top 32 of 63 standard experts per layer, and sum
the captured probability mass, averaged over the 16 layers. Control: a permutation of the cluster labels
&mdash; 32 <em>random</em> document sets with exactly the same sizes.</p>
<ul>
<li><strong>Cluster docs:</strong> the 32-expert working set captures {pct(cl[0])}&ndash;{pct(cl[-1])}
of router mass (median {pct(cl[len(cl) // 2])}); {n_lo} of 32 clusters sit below 75%.</li>
<li><strong>Random docs:</strong> {pct(min(rnd))}&ndash;{pct(max(rnd))} &mdash; indistinguishable from
the global routing profile ({pct(conc["global"]["top32_mass"])}), itself barely above the uniform
32/63&nbsp;=&nbsp;51%. Clustering, not subsetting, is what concentrates usage: it buys
+{(cl[0] - max(rnd)) * 100:.0f} to +{(cl[-1] - max(rnd)) * 100:.0f} points.</li>
<li>The residual {pct(1 - cl[-1])}&ndash;{pct(1 - cl[0])} of mass routed to the frozen 31 experts still
flows normally at train time &mdash; those experts just receive no gradient.</li>
</ul>
<div class="figrow">{img_tag(K32CPT / "expert_concentration.png",
    "Router mass captured by each cluster's own top-32 experts (blue, sorted) vs the same statistic "
    "for 32 random document sets of identical sizes (orange band) and the all-docs profile (dashed).")}
</div>"""
                         )

    drift_html = ""
    if drift:
        drift_html = _drift_card(drift)

    return intro + conc_html + drift_html + build_k32_freeze(k32)


def _drift_card(drift) -> str:
    last_cum = drift["cumulative"][-1]["groups"]
    first_int = drift["intervals"][0]["groups"]
    tok_lo = drift["cumulative"][-1]["from_step"] * drift["tokens_per_step"] / 1e9
    tok_hi = drift["cumulative"][-1]["to_step"] * drift["tokens_per_step"] / 1e9
    rows = [(g, f"{first_int[g]['rel_drift']:.3f}", f"{last_cum[g]['rel_drift']:.3f}",
             f"{last_cum[g]['cosine']:.4f}")
            for g in ("experts", "router", "attention", "embeddings", "norms", "lm_head")]
    return card("results", "Pre-flight 2 &mdash; how much do non-expert parameters move "
                           "during 64-expert training?", f"""
<p>Weight-space drift per parameter group across the extension run's permanent checkpoints
({tok_lo:.0f}B&nbsp;&rarr;&nbsp;{tok_hi:.0f}B tokens, one every ~100B): relative L2 change
&#8214;&Delta;&theta;&#8214;/&#8214;&theta;&#8214; per ~100B interval and cumulatively. A cheap proxy for
functional shift &mdash; read it comparatively (group vs group), not absolutely.</p>
{table(["group", "drift over first 100B interval", f"cumulative drift {tok_lo:.0f}B→{tok_hi:.0f}B",
        "cosine to 100B weights"], rows)}
<div class="figrow">{img_tag(K32CPT / "param_drift.png",
    "Relative L2 drift by parameter group, per ~100B-token interval (left) and cumulative since the "
    "100B checkpoint (right); log scale.")}
</div>""")


def build_per_cluster(soft, hard, part) -> str:
    hard_by_c = {c["cluster"]: c for c in hard["per_cluster"]}
    rows = []
    for c in soft["per_cluster"]:
        i = c["cluster"]
        pc = part.get(i, {})
        hv = hard_by_c.get(i, {}).get("verdict", "")
        sig = ", ".join(f"L{e['layer']}·E{e['expert']}" for e in c["signature_experts"][:3])
        top_src = "; ".join(
            f"{k.split('/')[0]}×{v}" for k, v in list(pc.get("top_sources", {}).items())[:2]
        )
        rows.append(
            f"<tr><td>{i}</td>"
            f"<td class='v-{c['verdict']}'>{c['verdict']}</td>"
            f"<td class='v-{hv}'>{hv}</td>"
            f"<td>{pc.get('num_docs', c['size']):,}</td>"
            f"<td>{pc.get('num_tokens', 0):,}</td>"
            f"<td>{c['effective_dims']:.1f}</td>"
            f"<td>{c['top5_mass']:.2f}</td>"
            f"<td>{c['best_dim']['auc']:.3f}</td>"
            f"<td>{c['full_pattern_auc']:.3f}</td>"
            f"<td>{c['effective_experts_used']:.1f}</td>"
            f"<td>{c['mean_cosine_to_centroid']:.2f}</td>"
            f"<td class='sig'>{html.escape(sig)}</td>"
            f"<td class='src'>{html.escape(top_src)}</td></tr>"
        )
    header = (
        "<th>cluster</th><th>verdict (soft)</th><th>verdict (hard)</th><th>#docs</th><th>#tokens</th>"
        "<th>eff. dims</th><th>top5 mass</th><th>best-AUC</th><th>full-AUC</th><th>exp. used</th>"
        "<th>cos</th><th>top signature experts</th><th>top sources</th>"
    )
    return f"""
{card("goal", "Every cluster, both signals", '''<p>Breadth and strength for all 64 clusters under soft
and hard routing, joined with document/token counts and dominant sources. Click any column header to
sort; see <strong>Method</strong> for the column definitions. Every row is
<span class="v-broad-redundant">broad-redundant</span> &mdash; the finding is uniform, and this table is
where you can check it cluster by cluster (e.g. sort by <code>best-AUC</code> or <code>eff. dims</code>
to see the spread).</p>''')}
<div class="tablewrap"><table id="ct" class="sortable"><thead><tr>{header}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>

{card("results", "Example routing profiles", fig_row(
    fig("doc_probs", "example_profiles.png",
        "16×63 per-cluster routing profiles, ordered most-concentrated → most-distributed. "
        "Even the most concentrated spreads its weight widely."),
))}
"""


# --------------------------------------------------------------------------
# Page assembly (CSS/JS kept in sync with models_routerfixed; mobile-safe)
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
p { max-width:78ch; }
.card { background:var(--card); border:1px solid var(--line); border-left:4px solid var(--line);
        border-radius:8px; padding:16px 20px; margin:16px 0; }
.card h3 { margin:0 0 8px; font-size:15px; text-transform:uppercase; letter-spacing:0.05em;
           scroll-margin-top:96px; }
.card.goal { border-left-color:#2563eb; } .card.goal h3 { color:#2563eb; }
.card.method { border-left-color:#7c3aed; } .card.method h3 { color:#7c3aed; }
.card.results { border-left-color:#059669; } .card.results h3 { color:#059669; }
table { border-collapse:collapse; margin:12px 0; font-size:14px; width:auto; max-width:100%; }
th, td { border:1px solid var(--line); padding:6px 12px; text-align:left; vertical-align:top; }
th { background:#f1f5f9; }
tbody tr:nth-child(even) { background:#f8fafc; }
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; margin:12px 0; }
.scroll table { margin:0; }
.note { font-size:13px; color:var(--muted); }
code { background:#eef2f7; padding:1px 5px; border-radius:4px; font-size:0.9em; }
ul { max-width:78ch; } li { margin:4px 0; }
.figrow { display:flex; gap:16px; flex-wrap:wrap; margin:14px 0; }
figure { flex:1 1 380px; min-width:0; max-width:560px; margin:0; }
figure img { width:100%; border:1px solid var(--line); border-radius:6px; cursor:zoom-in; background:#fff; }
figure img.zoom { max-width:none; width:auto; max-height:90vh; position:fixed; inset:0; margin:auto;
                  z-index:100; box-shadow:0 0 0 100vmax rgba(15,23,42,.75); cursor:zoom-out; }
figcaption { font-size:12.5px; color:var(--muted); margin-top:4px; }
details { margin:12px 0; }
summary { cursor:pointer; color:#2563eb; font-size:14px; }
.missing { color:#b91c1c; font-size:13px; }
.tablewrap { max-height:600px; overflow:auto; border:1px solid var(--line); border-radius:8px; margin:12px 0; }
.tablewrap table { margin:0; font-size:13px; width:100%; }
.tablewrap th { position:sticky; top:0; cursor:pointer; white-space:nowrap; }
.tablewrap td { white-space:nowrap; }
td.sig, td.src { color:var(--muted); font-size:12px; white-space:normal; }
.v-few-expert { color:#dc2626; font-weight:600; }
.v-broad-redundant { color:#7c3aed; font-weight:600; }
.v-subtle { color:#2563eb; font-weight:600; }
@media (max-width:640px) {
  header, nav, #subnav { padding-left:14px; padding-right:14px; }
  main { padding:16px 14px 60px; }
  nav button { padding:6px 10px; font-size:13px; }
  table { font-size:12.5px; }
  th, td { padding:5px 8px; }
  figure { flex-basis:100%; max-width:100%; }
}
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

// click-to-sort the per-cluster table
document.querySelectorAll('table.sortable').forEach(t => {
  t.querySelectorAll('th').forEach((th, i) => th.addEventListener('click', () => {
    const tb = t.tBodies[0], rows = [...tb.rows];
    const num = v => parseFloat(v.replace(/[^0-9.\\-]/g, ''));
    const asc = th.dataset.asc = th.dataset.asc === '1' ? '' : '1';
    rows.sort((a, b) => {
      const x = a.cells[i].innerText, y = b.cells[i].innerText;
      const nx = num(x), ny = num(y);
      const c = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : x.localeCompare(y);
      return asc ? c : -c;
    });
    rows.forEach(r => tb.appendChild(r));
  }));
});
"""


def _code_cluster(soft, part):
    """The starcoder/code-dominated cluster, for the takeaway spotlight."""
    best = None
    for c in soft["per_cluster"]:
        pc = part.get(c["cluster"], {})
        ts = list(pc.get("top_sources", {}).items())
        if ts and "starcoder" in ts[0][0].lower():
            best = {"cluster": c["cluster"], "num_docs": pc.get("num_docs", c["size"]),
                    "top": ts[0][0], "auc": c["best_dim"]["auc"]}
            break
    if best is None:  # fallback: largest cluster
        c = max(soft["per_cluster"], key=lambda c: c["size"])
        pc = part.get(c["cluster"], {})
        top = next(iter(pc.get("top_sources", {"data": 0}).items()))[0]
        best = {"cluster": c["cluster"], "num_docs": pc.get("num_docs", c["size"]),
                "top": top, "auc": c["best_dim"]["auc"]}
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "claude_outputs/modular_extension/report.html")
    args = parser.parse_args()

    soft = json.load(open(ATTR / "doc_probs/metrics.json"))
    hard = json.load(open(ATTR / "doc_topk_freq/metrics.json"))
    part = {c["cluster"]: c for c in json.load(open(PART))["clusters"]} if PART.exists() else {}
    ill_path = ATTR / "auc_illustration.json"
    ill = json.load(open(ill_path)) if ill_path.exists() else {}
    share_path = ATTR / "expert_sharing.json"
    share = json.load(open(share_path)) if share_path.exists() else {}
    n_docs = sum(c["size"] for c in soft["per_cluster"])
    code = _code_cluster(soft, part)

    joint = json.load(open(JOINT)) if JOINT.exists() else {}
    scores_path = SUBSTAB / "scores.json"
    scores = json.load(open(scores_path)) if scores_path.exists() else {}
    oracle_split = json.load(open(ORACLE_SPLIT)) if ORACLE_SPLIT.exists() else {}

    k32_path = JOINT.parent / "doc_clusters_k32_summary.json"
    k32 = json.load(open(k32_path)) if k32_path.exists() else {}
    conc_path = K32CPT / "expert_concentration.json"
    conc = json.load(open(conc_path)) if conc_path.exists() else {}
    drift_path = K32CPT / "param_drift.json"
    drift = json.load(open(drift_path)) if drift_path.exists() else {}

    tabs = [
        ("overview", "Overview", build_overview(soft, hard, n_docs)),
        ("method", "Method", build_method(soft, ill)),
        ("findings", "Findings", build_findings(soft, hard, code, ill, share)),
        ("per-cluster", "Per-cluster", build_per_cluster(soft, hard, part)),
        ("oracle-partition", "Oracle partition", build_substab(joint, scores, oracle_split)),
        ("k32-cpt", "k=32 CPT (no extension)", build_k32cpt(k32, conc, drift)),
        ("next-steps", "Next steps", build_nextsteps(soft)),
    ]
    nav = "".join(f'<button data-target="{tid}">{name}</button>' for tid, name, _ in tabs)
    sections = "".join(
        f'<section class="tab" id="{tid}">{body}</section>' for tid, _, body in tabs
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EMO modular_extension: adding experts during training under a fixed budget</title>
<style>{CSS}</style>
</head>
<body>
<header>
<a class="home-link" href="/">&larr; all reports</a>
<h1>EMO modular_extension: adding experts during training under a fixed budget</h1>
<p>modular_extension &mdash; grow a 64-expert EMO MoE toward more experts via cluster-partitioned continued
training &middot; Stage 1: what defines the emergent clusters (k={soft['k']}, 100B&ndash;110B window)
&middot; Oracle partition: the full 100B&ndash;130B window labeled, plus how stable the clustering is
&middot; generated by scripts/modular_extension/build_report.py</p>
</header>
<div class="topbar"><nav>{nav}</nav><div id="subnav"></div></div>
<main>{sections}</main>
<script>{JS}</script>
</body>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
