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
routes higher than every non-cluster doc (perfect separation). We compute it the fast, identical way (rank
all values, sum the in-pile's ranks: <code>AUC = (R_in &minus; n_in(n_in+1)/2) / (n_in&middot;n_out)</code>),
and fold both directions into a separability <code>|AUC&minus;0.5|+0.5</code> so an expert the cluster
routes <em>away</em> from counts too. (This win-rate equals the area under the ROC curve &mdash; same
quantity, ranking view.)</p>
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
<li><strong><code>full-AUC</code></strong> = the ceiling, using the whole joint pattern (cosine-to-centroid,
one-vs-rest). <strong><code>ratio</code></strong> = (best&minus;0.5)/(full&minus;0.5), the fraction of the
full separability one expert already recovers.</li>
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
         "whole-pattern ceiling, and how much one expert recovers",
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


def build_findings(soft, hard, code, ill) -> str:
    cc = ill.get("cluster", 5)
    be = ill.get("best_expert", {})
    st = ill.get("strong_experts", {})
    cnt = ill.get("counts", {})
    n_exp = soft["n_dims"]
    ge90 = cnt.get("ge_0.90", "—")
    ge95 = cnt.get("ge_0.95", "—")
    return f"""
{card("results", "Finding 1 &mdash; the signal is broad, not a few experts (H1 refuted)", f'''
<p>If a cluster were defined by a few signature experts, its deviation would pile onto those experts and
its effective-dimension count would be small. It does not. The top-5 experts hold only
~{pct(soft['median_top5_mass'])} of the median cluster's deviation (~{pct(hard['median_top5_mass'])}
under hard selection), and a median of <strong>~{soft['median_effective_dims']:.0f} of {soft['n_dims']}</strong>
experts carry meaningful deviation. The concentration curves are shallow and the deviation heatmap is
diffuse, not blocky &mdash; no small set of columns dominates.</p>
<p>This is not the same as &ldquo;the cluster uses every expert equally&rdquo;: clusters <em>are</em> more
peaked than the corpus, routing to an effective ~{soft['median_effective_experts_used']:.0f} of
{soft['n_experts_per_layer']} experts per layer vs the corpus's ~{soft['global_effective_experts']:.0f}.
The specialization is real; it is just <em>distributed</em> over dozens of experts, not a handful.</p>'''
+ fig_row(
    fig("doc_probs", "signature_concentration.png",
        "Share of each cluster's deviation carried by its top-m experts. Shallow curves ⇒ no small signature set."),
    fig("doc_probs", "deviation_heatmap.png",
        "Per-cluster expert deviation (z-scored), 64 clusters × 1008 experts. Diffuse, not blocky vertical stripes."),
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
({ge95} reach &ge;&nbsp;0.95); the &ldquo;best&rdquo; expert is merely the top of that pile. So the cluster's
identity is written <strong>redundantly</strong>: no single expert is necessary, because many carry the
same signal. Documents also route <em>consistently</em> (median within-cluster cosine to centroid
{soft['median_cosine_to_centroid']:.2f}), so these shifts are a stable property of the cluster, not noise
averaged over a loose bag of docs. Combined with Finding&nbsp;1, every cluster is
<strong>strong &amp; broad</strong> &rarr; <span class="v-broad-redundant">broad-redundant</span>
({soft['verdict_counts']['broad-redundant']}/{soft['k']}).</p>'''
+ fig_row(
    fig("doc_probs", "single_dim_vs_full_auc.png",
        "Best single expert vs full joint pattern, per cluster. Near the diagonal ⇒ one expert ≈ the whole pattern."),
    fig("doc_probs", "verdict_scatter.png",
        "Breadth (effective deviating experts, x) vs peakedness (y), sized by #docs. All clusters fall in the broad-redundant regime."),
))}

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
         nxt),
        ("3", "Add experts + partitioned training",
         "Per cluster partition: reuse the <em>n</em> most-relevant pool experts, initialize 64&minus;n new "
         "ones, train the 64-expert working set, then grow the pool with the new experts and write the reused "
         "ones back.",
         fut),
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
    n_docs = sum(c["size"] for c in soft["per_cluster"])
    code = _code_cluster(soft, part)

    tabs = [
        ("overview", "Overview", build_overview(soft, hard, n_docs)),
        ("method", "Method", build_method(soft, ill)),
        ("findings", "Findings", build_findings(soft, hard, code, ill)),
        ("per-cluster", "Per-cluster", build_per_cluster(soft, hard, part)),
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
