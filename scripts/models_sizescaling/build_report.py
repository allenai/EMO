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
        cov = statistics.mean(v for row in m["cov_topic_layer"] for v in row)
        e = m["num_standard_experts"]
        rows.append((lbl, e, f"{eff:.1f}", f"{100*eff/e:.1f}%", f"{100*cov:.1f}%"))
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
         "Fraction of pool", "Coverage &gt; uniform"], rows) + '''
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


def _match_rows(base: Path):
    rows = []
    for pair in PAIRS:
        path = base / "matching" / pair / "match_summary.json"
        if not path.is_file():
            continue
        s = json.loads(path.read_text())
        med = lambda k: statistics.median(l[k] for l in s["per_layer"])  # noqa: E731
        rows.append((
            f"{s['label_a']} &harr; {s['label_b']}",
            f"{med('matched_sim_median'):.2f}",
            f"{med('mean_splits_per_a_expert'):.1f}",
            f"{med('split_coherence'):.2f}",
            f"{100*med('frac_novel_b'):.1f}%",
            f"{med('nn_a_median'):.2f} / {med('nn_b_median'):.2f}",
        ))
    return rows


def build_matching(base: Path) -> str:
    ma = base / "matching"
    rows = _match_rows(base)
    featured = ma / "32e_vs_128e"
    other_details = []
    for pair in PAIRS[:-1]:
        d = ma / pair
        other_details.append(details(
            f"Pair {pair.replace('_vs_', ' &harr; ')}",
            fig_row(img_tag(d / "matched_sim_vs_layer.png", "Matched similarity by layer"),
                    img_tag(d / "splitting_novelty_vs_layer.png", "Splitting &amp; novelty by layer"))
            + fig_row(img_tag(d / "corr_heatmap_layers.png", "Correlation heatmaps (B columns sorted by Hungarian match)"),
                      img_tag(d / "nn_redundancy_vs_layer.png", "Within-model nearest-neighbor redundancy"))))
    return f"""
{card("goal", "Goal", '''<p>When the pool grows, is the larger model's
organization a <strong>refinement</strong> of the smaller's (each expert
<em>splits</em> into several), a set of <strong>copies</strong> (redundancy), or
<strong>new territory</strong> (novelty)? Because initializations are independent
(see Overview), any correspondence found here emerged from the data alone.</p>''')}

{card("method", "Method", '''<p>Each expert's <strong>fingerprint</strong> is its
usage vector over the 29,042 shared documents (valid because all models saw the
identical document set). Per layer, Pearson correlation between every
(A-expert, B-expert) pair gives a similarity matrix, from which we compute:</p>
<ul>
<li><strong>Matched similarity</strong> &mdash; Hungarian (optimal 1-to-1) matching score;</li>
<li><strong>Splits per A-expert</strong> &mdash; how many B-experts correlate &gt; 0.4 with it;</li>
<li><strong>Split coherence</strong> &mdash; how similar those B-experts are to <em>each other</em>
(high = redundant copies, low = they partition the A-expert's documents);</li>
<li><strong>Novelty</strong> &mdash; fraction of B-experts with no A-counterpart (max corr &lt; 0.3);</li>
<li><strong>NN redundancy</strong> &mdash; within-model nearest-neighbor correlation.</li>
</ul>''')}

{card("results", "Results", table(
        ["Pair", "Matched sim", "Splits per A-expert", "Split coherence",
         "Novel B-experts", "Within-model NN (A / B)"], rows) + '''
<p class="note">All values are medians across the 16 layers.</p>
<p><strong>The same organization at every scale.</strong> Matched correlation
holds at ~0.73 for every pair &mdash; including 32e&harr;128e directly &mdash; so the
32-expert model's structure survives recognizably inside a 4&times; larger pool.
Each 32e expert is covered by ~6&ndash;9.5 128e experts that correlate only
~0.34&ndash;0.40 with each other: they <strong>divide</strong> the parent's documents
rather than duplicate it (splitting, not redundancy). Novelty is near zero
(0.4&ndash;3.5%): 128 experts buy a finer partition of the same space, not new
specializations. Within-model redundancy creeps up mildly with pool size,
consistent with EMO's tolerance to expert pruning.</p>'''
+ fig_row(
    img_tag(featured / "corr_heatmap_layers.png",
            "32e &harr; 128e: cross-model correlation heatmaps (columns sorted by Hungarian match; bright diagonal band = matched structure)"),
    img_tag(featured / "matched_sim_vs_layer.png", "32e &harr; 128e: matched similarity by layer"))
+ fig_row(
    img_tag(featured / "splitting_novelty_vs_layer.png",
            "32e &harr; 128e: splits per A-expert, split coherence, novelty by layer"),
    img_tag(featured / "nn_redundancy_vs_layer.png",
            "Within-model nearest-neighbor redundancy by layer"))
+ "<h4>Other pairs</h4>" + "".join(other_details))}
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

    overview_rows = []
    ablation_rows = []
    for run, lbl, _ in MODELS:
        s = _attr_summary(base, run)
        if not s:
            continue
        c = s["concentration_mean"]
        overview_rows.append((
            lbl,
            s["n_dims"],
            f"{100*c['top1_mass']:.1f}%",
            f"{100*c['top5_mass']:.1f}%",
            f"{100*c['top32_mass']:.1f}%",
            f"{s['effective_dims_median']:.0f} ({100*s['effective_dims_median']/s['n_dims']:.0f}%)",
            f"{s['best_single_dim_auc']['median']:.3f}",
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

    featured_run, featured_lbl = MODELS[-1][0], MODELS[-1][1]
    other_details = []
    for run, lbl, _ in MODELS[:-1]:
        d = at / run
        other_details.append(details(
            f"{lbl} figures",
            fig_row(img_tag(d / "ablation_curves.png", f"{lbl}: drop/keep ablation curves"),
                    img_tag(d / "single_dim_vs_full_auc.png", f"{lbl}: single-dim vs full-pattern AUC"))
            + fig_row(img_tag(d / "signature_concentration.png", f"{lbl}: signature concentration"))))

    return f"""
{card("goal", "Goal", '''<p>The published clustering result shows that k-means on
router-probability embeddings yields semantically meaningful document clusters.
<strong>Did each cluster form because of a few individual, particular experts, or
because of the general expert-activation pattern?</strong> All tests run in the raw
(layer&nbsp;&times;&nbsp;expert) probability space, where every dimension is one
expert at one layer, on the same k=32 spherical k-means clustering recipe as the
published runs (<code>doc_probs</code>, mean-center &rarr; PCA 95% &rarr; L2).</p>''')}

{card("method", "Method", '''<p>Three complementary tests per model:</p>
<ol>
<li><strong>Signature concentration</strong> (descriptive) &mdash; per cluster, take the
centroid's deviation from the global mean and ask how much of its |mass| sits in the
top-m (layer, expert) dims. Few dims dominating &rArr; individual experts.</li>
<li><strong>Single-dim separability</strong> (markers) &mdash; for every cluster, the best
single dim's one-vs-rest AUC (rank-based, counting under-activation markers too),
vs the full-pattern baseline (cosine similarity to the cluster centroid).</li>
<li><strong>Drop / keep ablations</strong> (causal) &mdash; remove the union of every
cluster's top-m |deviation| dims, re-run the <em>entire</em> pipeline
(PCA &rarr; L2 &rarr; spherical k-means), and compare the new clustering to the
original (ARI) and to the 24 weborganizer topic labels (NMI). Matched random-dim
drops calibrate re-run instability; the converse keep-only-top-m run tests
sufficiency.</li>
</ol>
<p class="note">Calibration caveat: re-running k-means after <em>any</em>
perturbation lands at ARI ~0.6&ndash;0.75 (see the random-drop control), because the
solution is only marginally stable &mdash; for the 32e model even float-level noise
flips it to ARI 0.62. ARI must therefore be read against the random-control floor,
not against 1.0; NMI vs topics is the cleaner external anchor.</p>''')}

{card("results", "Results", table(
        ["Model", "Dims (layers &times; experts)", "Top-1 dim share of signature",
         "Top-5", "Top-32", "Effective dims (median)",
         "Best single-dim AUC (median)", "Full-pattern AUC"], overview_rows)
+ table(
        ["Model", "Rerun baseline ARI", "ARI after drop top-16/cluster (vs random)",
         "NMI topics: baseline", "NMI topics: drop top-16 (vs random)",
         "NMI topics: keep ONLY top-1/cluster (~32 dims)"], ablation_rows) + '''
<p><strong>Answer: the clusters are carried by the broad activation pattern, with
individual experts as redundant markers &mdash; not the cause.</strong></p>
<ul>
<li><strong>Signatures are highly distributed</strong> &mdash; a cluster's single
strongest dim carries only ~1.4&ndash;2.1% of its deviation mass, the top 32 dims
only ~20&ndash;29%, and the effective dimensionality is roughly <em>half the whole
space</em> (e.g. ~1,038 of 2,032 dims at 128e).</li>
<li><strong>Yet near-perfect marker experts exist</strong> &mdash; every cluster has
at least one single (layer, expert) dim with one-vs-rest AUC ~0.98&ndash;0.99,
barely below the full-pattern 0.996. Individual experts are excellent
<em>identifiers</em> of each cluster.</li>
<li><strong>But they are not necessary</strong> &mdash; deleting every cluster's
top-16 dims (~460 dims at 128e) hurts cluster identity more than random deletions
(ARI 0.39 vs 0.67) yet the re-formed clustering is <em>still almost as
topic-aligned</em> (NMI 0.40 vs 0.45 baseline): remove the markers and an
equivalent semantic organization re-emerges from the remaining pattern.</li>
<li><strong>And a few markers are nearly sufficient</strong> &mdash; keeping ONLY each
cluster's single top dim (~32 of 2,032 dims, 1.6%) already recovers NMI-vs-topics
0.41 of the 0.45 baseline; ~240 dims fully match it.</li>
<li><strong>Stable across pool sizes</strong> &mdash; the same picture holds at 32e,
64e, 96e and 128e; signatures spread slightly wider as the pool grows (top-5 mass
8.4% &rarr; 5.5%).</li>
</ul>
<p>In short: cluster information in the router space is <em>massively
redundant</em>. A handful of expert dims would suffice to reconstruct the clusters,
and no handful is load-bearing &mdash; the semantic structure lives in the
correlated activation pattern of hundreds of experts simultaneously.</p>'''
+ f"<h4>Featured: {featured_lbl}</h4>"
+ fig_row(
    img_tag(at / featured_run / "ablation_curves.png",
            f"{featured_lbl}: drop/keep ablations. Red (drop top-m) falls below gray (drop random) in ARI, but NMI-vs-topics (right) barely moves; green (keep only top-m) climbs to baseline with ~240 dims"),
    img_tag(at / featured_run / "single_dim_vs_full_auc.png",
            f"{featured_lbl}: every cluster has a single (layer, expert) dim with AUC ~0.96-1.0 (red), nearly matching the full pattern (blue)"))
+ fig_row(
    img_tag(at / featured_run / "signature_concentration.png",
            f"{featured_lbl}: even each cluster's top-32 dims carry only ~10-34% of its centroid-deviation mass"))
+ "<h4>Other models</h4>" + "".join(other_details))}
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
.card.method { border-left-color:#7c3aed; } .card.method h3 { color:#7c3aed; }
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
