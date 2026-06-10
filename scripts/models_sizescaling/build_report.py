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
nav { display:flex; gap:6px; flex-wrap:wrap; padding:10px 28px; background:#1e293b; position:sticky; top:0; z-index:10; }
nav button { border:0; border-radius:6px; padding:7px 14px; font-size:14px; cursor:pointer;
             background:transparent; color:#cbd5e1; }
nav button:hover { background:#334155; }
nav button.active { background:#3b82f6; color:#fff; }
main { max-width:1180px; margin:0 auto; padding:24px 28px 80px; }
section.tab { display:none; }
section.tab.active { display:block; }
.card { background:var(--card); border:1px solid var(--line); border-left:4px solid var(--line);
        border-radius:8px; padding:16px 20px; margin:16px 0; }
.card h3 { margin:0 0 8px; font-size:15px; text-transform:uppercase; letter-spacing:0.05em; }
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
function show(id) {
  document.querySelectorAll('section.tab').forEach(s => s.classList.toggle('active', s.id === id));
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('active', b.dataset.target === id));
  history.replaceState(null, '', '#' + id);
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
<h1>EMO size-scaling: how does expert specialization change with pool size?</h1>
<p>models_sizescaling &mdash; 32 / 64 / 96 / 128 experts, 1B active params, identical 130B-token recipe
&middot; generated by scripts/models_sizescaling/build_report.py</p>
</header>
<nav>{nav}</nav>
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
