#!/usr/bin/env python3
"""Build the modular_extension experiment report (self-contained HTML).

Renders the per-cluster expert-usage characterization
(modular_extension/cluster/emo100b_step23842/expert_attribution/) into
claude_outputs/modular_extension/report.html: headline verdict, the soft/hard contrast,
base64-embedded figures, and a sortable per-cluster metrics table joined with the
partition's per-cluster doc/token counts + top sources.

CSS/JS/tab structure kept in sync with scripts/models_routerfixed/build_report.py so this
report matches the rest of the experiment reports on https://emo-reports.pages.dev/.

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
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


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


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------


def build_overview(soft, hard) -> str:
    n_docs = sum(c["size"] for c in soft["per_cluster"])
    return f"""
<p><strong>Question:</strong> we partitioned {n_docs:,} documents of the EMO 100B
checkpoint's 100B&ndash;110B training window into <strong>{soft['k']} clusters</strong> by their
document-level router embeddings. What actually <em>makes</em> each cluster distinct &mdash; does a
cluster route <em>consistently and heavily to a few experts</em>, or does it use experts at
near-average rates and differ only through <em>subtle, distributed</em> patterns? And is that best
seen in <strong>hard expert selection</strong> (which experts tokens route to) or only in
<strong>soft router affinity</strong>?</p>

{card("goal", "Two axes we measure", '''
<p><strong>Breadth</strong> &mdash; is the distinguishing signal carried by a handful of experts, or
spread across most of them? (effective # of deviating experts, top-5 mass.) <strong>Strength</strong>
&mdash; does a <em>single</em> expert nearly identify the cluster on its own, or only the whole joint
pattern? (best single-expert one-vs-rest AUC vs full-pattern AUC.) Every cluster is then labelled
<span class="v-few-expert">few-expert</span> (sparse &amp; strong),
<span class="v-broad-redundant">broad-redundant</span> (many experts, each individually decisive), or
<span class="v-subtle">subtle</span> (many experts, only the joint pattern separates).</p>''')}

{card("results", "Answer &mdash; broad-redundant, under both signals", f'''
<p><strong>Neither &ldquo;a few experts&rdquo; nor &ldquo;subtle&rdquo;.</strong> The distinguishing
signal is spread over most of the {soft['n_dims']} layer&times;expert dimensions (the top-5 experts
carry only ~{soft['median_top5_mass']*100:.0f}% of a cluster's deviation), so clusters are <em>not</em>
defined by a small signature set. Yet the per-expert differences are large, not subtle: a single
well-chosen expert already separates a cluster with AUC ~{soft['median_best_single_dim_auc']:.2f}
&mdash; almost the full-pattern {soft['median_full_pattern_auc']:.2f}. Clusters concentrate routing
onto ~{soft['median_effective_experts_used']:.0f}/{soft['n_experts_per_layer']} experts per layer vs
the corpus average of ~{soft['global_effective_experts']:.0f}. So each cluster is a
<strong>broad, redundant expert-usage signature</strong>: many experts each shift consistently and
substantially, and any one of them already flags the cluster. This holds for both soft affinity and
hard selection (see the <strong>Soft vs hard</strong> tab).</p>''')}

{card("results", "Headline numbers", _headline_table(soft, hard) + '''
<p class="note"><strong>doc_probs</strong> = soft router affinity (mean softmax over experts, includes
non-selected experts); <strong>doc_topk_freq</strong> = hard expert selection (how often tokens
actually route to each expert). That the two rows agree is the answer to &ldquo;few experts vs subtle
patterns&rdquo;: the signature is broad and redundant whether you look at affinity or at real
selection.</p>''')}
"""


def _headline_table(soft, hard) -> str:
    def row(s):
        vc = s["verdict_counts"]
        return [
            f"<code>{s['embedding']}</code>",
            f'<span class="v-few-expert">{vc["few-expert"]}</span>',
            f'<span class="v-broad-redundant">{vc["broad-redundant"]}</span>',
            f'<span class="v-subtle">{vc["subtle"]}</span>',
            f"{s['median_effective_dims']:.1f} / {s['n_dims']}",
            f"{s['median_best_single_dim_auc']:.3f}",
            f"{s['median_full_pattern_auc']:.3f}",
            f"{s['median_effective_experts_used']:.1f} / {s['n_experts_per_layer']}",
        ]

    return table(
        ["signal", "few-expert", "broad-redundant", "subtle",
         "median eff. deviation dims", "median best-single AUC",
         "median full-pattern AUC", "median experts used / layer"],
        [row(soft), row(hard)],
    )


def build_method(soft) -> str:
    n_layers = soft["n_layers"]
    n_exp = soft["n_experts_per_layer"]
    return f"""
{card("goal", "Document-level router embeddings", f'''
<p>The published clustering pipeline works at the <em>token</em> level. Here we cluster whole
<em>documents</em>: for each document we run the EMO 100B checkpoint (step 23,842) with router logits
exposed and <strong>average the per-token routing over all of the document's tokens</strong>, per layer
and per expert. That gives each document a {n_layers}&times;{n_exp} = {soft['n_dims']}-dim embedding
(<strong>layer-major</strong>: dim <code>d</code> &rarr; layer <code>d//{n_exp}</code>, expert
<code>d%{n_exp}</code>). Two views are recorded per document:</p>
<ul>
<li><code>doc_probs</code> &mdash; mean softmax affinity per expert (each layer sums to 1; includes
experts the tokens never actually selected).</li>
<li><code>doc_topk_freq</code> &mdash; how often tokens actually selected each expert (each layer sums
to the routed top-k = 7).</li>
</ul>
<p>Documents are then clustered with the same recipe as the token pipeline &mdash; PCA to 95% variance
(&rarr; 150 dims) + L2 normalize, spherical k-means, <strong>k={soft['k']}</strong>, seed 42 &mdash; on
the <code>doc_probs</code> view. This report characterizes the resulting clusters; both embedding views
are re-analyzed against the <em>same</em> assignment.</p>''')}

{card("method", "What each metric means", table(
    ["Metric", "Definition", "Reads as"],
    [
        ("<code>eff. dims</code>",
         "effective # of experts carrying a cluster's deviation &delta; = &mu;<sub>c</sub> &minus; &mu;<sub>global</sub>, exp(entropy of |&delta;|)",
         "low &rArr; a few experts define the cluster; high &rArr; spread out"),
        ("<code>top5 mass</code>",
         "fraction of |&delta;| carried by the 5 largest-deviation experts",
         "high &rArr; concentrated signature"),
        ("<code>best-AUC</code>",
         "best single (layer,expert) one-vs-rest Mann&ndash;Whitney AUC",
         "how well the <em>single</em> most telling expert separates the cluster"),
        ("<code>full-AUC</code>",
         "cosine-to-centroid one-vs-rest AUC (whole pattern)",
         "the ceiling &mdash; using every expert jointly"),
        ("<code>exp. used</code>",
         "effective experts the cluster routes to per layer, exp(mean-layer entropy of &mu;<sub>c</sub>)",
         "absolute peakedness vs the corpus ~%.0f" % soft["global_effective_experts"]),
        ("<code>cos</code>",
         "mean cosine of the cluster's docs to their expert-space centroid",
         "within-cluster consistency (1 = identical routing)"),
    ]) + '''
<p class="note">The verdict combines breadth and strength: a cluster is <em>strong</em> when its best
single expert reaches AUC &ge; 0.8 and recovers &ge; 70% of the full-pattern separability
((best&minus;0.5)/(full&minus;0.5)); <em>sparse</em> when its deviation lives in &le; 10% of experts.
strong &amp; sparse &rarr; <span class="v-few-expert">few-expert</span>; strong &amp; broad &rarr;
<span class="v-broad-redundant">broad-redundant</span>; not strong &rarr;
<span class="v-subtle">subtle</span>.</p>''')}
"""


def build_softhard(soft, hard) -> str:
    return f"""
{card("goal", "Soft affinity vs hard selection", '''
<p>The direct discriminator for the user's dichotomy. <strong>Soft affinity</strong>
(<code>doc_probs</code>) can separate clusters through small, distributed shifts in how strongly tokens
<em>lean</em> toward experts &mdash; even experts they never select. <strong>Hard selection</strong>
(<code>doc_topk_freq</code>) only sees which experts tokens <em>actually route to</em>. If clusters are
&ldquo;subtle&rdquo;, they separate under soft affinity but wash out under hard selection; if they are a
real usage signature, they separate under both.</p>''')}

{card("results", "They separate equally well under both", fig_row(
    img_tag(ATTR / "hard_vs_soft.png",
            "Best single-expert separability per cluster: hard selection (x) vs soft affinity (y). "
            "Points on the diagonal = the actual expert selection already identifies the cluster; "
            "points high above it = only the soft-affinity signal separates it."),
    fig("doc_probs", "verdict_scatter.png",
        "Per-cluster verdict (doc_probs): effective deviation dims (x) vs routing peakedness (y); "
        "point size = #docs. All clusters land in the broad-redundant regime."),
) + f'''
<p>Points cluster on the diagonal: a cluster's best expert is about as telling under hard selection as
under soft affinity (median best-single AUC {hard['median_best_single_dim_auc']:.3f} hard vs
{soft['median_best_single_dim_auc']:.3f} soft). The distinguishing structure is in <em>which experts
tokens really go to</em>, not in a subtle soft-only signal &mdash; it is just spread across many
experts (median {hard['median_effective_dims']:.0f}/{hard['n_dims']} deviating experts under hard
selection).</p>''')}
"""


def build_figures() -> str:
    return f"""
{card("results", "doc_probs &mdash; soft affinity", fig_row(
    fig("doc_probs", "single_dim_vs_full_auc.png",
        "Best single expert vs full activation pattern. Small gap ⇒ few experts already define the cluster."),
    fig("doc_probs", "signature_concentration.png",
        "Fraction of each cluster's deviation carried by its top-m experts (flat curves ⇒ broad)."),
    fig("doc_probs", "deviation_heatmap.png",
        "Per-cluster expert deviation (z-scored). Blocky vertical stripes ⇒ shared few experts dominate; diffuse ⇒ broad."),
    fig("doc_probs", "example_profiles.png",
        "16×63 routing profiles for example clusters, most concentrated → most distributed."),
))}

{card("results", "doc_topk_freq &mdash; hard selection", fig_row(
    fig("doc_topk_freq", "single_dim_vs_full_auc.png",
        "Same, for actual expert selection frequency."),
    fig("doc_topk_freq", "deviation_heatmap.png",
        "Selection-frequency deviation per cluster."),
))}
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
{card("goal", "Per-cluster metrics", '''<p>Every cluster's breadth/strength metrics for both signals,
joined with its document/token counts and dominant sources. Click any column header to sort. See the
<strong>Method</strong> tab for definitions.</p>''')}
<div class="tablewrap"><table id="ct" class="sortable"><thead><tr>{header}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="note">Verdicts colored <span class="v-few-expert">few-expert</span> /
<span class="v-broad-redundant">broad-redundant</span> / <span class="v-subtle">subtle</span>.</p>
"""


# --------------------------------------------------------------------------
# Page assembly (CSS/JS kept in sync with models_routerfixed)
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
th, td { border:1px solid var(--line); padding:6px 12px; text-align:left; vertical-align:top; }
th { background:#f1f5f9; }
tbody tr:nth-child(even) { background:#f8fafc; }
.note { font-size:13px; color:var(--muted); }
code { background:#eef2f7; padding:1px 5px; border-radius:4px; font-size:0.9em; }
.figrow { display:flex; gap:16px; flex-wrap:wrap; margin:14px 0; }
figure { flex:1 1 380px; min-width:300px; max-width:560px; margin:0; }
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "claude_outputs/modular_extension/report.html")
    args = parser.parse_args()

    soft = json.load(open(ATTR / "doc_probs/metrics.json"))
    hard = json.load(open(ATTR / "doc_topk_freq/metrics.json"))
    part = {c["cluster"]: c for c in json.load(open(PART))["clusters"]} if PART.exists() else {}

    tabs = [
        ("overview", "Overview", build_overview(soft, hard)),
        ("method", "Method", build_method(soft)),
        ("soft-vs-hard", "Soft vs hard", build_softhard(soft, hard)),
        ("figures", "Figures", build_figures()),
        ("per-cluster", "Per-cluster", build_per_cluster(soft, hard, part)),
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
<title>EMO modular_extension: what distinguishes each document-router cluster?</title>
<style>{CSS}</style>
</head>
<body>
<header>
<a class="home-link" href="/">&larr; all reports</a>
<h1>EMO modular_extension: what distinguishes each document-router cluster?</h1>
<p>modular_extension &mdash; document-level router clustering of the EMO 100B&ndash;110B training window
(k={soft['k']}) &middot; generated by scripts/modular_extension/build_report.py</p>
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
