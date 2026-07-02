#!/usr/bin/env python3
"""Build the modular_extension experiment report (self-contained HTML).

Renders the per-cluster expert-usage characterization
(modular_extension/cluster/emo100b_step23842/expert_attribution/) into
claude_outputs/modular_extension/report.html: headline verdict, the soft/hard contrast,
base64-embedded figures, and a sortable per-cluster metrics table joined with the
partition's per-cluster doc/token counts + top sources.

Run:  python scripts/modular_extension/build_report.py
Registered in scripts/publish_reports.sh (deploys to https://emo-reports.pages.dev/).
"""
from __future__ import annotations
import base64, html, json, mimetypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATTR = ROOT / "modular_extension/cluster/emo100b_step23842/expert_attribution"
PART = ROOT / "modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-110B/doc_clusters_k64_summary.json"
OUT = ROOT / "claude_outputs/modular_extension/report.html"


def img(path: Path) -> str:
    if not path.exists():
        return f"<p class='missing'>missing: {html.escape(path.name)}</p>"
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"<img loading='lazy' src='data:{mime};base64,{b64}' alt='{html.escape(path.name)}'>"


def figure(emb: str, name: str, caption: str) -> str:
    return f"<figure>{img(ATTR / emb / name)}<figcaption>{caption}</figcaption></figure>"


def main():
    soft = json.load(open(ATTR / "doc_probs/metrics.json"))
    hard = json.load(open(ATTR / "doc_topk_freq/metrics.json"))
    part = {c["cluster"]: c for c in json.load(open(PART))["clusters"]} if PART.exists() else {}

    def headline(s):
        vc = s["verdict_counts"]
        return (f"<tr><td>{s['embedding']}</td><td>{vc['few-expert']}</td>"
                f"<td>{vc['broad-redundant']}</td><td>{vc['subtle']}</td>"
                f"<td>{s['median_effective_dims']:.1f} / {s['n_dims']}</td>"
                f"<td>{s['median_best_single_dim_auc']:.3f}</td>"
                f"<td>{s['median_full_pattern_auc']:.3f}</td>"
                f"<td>{s['median_effective_experts_used']:.1f} / {s['n_experts_per_layer']}</td></tr>")

    # per-cluster table joins soft metrics + hard verdict + partition counts
    hard_by_c = {c["cluster"]: c for c in hard["per_cluster"]}
    rows = []
    for c in soft["per_cluster"]:
        i = c["cluster"]
        pc = part.get(i, {})
        hv = hard_by_c.get(i, {}).get("verdict", "")
        sig = ", ".join(f"L{e['layer']}·E{e['expert']}" for e in c["signature_experts"][:3])
        top_src = "; ".join(f"{k.split('/')[0]}×{v}" for k, v in list(pc.get("top_sources", {}).items())[:2])
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

    intro = (
        f"Each of the <b>{soft['k']}</b> clusters partitions the {soft['n_dims'] // soft['n_experts_per_layer']}-layer × "
        f"{soft['n_experts_per_layer']}-expert routing space of the EMO 100B checkpoint over "
        f"{sum(c['size'] for c in soft['per_cluster']):,} documents (the 100B–110B training window). "
        f"<b>The answer is neither “a few experts” nor “subtle”:</b> the distinguishing signal is spread "
        f"over most of the {soft['n_dims']} layer-expert dimensions (top-5 carry only "
        f"~{soft['median_top5_mass']*100:.0f}%), yet a single well-chosen expert already separates a "
        f"cluster with AUC ~{soft['median_best_single_dim_auc']:.2f} (≈ the full-pattern "
        f"{soft['median_full_pattern_auc']:.2f}). Clusters concentrate routing onto "
        f"~{soft['median_effective_experts_used']:.0f}/{soft['n_experts_per_layer']} experts per layer "
        f"vs the corpus’s ~{soft['global_effective_experts']:.0f}. Each cluster is a "
        f"<b>broad, redundant expert-usage signature</b> — many experts each shifted consistently, any "
        f"one of them enough to flag the cluster."
    )

    page = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>modular_extension — cluster expert-usage</title>
<style>
:root{{--bg:#0f1115;--card:#181b22;--fg:#e6e8ec;--mut:#9aa3b2;--line:#2a2f3a;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}}
header{{background:linear-gradient(180deg,#1b2130,#12151c);padding:26px 22px;border-bottom:1px solid var(--line)}}
h1{{margin:0 0 6px;font-size:22px}} .sub{{color:var(--mut);max-width:70ch}}
main{{max-width:1180px;margin:0 auto;padding:22px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:0 0 20px}}
h2{{margin:0 0 12px;font-size:17px}} h3{{color:var(--mut);font-size:13px;text-transform:uppercase;letter-spacing:.04em;margin:18px 0 8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child,td.sig,td.src{{text-align:left}} th{{color:var(--mut);cursor:pointer;position:sticky;top:0;background:var(--card)}}
.tablewrap{{max-height:560px;overflow:auto;border:1px solid var(--line);border-radius:8px}}
td.sig,td.src{{color:var(--mut);font-size:12px;white-space:normal}}
.v-few-expert{{color:#ff6b6b;font-weight:600}} .v-broad-redundant{{color:#b392f0;font-weight:600}} .v-subtle{{color:#4d9fff;font-weight:600}}
figure{{margin:0 0 14px}} figure img{{width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}}
figcaption{{color:var(--mut);font-size:13px;margin-top:6px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} @media(max-width:820px){{.grid{{grid-template-columns:1fr}}}}
.hl td:first-child{{font-weight:600}} .missing{{color:#c66}}
code{{background:#0c0e12;padding:1px 5px;border-radius:4px;color:#c9d3e6}}
</style></head><body>
<header><h1>modular_extension — what distinguishes each document-router cluster?</h1>
<div class='sub'>{intro}</div></header>
<main>

<section><h2>Headline</h2>
<p class='sub'>Two independent axes. <b>Breadth</b>: is the distinguishing signal carried by a few experts
or spread broadly? <b>Strength</b>: does a single expert nearly identify the cluster, or only the whole
joint pattern? A cluster is <span class='v-few-expert'>few-expert</span> (sparse & strong),
<span class='v-broad-redundant'>broad-redundant</span> (many experts, each individually decisive), or
<span class='v-subtle'>subtle</span> (many experts, only the joint pattern separates).</p>
<table class='hl'><thead><tr><th>signal</th><th>few-expert</th><th>broad-redundant</th><th>subtle</th>
<th>median eff. deviation dims</th><th>median best-single AUC</th><th>median full-pattern AUC</th>
<th>median experts used/layer</th></tr></thead><tbody>
{headline(soft)}{headline(hard)}</tbody></table>
<p class='sub' style='margin-top:10px'><b>doc_probs</b> = soft router affinity (includes non-selected
experts); <b>doc_topk_freq</b> = hard expert selection (which experts tokens actually route to). The
gap between the two rows is the answer to “few experts vs subtle patterns”.</p>
</section>

<section><h2>Soft affinity vs hard selection</h2>
<div class='grid'>
<figure>{img(ATTR / 'hard_vs_soft.png')}<figcaption>Best single-expert separability per cluster:
hard selection (x) vs soft affinity (y). Points near the diagonal = the actual expert selection
already identifies the cluster; points high above it = the cluster is only separable via subtle
soft-affinity signal.</figcaption></figure>
{figure('doc_probs', 'verdict_scatter.png', 'Per-cluster verdict (doc_probs): effective deviation dims vs routing peakedness; size = #docs.')}
</div></section>

<section><h2>Figures — doc_probs (soft affinity)</h2><div class='grid'>
{figure('doc_probs','single_dim_vs_full_auc.png','Best single expert vs full activation pattern. Small gap ⇒ few experts define the cluster.')}
{figure('doc_probs','signature_concentration.png','Fraction of each cluster’s deviation carried by its top-m experts.')}
{figure('doc_probs','deviation_heatmap.png','Per-cluster expert deviation (z-scored). Blocky vertical stripes ⇒ shared few experts dominate; diffuse ⇒ subtle.')}
{figure('doc_probs','example_profiles.png','16×63 routing profiles for example clusters, most concentrated → most distributed.')}
</div></section>

<section><h2>Figures — doc_topk_freq (hard selection)</h2><div class='grid'>
{figure('doc_topk_freq','single_dim_vs_full_auc.png','Same, for actual expert selection.')}
{figure('doc_topk_freq','deviation_heatmap.png','Selection-frequency deviation per cluster.')}
</div></section>

<section><h2>Per-cluster metrics</h2>
<p class='sub'>Verdicts shown for both signals. <code>eff. dims</code> = effective # experts carrying the
deviation (low ⇒ few); <code>top5 mass</code> = fraction of deviation in the top 5 experts;
<code>best-AUC</code> = best single expert’s one-vs-rest separability; <code>full-AUC</code> =
whole-pattern separability; <code>exp. used</code> = effective experts used per layer;
<code>cos</code> = mean doc-to-centroid cosine (within-cluster consistency).</p>
<div class='tablewrap'><table id='t'><thead><tr>
<th>cluster</th><th>verdict (soft)</th><th>verdict (hard)</th><th>#docs</th><th>#tokens</th>
<th>eff. dims</th><th>top5 mass</th><th>best-AUC</th><th>full-AUC</th><th>exp. used</th><th>cos</th>
<th>top signature experts</th><th>top sources</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>

</main>
<script>
// click-to-sort table columns
const t=document.getElementById('t');
t.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{{
 const tb=t.tBodies[0], rows=[...tb.rows];
 const num=v=>parseFloat(v.replace(/[^0-9.\\-]/g,''));
 const asc=th.dataset.asc=th.dataset.asc==='1'?'':'1';
 rows.sort((a,b)=>{{const x=a.cells[i].innerText,y=b.cells[i].innerText;
  const nx=num(x),ny=num(y); const c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);
  return asc?c:-c;}});
 rows.forEach(r=>tb.appendChild(r));
}});
</script>
</body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
