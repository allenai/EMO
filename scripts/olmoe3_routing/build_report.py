# PARENT: "scripts/sparse_experts/build_report.py" (visual framework: CSS/JS/tab structure, card/table/figure helpers)
# DESCRIPTION:
#     Builds claude_outputs/olmoe3_routing/report.html: router statistics of the OLMoE3-ladder 275M
#     models trained in this repo (scripts/sparse_experts/olmoe3_275m.py; 512 experts, standard vs EMO
#     routing, 10B tokens) on 8,000 unseen training-stream instances (65.5M tokens, steps 19,076-38,146
#     of the same data order), under PREFIX-RESTRICTED document pools: layers 1-3 / 1-6 / all 9 forced
#     to a per-document pool of 32/64/128/256 experts while the remaining layers route freely. Tables
#     and figures come from claude_outputs/olmoe3_routing/runs/{metrics.json,figs/} produced by
#     scripts/sparse_experts/olmoe3_routing/analyze_routing.py (rerun that first).
#
#   python scripts/olmoe3_routing/build_report.py
##############################################################
import html
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "claude_outputs/olmoe3_routing"
RUNS = OUT / "runs"
RUNS1000 = OUT / "runs_1000"
_spec = importlib.util.spec_from_file_location("ml_report", ROOT / "scripts/meta_learning/build_report.py")
_ml = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ml)
card, table, img_tag, fig_row, CSS, JS = _ml.card, _ml.table, _ml.img_tag, _ml.fig_row, _ml.CSS, _ml.JS

LAYERS = [str(l) for l in range(1, 10)]
MODEL_LABEL = {"std": "standard MoE (512e)", "emo": "EMO (512e)", "std1000": "standard MoE (1000e)", "emo1000": "EMO (1000e)", "std128": "standard MoE (128e)", "emo128": "EMO (128e)"}


def cond_sort(c):
    return (c != "none", c.split(":")[0], int(c.split(":")[1]) if ":" in c else 0)


def fig(name, caption, d=RUNS):
    p = d / "figs" / name
    return img_tag(p if p.exists() else d / name, caption)


def f(x, nd=3):
    return "&mdash;" if x is None else f"{x:.{nd}f}"


def layer_table(res, key, nd=3, sub=None, subkey=None, conds=None):
    """rows = conditions (both models interleaved), cols = layers."""
    rows = []
    for model in ("std", "emo"):
        for cond in sorted((conds or res.get(model, {})), key=cond_sort):
            r = res[model].get(cond)
            if r is None: continue
            if sub:
                vals = [r[sub]["layers"][l][subkey] for l in LAYERS]
            else:
                vals = [r["layers"][l][key] for l in LAYERS]
            rows.append([MODEL_LABEL[model], cond, f(r["meta"]["mean_ce"])] + [f(v, nd) for v in vals])
    return table(["model", "restriction", "CE"] + [f"L{l}" for l in LAYERS], rows)


def ce_table(res, res1000):
    rows = []
    for model, conds in list(res.items()) + list(res1000.items()):
        base = conds["none"]["meta"]["mean_ce"]
        for cond in sorted(conds, key=cond_sort):
            r = conds[cond]; ce = r["meta"]["mean_ce"]
            grp = r.get("ce_by_group", {})
            rows.append([MODEL_LABEL[model], cond, f(ce), f"{ce-base:+.3f}"] + [f(grp.get(g)) for g in ("web", "pdf", "code", "math", "other")])
    return table(["model", "restriction", "CE (65.5M tok)", "&Delta; vs unrestricted", "web", "pdf", "code", "math", "other"], rows)


def split_findings(findings_html):
    """findings.html is a flat sequence of card divs (Finding 1..5); return them as a list so tabs can pick their own."""
    parts = [c for c in findings_html.split('<div class="card') if c.strip()]
    return ['<div class="card' + c for c in parts]


def build_overview():
    setup = (
        "<p>Two 512-expert MoE models at <b>276.7M active / 2.61B total</b> parameters, trained here for 10B tokens with a WSD schedule "
        "(constant LR after warmup, no decay). The architecture follows the 275M rung of the scaling-ladders repo unchanged. "
        "We also trained a <b>1000-expert</b> version by raising only the expert count with everything else fixed, so the total size grows "
        "to 4.90B (279.5M active), and a <b>128-expert</b> version the same way (0.80B total, 274.5M active). "
        "For each expert count we trained an <b>EMO</b> version and a <b>standard MoE</b> version.</p>"
        "<p>All routing statistics in this report come from forward passes of these checkpoints on 65.5M unseen training-stream tokens "
        "(8,000 instances from the 10B&ndash;20B token window of the same data order).</p>"
    )
    guide = (
        "<ul>"
        "<li><b>Method</b>: extraction pipeline, metric definitions, sanity checks.</li>"
        "<li><b>1</b>: what layers 7&ndash;9 do when layers 1&ndash;3 / 1&ndash;6 are forced to per-document pools; CE of every pass.</li>"
        "<li><b>2</b>: document-poolability and co-activation modularity of every layer, every condition.</li>"
        "<li><b>3</b>: do documents with the same early pool share later experts?</li>"
        "<li><b>4</b>: cross-layer expert dependence.</li>"
        "<li><b>5</b>: the 128- and 1000-expert arms.</li>"
        "<li><b>6</b>: lift and co-occurrence heatmaps.</li>"
        "<li><b>7</b>: spectral-k sweep and the layer-1 expert&rarr;document partition test.</li>"
        "</ul>"
    )
    return card("info", "Setup", setup) + card("info", "Where things are", guide)


def build_method():
    body = """
<p><b>Pipeline</b> (<code>scripts/sparse_experts/olmoe3_routing/</code>): <code>extract_stream.py</code> rebuilds the run's
dataset + loader from the checkpoint config with the pinned OLMo-core (fingerprint asserted equal to the checkpoint's; the run's own
global-index file is reused and guarded against rewrite) and dumps steps 19,074&ndash;38,147 (1.22M instances, 38 GB);
<code>sample_instances.py</code> draws the stratified 8k sample; <code>extract_routing.py</code> loads the OLMoDDP checkpoint natively
(flattened fp32 <code>module.&lt;name&gt;.main</code> optimizer tensors via the pinned HF-converter loader; first-batch CE guard &lt; 4),
runs bf16 forward passes on one H100 (~230k tok/s), applies the restriction, and accumulates counts on the GPU; <code>analyze_routing.py</code>
computes the metrics below.</p>
<p><b>Metrics per MoE layer</b> (all over the 65.5M-token pass):</p>
<ul>
<li><b>poolable top-P</b>: for each document, the share of its routed (token, expert) assignments in that layer that fall inside the
document's own top-P experts ranked by the document's summed router scores in that layer. Unweighted mean over documents with &ge; 64 tokens
(<code>_unw</code>) and token-weighted mean. 1.0 by construction in a restricted layer.</li>
<li><b>effective # experts per document</b>: exp(entropy of the document's expert-usage histogram).</li>
<li><b>Louvain / spectral Q</b>: Newman modularity of the token-level co-activation lift graph (log-lift, negatives clipped), with
a shuffled-label null (same recipe as the sparse_experts co-activation report).</li>
<li><b>early-pool conditioning</b>: documents are k-means clustered (k=8) by their layer-1 top-32 expert SET; for every layer we report
the normalized MI between cluster id and expert usage, and the mean Jaccard of documents' top-64 expert sets for pairs within vs across
clusters. A positive within-minus-across gap in a <em>later, unrestricted</em> layer means that documents whose early pools agree also
share later experts.</li>
<li><b>cross-layer NMI</b>: normalized mutual information between the expert chosen at layer l and at layer m (from the (E&times;E)
cross-layer assignment counts).</li>
</ul>
<p><b>Sanity checks</b>: router self-test (pool-masked standard router == EmoRouterV2 bit-for-bit; accumulators == brute force); a per-batch
assertion that restricted layers never use more than P experts in a document (this caught a config-aliasing bug in the first EMO pilot);
unrestricted CE 2.44 (std) / 2.46 (EMO) matches the runs' end-of-training loss.</p>"""
    return card("info", "Method", body)


def build_late_layers(res, res1000, findings):
    """The direct test: layers 7-9 (free) under unrestricted vs layers 1-3 vs 1-6 restricted, per pool size."""
    late = ["7", "8", "9"]
    def row(model, cond):
        x = res[model][cond]; ly = x["layers"]
        j = lambda k, nd: " / ".join(f"{ly[l][k]:.{nd}f}" for l in late)
        return [MODEL_LABEL[model], cond, f(x["meta"]["mean_ce"]), j("poolable_top32_unw", 2), j("poolable_top64_unw", 2),
                j("poolable_top128_unw", 2), j("doc_eff_experts_unw", 0), j("Q_louvain", 2)]
    hdr = ["model", "restriction", "CE", "top-32 share L7 / L8 / L9", "top-64 share", "top-128 share", "eff. experts / doc", "Louvain Q"]
    out = ("<p>Layers 7&ndash;9 are unrestricted in every row except the all-layer reference rows. <b>top-P share</b> = fraction of a "
           "document's routed assignments in that layer that fall inside the document's own top-P experts of that layer (unweighted mean over "
           "documents with &ge; 64 tokens); <b>eff. experts</b> = exp(entropy) of the document's expert-usage histogram in that layer.</p>")
    for model in ("emo", "std"):
        rows = []
        for P in (32, 64, 128):
            for cond in (["none"] if P == 32 else []) + [f"1-3:{P}", f"1-6:{P}", f"1-9:{P}"]:
                if cond in res[model]: rows.append(row(model, cond))
        out += card("info", f"{MODEL_LABEL[model]}: layers 7&ndash;9 under early-layer restriction", table(hdr, rows))
    out += fig_row(fig("poolable64_by_layer.png", "Top-64 poolability by layer (all conditions)."), fig("doc_eff_experts_by_layer.png", "Effective experts per document by layer (all conditions)."))
    out += card("info", "Cross-entropy of every pass", ce_table(res, res1000))
    return findings[1] + findings[2] + out


def build_poolability(res, findings):
    body = findings[0] + findings[3] + (
        "<p>Does restricting early layers make later layers more document-poolable? Rows are conditions, columns MoE layers; restricted "
        "layers read 1.000 by construction, so look at the unrestricted columns to the right of each restricted prefix.</p>"
        + card("info", "Share of routed assignments inside the document's top-64 experts (unweighted over docs)", layer_table(res, "poolable_top64_unw"))
        + card("info", "Same, top-128", layer_table(res, "poolable_top128_unw"))
        + card("info", "Effective # experts per document", layer_table(res, "doc_eff_experts_unw", nd=1))
        + fig_row(fig("poolable64_by_layer.png", "Top-64 poolability by layer; colour = pool size, line style = restricted prefix (solid 1-3, dashed 1-6, dotted 1-9)."),
                  fig("doc_eff_experts_by_layer.png", "Effective number of experts per document by layer."))
        + card("info", "Louvain modularity Q of the token co-activation graph", layer_table(res, "Q_louvain"))
        + card("info", "Spectral Q (k=8)", layer_table(res, "Q_spectral"))
        + fig_row(fig("q_louvain_by_layer.png", "Louvain Q by layer and condition."), fig("router_entropy_by_layer.png", "Mean router entropy by layer (nats; ln 512 = 6.24)."))
    )
    return body


def build_earlypool(res):
    body = (
        "<p>Documents clustered by their layer-1 top-32 expert set (k=8). For each layer: NMI(cluster ; expert usage), and the Jaccard of "
        "documents' top-64 expert sets within vs across clusters. The gap (within &minus; across) in the <em>free</em> layers is the direct "
        "test of the hypothesis that later-layer modularity appears once early routing is restricted.</p>"
        + card("info", "NMI(early-pool cluster ; expert usage)", layer_table(res, None, sub="earlypool", subkey="nmi_cluster_expert"))
        + card("info", "Jaccard within clusters", layer_table(res, None, nd=2, sub="earlypool", subkey="jaccard_within"))
        + card("info", "Jaccard across clusters", layer_table(res, None, nd=2, sub="earlypool", subkey="jaccard_across"))
        + fig_row(fig("earlypool_jaccard_gap.png", "Within-minus-across Jaccard of top-64 expert sets, by layer."), fig("earlypool_nmi.png", "NMI(early-pool cluster ; expert) by layer."))
    )
    return body


def build_cross(res):
    body = ("<p>Normalized MI between the expert selected at layer l (rows) and at layer m (columns), unrestricted vs the tightest restrictions.</p>"
            + fig("cross_layer_nmi.png", "Cross-layer expert NMI heatmaps (diagonal blanked)."))
    return body


def build_1000(res1000, res, findings):
    if not res1000:
        return card("warn", "1000-expert arms", "<p>No analysis found in runs_1000/.</p>")
    rows = []
    for model, conds in list(res.items()) + list(res1000.items()):
        r = conds["none"]
        rows.append([MODEL_LABEL[model], f(r["meta"]["mean_ce"])] + [f(r["layers"][l]["poolable_top64_unw"]) for l in LAYERS])
    rows2 = []
    for model, conds in list(res.items()) + list(res1000.items()):
        r = conds["none"]
        rows2.append([MODEL_LABEL[model]] + [f(r["layers"][l]["Q_louvain"]) for l in LAYERS])
    return (findings[4] + card("info", "Unrestricted pass: held-out CE and top-64 poolability by layer, all trained arms", table(["model", "CE", *[f"L{l}" for l in LAYERS]], rows))
            + card("info", "Louvain Q by layer, unrestricted", table(["model", *[f"L{l}" for l in LAYERS]], rows2))
            + fig_row(fig("poolable64_by_layer.png", "1000-expert arms: top-64 poolability (unrestricted only).", RUNS1000), fig("q_louvain_by_layer.png", "1000-expert arms: Louvain Q.", RUNS1000)))


HEAT = OUT / "heatmaps"
HEAT_TAGS = [("std512_full", "standard MoE, 512 experts, full routing"), ("emo512_full", "EMO, 512 experts, full routing"),
             ("std1000_full", "standard MoE, 1000 experts, full routing"), ("emo1000_full", "EMO, 1000 experts, full routing"),
             ("std512_pool64", "standard MoE, 512 experts, all 9 layers at pool 64"), ("emo512_pool64", "EMO, 512 experts, all 9 layers at pool 64")]


def build_heatmaps():
    body = ("<p>Per-layer expert co-activation on the same 65.5M-token pass, drawn exactly as in the sparse_experts report: log2 lift "
            "(observed / expected-under-independence pair counts, experts ordered by spectral cluster (k=8) then usage, clipped to &plusmn;3) and "
            "conditional co-activation P(E<sub>j</sub> | E<sub>i</sub>), each at token level (both experts in one token's top-16) and document "
            "level (both used at least once in a document). Full routing for all four trained arms, plus the 512-expert models with every layer "
            "pinned to a per-document pool of 64 (the analogue of the earlier report's pool-64 tab).</p>")
    for tag, label in HEAT_TAGS:
        qf = HEAT / f"{tag}_heatmaps.json"
        qtxt = ""
        if qf.exists():
            q = json.load(open(qf)); qtxt = " &middot; spectral Q by layer: " + ", ".join(f"{v:.2f}" for v in q["Q_spectral"])
        h = lambda name, cap: img_tag(HEAT / name, cap)
        sub = (fig_row(h(f"{tag}_lift_tok_grid.png", "log2 lift, token level"), h(f"{tag}_cond_tok_grid.png", "conditional co-activation P(j|i), token level"))
               + fig_row(h(f"{tag}_lift_doc_grid.png", "log2 lift, document level"), h(f"{tag}_cond_doc_grid.png", "conditional co-activation, document level"))
               + fig_row(h(f"{tag}_usage.png", "per-expert token usage by layer"), h(f"{tag}_lift_hist.png", "distribution of pairwise lift")))
        body += card("info", f"{label}{qtxt}", sub)
    return body


KS = OUT / "ksweep"


def build_ksweep():
    body = ("<p>Why k=8? It was inherited from the sparse_experts analysis, not chosen for this data; it only affects the display ordering and the "
            "spectral-Q number (the lift matrix is k-free). Here the spectral cluster count is swept over k = 4 / 8 / 16 / 32 / 64 on the EMO "
            "1000-expert full-routing pass, and for each k the experts of ONE layer are partitioned into k clusters and every document is assigned "
            "to the cluster that captures most of its routed assignments in that layer. <b>purity</b> = the share of the document's assignments in "
            "its assigned cluster (mean over documents with &ge; 64 tokens), against a random expert partition with the same cluster sizes; "
            "<b>lift within / across</b> = mean log2 lift among the experts the document actually uses (top-16 by usage) inside vs outside its "
            "cluster; NMI(cluster ; source) = how much the document clusters line up with the data source.</p>")
    body += fig(f"emo1000_full_Q_vs_k.png", "Spectral Q vs k per layer (EMO 1000, full routing); shuffled-label null shown for layer 1.", KS)
    hdr = ["k", "Q (this layer)", "purity", "purity null", "median purity", "docs with purity &gt; 0.5", "lift within", "lift across", "NMI(cluster;source)", "largest doc clusters"]
    for tag, layer, label in (("emo1000_full", 1, "EMO 1000, layer-1 partition"), ("emo1000_full", 5, "EMO 1000, layer-5 partition"), ("emo1000_full", 9, "EMO 1000, layer-9 partition"),
                              ("emo512_full", 1, "EMO 512, layer-1 partition"), ("std1000_full", 1, "standard 1000, layer-1 partition")):
        jf = KS / f"{tag}_L{layer}_ksweep.json"
        if not jf.exists(): continue
        r = json.load(open(jf)); rows = []
        for k, v in sorted(r.items(), key=lambda kv: int(kv[0])):
            rows.append([k, f(v["Q_spectral"][layer - 1]), f(v["purity"]), f(v["purity_null"]), f(v["purity_median"]), f"{v['frac_purity_gt_half']:.2f}", f(v["lift_within"], 2), f(v["lift_across"], 2), f(v["nmi_doc_cluster_source"]),
                         ", ".join(str(x) for x in sorted(v["doc_cluster_sizes"], reverse=True)[:5])])
        figs = fig_row(*[img_tag(KS / f"{tag}_L{layer}_k{k}_docpartition.png", f"k={k}: purity histogram vs null; source composition of the document clusters") for k in (4, 8, 32)])
        body += card("info", label, table(hdr, rows) + figs)
    body += card("info", "EMO 1000 full routing: log2 lift grids ordered by spectral clusters at each k",
                 "".join(fig(f"emo1000_full_k{k}_lift_tok_grid.png", f"k = {k}", KS) for k in (4, 8, 16, 32, 64)))
    return body


def build_next():
    return card("info", "Next steps", "<ul><li>Sweep every prefix length 1..8 at one pool size to locate where later-layer structure (if any) switches on.</li>"
                "<li>Replace the k-means early-pool clustering with the exact early pool identity (documents sharing the same top-32 set) once enough documents share pools.</li>"
                "<li>Repeat on the 1000-expert arms under restriction (only the unrestricted pass exists so far).</li>"
                "<li>Selective-expert finetuning on the pools found here, as in the sparse_experts plan.</li></ul>")


def main(findings_path=OUT / "findings.html"):
    res = json.load(open(RUNS / "metrics.json"))
    res1000 = json.load(open(RUNS1000 / "metrics.json")) if (RUNS1000 / "metrics.json").exists() else {}
    findings = split_findings(findings_path.read_text()) if findings_path.exists() else []
    findings += [card("warn", "Finding", "<p>findings.html not written yet.</p>")] * (5 - len(findings))
    tabs = [
        ("overview", "Overview", build_overview()),
        ("method", "Method", build_method()),
        ("late", "1 · Layers 7-9 under early restriction", build_late_layers(res, res1000, findings)),
        ("pool", "2 · Poolability & modularity, all layers", build_poolability(res, findings)),
        ("early", "3 · Early-pool conditioning", build_earlypool(res)),
        ("cross", "4 · Cross-layer NMI", build_cross(res)),
        ("e1000", "5 · 128- and 1000-expert arms", build_1000(res1000, res, findings)),
        ("heat", "6 · Co-activation heatmaps", build_heatmaps()),
        ("ksweep", "7 · k sweep & document partition", build_ksweep()),
        ("next", "Next steps", build_next()),
    ]
    nav = "".join(f'<button data-target="{tid}">{name}</button>' for tid, name, _ in tabs)
    sections = "".join(f'<section class="tab" id="{tid}">{body}</section>' for tid, _, body in tabs)
    title = "OLMoE3-ladder 275M: router statistics under prefix-restricted document pools (standard vs EMO)"
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style><style>.muted{{color:var(--muted);font-weight:normal;font-size:0.85em}}</style></head>
<body><header><a class="home-link" href="/">&larr; all reports</a><h1>{html.escape(title)}</h1>
<p>olmoe3_routing &mdash; the 512-expert standard-MoE and EMO 275M models trained here (10B tokens, ladder recipe) run on 65.5M unseen
training-stream tokens with layers 1&ndash;3 / 1&ndash;6 / 1&ndash;9 forced to per-document expert pools of 32/64/128/256 while the
remaining layers route freely; poolability, modularity, early-pool conditioning and cross-layer dependence per layer &middot;
generated by scripts/olmoe3_routing/build_report.py</p></header>
<div class="topbar"><nav>{nav}</nav><div id="subnav"></div></div><main>{sections}</main><script>{JS}</script></body></html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.html").write_text(page)
    print(f"Wrote {OUT/'report.html'} ({(OUT/'report.html').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
