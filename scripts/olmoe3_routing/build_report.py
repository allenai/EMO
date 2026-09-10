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


QUESTIONS = [
    ("q1", "Q1 · Out-of-the-box document structure",
     "Do the trained routers already send each document to a small pool of experts, and does EMO differ from standard MoE?"),
    ("q2", "Q2 · Early restriction &rarr; later-layer structure",
     "If the early layers are forced to per-document pools, do the free later layers become more document-poolable or modular, and at what cost?"),
    ("q3", "Q3 · Early-pool dependence",
     "Do documents that share an early-layer pool go on to share later-layer experts, i.e. does a layer's routing depend on the pool upstream?"),
    ("q4", "Q4 · Expert count",
     "How does the picture change with 128 / 512 / 1000 experts at fixed active size?"),
    ("q5", "Q5 · Joint expert / document partition",
     "Can one layer's experts be split into k blocks such that documents split the same way, and does the choice of k matter?"),
]


def build_overview():
    setup = (
        "<p>Two 512-expert MoE models at <b>276.7M active / 2.61B total</b> parameters, trained here for 10B tokens with a WSD schedule "
        "(constant LR after warmup, no decay). The architecture follows the 275M rung of the scaling-ladders repo unchanged. "
        "We also trained a <b>1000-expert</b> version by raising only the expert count with everything else fixed, so the total size grows "
        "to 4.90B (279.5M active), and a <b>128-expert</b> version the same way (0.80B total, 274.5M active). "
        "For each expert count we trained an <b>EMO</b> version and a <b>standard MoE</b> version.</p>"
        "<p>All routing statistics in this report come from forward passes of these checkpoints on 65.5M unseen training-stream tokens "
        "(8,000 instances from the 10B&ndash;20B token window of the same data order). Layer 0 of this family is dense; layers 1&ndash;9 are MoE.</p>"
    )
    rq = "<ol>" + "".join(f"<li><b>{name}</b> &mdash; {q}</li>" for _, name, q in QUESTIONS) + "</ol>"
    return card("info", "Setup", setup) + card("info", "Research questions (one tab each)", rq)


def question_card(qid):
    name, q = next((n, q) for i, n, q in QUESTIONS if i == qid)
    return card("ok", "Question", f"<p>{q}</p>")


METRIC_SPECS = {"top32": ("top-32 share", "poolable_top32_unw", 2), "top64": ("top-64 share", "poolable_top64_unw", 2),
                "top128": ("top-128 share", "poolable_top128_unw", 2), "eff": ("eff. experts / doc", "doc_eff_experts_unw", 0),
                "ql": ("Louvain Q", "Q_louvain", 2), "qs": ("spectral Q (k=8)", "Q_spectral", 2), "ent": ("router entropy (nats)", "mean_router_entropy", 2)}


def unrestricted_table(res, models=("std", "emo"), metrics=tuple(METRIC_SPECS)):
    """One row per (model, metric), columns = layers, unrestricted pass only."""
    rows = []
    for model in models:
        r = res[model]["none"]
        for m in metrics:
            label, key, nd = METRIC_SPECS[m]
            rows.append([MODEL_LABEL[model], label] + [f(r["layers"][l][key], nd) for l in LAYERS])
    return table(["model", "metric", *[f"L{l}" for l in LAYERS]], rows)


def section(title, what, result, takeaway):
    """One digestible result block: metric definition -> table/figure -> one-paragraph conclusion."""
    return card("info", title, f'<p class="what">{what}</p>{result}<p><b>Takeaway.</b> {takeaway}</p>')


HEAT = OUT / "heatmaps"


def heatmap_card(tag, label):
    qf = HEAT / f"{tag}_heatmaps.json"
    qtxt = ""
    if qf.exists():
        q = json.load(open(qf)); qtxt = " &middot; spectral Q by layer: " + ", ".join(f"{v:.2f}" for v in q["Q_spectral"])
    h = lambda name, cap: img_tag(HEAT / name, cap)
    sub = (fig_row(h(f"{tag}_lift_tok_grid.png", "log2 lift, token level"), h(f"{tag}_cond_tok_grid.png", "conditional co-activation P(j|i), token level"))
           + fig_row(h(f"{tag}_lift_doc_grid.png", "log2 lift, document level"), h(f"{tag}_cond_doc_grid.png", "conditional co-activation, document level"))
           + fig_row(h(f"{tag}_usage.png", "per-expert token usage by layer"), h(f"{tag}_lift_hist.png", "distribution of pairwise lift")))
    return card("info", f"Heatmaps: {label}{qtxt}", sub)


def build_q1(res, findings):
    body = question_card("q1")
    body += card("ok", "Short answer",
                 "<p><b>EMO: yes, from layer 2 on. Standard MoE: no.</b> In the EMO model a document's tokens in layers 2&ndash;9 mostly stay "
                 "inside a small subset of experts (about 60% of selections fall in the document's top-64 of 512), while the standard model "
                 "spreads a document over almost every expert in every layer. Both models reach the same held-out loss (CE 2.463 vs 2.444), "
                 "and the two routers are equally sharp per token, so EMO's concentration is a per-document property, not a sharper router. "
                 "EMO's first MoE layer behaves like the standard model.</p>")
    body += section("Top-P share: could a document be served by a pool of P experts?",
        "For one document and one layer, rank the experts by the total router weight the document's tokens gave them, keep the top P, and "
        "count the fraction of the document's actual selections that landed in those P. 1.0 means a P-expert pool would reproduce the routing "
        "exactly; spreading a document across the whole layer gives a small value. P = 32 / 64 / 128 of 512; mean over documents with &ge; 64 tokens.",
        unrestricted_table(res, metrics=("top32", "top64", "top128")),
        "In the standard model only 17&ndash;28% of a document's selections fall in its top-64 experts, and even the top-128 catch under half. "
        "In EMO the top-64 catch 55&ndash;67% and the top-128 catch 72&ndash;81% from layer 2 onward, rising slowly with depth. EMO layer 1 "
        "(29% top-64) looks like the standard model.")
    body += section("Effective experts per document: how many experts does a document really use?",
        "The exponential of the entropy of the document's expert-usage histogram in that layer: heavily used experts count fully, rarely used "
        "ones only a little. 16 would mean the same 16 experts for every token; 512 means every expert used equally.",
        unrestricted_table(res, metrics=("eff",)),
        "A document in the standard model effectively uses 360&ndash;450 of the 512 experts in every layer, i.e. routing is essentially "
        "token-level. In EMO it uses 140&ndash;210 from layer 2 on, falling to ~140 by layer 9, again with layer 1 (~385) as the exception.")
    body += section("Modularity Q and router entropy: are there expert groups, and is the router sharper?",
        "<b>Q</b>: experts are nodes, and the edge between two experts is how much more often they are selected on the same token than chance "
        "predicts (lift). Q measures how cleanly the graph splits into groups that co-fire within but rarely across, found by Louvain (free "
        "number of groups) or spectral clustering into k = 8; 0 means no structure beyond chance (a shuffled-label null sits at 0.00). "
        "<b>Router entropy</b>: how spread out the router's softmax is per token, averaged over tokens; ln 512 = 6.24 is uniform.",
        unrestricted_table(res, metrics=("ql", "qs", "ent")),
        "Token-level group structure is about the same in both models (Louvain Q 0.23&ndash;0.40 standard vs 0.28&ndash;0.38 EMO, both growing with "
        "depth), and the routers are equally spread out per token (entropy ~5.9&ndash;6.1 nats in both). So EMO does not create sharper "
        "per-token expert groups; its document-level concentration comes from the pool rule, which keeps a document within a subset of "
        "experts even though individual tokens still pick from a broad, near-uniform distribution.")
    body += card("info", "Co-activation heatmaps: which experts are used together?",
        "<p>Expert &times; expert grids per layer. <b>Lift</b>: how much more often two experts are used together than chance, in log2 "
        "(+1 = twice as often, 0 = independent, clipped to &plusmn;3), with experts ordered so that groups appear as blocks on the diagonal. "
        "<b>Conditional co-activation</b>: given that expert i is used, the probability that expert j is too. Both come at <em>token</em> level "
        "(the two experts are in one token's top-16) and <em>document</em> level (both are used somewhere in the same document).</p>")
    body += heatmap_card("std512_full", "standard MoE, 512 experts, full routing") + heatmap_card("emo512_full", "EMO, 512 experts, full routing")
    body += card("info", "Takeaway from the heatmaps",
        "<p>At token level both models show clear red blocks on the diagonal in every layer, of similar size and strength: which experts fire "
        "together on a token is organised the same way in both. The difference is at document level. The standard model's document-level "
        "grids are blank, because a document touches nearly every expert, so no pair is used together more or less often than chance. In "
        "EMO faint blocks appear from layer 2 and sharpen with depth (layer 9 clearest): sets of experts that are used by the same documents "
        "and, in the off-diagonal blue, sets that are rarely used by the same document.</p>")
    return body


def build_q2(res, res1000, findings):
    method = (
        "<p><b>Restriction</b>: <em>per document</em>, sum the softmax router scores over the document's tokens, keep the top-P experts, and "
        "route top-16 within them (the EMO pool rule; EMO checkpoints do it natively, the standard checkpoint gets a router wrapper self-tested "
        "to reproduce EmoRouterV2 bit-for-bit). A condition <code>A-B:P</code> applies pool P to MoE layers A..B and leaves the other layers "
        "unrestricted; P &isin; {32, 64, 128, 256}, prefixes 1&ndash;3 / 1&ndash;6 / 1&ndash;9 (the last one is the all-layer reference). A per-batch "
        "assertion checks that restricted layers never use more than P experts in a document (this caught a config-aliasing bug in the first "
        "EMO pilot). Metrics are those of Q1, read off in the <em>unrestricted</em> layers; restricted layers read 1.000 top-P share by "
        "construction. CE is the pass's mean token cross-entropy (also split by data source).</p>"
    )
    late = ["7", "8", "9"]
    def row(model, cond):
        x = res[model][cond]; ly = x["layers"]
        j = lambda k, nd: " / ".join(f"{ly[l][k]:.{nd}f}" for l in late)
        return [MODEL_LABEL[model], cond, f(x["meta"]["mean_ce"]), j("poolable_top32_unw", 2), j("poolable_top64_unw", 2),
                j("poolable_top128_unw", 2), j("doc_eff_experts_unw", 0), j("Q_louvain", 2)]
    hdr = ["model", "restriction", "CE", "top-32 share L7 / L8 / L9", "top-64 share", "top-128 share", "eff. experts / doc", "Louvain Q"]
    body = question_card("q2") + card("info", "Method", method) + findings[1] + findings[2] + findings[3]
    body += "<p>The direct test: layers 7&ndash;9 (unrestricted in every row except the 1-9 reference rows) under no restriction vs layers 1&ndash;3 vs 1&ndash;6 restricted, per pool size.</p>"
    for model in ("emo", "std"):
        rows = []
        for P in (32, 64, 128):
            for cond in (["none"] if P == 32 else []) + [f"1-3:{P}", f"1-6:{P}", f"1-9:{P}"]:
                if cond in res[model]: rows.append(row(model, cond))
        body += card("info", f"{MODEL_LABEL[model]}: layers 7&ndash;9 under early-layer restriction", table(hdr, rows))
    body += card("info", "Cross-entropy of every pass", ce_table(res, res1000))
    body += "<p>All layers, all conditions (rows = conditions, columns = MoE layers):</p>"
    body += (card("info", "Share of routed assignments inside the document's top-64 experts (unweighted over docs)", layer_table(res, "poolable_top64_unw"))
             + card("info", "Same, top-128", layer_table(res, "poolable_top128_unw"))
             + card("info", "Effective # experts per document", layer_table(res, "doc_eff_experts_unw", nd=1))
             + fig_row(fig("poolable64_by_layer.png", "Top-64 poolability by layer; colour = pool size, line style = restricted prefix (solid 1-3, dashed 1-6, dotted 1-9)."),
                       fig("doc_eff_experts_by_layer.png", "Effective number of experts per document by layer."))
             + card("info", "Louvain modularity Q of the token co-activation graph", layer_table(res, "Q_louvain"))
             + card("info", "Spectral Q (k=8)", layer_table(res, "Q_spectral"))
             + fig_row(fig("q_louvain_by_layer.png", "Louvain Q by layer and condition."), fig("router_entropy_by_layer.png", "Mean router entropy by layer (nats; ln 512 = 6.24)."), fig("ce_by_condition.png", "CE by condition.")))
    body += heatmap_card("std512_pool64", "standard MoE, 512 experts, all 9 layers at pool 64") + heatmap_card("emo512_pool64", "EMO, 512 experts, all 9 layers at pool 64")
    return body


def build_q3(res):
    method = (
        "<ul><li><b>early-pool conditioning</b>: documents are k-means clustered (k=8) by their layer-1 top-32 expert SET; for every layer we "
        "report the normalized MI between cluster id and expert usage, and the mean Jaccard of documents' top-64 expert sets for pairs within vs "
        "across clusters. A positive within-minus-across gap in a <em>later, unrestricted</em> layer means that documents whose early pools agree "
        "also share later experts.</li>"
        "<li><b>cross-layer NMI</b>: normalized mutual information between the expert chosen at layer l and at layer m, from the E&times;E "
        "cross-layer assignment counts.</li></ul>"
    )
    result = ("<p><b>Result.</b> No early-pool&rarr;late-expert dependence appears in either model: the within-minus-across Jaccard gap in the "
              "free layers stays at +0.02&ndash;0.03 and NMI(cluster ; expert) &le; 0.01 under every restriction. Cross-layer expert dependence is "
              "higher in EMO than in the standard model (late&ndash;late NMI 0.085 vs 0.055) and grows under restriction (0.15 at all layers @ 32), "
              "but this is the restricted layers agreeing with each other, not the free layers following them.</p>")
    body = question_card("q3") + card("info", "Method & metrics", method) + card("ok", "Finding", result)
    body += (card("info", "NMI(early-pool cluster ; expert usage)", layer_table(res, None, sub="earlypool", subkey="nmi_cluster_expert"))
             + card("info", "Jaccard within clusters", layer_table(res, None, nd=2, sub="earlypool", subkey="jaccard_within"))
             + card("info", "Jaccard across clusters", layer_table(res, None, nd=2, sub="earlypool", subkey="jaccard_across"))
             + fig_row(fig("earlypool_jaccard_gap.png", "Within-minus-across Jaccard of top-64 expert sets, by layer."), fig("earlypool_nmi.png", "NMI(early-pool cluster ; expert) by layer."))
             + card("info", "Cross-layer expert NMI", "<p>Rows = layer l, columns = layer m, unrestricted vs the tightest restrictions (diagonal blanked).</p>" + fig("cross_layer_nmi.png", "Cross-layer expert NMI heatmaps.")))
    return body


def build_q4(res1000, res, findings):
    if not res1000:
        return question_card("q4") + card("warn", "128- and 1000-expert arms", "<p>No analysis found in runs_1000/.</p>")
    method = ("<p>Same pass and metrics as Q1 (unrestricted routing only) on the 128- and 1000-expert arms; top-P share is at a fixed P, so it "
              "dilutes mechanically as the expert count grows &mdash; compare the ordering between routings and the effective expert count.</p>")
    rows, rows2 = [], []
    for model, conds in list(res.items()) + list(res1000.items()):
        r = conds["none"]
        rows.append([MODEL_LABEL[model], f(r["meta"]["mean_ce"])] + [f(r["layers"][l]["poolable_top64_unw"]) for l in LAYERS])
        rows2.append([MODEL_LABEL[model]] + [f(r["layers"][l]["Q_louvain"]) for l in LAYERS])
    body = question_card("q4") + card("info", "Method", method) + findings[4]
    body += (card("info", "Unrestricted pass: held-out CE and top-64 poolability by layer, all trained arms", table(["model", "CE", *[f"L{l}" for l in LAYERS]], rows))
             + card("info", "Louvain Q by layer, unrestricted", table(["model", *[f"L{l}" for l in LAYERS]], rows2))
             + card("info", "1000-expert arms: per-layer routing statistics", unrestricted_table(res1000, ("std1000", "emo1000")))
             + card("info", "128-expert arms: per-layer routing statistics", unrestricted_table(res1000, ("std128", "emo128")))
             + fig_row(fig("poolable64_by_layer.png", "128/1000-expert arms: top-64 poolability (unrestricted only).", RUNS1000), fig("q_louvain_by_layer.png", "128/1000-expert arms: Louvain Q.", RUNS1000)))
    body += heatmap_card("std1000_full", "standard MoE, 1000 experts, full routing") + heatmap_card("emo1000_full", "EMO, 1000 experts, full routing")
    return body


KS = OUT / "ksweep"


def build_q5():
    method = ("<p>The spectral cluster count k = 8 used elsewhere was inherited from the sparse_experts analysis, not chosen for this data; it only "
              "affects the display ordering and the spectral-Q number (the lift matrix is k-free). Here k is swept over 4 / 8 / 16 / 32 / 64 on the "
              "EMO 1000-expert full-routing pass. For each k the experts of ONE layer are partitioned into k spectral clusters and every document "
              "is assigned to the cluster that captures most of its routed assignments in that layer. <b>purity</b> = the share of the document's "
              "assignments in its assigned cluster (mean over documents with &ge; 64 tokens), against a random expert partition with the same "
              "cluster sizes; <b>lift within / across</b> = mean log2 lift among the experts the document actually uses (top-16 by usage) inside "
              "vs outside its cluster; NMI(cluster ; source) = how much the document clusters line up with the data source.</p>")
    result = ("<p><b>Result.</b> Block structure exists at every k (Q stays well above the null), but a <em>layer-1</em> expert partition does not "
              "partition documents at any k: purity matches the random-partition null. Later layers do: e.g. at layer 9, k = 4 gives purity 0.55 "
              "vs 0.32 null. So document-level partitions live in the later layers, consistent with Q1.</p>")
    body = question_card("q5") + card("info", "Method", method) + card("ok", "Finding", result)
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
        ("q1", QUESTIONS[0][1], build_q1(res, findings)),
        ("q2", QUESTIONS[1][1], build_q2(res, res1000, findings)),
        ("q3", QUESTIONS[2][1], build_q3(res)),
        ("q4", QUESTIONS[3][1], build_q4(res1000, res, findings)),
        ("q5", QUESTIONS[4][1], build_q5()),
        ("next", "Next steps", build_next()),
    ]
    nav = "".join(f'<button data-target="{tid}">{name}</button>' for tid, name, _ in tabs)
    sections = "".join(f'<section class="tab" id="{tid}">{body}</section>' for tid, _, body in tabs)
    import datetime; stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = "OLMoE3-ladder 275M: document-level routing structure, standard MoE vs EMO"
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style><style>.muted{{color:var(--muted);font-weight:normal;font-size:0.85em}} .what{{color:#475569;font-size:0.93em;border-left:3px solid #cbd5e1;padding-left:10px;margin:4px 0 10px}}</style></head>
<body><header><a class="home-link" href="/">&larr; all reports</a><h1>{html.escape(title)}</h1>
<p>olmoe3_routing &mdash; router statistics of the 275M-active OLMoE3-ladder models trained here (standard MoE vs EMO; 128 / 512 / 1000 experts)
on 65.5M unseen training-stream tokens, unrestricted and with early layers forced to per-document expert pools &middot;
generated by scripts/olmoe3_routing/build_report.py on {stamp}</p></header>
<div class="topbar"><nav>{nav}</nav><div id="subnav"></div></div><main>{sections}</main><script>{JS}</script></body></html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.html").write_text(page)
    print(f"Wrote {OUT/'report.html'} ({(OUT/'report.html').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
