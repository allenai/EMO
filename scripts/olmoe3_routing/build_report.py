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
    sub = (fig_row(h(f"{tag}_lift_tok_grid.png", "log2 lift"), h(f"{tag}_cond_tok_grid.png", "conditional co-activation P(j|i)"))
           + fig_row(h(f"{tag}_lift_hist.png", "distribution of pairwise lift")))
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
        "<b>Conditional co-activation</b>: given that expert i is used on a token, the probability that expert j is in the same token's top-16 "
        "(row = conditioning expert; not symmetric, so a dark column is simply a rarely used expert). Two experts count as co-activated when "
        "they are both in one token's top-16.</p>")
    body += heatmap_card("std512_full", "standard MoE, 512 experts, full routing") + heatmap_card("emo512_full", "EMO, 512 experts, full routing")
    body += card("info", "Takeaway from the heatmaps",
        "<p>Both models show clear red blocks on the diagonal in every layer, of similar size and strength, and the lift histograms have the "
        "same shape: which experts fire together on a token is organised the same way in both models. EMO's off-block region is somewhat "
        "bluer (pairs from different groups co-fire less than chance), but the group structure itself is not stronger, matching the Q values "
        "above. Nothing in these token-level grids reveals EMO's per-document concentration, which is a statement about which experts a "
        "whole document uses, not about which experts a single token uses together.</p>")
    return body


RESTRICTION_CARD = card("info", "How the restriction works",
    "<p>Per document, sum the router's softmax scores over the document's tokens, keep the top-P experts, and route each token top-16 "
    "within them (EMO's own pool rule; the standard checkpoint gets a router wrapper that reproduces it exactly). A condition "
    "<code>A-B:P</code> applies pool P to MoE layers A&ndash;B and leaves the others free: prefixes 1&ndash;3 and 1&ndash;6 are the "
    "experiment, 1&ndash;9 (every layer pooled) is the reference. P = 32 / 64 / 128 / 256 of 512. Metrics are those of Q1, read in the "
    "<em>free</em> layers; a pooled layer reads 1.00 top-P share by construction.</p>")


def compact_ce_table(res):
    conds = sorted(res["std"], key=cond_sort)
    rows = []
    for c in conds:
        cells = [c]
        for m in ("std", "emo"):
            ce = res[m][c]["meta"]["mean_ce"]; base = res[m]["none"]["meta"]["mean_ce"]
            cells += [f(ce), f"{ce-base:+.3f}"]
        rows.append(cells)
    return table(["restriction", "standard CE", "&Delta;", "EMO CE", "&Delta;"], rows)


def build_q2(res, res1000, findings):
    body = question_card("q2")
    body += card("ok", "Short answer",
                 "<p><b>Only slightly, and at a large cost, in the standard model; EMO is already there and pays almost nothing.</b> Pooling "
                 "the standard model's layers 1&ndash;6 to 32 experts per document lifts layers 7&ndash;9 from 22&ndash;28% to 33&ndash;39% top-64 share, "
                 "far short of EMO's 62&ndash;67%, leaves their expert-group structure unchanged, and raises CE from 2.44 to 3.59. The same "
                 "restriction on EMO costs 0.06 CE and changes the free layers by a few points. Later-layer document structure is learned "
                 "during training, not induced by restricting what comes before.</p>")
    body += RESTRICTION_CARD
    late = ["7", "8", "9"]
    def row(model, cond):
        x = res[model][cond]; ly = x["layers"]
        j = lambda k, nd: " / ".join(f"{ly[l][k]:.{nd}f}" for l in late)
        return [cond, f(x["meta"]["mean_ce"]), j("poolable_top32_unw", 2), j("poolable_top64_unw", 2), j("doc_eff_experts_unw", 0), j("Q_louvain", 2)]
    hdr = ["restriction", "CE", "top-32 share L7 / L8 / L9", "top-64 share L7 / L8 / L9", "eff. experts / doc L7 / L8 / L9", "Louvain Q L7 / L8 / L9"]
    tables = ""
    for model in ("std", "emo"):
        rows = []
        for P in (32, 64, 128):
            for cond in (["none"] if P == 32 else []) + [f"1-3:{P}", f"1-6:{P}", f"1-9:{P}"]:
                if cond in res[model]: rows.append(row(model, cond))
        tables += f"<p><b>{MODEL_LABEL[model]}</b></p>" + table(hdr, rows)
    body += section("Layers 7&ndash;9 when layers 1&ndash;3 or 1&ndash;6 are pooled",
        "The direct test. Layers 7&ndash;9 route freely in every row except the 1&ndash;9 reference rows; each cell lists the three layers. "
        "Top-P share and effective experts per document are defined in Q1.",
        tables,
        "Standard model: pooling layers 1&ndash;3 at 32 moves the top-64 share of layers 7&ndash;9 by +0.02&ndash;0.03; pooling 1&ndash;6 by "
        "+0.11 and cuts effective experts from ~400 to ~320. The shift grows with tighter pools and deeper prefixes but never approaches "
        "EMO (0.62&ndash;0.67), and Louvain Q in those layers does not move (0.38&ndash;0.40 in every row). EMO: the free layers gain "
        "+0.02&ndash;0.04 and lose ~15 effective experts, i.e. they were already pooled and barely notice.")
    body += section("What the restriction costs",
        "Mean token cross-entropy of the whole pass under each condition, and its change from the unrestricted pass.",
        compact_ce_table(res),
        "The standard model pays 0.43 CE for pooling layers 1&ndash;3 at 32, 1.15 for 1&ndash;6 and 1.50 for all nine; even pool 256 in "
        "three layers costs 0.05. EMO pays 0.003 / 0.06 / 0.11 for the same three, and pool 64 in layers 1&ndash;3 is free. The cost scales "
        "with every data source alike (standard 1&ndash;3 at 32: code 1.45&rarr;1.70, web 2.95&rarr;3.48). Restricting a router trained "
        "without pools breaks it; restricting one trained with pools does not.")
    body += section("Every layer under every condition",
        "Rows are conditions, columns MoE layers. Pooled layers read 1.00 / P by construction; read the free columns to the right of "
        "each pooled prefix.",
        "<p><b>Top-64 share</b></p>" + layer_table(res, "poolable_top64_unw")
        + "<p><b>Effective experts per document</b></p>" + layer_table(res, "doc_eff_experts_unw", nd=1)
        + fig_row(fig("poolable64_by_layer.png", "Top-64 share by layer; colour = pool size, line style = pooled prefix (solid 1-3, dashed 1-6, dotted 1-9)."),
                  fig("doc_eff_experts_by_layer.png", "Effective experts per document by layer.")),
        "The same picture layer by layer: in the standard model the free layers stay at 0.2&ndash;0.4 top-64 share whatever happens "
        "upstream, with a small monotone lift; in EMO they sit at 0.55&ndash;0.71 in every condition. A pooled layer's own numbers are "
        "mechanical (pool 32 &rArr; 31 effective experts).")
    body += section("Do the pools line up with the experts' natural groups?",
        "Louvain Q of the token co-activation graph (Q1) inside the <em>pooled</em> layers. If a document's top-P experts by router score "
        "form a coherent group, pooling makes co-firing blockier and Q rises; if the top-P slice across the natural groups, Q falls.",
        layer_table(res, "Q_louvain") + fig_row(fig("q_louvain_by_layer.png", "Louvain Q by layer and condition."), fig("router_entropy_by_layer.png", "Mean router entropy by layer.")),
        "With the grain in EMO, against it in the standard model. EMO's pooled layers become <em>more</em> modular (layer 1: 0.28 &rarr; 0.40 "
        "at pool 32; layers 4&ndash;6: 0.37&ndash;0.38 &rarr; 0.43&ndash;0.44), the standard model's less (layer 1: 0.23 &rarr; 0.10; layers 4&ndash;6: "
        "0.32&ndash;0.39 &rarr; 0.17&ndash;0.24). A standard document's top-P experts are a slice through several co-firing groups, so forcing "
        "it onto them destroys the group structure; an EMO document's top-P already is a group.")
    body += card("info", "Co-activation heatmaps with every layer pooled at 64",
                 "<p>Token-level lift and conditional co-activation as in Q1, for the 1&ndash;9:64 condition of both 512-expert models.</p>")
    body += heatmap_card("std512_pool64", "standard MoE, 512 experts, all 9 layers at pool 64") + heatmap_card("emo512_pool64", "EMO, 512 experts, all 9 layers at pool 64")
    body += card("info", "Takeaway from the heatmaps",
                 "<p>The same result in pictures. EMO's blocks under pooling look like its unrestricted blocks (spectral Q 0.34&ndash;0.40 vs "
                 "0.27&ndash;0.33). The standard model's blocks dissolve into a near-uniform grid (spectral Q 0.09&ndash;0.22 vs 0.22&ndash;0.36): "
                 "the pools mix experts from different groups and split experts that used to fire together.</p>")
    return body


def gap_table(res, conds):
    rows = []
    for model in ("std", "emo"):
        for cond in sorted(conds, key=cond_sort):
            ep = res[model][cond]["earlypool"]["layers"]
            rows.append([MODEL_LABEL[model], cond] + [f"{ep[l]['jaccard_within'] - ep[l]['jaccard_across']:+.2f}" for l in LAYERS])
    return table(["model", "restriction", *[f"L{l}" for l in LAYERS]], rows)


def cross_table(res, conds):
    import numpy as np
    rows = []
    for model in ("std", "emo"):
        for cond in sorted(conds, key=cond_sort):
            X = np.array(res[model][cond]["cross_nmi"]); off = lambda A: A[~np.eye(A.shape[0], dtype=bool)].mean()
            rows.append([MODEL_LABEL[model], cond, f(np.mean([X[i, i + 1] for i in range(8)])), f(off(X[0:3, 0:3])), f(off(X[6:9, 6:9])), f(X[0:3, 6:9].mean()), f(X[0, 8])])
    return table(["model", "restriction", "adjacent layers", "within L1&ndash;3", "within L7&ndash;9", "L1&ndash;3 vs L7&ndash;9", "L1 vs L9"], rows)


def build_q3(res):
    conds = ["none", "1-3:32", "1-6:32", "1-9:32", "1-9:256"]
    body = question_card("q3")
    body += card("ok", "Short answer",
                 "<p><b>Not in the standard model, and only weakly in EMO, where it is present with or without restriction.</b> Documents "
                 "that share a layer-1 pool share later experts no more than random documents in the standard model (Jaccard gap +0.02, "
                 "unchanged by any restriction). In EMO the gap is +0.10&ndash;0.15 in every later layer, but it is the same before and after "
                 "pooling the early layers, so it reflects documents that look alike routing alike at every depth, not later layers "
                 "following the early pool.</p>")
    body += section("Early-pool conditioning: do documents with the same early pool share later experts?",
        "Cluster documents by their layer-1 top-32 expert set (k-means, k = 8). For each later layer, compare how similar two documents' "
        "top-64 expert sets are (Jaccard) when they are in the same early cluster vs in different ones. The table shows the difference, "
        "within minus across: 0 means the early pool tells you nothing about later routing. Shown for pool 32 at each prefix and pool 256 "
        "for every layer.",
        gap_table(res, conds) + fig_row(fig("earlypool_jaccard_gap.png", "Within-minus-across Jaccard of top-64 expert sets, by layer, all conditions."), fig("earlypool_nmi.png", "Normalized mutual information between early-pool cluster and expert usage, by layer.")),
        "Standard model: +0.01&ndash;0.03 in every free layer under every condition, i.e. none. EMO: +0.10&ndash;0.15 in layers 2&ndash;9, already "
        "in the unrestricted pass, and pooling layers 1&ndash;3 or 1&ndash;6 leaves the free layers' gap where it was. Within a pooled prefix "
        "the gap drops (+0.07: every document is on 32 experts, so sets are small either way). Nothing here is created by restriction.")
    body += section("Cross-layer dependence: does the expert picked at one layer predict the expert picked at another?",
        "Normalized mutual information between the expert a token selects at layer l and at layer m, from the expert &times; expert "
        "cross-layer counts. 0 = independent choices, 1 = one determines the other. Averaged over adjacent pairs, within the early block, "
        "within the late block, and between the two.",
        cross_table(res, conds) + fig("cross_layer_nmi.png", "Cross-layer expert NMI, all layer pairs (diagonal blanked), unrestricted vs the tightest restrictions."),
        "Dependence is weak everywhere (&le; 0.15) and higher in EMO than in the standard model (adjacent layers 0.085 vs 0.058). Under "
        "pooling the two models move in opposite directions: EMO's early&ndash;late NMI rises (0.055 &rarr; 0.076&ndash;0.136) because "
        "pooled layers now agree with each other on a per-document set, while the standard model's falls (0.034 &rarr; 0.004) because "
        "its pooled early layers are forced onto experts that carry no information about what the free layers will choose.")
    return body


def arms_table(res, res1000, key, nd=2):
    rows = []
    for model, conds in list(res.items()) + list(res1000.items()):
        r = conds["none"]
        rows.append([MODEL_LABEL[model], f(r["meta"]["mean_ce"])] + [f(r["layers"][l][key], nd) for l in LAYERS])
    return table(["model", "CE", *[f"L{l}" for l in LAYERS]], rows)


def build_q4(res1000, res, findings):
    if not res1000:
        return question_card("q4") + card("warn", "128- and 1000-expert arms", "<p>No analysis found in runs_1000/.</p>")
    body = question_card("q4")
    body += card("ok", "Short answer",
                 "<p><b>It does not.</b> At every expert count the standard model spreads a document over most of the layer and EMO keeps "
                 "it on roughly a fifth to a quarter of the experts from layer 2 on, at a cost of about 0.02 CE. More experts help both "
                 "routings equally (CE 2.51 / 2.44 / 2.42 standard, 2.53 / 2.46 / 2.44 EMO at 128 / 512 / 1000), and expert-group "
                 "structure (Q) is the same in all six models.</p>")
    body += section("Held-out CE and top-64 share, all six arms",
        "Unrestricted pass. Top-64 share (Q1) is at a fixed P, so it falls mechanically as the expert count grows: 64 is half of 128 "
        "experts but 6% of 1000. Compare the two routings at the same count, not across counts.",
        arms_table(res, res1000, "poolable_top64_unw") + fig_row(fig("poolable64_by_layer.png", "128- and 1000-expert arms: top-64 share by layer.", RUNS1000)),
        "EMO above standard at every count and layer from 2 on: 0.88&ndash;0.92 vs 0.58&ndash;0.67 at 128, 0.55&ndash;0.67 vs 0.17&ndash;0.28 at 512, "
        "0.38&ndash;0.57 vs 0.09&ndash;0.17 at 1000. EMO's layer-1 exception holds at every count. Going 512 &rarr; 1000 buys 0.02&ndash;0.03 CE "
        "for both routings.")
    body += section("Effective experts per document, all six arms",
        "Exp-entropy of the document's usage histogram (Q1); the natural scale-free comparison across expert counts.",
        arms_table(res, res1000, "doc_eff_experts_unw", nd=0),
        "Standard: 110&ndash;124 of 128, 360&ndash;450 of 512, 600&ndash;810 of 1000, i.e. 60&ndash;95% of the layer at any size. EMO from layer 2: "
        "60&ndash;72 of 128, 140&ndash;210 of 512, 210&ndash;400 of 1000, i.e. 20&ndash;50%, and the share shrinks with depth at every count.")
    body += section("Expert-group structure, all six arms",
        "Louvain Q of the token co-activation graph (Q1).",
        arms_table(res, res1000, "Q_louvain") + fig_row(fig("q_louvain_by_layer.png", "128- and 1000-expert arms: Louvain Q by layer.", RUNS1000)),
        "Q rises with depth from ~0.2&ndash;0.3 to ~0.4 in every model and does not separate the routings or the expert counts. As in Q1, "
        "EMO's document-level concentration is not visible as stronger token-level groups.")
    body += card("info", "Co-activation heatmaps, 1000 experts", "<p>Token-level lift and conditional co-activation as in Q1.</p>")
    body += heatmap_card("std1000_full", "standard MoE, 1000 experts, full routing") + heatmap_card("emo1000_full", "EMO, 1000 experts, full routing")
    body += card("info", "Takeaway from the heatmaps",
                 "<p>Same blocks, same strength as at 512 experts (spectral Q 0.18&ndash;0.38 standard, 0.23&ndash;0.35 EMO), just twice as many "
                 "experts per block. Nothing new appears at 1000 experts.</p>")
    return body


KS = OUT / "ksweep"


def ksweep_table(tag, layer):
    jf = KS / f"{tag}_L{layer}_ksweep.json"
    if not jf.exists(): return "<p class=\"muted\">not run</p>"
    r = json.load(open(jf)); rows = []
    for k, v in sorted(r.items(), key=lambda kv: int(kv[0])):
        rows.append([k, f(v["purity"], 2), f(v["purity_null"], 2), f"{v['frac_purity_gt_half']:.2f}", f(v["lift_within"], 2), f(v["lift_across"], 2)])
    return table(["k", "purity", "purity, random partition", "docs with purity &gt; 0.5", "lift within", "lift across"], rows)


def build_q5():
    body = question_card("q5")
    body += card("ok", "Short answer",
                 "<p><b>Yes in EMO's later layers, no in EMO's layer 1, and never in the standard model; k does not change the answer.</b> "
                 "Experts form co-firing blocks at every k from 4 to 64 in every model. Splitting a layer's experts into k blocks splits the "
                 "documents only for EMO at layers 5 and 9: there the average document puts about half of its routing in one block at k = 4 "
                 "(random partition: a third), with identical numbers at 512 and 1000 experts. For EMO layer 1 and for every layer of the "
                 "standard 1000-expert model, a document's routing is spread over the blocks exactly as under a random partition.</p>")
    body += section("Does k matter for the expert clustering?",
        "The k = 8 used elsewhere was inherited from the sparse_experts analysis. Here the spectral clustering of the token lift graph is "
        "re-run with k = 4 / 8 / 16 / 32 / 64 on the EMO 1000-expert unrestricted pass, and Q is reported per layer, against a "
        "shuffled-label null.",
        fig("emo1000_full_Q_vs_k.png", "Spectral Q vs k per layer (EMO 1000, full routing); null shown for layer 1.", KS)
        + "<p><b>EMO 1000, lift grids ordered by the k clusters</b></p>" + "".join(fig(f"emo1000_full_k{k}_lift_tok_grid.png", f"k = {k}", KS) for k in (4, 8, 16, 32, 64)),
        "Q is far above the null at every k and declines slowly as k grows, as expected when a fixed amount of block structure is cut "
        "into more pieces. The grids show the same diagonal blocks subdivided: k changes the granularity, not the picture.")
    body += section("Does an expert partition partition the documents?",
        "For one layer, split its experts into k spectral clusters and assign each document to the cluster that receives most of its routing "
        "in that layer. <b>Purity</b> = the share of the document's routing that lands in its own cluster, compared with a random expert "
        "partition of the same sizes. <b>Lift within / across</b> = mean log2 lift among the experts the document actually uses, inside vs "
        "outside its cluster. Shown for three models at layers 1, 5 and 9.",
        "".join(f"<p><b>{MODEL_LABEL[m]}, layer {layer}</b></p>" + ksweep_table(tag, layer)
                + fig_row(*[img_tag(KS / f"{tag}_L{layer}_k{k}_docpartition.png", f"k = {k}: document purity vs a random partition") for k in (4, 8, 32)])
                for tag, m in (("emo512_full", "emo"), ("std1000_full", "std1000"), ("emo1000_full", "emo1000")) for layer in (1, 5, 9)),
        "Standard 1000, every layer: purity equals the random-partition value at every k (layer 9: 0.32 vs 0.31 at k = 4), no document "
        "puts more than half its routing in one block, and experts from different blocks anti-correlate (lift across &lt; 0). Its blocks "
        "are groups of experts that fire on the same <em>tokens</em>, and every document contains tokens of every kind. EMO layer 1 looks "
        "the same. EMO layers 5 and 9: purity is well above random at every k (layer 9: 0.54&ndash;0.55 / 0.41&ndash;0.43 / 0.27&ndash;0.30 vs "
        "0.32 / 0.18 / 0.12 at k = 4 / 8 / 16, for 512 and 1000 experts alike), about 60% of documents put more than half their routing in "
        "one block at k = 4, and experts from different blocks still co-fire (lift across &gt; 0), i.e. the blocks are document-level groups "
        "that tokens cross freely. A document-level expert partition exists only where EMO's per-document pooling acts (layers 2&ndash;9), "
        "not as a consequence of depth alone.")
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
