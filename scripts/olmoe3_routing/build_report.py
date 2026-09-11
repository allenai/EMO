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
MODEL_LABEL = {"std": "standard MoE (512e)", "emo": "EMO (512e)", "std1000": "standard MoE (1000e)", "emo1000": "EMO (1000e)", "std128": "standard MoE (128e)", "emo128": "EMO (128e)", "emo2000": "EMO (2000e)", "std2000": "standard MoE (2000e)"}


def cond_sort(c):
    return (c != "none", c.split(":")[0], int(c.split(":")[1]) if ":" in c else 0)


def fig(name, caption, d=RUNS):
    p = d / "figs" / name
    return img_tag(p if p.exists() else d / name, caption)


def f(x, nd=3):
    return "&mdash;" if x is None else f"{x:.{nd}f}"


QUESTIONS = [
    ("q1", "Q1 · Joint expert / document partition",
     "Can one layer's experts be split into k blocks such that documents split the same way, and does the choice of k matter?"),
    ("q2", "Q2 · Block agreement across layers",
     "Do documents that sit cleanly in one layer-8 expert block also sit together in a single layer-9 block, i.e. are the document-level blocks the same groups of documents from layer to layer?"),
    ("q3", "Q3 · Train the squares separately, then merge",
     "Can EMO 512e be split into four block-group sub-models (k = 4, layers 2&ndash;9), each trained on its own documents for the next 10B tokens, and merged back into a model that matches simply continuing the full model?"),
]


def build_overview():
    setup = (
        "<p>Two 512-expert MoE models at <b>276.7M active / 2.61B total</b> parameters, trained here for 10B tokens with a WSD schedule "
        "(constant LR after warmup, no decay). The architecture follows the 275M rung of the scaling-ladders repo unchanged. "
        "We also trained <b>1000-</b> and <b>2000-expert</b> versions by raising only the expert count with everything else fixed, so the total size grows "
        "to 4.90B (279.5M active) and 9.61B (285.2M active), and a <b>128-expert</b> version the same way (0.80B total, 274.5M active). "
        "For each expert count we trained an <b>EMO</b> version and a <b>standard MoE</b> version.</p>"
        "<p>All routing statistics in this report come from forward passes of these checkpoints on 65.5M unseen training-stream tokens "
        "(8,000 instances from the 10B&ndash;20B token window of the same data order). Layer 0 of this family is dense; layers 1&ndash;9 are MoE.</p>"
    )
    rq = "<ol>" + "".join(f"<li><b>{name}</b> &mdash; {q}</li>" for _, name, q in QUESTIONS) + "</ol>"
    return card("info", "Setup", setup) + card("info", "Research questions (one tab each)", rq)


def question_card(qid):
    name, q = next((n, q) for i, n, q in QUESTIONS if i == qid)
    return card("ok", "Question", f"<p>{q}</p>")


def section(title, what, result, takeaway):
    """One digestible result block: metric definition -> table/figure -> one-paragraph conclusion."""
    return card("info", title, f'<p class="what">{what}</p>{result}<p><b>Takeaway.</b> {takeaway}</p>')


KS = OUT / "ksweep"


KS_MODELS = (("emo2000_full", "emo2000"), ("emo1000_full", "emo1000"), ("emo512_full", "emo"), ("emo128_full", "emo128"), ("std2000_full", "std2000"), ("std1000_full", "std1000"))
KS_LAYERS = (1, 5, 9)


def purity_grid():
    """Document-purity histograms as a model x layer grid, one grid per k, with a k toggle."""
    css = ("<style>.pgrid table{border-collapse:separate;border-spacing:4px}.pgrid th{font-size:12px;font-weight:600}"
           ".pgrid td.h{font-size:12px;white-space:nowrap;text-align:right;padding-right:6px}.pgrid img{width:100%;max-width:420px;display:block}"
           ".pgrid .pg{display:none}.pgrid .pg.on{display:block}.pgrid .ctl{margin:8px 0;font-size:0.95em}</style>")
    html_ = [css, '<div class="pgrid"><div class="ctl"><b>k:</b> ' + " ".join(f'<label><input type="radio" name="pgk" value="{k}"{" checked" if k == 4 else ""}> {k}</label>' for k in (4, 8, 16, 32, 64)) + "</div>"]
    for k in (4, 8, 16, 32, 64):
        html_.append(f'<div class="pg{" on" if k == 4 else ""}" data-k="{k}"><table><tr><th></th>' + "".join(f"<th>layer {l}</th>" for l in KS_LAYERS) + "</tr>")
        for tag, m in KS_MODELS:
            html_.append(f'<tr><td class="h">{MODEL_LABEL[m]}</td>' + "".join(f"<td>{img_tag(KS / f'{tag}_L{l}_k{k}_docpartition.png', '')}</td>" for l in KS_LAYERS) + "</tr>")
        html_.append("</table></div>")
    html_.append("</div><script>document.querySelectorAll('.pgrid input[name=pgk]').forEach(r=>r.addEventListener('change',e=>{document.querySelectorAll('.pgrid .pg').forEach(d=>d.classList.toggle('on',d.dataset.k===e.target.value));}));</script>")
    return ("<p><b>Document purity, model &times; layer</b>: histogram of each document's purity under the layer's expert blocks (blue) vs a "
            "random partition of the same block sizes (orange); pick k with the buttons.</p>" + "".join(html_))


def build_q5():
    body = question_card("q1")
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
        purity_grid(),
        "Standard 1000, every layer: purity equals the random-partition value at every k (layer 9: 0.32 vs 0.31 at k = 4), no document "
        "puts more than half its routing in one block, and experts from different blocks anti-correlate (lift across &lt; 0). Its blocks "
        "are groups of experts that fire on the same <em>tokens</em>, and every document contains tokens of every kind. EMO layer 1 looks "
        "the same. EMO layers 5 and 9: purity is well above random at every k (layer 9: 0.54&ndash;0.55 / 0.41&ndash;0.43 / 0.27&ndash;0.30 vs "
        "0.32 / 0.18 / 0.12 at k = 4 / 8 / 16, for 512 and 1000 experts alike), about 60% of documents put more than half their routing in "
        "one block at k = 4, and experts from different blocks still co-fire (lift across &gt; 0), i.e. the blocks are document-level groups "
        "that tokens cross freely. A document-level expert partition exists only where EMO's per-document pooling acts (layers 2&ndash;9), "
        "not as a consequence of depth alone.")
    return body


LP = OUT / "layerpair"


def build_q6():
    body = question_card("q2")
    body += card("ok", "Short answer",
                 "<p><b>Yes.</b> In the EMO models, documents that route cleanly into one expert block at any layer from 2 onward land "
                 "together in a single layer-9 block: the heatmaps below are near-permutation matrices. Layer 1 does not, and the standard "
                 "model's agreement is much weaker. The document-level blocks are one partition of the documents that persists across the "
                 "later layers.</p>")
    body += interactive_grid()
    return body


GRID_MODELS = (("emo2000_full", "EMO (2000e)"), ("emo1000_full", "EMO (1000e)"), ("emo512_full", "EMO (512e)"), ("emo128_full", "EMO (128e)"), ("std2000_full", "standard MoE (2000e)"), ("std1000_full", "standard MoE (1000e)"))
GRID_JS = r"""
(function(){
  const Blues = v => { const t=Math.max(0,Math.min(1,v)); const r=Math.round(247-200*t), g=Math.round(251-170*t), b=Math.round(255-100*t); return `rgb(${r},${g},${b})`; };
  document.querySelectorAll('.lpgrid').forEach(root => {
    const data = JSON.parse(root.querySelector('script[type="application/json"]').textContent);
    const k = data.k, grid = data.n_grid, slider = root.querySelector('input[type=range]'), label = root.querySelector('.lp-label');
    const modeInputs = root.querySelectorAll('input[type=radio]');
    const cells = {};
    root.querySelectorAll('.lp-cell').forEach(c => { cells[c.dataset.model + '|' + c.dataset.layer] = c; });
    function render(){
      const ni = +slider.value, n = grid[ni]; const pct = [...modeInputs].find(r => r.checked).value === 'pct';
      label.textContent = (n >= 100000 ? 'all documents' : 'top ' + n.toLocaleString() + ' documents per layer-i block');
      for (const [tag, m] of Object.entries(data.models)) for (const [l, L] of Object.entries(m.layers)) {
        const cell = cells[tag + '|' + l]; if (!cell) continue;
        const T = L.tables[ni]; const tbl = cell.querySelector('table');
        for (let i=0;i<k;i++){ const row=T[i], tot=Math.max(row.reduce((a,b)=>a+b,0),1);
          for (let j=0;j<k;j++){ const td=tbl.rows[i].cells[j]; const f=row[j]/tot; td.style.background=Blues(f); td.style.color = f>0.6?'#fff':'#111';
            td.textContent = pct ? (f>=0.995 ? '100' : Math.round(100*f).toString()) : row[j].toLocaleString();
            td.title = `${row[j].toLocaleString()} docs (${(100*f).toFixed(1)}%)`; } }
        cell.querySelector('.lp-stat').textContent = `${L.stats[ni].n.toLocaleString()} docs`;
      }
    }
    slider.addEventListener('input', render); modeInputs.forEach(r => r.addEventListener('change', render)); render();
  });
})();
"""
GRID_CSS = """
.lpgrid{overflow-x:auto}.lpgrid .lp-controls{display:flex;gap:14px;align-items:center;margin:6px 0 10px;font-size:0.95em}
.lpgrid input[type=range]{width:360px}.lpgrid .lp-table{border-collapse:separate;border-spacing:6px}
.lpgrid .lp-cell{vertical-align:top;text-align:center;padding:0}.lpgrid .lp-cell table{border-collapse:collapse;margin:0 auto}
.lpgrid .lp-cell td{border:1px solid #fff;text-align:center;font-family:ui-monospace,Menlo,monospace;padding:0}
.lpgrid .lp-k4 td{width:34px;height:24px;font-size:10px}.lpgrid .lp-k8 td{width:19px;height:16px;font-size:8px}
.lpgrid .lp-stat{font-size:9.5px;color:#64748b;margin-top:3px}
.lpgrid th{font-weight:600;font-size:12px}.lpgrid .lp-rowhead{text-align:right;padding-right:8px;font-size:12px;white-space:nowrap}
"""


def interactive_grid():
    out = ("<style>" + GRID_CSS + "</style>"
           "<p>Each heatmap is the contingency table of <em>layer-i block</em> (rows: the block that receives most of a document's layer-i "
           "routing) against <em>layer-9 block</em> (columns) for the top-N purity documents per layer-i block; colour is the row share. "
           "Rows are ordered once so the diagonal is the best match; layer 9 &rarr; layer 9 is the identity check. The slider admits more "
           "documents per block (from the cleanest down to all); the toggle switches the cell values between document counts and row "
           "percentages.</p>")
    for k in (4, 8):
        models = {}
        for tag, _ in GRID_MODELS:
            f = LP / f"{tag}_k{k}_grid.json"
            if f.exists(): models[tag] = json.load(open(f))
        if not models: continue
        any_ = next(iter(models.values()))
        payload = dict(k=k, n_grid=any_["n_grid"], target=any_["target"], models={tag: dict(layers={l: dict(tables=v["tables"], stats=v["stats"]) for l, v in m["layers"].items()}) for tag, m in models.items()})
        default = any_["n_grid"].index(1000)
        html_ = [f'<div class="lpgrid" id="lpgrid-k{k}"><div class="lp-controls"><b>k = {k} blocks</b> &nbsp; N: <input type="range" min="0" max="{len(any_["n_grid"])-1}" value="{default}" step="1"> <span class="lp-label"></span>'
                 f' &nbsp;|&nbsp; <label><input type="radio" name="lpmode{k}" value="abs" checked> counts</label> <label><input type="radio" name="lpmode{k}" value="pct"> % of row</label></div>']
        html_.append('<table class="lp-table"><tr><th></th>' + "".join(f"<th>layer {l} &rarr; layer 9</th>" for l in range(1, 10)) + "</tr>")
        for tag, label in GRID_MODELS:
            if tag not in models: continue
            html_.append(f'<tr><td class="lp-rowhead">{label}</td>')
            for l in range(1, 10):
                cell_tbl = "<table>" + "".join("<tr>" + "".join("<td></td>" for _ in range(k)) + "</tr>" for _ in range(k)) + "</table>"
                html_.append(f'<td class="lp-cell lp-k{k}" data-model="{tag}" data-layer="{l}">{cell_tbl}<div class="lp-stat"></div></td>')
            html_.append("</tr>")
        html_.append("</table>")
        html_.append('<script type="application/json">' + json.dumps(payload, separators=(",", ":")) + "</script></div>")
        out += card("info", f"Layer i &rarr; layer 9 block agreement, k = {k}, all layers, slider over N", "".join(html_))
    out += "<script>" + GRID_JS + "</script>"
    return out


SQ = ROOT / "sparse_experts/olmoe3_squares"; SQO = OUT / "squares"


def build_q3():
    body = question_card("q3")
    G = json.load(open(SQ / "groups.json")) if (SQ / "groups.json").exists() else None
    stats = json.load(open(SQO / "stats.json")) if (SQO / "stats.json").exists() else None
    body += card("info", "Plan",
        "<ol><li><b>Partition</b> (done locally). Spectral blocks of the EMO 512e experts, k = 4, in layers 2&ndash;9, matched to the "
        "layer-9 blocks with the Q2 rule. Layer 1 is left whole in every sub-model (its blocks show no document agreement). "
        "Each group therefore owns 97&ndash;162 experts per layer from 2 on and all 512 in layer 1.</li>"
        "<li><b>Assign documents</b> (1 forward pass of the full model over the next 10B tokens, the 10B&ndash;20B window in training order). "
        "A document goes to the group whose experts receive most of its top-16 selections in layers 2&ndash;9. The full per-layer "
        "(group &times; count) table is kept per document, so the share of routing a sub-model cannot serve is known.</li>"
        "<li><b>Train four sub-models</b> from the step-19074 weights (experts and router rows sliced to the group; everything else copied), "
        "each on its own documents re-packed in training order, same EMO loss, same 64 &times; 8192 batch, constant LR 8e-4 (the WSD trunk). "
        "Token budgets are the groups' shares of the 10B, so total compute equals the baseline. Checkpoints at the same fractions of "
        "progress as the baseline's (steps 20000 / 25000 / 30000 / 35000 / 38148).</li>"
        "<li><b>Baseline</b>: the full EMO 512e continued from step 19074 on the same 10B tokens.</li>"
        "<li><b>Merge</b> at every matched checkpoint: experts concatenated, shared parameters (and layer 1) averaged with token-share weights; "
        "evaluated as-is (one 512-way router) and with oracle group routing, on the v3-small validation sets and a fresh held-out sample.</li></ol>")
    if G:
        rows = [[f"layer {l}", *[str(s) for s in G["sizes"][str(l)]], f(G["layer9_agreement"].get(str(l)), 2) if str(l) in G["layer9_agreement"] else "&mdash; (whole)"] for l in range(1, 10)]
        pv = G["preview"]
        body += section("Stage 0: the four block-groups",
            "Experts per group and layer, and how often a document's block in that layer agrees with its layer-9 block (the Q2 alignment).",
            table(["layer", "group 0", "group 1", "group 2", "group 3", "agreement with layer 9"], rows),
            f"On the 8k-instance routing sample the groups would receive {' / '.join(f'{100*x:.0f}%' for x in pv['token_share'])} of the tokens, "
            f"and a document keeps on average {100*pv['in_group_share_mean']:.0f}% of its layers 2&ndash;9 routing inside its chosen group "
            f"(median {100*pv['in_group_share_median']:.0f}%, random assignment would give 25%). The other half is what a sub-model cannot serve.")
    if stats:
        body += section("Stage 1: assigning the next 10B tokens",
            "Every document of the 10B&ndash;20B training window routed through the full model; in-group share = the fraction of its top-16 selections "
            "(layers 2&ndash;9) that fall on its group's experts.",
            table(["group", "documents", "tokens", "token share", "mean in-group share", "full-model CE"],
                  [[f"group {g}", f"{stats['docs_per_group'][g]:,}", f"{stats['tokens_per_group'][g]/1e9:.2f}B", f"{100*stats['token_share'][g]:.1f}%", f(stats['in_group_share_by_group'][str(g)], 2), f(stats['ce_by_group'][str(g)], 3)] for g in range(4)])
            + "<p>In-group share by layer: " + ", ".join(f"L{l} {v:.2f}" for l, v in stats["in_group_share_by_layer"].items()) + "</p>",
            f"{stats['n_docs']:,} documents, {stats['n_tokens']/1e9:.2f}B tokens. Token-weighted, {100*stats['in_group_share_token_weighted']:.0f}% of the "
            f"selections a sub-model's documents make are inside its own expert group.")
    else:
        body += card("warn", "Stage 1", "<p>Assignment pass running.</p>")
    body += squares_results()
    Gs = SQ.parent / "olmoe3_squares_std" / "groups.json"; Ss = OUT / "olmoe3_squares_std" / "stats.json"
    std_stats_src = ROOT / "sparse_experts/olmoe3_squares_std/pack/stats.json"
    if std_stats_src.exists():
        (OUT / "olmoe3_squares_std").mkdir(parents=True, exist_ok=True)
        if not Ss.exists() or Ss.stat().st_mtime < std_stats_src.stat().st_mtime: Ss.write_text(std_stats_src.read_text())
    if Gs.exists():
        G2 = json.load(open(Gs)); pv = G2["preview"]
        rows = [[f"layer {l}", *[str(s) for s in G2["sizes"][str(l)]], f(G2["layer9_agreement"].get(str(l)), 2) if str(l) in G2["layer9_agreement"] else "&mdash; (whole)"] for l in range(1, 10)]
        std_intro = ("<p><b>Same experiment on the standard-routing 512-expert model</b> (olmoe3_275m_10b, no EMO), as a second baseline. Its k = 4 blocks "
                     "are token-level (Q1), so documents were assigned by routing mass <em>per expert</em> of each group (raw mass would send 99% of "
                     "documents to the two biggest blocks); the same rule on the EMO model changes its group shares only slightly (21/35/26/18% vs "
                     "23/40/15/22%) and its in-group share not at all (0.50 vs 0.51).</p>")
        body += card("info", "Standard MoE: the four block-groups", std_intro + table(["layer", "group 0", "group 1", "group 2", "group 3", "agreement with layer 9"], rows)
                     + f"<p>Routing-sample preview: token shares {' / '.join(f'{100*x:.0f}%' for x in pv['token_share'])}, mean in-group share {100*pv['in_group_share_mean']:.0f}% "
                     f"(random 25%): a standard-model document keeps almost none of its routing inside any one group.</p>")
        if Ss.exists():
            st = json.load(open(Ss))
            body += card("info", "Standard MoE: assigning the next 10B tokens",
                         table(["group", "documents", "tokens", "token share", "mean in-group share", "full-model CE"],
                               [[f"group {g}", f"{st['docs_per_group'][g]:,}", f"{st['tokens_per_group'][g]/1e9:.2f}B", f"{100*st['token_share'][g]:.1f}%", f(st['in_group_share_by_group'][str(g)], 2), f(st['ce_by_group'][str(g)], 3)] for g in range(4)]))
        body += squares_results(HELD=ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b_std", PPL=ROOT / "sparse_experts/olmoe3_squares_std/ppl_validation", SQO=OUT / "olmoe3_squares_std",
                                start="std_step19074", start_ppl="olmoe3_275m_10b", base_runs=("olmoe3_275m_20b_1node",), label=" (standard MoE)", take_main=STD_TAKE, take_pw=STD_PW_TAKE)
    return body


HELD = ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b"; PPL = ROOT / "sparse_experts/olmoe3_squares/ppl_validation"
MATCH = [(20000, "match20000"), (25000, "match25000"), (30000, "match30000"), (35000, "match35000"), (38148, "match38148")]


def _ce(d):
    f = d / "meta.json"
    return json.load(open(f))["mean_ce"] if f.exists() else None


def _ppl(f):
    if not f.exists(): return None
    d = json.load(open(f))["per_set"]; return sum(v["CE loss"] for v in d.values()) / len(d)


EMO_TAKE = ("Merged beats the baseline only at 5% (2.424 vs 2.455) and then falls behind monotonically, ending 0.15 above it and 0.09 above the start model. "
            "The oracle-routed merged model keeps improving, so the loss is not in serving a document from one group.")
EMO_PW_TAKE = ("At 5% the squares alone (2.509) equal the merged oracle number (2.511): averaging is harmless while the four copies of the shared "
            "parameters are still nearly identical, and free routing across groups then adds a large gain (2.424). At 31% the squares alone keep "
            "improving (2.437, about the baseline's 2.434) but the merged model does not (2.530 under oracle routing): the averaged shared "
            "parameters now cost ~0.09, and the damage is concentrated on group 0, the code group, whose own square scores 1.50 while the "
            "merged model scores 1.77 on the same documents. The partition and the sub-models are not the problem; averaging diverged shared "
            "parameters is.")
STD_TAKE = ("The standard merge never beats its baseline: 2.482 vs 2.436 at 5%, then flat around 2.46 while the baseline keeps improving to 2.374, "
            "ending 0.09 behind. Unlike EMO it does not degrade with more separate training, and its oracle number improves steadily, but the "
            "restriction cost of the standard partition stays huge for the baseline (oracle 3.31 vs 2.37).")
STD_PW_TAKE = ("For the standard model the squares themselves are the problem: each square is worse than the start model on its own documents even "
            "at 31% (piecewise 2.506 vs the baseline's 2.413), because a quarter of token-level experts cannot serve the group's documents. "
            "Averaging the shared parameters costs almost nothing here (piecewise 2.506 vs merged oracle 2.516), the reverse of the EMO case, "
            "and free routing across groups recovers most of the rest (2.463).")


def squares_results(HELD=HELD, PPL=PPL, SQO=SQO, start="emo_step19074", start_ppl="olmoe3_275m_emo_10b", base_runs=("olmoe3_275m_emo_20b", "olmoe3_275m_emo_20b_filler", "olmoe3_275m_emo_20b_1node"), label="", take_main=EMO_TAKE, take_pw=EMO_PW_TAKE):
    ref_none, ref_orc = _ce(HELD / f"{start}/none"), _ce(HELD / f"{start}/oracle")
    ref_ppl = _ppl(ROOT / f"claude_outputs/debug_validation/ppl_validation/{start_ppl}/step19074.json")
    rows = [["start (step 19074, full model)", "&mdash;", f(ref_none), f(ref_orc), f(ref_ppl), "&mdash;", "&mdash;", "&mdash;"]]
    have = False
    for step, name in MATCH:
        b_none, b_orc = _ce(HELD / f"baseline_step{step}/none"), _ce(HELD / f"baseline_step{step}/oracle")
        b_ppl = next((_ppl(PPL / run / f"step{s}.json") for run in base_runs for s in (step, step - 1) if (PPL / run / f"step{s}.json").exists()), None)  # final ckpt is step38147
        m_none, m_orc = _ce(HELD / f"merged_{name}/none"), _ce(HELD / f"merged_{name}/oracle")
        m_ppl = _ppl(PPL / "merged" / f"{name}.json")
        if any(x is not None for x in (b_none, m_none, b_ppl, m_ppl)): have = True
        frac = (step - 19074) / 19074
        rows.append([f"baseline step {step:,} ({100*frac:.0f}% of the 10B)", "baseline", f(b_none), f(b_orc), f(b_ppl), "", "", ""])
        rows.append([f"merged squares @ {100*frac:.0f}%", "merged", f(m_none), f(m_orc), f(m_ppl),
                     f"{m_none-b_none:+.3f}" if (m_none is not None and b_none is not None) else "&mdash;",
                     f"{m_orc-b_orc:+.3f}" if (m_orc is not None and b_orc is not None) else "&mdash;",
                     f"{m_ppl-b_ppl:+.3f}" if (m_ppl is not None and b_ppl is not None) else "&mdash;"])
    # post-merge finetuning: +0.5B tokens (steps 38548-39502) for the 100% merge and, for fairness, for the baseline
    ft = {tag: (_ce(HELD / f"{tag}/none"), _ce(HELD / f"{tag}/oracle")) for tag in ("baseline_ft", "merged_ft")}
    ppl_ft = {}
    for r in PPL.glob("*_ft"):
        if (r / "step39502.json").exists(): ppl_ft["merged_ft" if "merged" in r.name else "baseline_ft"] = _ppl(r / "step39502.json")
    b_ft, m_ft = ft["baseline_ft"], ft["merged_ft"]
    if any(x is not None for x in b_ft[:2] + m_ft[:2]) or ppl_ft:
        rows.append(["baseline + 0.5B finetune (steps 38548&ndash;39502)", "baseline", f(b_ft[0]), f(b_ft[1]), f(ppl_ft.get("baseline_ft")), "", "", ""])
        rows.append(["merged @ 100% + 0.5B finetune (Adam state merged, same tokens)", "merged", f(m_ft[0]), f(m_ft[1]), f(ppl_ft.get("merged_ft")),
                     f"{m_ft[0]-b_ft[0]:+.3f}" if (m_ft[0] is not None and b_ft[0] is not None) else "&mdash;",
                     f"{m_ft[1]-b_ft[1]:+.3f}" if (m_ft[1] is not None and b_ft[1] is not None) else "&mdash;",
                     f"{ppl_ft['merged_ft']-ppl_ft['baseline_ft']:+.3f}" if ("merged_ft" in ppl_ft and "baseline_ft" in ppl_ft) else "&mdash;"])
    tbl = table(["checkpoint", "model", "held-out CE (20B window, 65M tok)", "held-out CE, oracle group routing", "v3-small ppl sets, mean CE", "&Delta; CE vs baseline", "&Delta; oracle", "&Delta; ppl"], rows)
    intro = ("<b>Post-merge finetuning</b> rows (when present): the 100% merged model, with the four squares' Adam moments merged the same way as the weights, "
             "trained for 0.5B more tokens (steps 38,548&ndash;39,502 of the training stream, past the held-out window) at the same constant LR; the baseline "
             "is continued on exactly the same tokens so the comparison stays at equal data. "
             "<b>Held-out CE</b>: mean token cross-entropy of 7,991 unseen instances (65.5M tokens) sampled from the 20B&ndash;30B window of the "
             "training stream, which neither the baseline nor the sub-models see. <b>Oracle group routing</b>: the same pass, but every document is "
             "restricted in layers 2&ndash;9 to the experts of the one group that receives most of its own unrestricted routing (the sub-model that "
             "would have served it); on the start model this measures the cost of the partition alone. <b>v3-small ppl sets</b>: OLMo-core's 11 "
             "validation sets (c4, dolma books / common-crawl / pes2o / reddit / stack / wiki, ice, m2d2, pile, wikitext), mean CE over sets. "
             "Baseline = the full model continued from step 19,074 on the same 10B tokens; merged = the four sub-models at the same fraction of "
             "progress, experts concatenated and shared parameters averaged with token-share weights.")
    take = (f"Restricting the start model to one group per document costs {ref_orc-ref_none:+.3f} CE before any training ({f(ref_none)} &rarr; {f(ref_orc)}): "
            "that is the routing the squares give up." if (ref_none and ref_orc) else "Reference passes pending.")
    if not have: take += " Baseline and merged evaluations are pending."
    else: take += " " + take_main
    out = section(f"Stages 2&ndash;4: merged squares vs continued baseline{label}", intro, tbl, take)
    # piecewise diagnostic: each held-out document scored by its own square, no merge
    pw = {name: json.load(open(SQO / f"piecewise_{name}.json")) for _, name in MATCH if (SQO / f"piecewise_{name}.json").exists() and json.load(open(SQO / f"piecewise_{name}.json")).get("piecewise")}
    if pw:
        rows = []
        for name, d in pw.items():
            step = int(name[5:]); frac = (step - 19074) / 19074
            b_none = _ce(HELD / f"baseline_step{step}/none")
            rows.append([f"{100*frac:.0f}%", f(d["piecewise"]), f(d[f"merged_{name}_oracle"]), f(d[f"merged_{name}"]), f(b_none)])
        grp_rows = []
        for name, d in pw.items():
            step = int(name[5:]); frac = (step - 19074) / 19074
            for g in range(4):
                grp_rows.append([f"{100*frac:.0f}%", f"group {g}", f"{d['docs_per_group'][g]:,}", f(d["start_full_by_group"][str(g)]), f(d["sub_on_group"][str(g)][str(g)]), f(d[f"merged_{name}_oracle_by_group"][str(g)]), f(d[f"merged_{name}_by_group"][str(g)])])
        out += section(f"Where the merge loses: each square alone vs the merged model{label}",
            "<b>Piecewise CE</b>: every held-out document is scored by the sub-model of its own group (group = the start model's routing), with "
            "no merging at all. Compared with the merged model under oracle routing (same documents, same expert groups, but shared parameters "
            "averaged) it isolates the cost of averaging; compared with the merged model's free routing it shows what cross-group experts add.",
            table(["progress", "piecewise CE (own square, no merge)", "merged, oracle routing", "merged, free routing", "baseline"], rows)
            + "<p><b>By document group</b> (held-out documents of each group; sub-model g on its own group vs the merged model on the same documents):</p>"
            + table(["progress", "group", "docs", "start model", "own square", "merged, oracle", "merged, free"], grp_rows),
            take_pw)
    return out


def build_next():
    return card("info", "Next steps", "<ul><li>Add the 2000-expert arms (EP=2) to the partition and block-agreement grids once their 10B runs finish.</li>"
                "<li>Use the persistent document-level blocks as the expert subsets for selective-expert finetuning, as in the sparse_experts plan.</li></ul>")


def main():
    tabs = [
        ("overview", "Overview", build_overview()),
        ("q1", QUESTIONS[0][1], build_q5()),
        ("q2", QUESTIONS[1][1], build_q6()),
        ("q3", QUESTIONS[2][1], build_q3()),
        ("next", "Next steps", build_next()),
    ]
    nav = "".join(f'<button data-target="{tid}">{name}</button>' for tid, name, _ in tabs)
    sections = "".join(f'<section class="tab" id="{tid}">{body}</section>' for tid, _, body in tabs)
    import datetime; stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = "OLMoE3-ladder 275M: document-level expert blocks, standard MoE vs EMO"
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style><style>.muted{{color:var(--muted);font-weight:normal;font-size:0.85em}} .what{{color:#475569;font-size:0.93em;border-left:3px solid #cbd5e1;padding-left:10px;margin:4px 0 10px}}</style></head>
<body><header><a class="home-link" href="/">&larr; all reports</a><h1>{html.escape(title)}</h1>
<p>olmoe3_routing &mdash; document-level expert blocks in the 275M-active OLMoE3-ladder models trained here (standard MoE vs EMO; 128 / 512 / 1000 experts),
from routing statistics on 65.5M unseen training-stream tokens &middot;
generated by scripts/olmoe3_routing/build_report.py on {stamp}</p></header>
<div class="topbar"><nav>{nav}</nav><div id="subnav"></div></div><main>{sections}</main><script>{JS}</script></body></html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.html").write_text(page)
    print(f"Wrote {OUT/'report.html'} ({(OUT/'report.html').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
