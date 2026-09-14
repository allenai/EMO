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
MODEL_LABEL = {"std": "standard MoE (512e)", "emo": "EMO (512e)", "std1000": "standard MoE (1000e)", "emo1000": "EMO (1000e)", "std128": "standard MoE (128e)", "emo128": "EMO (128e)", "emo2000": "EMO (2000e)", "std2000": "standard MoE (2000e)", "emo_pool64or512": "EMO (512e, pools {64, 512})", "emo_learnedd": "EMO (512e, learned pools)"}


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
    ("q4", "Q4 · Learn the pool size per document",
     "Instead of sampling each document's expert-pool size d at random, can the model learn d?"),
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


KS_MODELS = (("emo2000_full", "emo2000"), ("emo1000_full", "emo1000"), ("emo512_full", "emo"), ("emo_pool64or512_full", "emo_pool64or512"), ("emo_learnedd_full", "emo_learnedd"), ("emo128_full", "emo128"), ("std2000_full", "std2000"), ("std1000_full", "std1000"))
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
            if not all((KS / f"{tag}_L{l}_k{k}_docpartition.png").exists() for l in KS_LAYERS): continue
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


GRID_MODELS = (("emo2000_full", "EMO (2000e)"), ("emo1000_full", "EMO (1000e)"), ("emo512_full", "EMO (512e)"), ("emo_pool64or512_full", "EMO (512e, pools {64, 512})"), ("emo_learnedd_full", "EMO (512e, learned pools)"), ("emo128_full", "EMO (128e)"), ("std2000_full", "standard MoE (2000e)"), ("std1000_full", "standard MoE (1000e)"))
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


VARIANT_CSS = ("<style>.variant{margin:34px 0 10px;padding:0 0 4px 16px;border-left:6px solid var(--vc)}.variant>h2{margin:0 0 4px;font-size:20px;color:var(--vc)}"
               ".variant>h2 .vb{display:inline-block;width:28px;height:28px;line-height:28px;border-radius:6px;background:var(--vc);color:#fff;text-align:center;margin-right:10px;font-size:15px}"
               ".variant>p.lead{margin:0 0 12px;color:#475569}.q3index li{margin:3px 0}</style>")
VARIANTS = [("A", "EMO 512e, 4 sub-models", "#2563eb", "the main experiment"),
            ("B", "Standard MoE 512e, 4 sub-models", "#dc2626", "same pipeline on the standard-routing model, as a second baseline"),
            ("C", "EMO 512e, 4 sub-models trained without the EMO loss", "#7c3aed", "same partition and start checkpoints as A; the sub-models use plain top-16 routing"),
            ("D", "EMO 512e, 8 sub-models", "#059669", "same as A with k = 8 blocks per layer")]


def variant(letter, inner):
    _, title, color, blurb = next(v for v in VARIANTS if v[0] == letter)
    return (f'<div class="variant" id="q3-{letter}" style="--vc:{color}"><h2><span class="vb">{letter}</span>{title}</h2>'
            f'<p class="lead">{blurb}.</p>{inner}</div>')


def _assign_table(st, k):
    return table(["group", "documents", "tokens", "token share", "mean in-group share", "full-model CE"],
                 [[f"group {g}", f"{st['docs_per_group'][g]:,}", f"{st['tokens_per_group'][g]/1e9:.2f}B", f"{100*st['token_share'][g]:.1f}%", f(st['in_group_share_by_group'][str(g)], 2), f(st['ce_by_group'][str(g)], 3)] for g in range(k)])


def _groups_table(G, k):
    return table(["layer", *[f"group {g}" for g in range(k)], "agreement with layer 9"],
                 [[f"layer {l}", *[str(x) for x in G["sizes"][str(l)]], f(G["layer9_agreement"].get(str(l)), 2) if str(l) in G["layer9_agreement"] else "&mdash; (whole)"] for l in range(1, 10)])


def build_q3():
    body = question_card("q3")
    G = json.load(open(SQ / "groups.json")) if (SQ / "groups.json").exists() else None
    stats = json.load(open(SQO / "stats.json")) if (SQO / "stats.json").exists() else None
    body += card("info", "Plan",
        "<ol>"
        "<li><b>Split.</b> For each model, split each layer's experts into 4 spectral blocks and, using the block agreement across layers (Q2), "
        "build 4 sub-models, each owning one block per layer. Layer 1 is kept whole in every sub-model.</li>"
        "<li><b>Assign.</b> Assign the documents we are going to train on to the four sub-models: each document goes to the sub-model whose "
        "experts it routes to most.</li>"
        "<li><b>Train.</b> Train each sub-model on its own documents.</li>"
        "<li><b>Merge.</b> Merge the four sub-models back into one model: experts side by side, shared parameters averaged.</li>"
        "<li><b>Baseline.</b> The original model, continuously trained on all the documents together, without splitting into sub-models. "
        "Both routes see the same 10B tokens; the merged model is compared with the baseline at matching points of that training.</li>"
        "</ol>")
    present = ["A"] + (["B"] if (SQ.parent / "olmoe3_squares_std" / "groups.json").exists() else []) \
              + (["C"] if (ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b_noemo").exists() else []) \
              + (["D"] if (ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b_k8").exists() else [])
    body += VARIANT_CSS + card("info", "Four runs of this plan",
        '<ul class="q3index">' + "".join(f'<li><a href="#q3-{L}"><b>{L}</b> &middot; {t}</a> &mdash; {bl}</li>' for L, t, _, bl in VARIANTS if L in present)
        + "</ul><p>Each run has the same three blocks: the block-groups (stage 0), the document assignment (stage 1), and the merged squares "
        "against the continued baseline (stages 2&ndash;4) with the piecewise diagnostic.</p>")
    # ---- A: EMO 512e, k = 4 ----
    inner = ""
    if G:
        inner += section("Stage 0: the four block-groups",
            "Number of experts per group in each layer. The last column is how often a document's block in that layer agrees with its "
            "layer-9 block (the Q2 alignment used to match blocks across layers).",
            _groups_table(G, 4),
            "Groups are uneven (97&ndash;162 experts per layer), so the four sub-models differ in size; layer 1 keeps all 512 experts in every group.")
    if stats:
        inner += section("Stage 1: assigning the next 10B tokens",
            "Every document of the 10B&ndash;20B training window routed through the full model; in-group share = the fraction of its top-16 selections "
            "(layers 2&ndash;9) that fall on its group's experts.",
            _assign_table(stats, 4) + "<p>In-group share by layer: " + ", ".join(f"L{l} {v:.2f}" for l, v in stats["in_group_share_by_layer"].items()) + "</p>",
            f"{stats['n_docs']:,} documents, {stats['n_tokens']/1e9:.2f}B tokens. Token-weighted, {100*stats['in_group_share_token_weighted']:.0f}% of the "
            f"selections a sub-model's documents make are inside its own expert group.")
    else:
        inner += card("warn", "Stage 1", "<p>Assignment pass running.</p>")
    inner += squares_results()
    body += variant("A", inner)
    # ---- B: standard MoE, k = 4 ----
    Gs = SQ.parent / "olmoe3_squares_std" / "groups.json"; Ss = OUT / "olmoe3_squares_std" / "stats.json"
    std_stats_src = ROOT / "sparse_experts/olmoe3_squares_std/pack/stats.json"
    if std_stats_src.exists():
        (OUT / "olmoe3_squares_std").mkdir(parents=True, exist_ok=True)
        if not Ss.exists() or Ss.stat().st_mtime < std_stats_src.stat().st_mtime: Ss.write_text(std_stats_src.read_text())
    if Gs.exists():
        G2 = json.load(open(Gs))
        inner = section("Stage 0: the four block-groups",
            "Same spectral blocks and layer-9 alignment as in A, on the standard-routing model (olmoe3_275m_10b).",
            _groups_table(G2, 4),
            "The blocks are token-level (Q1): a document spreads its selections over the four groups almost in proportion to their size.")
        if Ss.exists():
            inner += section("Stage 1: assigning the next 10B tokens",
                "Documents were assigned by routing mass <em>per expert</em> of each group (selections on the group divided by its expert count). "
                "The raw count used in A is degenerate here: it sends 99% of the documents to the two biggest blocks, because every document routes to "
                "the groups in proportion to their size. On the EMO model the per-expert rule changes the group shares only slightly (21/35/26/18% vs "
                "23/40/15/22%) and the in-group share not at all (0.50 vs 0.51).",
                _assign_table(json.load(open(Ss)), 4),
                "Even with the per-expert rule only 27% of a document's selections fall inside its own group (52% for EMO): the standard model's "
                "documents are not tied to a block.")
        inner += squares_results(HELD=ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b_std", PPL=ROOT / "sparse_experts/olmoe3_squares_std/ppl_validation", SQO=OUT / "olmoe3_squares_std",
                                 start="std_step19074", start_ppl="olmoe3_275m_10b", base_runs=("olmoe3_275m_20b_1node",), label="", take_main=STD_TAKE, take_pw=STD_PW_TAKE)
        body += variant("B", inner)
    # ---- C: EMO squares without the EMO loss ----
    if (ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b_noemo").exists():
        inner = card("info", "Setup", "<p>Same EMO 512e partition, document assignment and sliced start checkpoints as A (stages 0 and 1 are identical), but the four "
                     "sub-models and the post-merge 0.5B finetune train with plain top-16 routing (no per-document pool rule, instance-level load "
                     "balancing). Compared against the same EMO baseline as A.</p>")
        inner += squares_results(HELD=ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b_noemo", PPL=ROOT / "sparse_experts/olmoe3_squares_noemo/ppl_validation", SQO=OUT / "olmoe3_squares_noemo",
                                 start="emo_step19074", start_ppl="olmoe3_275m_emo_10b", base_runs=("olmoe3_275m_emo_20b_1node",), label="", take_main="", take_pw="")
        body += variant("C", inner)
    # ---- D: EMO, k = 8 ----
    if (ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b_k8").exists():
        G8 = json.load(open(ROOT / "sparse_experts/olmoe3_squares_k8/groups.json")) if (ROOT / "sparse_experts/olmoe3_squares_k8/groups.json").exists() else None
        inner = section("Stage 0: the eight block-groups",
            "Same stages as A on the EMO 512e model, with each layer's experts split into 8 spectral blocks instead of 4 (layer 1 kept whole), so each "
            "sub-model owns about an eighth of the experts and trains on about an eighth of the tokens. Compared against the same EMO baseline as A.",
            _groups_table(G8, 8) if G8 else "",
            "Blocks of 46&ndash;84 experts per layer; the agreement with layer 9 is lower than at k = 4.")
        k8_stats = ROOT / "sparse_experts/olmoe3_squares_k8/pack/stats.json"
        if k8_stats.exists():
            st = json.load(open(k8_stats))
            inner += section("Stage 1: assigning the next 10B tokens", "Raw-count assignment as in A.", _assign_table(st, 8),
                             f"Token-weighted, {100*st['in_group_share_token_weighted']:.0f}% of the selections a sub-model's documents make are inside its own expert group.")
        inner += squares_results(HELD=ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b_k8", PPL=ROOT / "sparse_experts/olmoe3_squares_k8/ppl_validation", SQO=OUT / "olmoe3_squares_k8",
                                 start="emo_step19074", start_ppl="olmoe3_275m_emo_10b", base_runs=("olmoe3_275m_emo_20b_1node",), label="", take_main="", take_pw="")
        body += variant("D", inner)
    return body


HELD = ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b"; PPL = ROOT / "sparse_experts/olmoe3_squares/ppl_validation"
MATCH = [(20000, "match20000"), (25000, "match25000"), (30000, "match30000"), (35000, "match35000"), (38148, "match38148")]


ROBUST_NOTE = []


def _ce(d):
    """mean CE of a held-out pass; if non-finite (a few documents blew up), fall back to the token-weighted CE over
    documents with finite CE < 10 and record the note (the std baseline finetune hit this: 31 of 56,547 docs)."""
    f = d / "meta.json"
    if not f.exists(): return None
    ce = json.load(open(f))["mean_ce"]
    if ce == ce and ce < 50: return ce
    import numpy as np
    s = np.load(d / "doc_stats.npz"); dc = s["ce_sum"] / np.maximum(s["ce_len"], 1); ok = np.isfinite(dc) & (dc < 10)
    ROBUST_NOTE.append(f"{d.parent.name}/{d.name}: {int((~ok).sum())} of {len(dc):,} documents non-finite or CE &gt; 10; number shown excludes them ({100*s['ce_len'][ok].sum()/s['ce_len'].sum():.2f}% of tokens kept)")
    return float(s["ce_sum"][ok].sum() / s["ce_len"][ok].sum())


def _ppl(f):
    if not f.exists(): return None
    d = json.load(open(f))["per_set"]; return sum(v["CE loss"] for v in d.values()) / len(d)


CHART_JS = """<script>(function(){function init(w){const d=JSON.parse(w.dataset.chart);const svg=w.querySelector('svg');const tip=w.querySelector('.lc-tip');const vl=w.querySelector('.lc-vline');
const px=i=>d.x0+(d.xs[i]-d.xmin)/(d.xmax-d.xmin)*d.pw;
svg.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect();const mx=(e.clientX-r.left)*(d.W/r.width);let best=0,bd=1e9;d.xs.forEach((x,i)=>{const dd=Math.abs(px(i)-mx);if(dd<bd){bd=dd;best=i;}});
vl.setAttribute('x1',px(best));vl.setAttribute('x2',px(best));vl.style.display='block';
tip.style.display='block';tip.innerHTML='<b>'+d.xlab[best]+'</b><br>'+d.series.map(s=>{const v=s.y[best];return v==null?'':'<span style="color:'+s.c+'">&#9632;</span> '+s.n+': '+v.toFixed(3);}).filter(t=>t).join('<br>');
const tx=(e.clientX-r.left),ty=(e.clientY-r.top);tip.style.left=(tx+14)+'px';tip.style.top=(ty-10)+'px';});
svg.addEventListener('mouseleave',()=>{tip.style.display='none';vl.style.display='none';});}
document.querySelectorAll('.lc').forEach(init);})();</script>"""
CHART_CSS = "<style>.lc{position:relative;display:inline-block;vertical-align:top;margin:4px 10px 8px 0}.lc svg{max-width:100%;height:auto;font-family:inherit}.lc .lc-tip{display:none;position:absolute;background:#fff;border:1px solid #cbd5e1;border-radius:4px;padding:6px 8px;font-size:12px;pointer-events:none;box-shadow:0 2px 6px rgba(0,0,0,.12);white-space:nowrap;z-index:5}.lc .lc-title{font-size:13px;font-weight:600;margin:0 0 2px 44px}</style>"
SUBNAV_JS = """<script>
// Sub-navigation: only the top level of each tab (direct cards, and the A-D experiment banners of Q3), not every nested card.
buildSubnav = function(id) {
  const sec = document.getElementById(id), sub = document.getElementById('subnav'); sub.innerHTML = ''; if (!sec) return;
  const add = (el, text) => { if (!el.id) el.id = id + '--' + slug(text); const a = document.createElement('a'); a.href = '#' + el.id; a.textContent = text;
    a.addEventListener('click', e => { e.preventDefault(); el.scrollIntoView({behavior:'smooth', block:'start'}); }); sub.appendChild(a); };
  sec.querySelectorAll(':scope > .card > h3, :scope > .variant > h2').forEach(h => {
    if (h.tagName === 'H2') { const b = h.querySelector('.vb'); add(h.parentElement, (b ? b.textContent + ' \u00b7 ' : '') + h.textContent.replace(b ? b.textContent : '', '').trim()); }
    else add(h, h.textContent); });
};
show(location.hash && document.getElementById(location.hash.slice(1)) ? location.hash.slice(1) : 'overview');
</script>"""
_LC_COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]


def line_chart(xs, series, *, title="", y_label="CE", x_label="progress through the 10B tokens", xfmt=lambda v: f"{v:.0f}%", W=480, H=280, xlabels=None, shade=None):
    """Interactive line chart (inline SVG; hovering shows every series' value at the nearest x).
    xs: x values; series: dicts(name, y=list with None for missing, dashed=False, const=False (horizontal reference line), color);
    xlabels: explicit tick labels; shade=(x_from, label): tint the plot area from x_from rightwards (e.g. a finetuning stage)."""
    import math
    x0, y0, pw, ph = 46, 14, W - 60, H - 52
    ys = [v for s_ in series for v in s_["y"] if v is not None]
    if not ys: return ""
    ymin, ymax = min(ys), max(ys); pad = max(0.02, 0.08 * (ymax - ymin)); ymin -= pad; ymax += pad
    xmin, xmax = min(xs), max(xs)
    if xmax == xmin: xmax = xmin + 1
    X = lambda v: x0 + (v - xmin) / (xmax - xmin) * pw
    Y = lambda v: y0 + (ymax - v) / (ymax - ymin) * ph
    labels = xlabels or [xfmt(v) for v in xs]
    g = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
    if shade:
        xa = (X(shade[0]) + X(max(v for v in xs if v < shade[0]))) / 2 if any(v < shade[0] for v in xs) else X(shade[0])
        g.append(f'<rect x="{xa:.1f}" y="{y0}" width="{x0+pw-xa:.1f}" height="{ph}" fill="#f1f5f9"/>'
                 f'<text x="{(xa+x0+pw)/2:.1f}" y="{y0+11}" font-size="10" text-anchor="middle" fill="#64748b">{shade[1]}</text>')
    step = (ymax - ymin) / 4
    mag = 10 ** math.floor(math.log10(step)); step = math.ceil(step / mag) * mag
    t = math.ceil(ymin / step) * step
    while t <= ymax:
        g.append(f'<line x1="{x0}" x2="{x0+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="#e5e7eb"/><text x="{x0-6}" y="{Y(t)+4:.1f}" font-size="11" text-anchor="end" fill="#475569">{t:.2f}</text>')
        t += step
    for v, lab in zip(xs, labels):
        g.append(f'<text x="{X(v):.1f}" y="{y0+ph+16}" font-size="11" text-anchor="middle" fill="#475569">{lab}</text>')
    g.append(f'<line x1="{x0}" x2="{x0+pw}" y1="{y0+ph}" y2="{y0+ph}" stroke="#94a3b8"/><line x1="{x0}" x2="{x0}" y1="{y0}" y2="{y0+ph}" stroke="#94a3b8"/>')
    g.append(f'<text x="{x0+pw/2:.0f}" y="{H-4}" font-size="11" text-anchor="middle" fill="#475569">{x_label}</text>')
    g.append(f'<text transform="translate(12,{y0+ph/2:.0f}) rotate(-90)" font-size="11" text-anchor="middle" fill="#475569">{y_label}</text>')
    data = {"W": W, "x0": x0, "pw": pw, "xmin": xmin, "xmax": xmax, "xs": xs, "xlab": labels, "series": []}
    legend = []
    for i, s_ in enumerate(series):
        c = s_.get("color") or _LC_COLORS[i % len(_LC_COLORS)]; dash = ' stroke-dasharray="5,4"' if s_.get("dashed") else ""
        y = s_["y"]
        if s_.get("const"):
            v = next((v for v in y if v is not None), None)
            if v is None: continue
            g.append(f'<line x1="{x0}" x2="{x0+pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{c}" stroke-width="1.6"{dash}/>')
            y = [v] * len(xs)
        else:
            pts = [(X(x), Y(v)) for x, v in zip(xs, y) if v is not None]
            if len(pts) > 1: g.append(f'<polyline fill="none" stroke="{c}" stroke-width="2"{dash} points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
            for a, b in pts: g.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.2" fill="{c}"/>')
        data["series"].append({"n": s_["name"], "y": y, "c": c})
        legend.append(f'<span style="display:inline-block;margin-right:10px"><span style="display:inline-block;width:18px;border-top:2px {"dashed" if s_.get("dashed") else "solid"} {c};vertical-align:middle"></span> {s_["name"]}</span>')
    g.append(f'<line class="lc-vline" x1="0" x2="0" y1="{y0}" y2="{y0+ph}" stroke="#94a3b8" stroke-dasharray="2,3" style="display:none"/></svg>')
    payload = html.escape(json.dumps(data), quote=True)
    return (f'<div class="lc" data-chart="{payload}"><div class="lc-title">{title}</div>' + "".join(g)
            + f'<div style="font-size:12px;margin-left:44px">{"".join(legend)}</div><div class="lc-tip"></div></div>')


EMO_TAKE = ("The merged model beats the baseline only at 5% (2.424 vs 2.455) and then falls behind monotonically, ending 0.15 above it and "
            "0.09 above the start model on the held-out sample; the v3-small sets tell the same story.")
EMO_PW_TAKE = ("The squares on their own track the baseline the whole way and end slightly below it (2.385 vs 2.398 at 100%), while the merged "
            "model drifts up to 2.551. Only at 5% is merging a gain (2.424 vs 2.509 piecewise): the four copies of the shared parameters are still "
            "nearly identical, so averaging is free and routing across groups adds experts. From 31% on, averaging diverged shared parameters is "
            "what costs, and the damage is concentrated on group 0, the code group: its own square improves from 1.56 to 1.45 while the merged "
            "model on the same documents worsens from 1.53 to 1.91. The partition and the sub-models are not the problem; averaging is.")
STD_TAKE = ("The standard merge never beats its baseline: 2.482 vs 2.436 at 5%, then flat around 2.46 while the baseline keeps improving to 2.374, "
            "ending 0.09 behind. Unlike EMO it does not degrade with more separate training.")
STD_PW_TAKE = ("For the standard model the squares themselves are the problem: on their own they start far behind (2.614 at 5% vs the start "
            "model's 2.441) and only get back to the start model by 100% (2.448), never near the baseline (2.374), because a quarter of "
            "token-level experts cannot serve a group's documents. Merging costs almost nothing here (2.463 at 100% vs 2.448 piecewise), the "
            "reverse of the EMO case, and is a gain early on while the squares are still weak.")


def squares_results(HELD=HELD, PPL=PPL, SQO=SQO, start="emo_step19074", start_ppl="olmoe3_275m_emo_10b", base_runs=("olmoe3_275m_emo_20b", "olmoe3_275m_emo_20b_filler", "olmoe3_275m_emo_20b_1node"), label="", take_main=EMO_TAKE, take_pw=EMO_PW_TAKE):
    ref_none = _ce(HELD / f"{start}/none")
    ref_ppl = _ppl(ROOT / f"claude_outputs/debug_validation/ppl_validation/{start_ppl}/step19074.json")
    xs, b_h, m_h, b_p, m_p = [], [], [], [], []
    for step, name in MATCH:
        frac = 100 * (step - 19074) / 19074; xs.append(round(frac))
        b_h.append(_ce(HELD / f"baseline_step{step}/none")); m_h.append(_ce(HELD / f"merged_{name}/none"))
        b_p.append(next((_ppl(PPL / run / f"step{s}.json") for run in base_runs for s in (step, step - 1) if (PPL / run / f"step{s}.json").exists()), None))  # final ckpt is step38147
        m_p.append(_ppl(PPL / "merged" / f"{name}.json"))
    have = any(v is not None for v in b_h + m_h)
    # post-merge finetuning: +0.5B tokens (steps 38548-39502) for the 100% merge and, for fairness, for the baseline -> right-most point
    ft = {tag: (_ce(HELD / f"{tag}_rerun/none") if (HELD / f"{tag}_rerun/none/meta.json").exists() else _ce(HELD / f"{tag}/none")) for tag in ("baseline_ft", "merged_ft")}  # a rerun pass supersedes a pass with blown-up documents
    ppl_ft = {}
    for r in PPL.glob("*_ft"):
        if (r / "step39502.json").exists(): ppl_ft["merged_ft" if "merged" in r.name else "baseline_ft"] = _ppl(r / "step39502.json")
    # second finetuning stage (+1B total): steps 39502 -> 40456
    ft2 = {tag: _ce(HELD / f"{tag}2/none") for tag in ("baseline_ft", "merged_ft")}
    for r in PPL.glob("*_ft2"):
        if (r / "step40456.json").exists(): ppl_ft["merged_ft2" if "merged" in r.name else "baseline_ft2"] = _ppl(r / "step40456.json")
    has_ft = any(v is not None for v in ft.values()) or bool(ppl_ft)
    has_ft2 = any(v is not None for v in ft2.values()) or any(k.endswith("_ft2") for k in ppl_ft)
    extra_x = ([125] if has_ft else []) + ([150] if has_ft2 else []); extra_lab = (["+0.5B ft"] if has_ft else []) + (["+1B ft"] if has_ft2 else [])
    xs_all = xs + extra_x; labels = [f"{v}%" for v in xs] + extra_lab
    shade = (125, "finetuned on 0.5B / 1B more tokens" if has_ft2 else "finetuned 0.5B") if has_ft else None
    def ext(y, v1, v2): return y + ([v1] if has_ft else []) + ([v2] if has_ft2 else [])
    charts = ""
    if have:
        charts = CHART_CSS + line_chart(xs_all, [{"name": "baseline (full model, continued)", "y": ext(b_h, ft["baseline_ft"], ft2["baseline_ft"])}, {"name": "merged squares", "y": ext(m_h, ft["merged_ft"], ft2["merged_ft"])},
                                                {"name": "start model (step 19,074)", "y": [ref_none] * len(xs_all), "const": True, "dashed": True, "color": "#64748b"}],
                                       title="Held-out CE (20B window, 65M tokens)", xlabels=labels, shade=shade)
        if any(v is not None for v in b_p + m_p):
            charts += line_chart(xs_all, [{"name": "baseline (full model, continued)", "y": ext(b_p, ppl_ft.get("baseline_ft"), ppl_ft.get("baseline_ft2"))}, {"name": "merged squares", "y": ext(m_p, ppl_ft.get("merged_ft"), ppl_ft.get("merged_ft2"))},
                                          {"name": "start model (step 19,074)", "y": [ref_ppl] * len(xs_all), "const": True, "dashed": True, "color": "#64748b"}],
                                 title="v3-small ppl sets, mean CE", xlabels=labels, shade=shade)
    ft_html = ""
    if has_ft:
        ft_html = ("<p><b>Finetuning</b> (shaded): the 100% merged model (Adam moments merged like the weights) and the baseline each trained on the same "
                   "0.5B more tokens (steps 38,548&ndash;39,502) at the same constant LR" + (", then on a further 0.5B (steps 39,502&ndash;40,456; 1B in total)" if has_ft2 else "")
                   + ". The held-out sample is steps 38,148&ndash;38,547 of the stream, so no finetuning token is in it.</p>")
    intro = ("<b>Held-out CE</b>: mean token cross-entropy of 7,991 unseen instances (65.5M tokens) from the 20B&ndash;30B window of the training "
             "stream, which neither the baseline nor the sub-models see. <b>v3-small ppl sets</b>: OLMo-core's 11 validation sets, mean CE over sets. "
             "Baseline = the full model continued from step 19,074 on the same 10B tokens; merged = the four sub-models at the same fraction of "
             "progress, experts concatenated and shared parameters averaged with token-share weights. Hover over a plot for the values.")
    take = take_main if have else "Baseline and merged evaluations are pending."
    if ROBUST_NOTE: take += " <b>Caveat:</b> " + "; ".join(ROBUST_NOTE) + "."; ROBUST_NOTE.clear()
    out = section(f"Stages 2&ndash;4: merged squares vs continued baseline{label}", intro, charts + ft_html, take)
    # piecewise diagnostic: each held-out document scored by its own square, no merge
    pw = {name: json.load(open(SQO / f"piecewise_{name}.json")) for _, name in MATCH if (SQO / f"piecewise_{name}.json").exists() and json.load(open(SQO / f"piecewise_{name}.json")).get("piecewise")}
    if pw:
        pxs = [round(100 * (int(n[5:]) - 19074) / 19074) for n in pw]
        D = list(pw.values()); K = len(D[0]["docs_per_group"])
        avg = line_chart(pxs, [{"name": "piecewise (own square, no merge)", "y": [d["piecewise"] for d in D]},
                               {"name": "merged, free routing", "y": [d[f"merged_{d['name']}"] for d in D]},
                               {"name": "baseline", "y": [d.get("baseline") for d in D], "dashed": True, "color": "#059669"},
                               {"name": "start model", "y": [D[0]["start_full"]] * len(D), "const": True, "dashed": True, "color": "#64748b"}],
                         title="All held-out documents")
        grp = ""
        for g in range(K):
            grp += line_chart(pxs, [{"name": "own square", "y": [d["sub_on_group"][str(g)][str(g)] for d in D]},
                                    {"name": "merged, free routing", "y": [d[f"merged_{d['name']}_by_group"][str(g)] for d in D]},
                                    {"name": "baseline", "y": [(d.get("baseline_by_group") or {}).get(str(g)) for d in D], "dashed": True, "color": "#059669"},
                                    {"name": "start model", "y": [D[0]["start_full_by_group"][str(g)]] * len(D), "const": True, "dashed": True, "color": "#64748b"}],
                              title=f"Group {g} documents ({D[0]['docs_per_group'][g]:,} held-out docs)", W=400, H=250)
        out += section(f"Where the merge loses: each square alone vs the merged model{label}",
            "<b>Piecewise CE</b>: every held-out document is scored by the sub-model of its own group (group = the start model's routing), with no "
            "merging at all; the merged model is scored on the same documents. Points where the squares' passes are still running are missing.",
            CHART_CSS + avg + "<p><b>By document group</b> (each group's held-out documents: its own square vs the merged model on the same documents):</p>" + grp,
            take_pw)
    return out


LD = ROOT / "sparse_experts/learnedd_sweep"


def build_q4():
    body = question_card("q4")
    body += card("info", "Method",
        "<ol>"
        "<li><b>Predict d.</b> In each MoE layer a small linear head reads the document's mean hidden state and outputs its pool size d, "
        "between 16 (the top-k) and 512 (all experts). Routing then works exactly as in EMO with that pool size.</li>"
        "<li><b>Keep pools small.</b> A penalty on d pushes every pool smaller; <b>&lambda;<sub>d</sub></b> sets how hard.</li>"
        "<li><b>Let pools grow where needed.</b> Something has to push back, or every pool shrinks to 16. Two choices of signal:"
        "<ul>"
        "<li><b>STE</b> uses the language-model loss. The edge of the pool is made soft over <b>T</b> expert ranks, so the loss can say whether "
        "the experts just inside the edge are helping.</li>"
        "<li><b>Coverage</b> uses the router itself. The pool grows while the expert at its edge still receives more than a threshold of the "
        "document's router mass; the threshold is &lambda;<sub>d</sub> / (<b>&lambda;<sub>cov</sub></b> &middot; 496), so a larger "
        "&lambda;<sub>d</sub> shrinks pools and a larger &lambda;<sub>cov</sub> grows them.</li>"
        "</ul></li>"
        "<li><b>Start gently.</b> Training begins with all 512 experts and no penalty; over the first <b>W</b> steps (<b>warm-up</b>) the "
        "allowed pool shrinks to 16 and the penalty ramps up to &lambda;<sub>d</sub>.</li>"
        "<li><b>Let d move.</b> The head trains with a learning-rate multiplier (<b>head LR</b>); at the base rate d barely moves, because Adam "
        "steps a scalar by about one learning rate per step.</li>"
        "<li><b>Evaluate.</b> Route each document with its predicted d (default), or with a fixed pool for comparison.</li>"
        "</ol>")
    tab = LD / "sweep_table.json"
    if tab.exists():
        import re as _re
        T = {r["run"].split("sweep_", 1)[1]: r for r in json.load(open(tab))}
        def parse(key):
            if key == "control_uniform": return None
            m = _re.match(r"T([\d.]+)_l([\d.]+)_w(\d+)_m([\d.]+)(?:_coveragec([\d.]+))?", key)
            T_, l, w, mult, cov = m.groups()
            return {"signal": "coverage" if cov else "STE", "T": float(T_), "lambda": float(l), "W": int(w), "mult": float(mult), "cov": float(cov) if cov else None}
        def row(key):
            r = T[key]; h = parse(key)
            dm = r.get("d_soft_mean") or {}; fr = r.get("d_frac_le64") or {}
            ds = [dm.get(str(l)) for l in range(2, 10)]; ds = [d for d in ds if d is not None]; d1 = dm.get("1")
            dtxt = ("16 in every layer" if ds and max(ds) < 16.5 and (d1 or 0) < 16.5 else f"{min(ds):.0f}&ndash;{max(ds):.0f} (L1 {d1:.0f})") if ds else "&mdash;"
            f5 = fr.get("5"); ev = r.get("eval_ce", {}).get("2000"); e2 = sum(ev.values()) / len(ev) if ev else None
            if h is None:
                hp = ["uniform EMO (control)", "&mdash;", "&mdash;", "&mdash;", "&mdash;", "&mdash;"]
            else:
                thr = f"{h['lambda']/(h['cov']*496):.3f}" if h["cov"] else "&mdash;"
                hp = [h["signal"], f"{h['T']:g}", f"{h['lambda']:g}", str(h["W"]), f"&times;{h['mult']:g}", thr]
            return hp + [dtxt, f"{100*f5:.0f}%" if f5 is not None else "&mdash;", f(r.get("train_ce_last100"), 3), f(e2, 3), str(r.get("skipped_steps", 0))]
        order = sorted(T, key=lambda k: (0 if k == "control_uniform" else 1 if "coverage" not in k else 2, parse(k)["lambda"] if parse(k) else 0, parse(k)["T"] if parse(k) else 0, parse(k)["W"] if parse(k) else 0, parse(k)["mult"] if parse(k) else 0))
        rows = [row(k) for k in order]
        body += section("2000-step sweep",
            "One run per row, all from scratch on the same 2000 steps (1.05B tokens) as the first 2000 steps of the 10B arms. d = mean predicted pool at "
            "step 2000 over layers 2&ndash;9 (layer 1 in brackets); docs &le; 64 = share of documents whose layer-5 pool is at most 64 experts; eval CE = mean "
            "over the 11 v3-small validation sets at step 2000, routing with the predicted d.",
            table(["signal", "T", "&lambda;<sub>d</sub>", "warm-up", "head LR", "mass threshold", "d at step 2000", "docs &le; 64", "train CE", "eval CE", "skipped steps"], rows),
            "Every STE arm collapses to d = 16 in every layer within a few hundred steps, whatever the temperature, penalty (even 0), warm-up or head "
            "LR: the STE gradient only sees the selected experts, and a bigger pool just flattens their weights, so the LM loss always asks for a smaller d. "
            "That costs +0.05 eval CE. The coverage signal gives a tunable pool size: at threshold 0.002 the pools average 110&ndash;130 experts, two thirds "
            "of the documents use at most 64, and the eval CE matches the uniform-pool control.")
    else:
        body += card("warn", "Sweep", "<p>Sweep results not collected yet.</p>")
    body += learnedd_10b()
    return body


def learnedd_10b():
    intro = ("<p>EMO 512e trained from scratch for 10B tokens with the coverage signal at threshold 0.002 (T = 2, warm-up 500 steps, head LR &times;10), "
             "otherwise the recipe of the other 10B arms (run olmoe3_275m_emo_learnedd_10b).</p>")
    wb = LD / "learnedd_10b_wandb.json"
    if not wb.exists():
        return card("info", "10B run", intro + "<p>Running.</p>")
    J = json.load(open(wb))
    out = card("info", "10B run", intro)
    rows = []
    for e in J["d_traj"]:
        if e["step"] in (1000, 5000, 10000, 15000, 19074):
            d = e["d"]; fr = e["frac_le64"]
            rows.append([f"{e['step']:,}", *[f"{v:.0f}" for v in d], f"{100*sum(fr[1:])/len(fr[1:]):.0f}%"])
    out += section("Learned pool size during training",
        "Mean predicted d per layer at the logged step (the training batch's documents), and the share of documents whose pool is at most 64 experts, "
        "averaged over layers 2&ndash;9.",
        table(["step", *[f"L{l}" for l in range(1, 10)], "docs &le; 64"], rows),
        "The pools settle within the first 1000 steps and stay at roughly 110&ndash;190 experts for the rest of training, with about 60% of the "
        "documents at 64 or fewer; they do not keep shrinking as the router sharpens. Layer 1 keeps the largest pools.")
    ev = J.get("eval", {}); ref = J.get("ref_eval", {})
    steps = ["5000", "10000", "15000", "19074"]
    def m(d, st):
        v = d.get(st) if d.get(st) is not None else d.get(int(st))
        return sum(v.values()) / len(v) if isinstance(v, dict) else v
    rows = [["EMO, learned pools (routing with the predicted d)", *[f(m(ev, st), 3) for st in steps]],
            ["EMO, uniform pools [16, 512] (routing with all 512)", *[f(m(ref.get("olmoe3_275m_emo_10b", {}), st), 3) for st in steps]],
            ["standard MoE", *[f(m(ref.get("olmoe3_275m_10b", {}), st), 3) for st in steps]]]
    out += section("Validation CE during training",
        "Mean CE over the 11 v3-small validation sets (one padded instance per document) at the checkpoint steps; the learned-pool model "
        "routes each validation document with its predicted d, the uniform-pool EMO model with the full expert set.",
        table(["model", *[f"step {int(st):,}" for st in steps]], rows),
        "With pools of about 140 experts on average (60% of documents at 64 or fewer), the learned-pool model ends at the standard model's "
        "validation CE and 0.028 below the uniform-pool EMO model evaluated with all 512 experts. Train CE over the last 100 steps: 2.452 vs "
        "2.443 (uniform EMO) and 2.434 (standard).")
    H = ROOT / "sparse_experts/olmoe3_routing/runs/emo_learnedd"
    hp = H / "none/meta.json"; hf = H / "fixed512/meta.json"
    if hp.exists() or hf.exists():
        rows = []
        if hp.exists():
            mp = json.load(open(hp)); pp = mp.get("pred_pool_mean") or {}
            rows.append(["predicted d", f(mp["mean_ce"], 3), ", ".join(f"L{l} {v:.0f}" for l, v in sorted(pp.items(), key=lambda kv: int(kv[0])))])
        if hf.exists():
            rows.append(["all 512 experts", f(json.load(open(hf))["mean_ce"], 3), "512"])
        refs = {"EMO, uniform pools": ROOT / "sparse_experts/olmoe3_routing/runs/emo/none/meta.json", "standard MoE": ROOT / "sparse_experts/olmoe3_routing/runs/std/none/meta.json"}
        for lab, pth in refs.items():
            if pth.exists(): rows.append([lab, f(json.load(open(pth))["mean_ce"], 3), "512"])
        out += section("Held-out CE, both eval modes",
            "The same 65.5M unseen training-stream tokens as the rest of the report, through the final checkpoint routing with the predicted "
            "d per document and, separately, with every expert available (the fixed eval pool); mean token CE.",
            table(["routing", "held-out CE", "mean pool per layer (tokens)"], rows),
            "On the held-out sample the learned pools do better than opening every expert (2.460 vs 2.474): the model was trained with its "
            "pools and prefers them. Against the other arms it matches the uniform-pool EMO model (2.463) and stays 0.016 behind the standard "
            "model (2.444); the token-weighted mean pool is 320&ndash;360 experts because long documents get the large pools. "
            "Its document-level structure is stronger than the uniform-pool model's (Q1 purity at k = 4, layers 1/5/9: 0.52/0.60/0.55 vs "
            "0.34/0.51/0.54) but less aligned across layers (Q2 block agreement with layer 9, NMI 0.65&ndash;0.94 vs 0.85&ndash;1.00); both "
            "rows are in the Q1 and Q2 grids.")
    else:
        out += card("info", "Held-out CE", "<p>Routing passes on the held-out sample (predicted d and full pool) running.</p>")
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
        ("q4", QUESTIONS[3][1], build_q4()),
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
<div class="topbar"><nav>{nav}</nav><div id="subnav"></div></div><main>{sections}</main><script>{JS}</script>{CHART_JS}{SUBNAV_JS}</body></html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.html").write_text(page)
    print(f"Wrote {OUT/'report.html'} ({(OUT/'report.html').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
