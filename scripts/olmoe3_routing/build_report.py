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
            ("D", "EMO 512e, 8 sub-models", "#059669", "same as A with k = 8 blocks per layer"),
            ("E", "EMO 512e with pools {64, 512}, 4 sub-models", "#d97706", "same pipeline as A on the arm whose training pools were a random choice of 64 or 512 experts per document"),
            ("F", "Standard MoE 128e, random controls with 4 and 8 sub-models", "#0d9488", "the random-partition control of block B on a 128-expert model of the same expert size (top-16 of 128), jointly trained to 10B then continued to 130B as the baseline")]


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


def squares_block(tag, full_run, start_ppl_run, base_run, take_main="", take_pw="", stage0_take="", stage1_take=""):
    """Stage 0/1 tables + stages 2-4 charts + piecewise for an extra k=4 run (dirs olmoe3_squares_<tag>, runs_heldout20b_<tag>)."""
    SQd = ROOT / f"sparse_experts/olmoe3_squares_{tag}"; HRd = ROOT / f"sparse_experts/olmoe3_routing/runs_heldout20b_{tag}"
    inner = ""
    if (SQd / "groups.json").exists():
        G = json.load(open(SQd / "groups.json"))
        inner += section("Stage 0: the four block-groups", "Same spectral blocks and layer-9 alignment as in A, on this model's own routing pass.",
                         _groups_table(G, 4), stage0_take or "Experts per group in each layer; layer 1 keeps all 512 experts in every group.")
    if (SQd / "pack/stats.json").exists():
        st = json.load(open(SQd / "pack/stats.json"))
        inner += section("Stage 1: assigning the next 10B tokens", "Raw-count assignment as in A (each document goes to the group receiving most of its layer 2&ndash;9 selections).",
                         _assign_table(st, 4), stage1_take or f"Token-weighted, {100*st['in_group_share_token_weighted']:.0f}% of the selections a sub-model's documents make are inside its own expert group.")
    else:
        inner += card("info", "Stage 1", "<p>Assignment pass running.</p>")
    (OUT / f"olmoe3_squares_{tag}").mkdir(parents=True, exist_ok=True)
    inner += squares_results(HELD=HRd, PPL=SQd / "ppl_validation", SQO=OUT / f"olmoe3_squares_{tag}", start=f"{tag}_step19074", start_ppl=start_ppl_run,
                             base_runs=(base_run,), label="", take_main=take_main, take_pw=take_pw)
    return inner


WINDOW2_FT_TAKE = ("Joint finetuning (dotted, shaded) recovers about 40% of the merge loss after 0.5B tokens at either window end and adds little in the "
                   "next 0.5B; the merged model stays behind the baseline.")
STDRAND_TAKE = ("With no structure in the split the merge does <i>better</i> than with the routing split and keeps improving through the first two "
                "windows: 2.483 &rarr; 2.430 at 20B &rarr; 2.408 at 30B on the held-out sample, against 2.462 &rarr; 2.460 for the routing split and 2.351 for "
                "the baseline at 30B, its gap to the baseline staying near 0.05 while the routing split's grows to 0.11. The squares themselves are much weaker "
                "(2.63 &rarr; 2.50 &rarr; 2.47, below the start model until 30B, and the four are indistinguishable). Over the next 100B the picture changes: the "
                "merge stops improving at 2.37 (87B) and drifts back up to 2.38 at 130B while the squares keep improving (2.47 &rarr; 2.39) and the baseline "
                "pulls away (2.28 at 130B, a gap of 0.10), so by 130B a single random square is within 0.015 of the merge. Merging helps while the squares are alike; with "
                "enough separate training even random squares diverge until averaging their shared parameters stops paying, and the merged standard model "
                "stays behind the baseline throughout.")


def std_window2():
    """Block B continued to 30B tokens: separate training of the squares on the next 10B (window 2) vs the baseline continued,
    both windows re-evaluated on one held-out sample taken 300B tokens into the stream; x-axis = tokens trained. The joint
    finetunes from both window ends sit in a shaded region at the right; the random-partition control follows."""
    HR = ROOT / "sparse_experts/olmoe3_routing/runs_heldout300b_std"; PPL = ROOT / "sparse_experts/olmoe3_squares_std/ppl_validation"; PW = OUT / "olmoe3_squares_std_w2"
    if not HR.exists(): return ""
    W1 = [19074, 20000, 25000, 30000, 35000, 38148]; W2 = [39073, 44073, 49073, 54073, 57221]; steps = W1 + W2  # 19074 = start model / untrained slices merged
    toks = [round(st * 524288 / 1e9, 1) for st in steps]  # tokens trained (B) at each matched point
    hv = lambda tag: _ce(HR / f"{tag}/none"); pv = lambda f: _ppl(PPL / f)
    start = hv("std_step19074")
    b_h = [hv(f"baseline_step{st}") for st in steps]; m_h = [hv(f"merged_match{st}") for st in steps]
    ref_ppl = _ppl(ROOT / "claude_outputs/debug_validation/ppl_validation/olmoe3_275m_10b/step19074.json")
    def bppl(st):
        if st == 19074: return ref_ppl
        for run in ("olmoe3_275m_20b_1node", "olmoe3_275m_30b_1node", "olmoe3_275m_130b"):
            for s_ in (st, st - 1):
                if (PPL / run / f"step{s_}.json").exists(): return _ppl(PPL / run / f"step{s_}.json")
        return None
    b_p = [bppl(st) for st in steps]; m_p = [_ppl(PPL / "merged" / f"match{st}.json") for st in steps]
    if not any(v is not None for v in b_h + m_h): return card("info", "Window 2 (20B &rarr; 30B)", "<p>Evaluations on the 300B-token held-out sample running.</p>")
    # joint finetuning from both window ends: dotted branches right after the 20B and the 30B point (+0.5B, +1B of tokens)
    ft = {t: hv(t) for t in ("merged30_ft", "merged30_ft2", "baseline30_ft", "baseline30_ft2", "merged_ft", "merged_ft2", "baseline_ft", "baseline_ft2")}
    ftp = {"merged30_ft": pv("olmoe3_275m_merged30_ft/step58175.json"), "merged30_ft2": pv("olmoe3_275m_merged30_ft2/step59129.json"),
           "baseline30_ft": pv("olmoe3_275m_baseline30_ft/step58175.json"), "baseline30_ft2": pv("olmoe3_275m_baseline30_ft2/step59129.json"),
           "merged_ft": pv("olmoe3_275m_merged_ft/step39502.json"), "merged_ft2": pv("olmoe3_275m_merged_ft2/step40456.json"),
           "baseline_ft": pv("olmoe3_275m_baseline_ft/step39502.json"), "baseline_ft2": pv("olmoe3_275m_baseline_ft2/step40456.json")}
    has_ft = any(v is not None for v in ft.values())
    toks = [st * 524288 / 1e9 for st in steps]  # exact token counts, so the finetune x positions (20.5 / 21.0 / 30.5 / 31.0) stay distinct
    i20, i30 = steps.index(38148), steps.index(57221)
    ftx = [toks[i20] + 0.5, toks[i20] + 1.0, toks[i30] + 0.5, toks[i30] + 1.0] if has_ft else []
    xs_all = sorted(toks + ftx); pos = {x: i for i, x in enumerate(xs_all)}
    labels = [f"{x:.3g}B" if x in toks else "" for x in xs_all]
    def ser(b, m, F):
        col = lambda vals: [vals.get(x) for x in xs_all]
        out = [{"name": "baseline (full model, continued)", "y": col(dict(zip(toks, b)))}, {"name": "merged squares", "y": col(dict(zip(toks, m)))}]
        if has_ft:
            out += [{"name": "merged, joint finetuning (+0.5B, +1B)", "color": "#dc2626", "dotted": True,
                     "y": col({toks[i20]: m[i20], ftx[0]: F["merged_ft"], ftx[1]: F["merged_ft2"], toks[i30]: m[i30], ftx[2]: F["merged30_ft"], ftx[3]: F["merged30_ft2"]})},
                    {"name": "baseline, joint finetuning (+0.5B, +1B)", "color": "#2563eb", "dotted": True,
                     "y": col({toks[i20]: b[i20], ftx[0]: F["baseline_ft"], ftx[1]: F["baseline_ft2"], toks[i30]: b[i30], ftx[2]: F["baseline30_ft"], ftx[3]: F["baseline30_ft2"]})}]
        return out
    bands = [(toks[i20], ftx[1], "joint finetuning"), (toks[i30], ftx[3], "joint finetuning")] if has_ft else None
    charts = CHART_CSS + line_chart(xs_all, ser(b_h, m_h, ft) + [{"name": "start model (10B)", "y": [start] * len(xs_all), "const": True, "dashed": True, "color": "#64748b"}],
                                   title="Held-out CE (sample from 300B tokens into the stream)", xlabels=labels, x_label="tokens trained", shade=bands)
    if any(v is not None for v in b_p + m_p):
        charts += line_chart(xs_all, ser(b_p, m_p, ftp) + [{"name": "start model (10B)", "y": [ref_ppl] * len(xs_all), "const": True, "dashed": True, "color": "#64748b"}],
                             title="v3-small ppl sets, mean CE", xlabels=labels, x_label="tokens trained", shade=bands)
    out = section("Stages 2&ndash;4: merged squares vs continued baseline, 10B &rarr; 30B tokens, with joint finetuning",
        "The squares train separately on their own documents for 20B tokens in two 10B windows (stream steps 19,074&ndash;38,147, then "
        "38,147&ndash;57,221; the second window's documents assigned with the same rule and the same 10B start model, each square continuing from "
        "its own checkpoint), with checkpoints at five progress fractions per window; the baseline is the full model continued to 30B. "
        "<b>Held-out CE</b>: 7,993 instances taken 300B tokens into the stream (steps 572,205&ndash;572,605), unseen by every model. x-axis: tokens "
        "trained. <b>Dotted, shaded</b>: joint finetuning of the merged model (Adam moments merged like the weights) and of the baseline on the stream right "
        "after each window end, +0.5B = 954 steps and +1B = 1,908 steps at the same constant LR (from 20B: steps 38,548&ndash;40,456, the other "
        "blocks' finetuning tokens; from 30B: steps 57,221&ndash;59,129, seen by no square).",
        charts,
        "The merged model never beats the baseline and is flat at 2.46 from 15.7B to 30B (2.464 &rarr; 2.460) while the baseline keeps improving "
        "(2.399 &rarr; 2.351), so the gap widens only through the baseline's progress, from 0.065 at 15.7B to 0.11 at 30B: longer separate "
        "training does not make the merge worse in absolute terms. On the v3-small sets the merged model still improves slowly (2.970 &rarr; "
        "2.944) and the gap grows from 0.045 to 0.069. " + WINDOW2_FT_TAKE)
    pw = {st: d for st in steps if (d := _jload(PW / f"piecewise_match{st}.json")) and d.get("piecewise")}
    if pw:
        pxs = [st * 524288 / 1e9 for st in pw]; D = list(pw.values())
        avg = line_chart(pxs, [{"name": "piecewise (own square, no merge)", "y": [d["piecewise"] for d in D]},
                               {"name": "merged, free routing", "y": [d.get(f"merged_{d['name']}") for d in D]},
                               {"name": "baseline", "y": [d.get("baseline") for d in D], "dashed": True, "color": "#059669"},
                               {"name": "start model", "y": [D[0]["start_full"]] * len(D), "const": True, "dashed": True, "color": "#64748b"}],
                         title="All held-out documents (300B sample)", xlabels=[f"{t:.3g}B" for t in pxs], x_label="tokens trained")
        grp = ""
        for g in range(4):
            grp += line_chart(pxs, [{"name": "own square", "y": [d["sub_on_group"][str(g)][str(g)] for d in D]},
                                    {"name": "merged, free routing", "y": [(d.get(f"merged_{d['name']}_by_group") or {}).get(str(g)) for d in D]},
                                    {"name": "baseline", "y": [(d.get("baseline_by_group") or {}).get(str(g)) for d in D], "dashed": True, "color": "#059669"},
                                    {"name": "start model", "y": [D[0]["start_full_by_group"][str(g)]] * len(D), "const": True, "dashed": True, "color": "#64748b"}],
                              title=f"Group {g} documents ({D[0]['docs_per_group'][g]:,} held-out docs)", W=400, H=250, xlabels=[f"{t:.3g}B" for t in pxs], x_label="tokens trained")
        out += section("Where the merge loses: each square alone vs the merged model",
            "Piecewise CE (each held-out document scored by its own square, group = the start model's routing) vs the merged model on the same "
            "documents, over 10B &rarr; 30B tokens, on the 300B-token held-out sample.",
            CHART_CSS + avg + "<p><b>By document group</b>:</p>" + grp,
            "The standard squares on their own keep improving through window 2 (2.448 at 20B &rarr; 2.411 at 30B) but never reach the baseline "
            "(2.351 at 30B); the merged model sits above them at 2.46. So for the standard model both parts cost: each square is weaker than the "
            "full model on its own documents, and merging adds another 0.05 on top, both roughly constant over the second window.")
    out += random_control("std")
    return out


RANDOM_CONTROL = {  # random expert groups + random document split; arms = (sub-model count, dirs, colour), one entry per start model
    "std": dict(hrb="runs_heldout300b_std", start="std_step19074", ppl_dirs=("olmoe3_squares_std/ppl_validation",),
                base_runs=("olmoe3_275m_20b_1node", "olmoe3_275m_30b_1node", "olmoe3_275m_130b"), start_ppl="olmoe3_275m_10b", model="standard-routing 512e", E=512,
                arms=[dict(k=4, sqn="olmoe3_squares_stdrand", hr="runs_heldout300b_stdrand", color=None), dict(k=8, sqn="olmoe3_squares_stdrand8", hr="runs_heldout300b_stdrand8", color="#0d9488")],
                remerge=True, take=lambda: STDRAND_TAKE),
    "emo": dict(hrb="runs_heldout300b_emo", start="emo_step19074", ppl_dirs=("olmoe3_squares/ppl_validation", "olmoe3_squares_emorand/ppl_validation"),
                base_runs=("olmoe3_275m_emo_20b_1node", "olmoe3_275m_emo_20b", "olmoe3_275m_emo_30b_1node", "olmoe3_275m_emo_130b"), start_ppl="olmoe3_275m_emo_10b", model="EMO 512e", E=512,
                arms=[dict(k=4, sqn="olmoe3_squares_emorand", hr="runs_heldout300b_emorand", color=None), dict(k=8, sqn="olmoe3_squares_emorand8", hr="runs_heldout300b_emorand8", color="#0d9488")], remerge=False, take=lambda: EMORAND_TAKE),
    "s128": dict(hrb="runs_heldout300b_s128", start="s128_step19074", ppl_dirs=("olmoe3_squares_s128rand4/ppl_validation",),
                 base_runs=("olmoe3_275m_128e_130b",), start_ppl="olmoe3_275m_128e_10b", model="standard-routing 128e", E=128,
                 arms=[dict(k=4, sqn="olmoe3_squares_s128rand4", hr="runs_heldout300b_s128rand4", color=None), dict(k=8, sqn="olmoe3_squares_s128rand8", hr="runs_heldout300b_s128rand8", color="#0d9488")],
                 remerge=False, take=lambda: S128_TAKE),
}
W3_STEPS = (66481, 116479, 166478, 216477, 247956)
EMORAND_TAKE = "Square trainings running."
S128_TAKE = "Baseline and square trainings running."


def random_control(which):
    """Control for a squares block: K random expert groups of equal size and documents split uniformly at random, same 300B-token
    held-out sample. Squares are scored on ALL held-out documents (a random partition gives a held-out document no 'own' square),
    so the square line is the mean of the K (plus the best one). Several arms (K = 4, 8) share the charts."""
    C = RANDOM_CONTROL[which]; R = ROOT / "sparse_experts/olmoe3_routing"; HRB = R / C["hrb"]
    arms = [dict(a, HR=R / a["hr"], SQ=ROOT / "sparse_experts" / a["sqn"]) for a in C["arms"] if (ROOT / "sparse_experts" / a["sqn"] / "groups.json").exists()]
    if not arms: return ""
    hb = lambda tag: _ce(HRB / f"{tag}/none")
    W1 = [19074, 20000, 25000, 30000, 35000, 38148]; W2 = [39073, 44073, 49073, 54073, 57221]  # 19074 = start model / untrained slices merged
    def has(a, st): return _ce(a["HR"] / f"merged_match{st}/none") is not None or any(_ce(a["HR"] / f"sub{g}_match{st}/none") is not None for g in range(a["k"]))
    steps = W1 + [st for st in W2 if any(has(a, st) for a in arms)] + [st for st in W3_STEPS if any(has(a, st) for a in arms)]
    w2 = len(steps) > 6; w3 = len(steps) > 11
    start = hb(C["start"]); ref_ppl = _ppl(ROOT / f"claude_outputs/debug_validation/ppl_validation/{C['start_ppl']}/step19074.json")
    def bppl(st):
        if st == 19074: return ref_ppl
        for d in C["ppl_dirs"]:
            for run in C["base_runs"]:
                for s_ in (st, st - 1):
                    if (ROOT / "sparse_experts" / d / run / f"step{s_}.json").exists(): return _ppl(ROOT / "sparse_experts" / d / run / f"step{s_}.json")
        return None
    b_h = [hb(f"baseline_step{st}") for st in steps]; b_p = [bppl(st) for st in steps]
    for a in arms:  # per-arm series
        hv = lambda tag, a=a: _ce(a["HR"] / f"{tag}/none")
        a["mr"] = [hv(f"merged_match{st}") for st in steps]; sq = [[hv(f"sub{g}_match{st}") for g in range(a["k"])] for st in steps]
        a["sqm"] = [sum(v) / a["k"] if all(x is not None for x in v) else None for v in sq]
        a["sqb"] = [min(x for x in v if x is not None) if any(x is not None for x in v) else None for v in sq]
        a["mp"] = [_ppl(a["SQ"] / "ppl_validation" / "merged" / f"match{st}.json") for st in steps]
        a["lab"] = f", {a['k']} sub-models" if len(arms) > 1 else ""; a["stt"] = _jload(a["SQ"] / "pack/stats.json")
    arms = [a for i, a in enumerate(arms) if i == 0 or any(v is not None for v in a["mr"] + a["sqm"])]  # arms without any result yet stay out of the charts
    toks = [st * 524288 / 1e9 for st in steps]; labels = [f"{t:.3g}B" for t in toks]
    end = "130B" if w3 else ("30B" if w2 else "20B"); E = C["E"]; ks = " and ".join(str(a["k"]) for a in arms)
    groups_txt = " / ".join(f"{a['k']} groups of {E // a['k']}" for a in arms)
    setup = (f"Same {C['model']} start model (10B), same window" + (" and the same second window (stream steps 38,147&ndash;57,221, each square continuing from its own window-1 final on a fresh random quarter)" if w2 else "")
             + (", then a third window of 100B tokens (stream steps 57,221&ndash;247,956) in which each square continues on a contiguous slice of the stream (1/K of the "
                "window), a uniformly random share of its documents since the stream is a global shuffle, while the baseline continues to 130B" if w3 else "") + " (stream steps 19,074&ndash;38,147), same held-out sample, but the split has no "
             f"structure: in every layer 2&ndash;9 the {E} experts are shuffled into {groups_txt} (layer 1 whole, as before), and every document "
             f"of the window goes to a uniformly random group, so each of the K = {ks} squares owns 1/K of the experts and a random 1/K of the tokens. Squares are merged with equal weights. "
             "Because a held-out document has no 'own' square under a random split, the square line is the mean CE of the squares each scored "
             "on all held-out documents." + (" The same random groups and the same random document packs are used for every start model." if which != "std" else ""))
    title = f"Control: random expert groups, random document split (10B &rarr; {end})"
    if not any(v is not None for a in arms for v in a["mr"] + a["sqm"]):
        return section(title, setup, "", C["take"]())
    # re-merge + re-partition at 61.1B (standard control only): the 116,479 merge re-split into new random groups, trained on to 130B
    rm_h = rm_p = rm_sq = None; a0 = arms[0]
    RM = R / "runs_heldout300b_stdremerge"; RMP = ROOT / "sparse_experts/olmoe3_squares_stdremerge/ppl_validation"
    if C.get("remerge") and RM.exists() and 116479 in steps:
        rv = lambda tag: _ce(RM / f"{tag}/none"); i0 = steps.index(116479)
        rm_h = [None] * i0 + [a0["mr"][i0]] + [rv(f"merged_match{st}") for st in steps[i0 + 1:]]
        rm_p = [None] * i0 + [a0["mp"][i0]] + [_ppl(RMP / "merged" / f"match{st}.json") for st in steps[i0 + 1:]]
        rsq = [[rv(f"sub{g}_match{st}") for g in range(4)] for st in steps[i0 + 1:]]
        rm_sq = [None] * i0 + [a0["sqm"][i0]] + [sum(v) / 4 if all(x is not None for x in v) else None for v in rsq]
        if not any(v is not None for v in rm_h[i0 + 1:]): rm_h = rm_p = rm_sq = None
    rml = lambda y, name: [{"name": name, "y": y, "color": "#7c3aed"}] if y is not None else []
    ser_m = lambda key: [{"name": "merged squares" + a["lab"], "y": a[key], **({"color": a["color"]} if a["color"] else {})} for a in arms]
    charts = CHART_CSS + line_chart(toks, [{"name": "baseline (full model, continued)", "y": b_h}, *ser_m("mr"), *rml(rm_h, "merged squares, re-merged and re-partitioned at 61B"),
                                          {"name": "start model (10B)", "y": [start] * len(toks), "const": True, "dashed": True, "color": "#64748b"}],
                                   title="Held-out CE (300B sample)", xlabels=labels, x_label="tokens trained")
    if any(v is not None for a in arms for v in a["mp"]):
        charts += line_chart(toks, [{"name": "baseline (full model, continued)", "y": b_p}, *ser_m("mp"), *rml(rm_p, "merged squares, re-merged and re-partitioned at 61B"),
                                    {"name": "start model (10B)", "y": [ref_ppl] * len(toks), "const": True, "dashed": True, "color": "#64748b"}],
                             title="v3-small ppl sets, mean CE", xlabels=labels, x_label="tokens trained")
    sq_series = []
    for a in arms:
        c = {"color": a["color"]} if a["color"] else {}
        sq_series += [{"name": f"squares (mean of {a['k']}, no merge)", "y": a["sqm"], **({"dashed": True} if a["color"] else {}), **c},
                      *([{"name": "best square (no merge)", "y": a["sqb"], "color": "#d97706"}] if a is a0 else []),
                      {"name": "merged, free routing" + a["lab"], "y": a["mr"], **c}]
    charts += "<p><b>Where the merge stands against the squares</b> (each square scored on all held-out documents, mean of the K):</p>" + line_chart(toks,
        sq_series + rml(rm_h, "merged, re-partitioned at 61B") + ([{"name": "squares, re-partitioned at 61B (mean of 4)", "y": rm_sq, "color": "#7c3aed", "dashed": True}] if rm_sq is not None else [])
        + [{"name": "baseline", "y": b_h, "dashed": True, "color": "#059669"}, {"name": "start model", "y": [start] * len(toks), "const": True, "dashed": True, "color": "#64748b"}],
        title="All held-out documents (300B sample)", xlabels=labels, x_label="tokens trained")
    return section(title, setup, charts, C["take"]()) + routing_similarity_section(arms[0]["SQ"], C["model"], arms[0]["k"])


ROUTING_SIM_TAKE = {"olmoe3_squares_stdrand": (
    "The merged model's routing leaves the baseline's at once and then drifts slowly: after the first 0.5B of separate training it keeps 62% of the "
    "baseline's top-64 experts per document and 85% of its 90%-mass set (KL 0.16); by 130B these are 51% and 81% (KL 0.26), while the raw expert "
    "sets overlap 96&ndash;97% throughout because a standard-routing document touches almost every expert. And the merged model is not aligned with "
    "its squares at all, at any checkpoint: the most-used square carries 26% of a document's routing (uniform = 25%, the baseline reads 27% on the "
    "same random partition), the mean share of a token's 16 experts in the dominant square is 26%, and not a single token routes 13 or more of its "
    "16 experts into one square. With a random split the squares' experts are interleaved and the merged model spreads its routing across them "
    "exactly as the full model does, so the merge loss here is not a routing story: it is the slow divergence of the averaged shared parameters, "
    "the same thing the re-merge test probes.")}


def routing_similarity_section(SQ, model, k):
    """Routing of the merged random-split squares vs the continued baseline on the held-out sample across the matched checkpoints
    (routing_similarity.py): expert-set overlap (raw / top-64 / 90%-mass), KL, document-level square coverage, token-level square exclusivity."""
    J = _jload(OUT / SQ.name / "routing_similarity.json")
    if not J or not J.get("points"): return ""
    P = J["points"]; xs = [p["tokens_b"] for p in P]; labels = [f"{x:.3g}B" for x in xs]; A = [p["avg"] for p in P]
    g = lambda key: [a.get(key) for a in A]
    ch = CHART_CSS
    ch += line_chart(xs, [{"name": "all experts the baseline uses (&ge; 1 token)", "y": g("recall")}, {"name": "baseline's top-64 experts by routing mass", "y": g("recall_top64")},
                          {"name": "baseline's smallest set covering 90% of its routing", "y": g("recall_mass90")}],
                     title="Recall: share of the baseline's expert set that the merged model also uses (per document, mean)", y_label="recall", xlabels=labels, x_label="tokens trained")
    ch += line_chart(xs, [{"name": f"overlap of the top-64 sets (of 64)", "y": g("overlap_top64")}, {"name": "overlap of the 90%-mass sets", "y": g("overlap_mass90")},
                          {"name": "size of the baseline's 90%-mass set", "y": g("n_mass90"), "dashed": True, "color": "#64748b"}],
                     title="Absolute overlap (experts per document per layer, mean)", y_label="experts", xlabels=labels, x_label="tokens trained")
    ch += line_chart(xs, [{"name": "KL(baseline &#8214; merged), per-document routing distribution", "y": g("kl")}],
                     title="KL divergence of the routing distributions (nats, mean over documents)", y_label="KL", xlabels=labels, x_label="tokens trained")
    ch += line_chart(xs, [{"name": "merged squares", "y": g("coverage_merged")}, {"name": "baseline", "y": g("coverage_baseline")},
                          {"name": f"uniform over {k} squares", "y": [1 / k] * len(xs), "const": True, "dashed": True, "color": "#64748b"}],
                     title="Document-level share of routing on the document's most-used square (max over squares, mean)", y_label="share", xlabels=labels, x_label="tokens trained")
    ch += line_chart(xs, [{"name": "merged: mean share of a token's 16 experts in the document's dominant square", "y": g("token_merged_share")},
                          {"name": "merged: tokens with &ge; 80% (13/16) in the dominant square", "y": g("token_merged_ge80")},
                          {"name": "merged: tokens with &ge; 90% (15/16)", "y": g("token_merged_ge90")}, {"name": "merged: tokens routed exclusively (16/16)", "y": g("token_merged_exclusive")},
                          {"name": "baseline: mean share", "y": g("token_baseline_share"), "dashed": True, "color": "#059669"}],
                     title="Token-level alignment with the squares (first 200 held-out instances)", y_label="fraction", xlabels=labels, x_label="tokens trained")
    last = P[-1]; pl = last["per_layer"]
    rows = [[f"layer {l}", f(pl[str(l)]["recall"], 3), f(pl[str(l)]["recall_top64"], 3), f(pl[str(l)]["recall_mass90"], 3), f(pl[str(l)]["kl"], 3), f(pl[str(l)]["coverage_merged"], 3), f(pl[str(l)]["coverage_baseline"], 3),
             f(last["token"]["merged"][str(l)]["share"], 3)] for l in J["layers"]]
    tbl = table(["", "recall (all)", "recall (top-64)", "recall (90% mass)", "KL", "coverage merged", "coverage baseline", "token share merged"], rows)
    what = (f"For every held-out document (&ge; {J['min_tokens']} tokens; {last['n_docs']:,} documents) and every partitioned layer (2&ndash;9, mean over layers in the charts), the "
            f"routing of the merged squares is compared with the routing of the baseline at the same matched checkpoint; at 10B both are the start model. "
            "<b>Recall</b>: the fraction of the baseline's expert set for the document that the merged model also uses, for three definitions of the set: every expert "
            "that receives at least one of the document's tokens (a standard-routing document of ~1,150 tokens touches nearly all 512 experts, so this is weak), the "
            "baseline's 64 most-used experts, and its smallest set covering 90% of its routing. <b>KL</b>: KL(baseline &#8214; merged) of the per-document distribution of "
            "expert selections, the merged distribution smoothed by half a count per expert (so the 10B point reads 0.003 rather than 0). <b>Coverage</b>: the share "
            f"of a document's routing that falls on whichever of the {k} squares it uses most (1/{k} = no alignment). <b>Token level</b> (first 200 instances, "
            "~1,400 documents): for each token, how many of its 16 experts belong to the document's dominant square; the mean share and the fraction of tokens "
            "with at least 13, 15 or all 16 of them there.")
    return section("Routing of the merged squares vs the baseline on the held-out sample", what, ch + "<p><b>Per layer at the last checkpoint</b> (" + f"{last['tokens_b']:.3g}B):</p>" + tbl,
                   ROUTING_SIM_TAKE.get(SQ.name, "Analysis running."))


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
    present = ["A"] + (["B"] if (SQ.parent / "olmoe3_squares_std" / "groups.json").exists() else []) + (["F"] if (ROOT / "sparse_experts/olmoe3_squares_s128rand4/groups.json").exists() else []) \
              + (["C"] if (ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b_noemo").exists() else []) \
              + (["D"] if (ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b_k8").exists() else []) \
              + (["E"] if (ROOT / "sparse_experts/olmoe3_squares_pool64or512/groups.json").exists() else [])
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
    inner += random_control("emo")
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
        inner += std_window2()
        body += variant("B", inner)
    # ---- C: EMO squares without the EMO loss ----
    if (ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b_noemo").exists():
        inner = card("info", "Setup", "<p>Same EMO 512e partition, document assignment and sliced start checkpoints as A (stages 0 and 1 are identical), but the four "
                     "sub-models and the post-merge 0.5B finetune train with plain top-16 routing (no per-document pool rule, instance-level load "
                     "balancing). Compared against the same EMO baseline as A.</p>")
        inner += squares_results(HELD=ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b_noemo", PPL=ROOT / "sparse_experts/olmoe3_squares_noemo/ppl_validation", SQO=OUT / "olmoe3_squares_noemo",
                                 start="emo_step19074", start_ppl="olmoe3_275m_emo_10b", base_runs=("olmoe3_275m_emo_20b_1node",), label="",
                                 take_main="Dropping the EMO loss from the sub-models changes nothing: the merge beats the baseline only at 5% (2.415 vs 2.455), "
                                           "ends 0.15 above it (2.549 vs 2.398; A: 2.557) and finetunes back to 2.422 (A: 2.434).", take_pw="")
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
                                 start="emo_step19074", start_ppl="olmoe3_275m_emo_10b", base_runs=("olmoe3_275m_emo_20b_1node",), label="",
                                 take_main="With eight sub-models the merge is worse at every point: it never beats the baseline (2.442 vs 2.455 at 5%) and ends "
                                           "0.19 above it (2.587 vs 2.398; A with four: 2.557). Finetuning brings it to 2.445 / 2.444, close to A's 2.434 / 2.433.", take_pw="")
        body += variant("D", inner)
    # ---- E: pool-{64,512} arm, k = 4 ----
    if (ROOT / "sparse_experts/olmoe3_squares_pool64or512/groups.json").exists():
        body += variant("E", squares_block("pool64or512", "olmoe3_275m_emo_pool64or512_10b", "olmoe3_275m_emo_pool64or512_10b", "olmoe3_275m_pool64or512_20b_1node",
            take_main="Same shape as A: the merged model beats the baseline only at 5% (2.425 vs 2.457) and then drifts up while the baseline keeps "
                      "improving, ending 0.14 above it (2.535 vs 2.397), a slightly smaller gap than A's 0.16; finetuning brings it to 2.428 / 2.426.",
            take_pw="As in A, the squares on their own track and then beat the baseline (2.376 vs 2.397 at 100%), so the loss is in averaging the "
                    "diverged shared parameters, not in the partition or the sub-models.",
            stage1_take="46% of the selections a sub-model's documents make fall inside its own group (52% for the uniform-pool model in A)."))
    if (ROOT / "sparse_experts/olmoe3_squares_s128rand4/groups.json").exists():
        body += variant("F", card("info", "Setup", "<p>The 128-expert standard-routing model of the expert-count ladder (same expert size as the 512e model, top-16 of 128, "
                                  "so a quarter of the total parameters): its 10B checkpoint is continued jointly to 130B as the baseline, and split into 4 and into 8 random "
                                  "sub-models with the same random document groups as the 512e controls, at exactly the 512e controls' matched checkpoints.</p>") + random_control("s128"))
    return body


HELD = ROOT / "sparse_experts/olmoe3_routing/runs_heldout20b"; PPL = ROOT / "sparse_experts/olmoe3_squares/ppl_validation"
MATCH = [(19074, "match19074"), (20000, "match20000"), (25000, "match25000"), (30000, "match30000"), (35000, "match35000"), (38148, "match38148")]  # 19074 = the untrained slices merged (0%)


ROBUST_NOTE = []


def _jload(f):
    """json.load that returns None for a missing or partially written file (drivers write results while the report builds)."""
    try: return json.load(open(f))
    except (FileNotFoundError, json.JSONDecodeError): return None


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
    xlabels: explicit tick labels; shade=(x_from, label) tints the plot area from x_from rightwards (e.g. a finetuning stage);
    a list of (x_from, x_to, label) tints bounded bands."""
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
    for band in ([shade] if shade and not isinstance(shade, list) else (shade or [])):
        if len(band) == 3:  # (x_from, x_to, label): a bounded band, padded a little so the end points sit inside it
            xa, xb = X(band[0]) - 5, min(X(band[1]) + 8, x0 + pw)
        else:  # (x_from, label): from x_from (half-way to the previous x) to the right edge
            xa = (X(band[0]) + X(max(v for v in xs if v < band[0]))) / 2 if any(v < band[0] for v in xs) else X(band[0]); xb = x0 + pw
        g.append(f'<rect x="{xa:.1f}" y="{y0}" width="{xb-xa:.1f}" height="{ph}" fill="#f1f5f9"/>'
                 f'<text x="{(xa+xb)/2:.1f}" y="{y0+11}" font-size="10" text-anchor="middle" fill="#64748b">{band[-1]}</text>')
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
        c = s_.get("color") or _LC_COLORS[i % len(_LC_COLORS)]; dash = ' stroke-dasharray="2,3"' if s_.get("dotted") else (' stroke-dasharray="5,4"' if s_.get("dashed") else "")
        y = s_["y"]
        if s_.get("const"):
            v = next((v for v in y if v is not None), None)
            if v is None: continue
            g.append(f'<line x1="{x0}" x2="{x0+pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{c}" stroke-width="1.6"{dash}/>')
            y = [v] * len(xs)
        else:
            pts = [(X(x), Y(v)) for x, v in zip(xs, y) if v is not None]
            segs, cur = [], []  # a None breaks the line (a series may hold several separate branches)
            for x, v in zip(xs, y):
                if v is None: segs.append(cur); cur = []
                else: cur.append((X(x), Y(v)))
            segs.append(cur)
            for seg in segs:
                if len(seg) > 1: g.append(f'<polyline fill="none" stroke="{c}" stroke-width="2"{dash} points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in seg) + '"/>')
            for a, b in pts: g.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.2" fill="{c}"/>')
        data["series"].append({"n": s_["name"], "y": y, "c": c})
        legend.append(f'<span style="display:inline-block;margin-right:10px"><span style="display:inline-block;width:18px;border-top:2px {"dotted" if s_.get("dotted") else ("dashed" if s_.get("dashed") else "solid")} {c};vertical-align:middle"></span> {s_["name"]}</span>')
    g.append(f'<line class="lc-vline" x1="0" x2="0" y1="{y0}" y2="{y0+ph}" stroke="#94a3b8" stroke-dasharray="2,3" style="display:none"/></svg>')
    payload = html.escape(json.dumps(data), quote=True)
    return (f'<div class="lc" data-chart="{payload}"><div class="lc-title">{title}</div>' + "".join(g)
            + f'<div style="font-size:12px;margin-left:44px">{"".join(legend)}</div><div class="lc-tip"></div></div>')


EMO_TAKE = ("The merged model beats the baseline only at 5% (2.424 vs 2.455) and then falls behind monotonically, ending 0.16 above it "
            "(2.557 vs 2.398) and 0.10 above the start model on the held-out sample; the v3-small sets tell the same story. Finetuning the merge "
            "closes most of the gap (2.434 after 0.5B, 2.433 after 1B, vs the baseline's 2.396 / 2.395); the second 0.5B adds nothing. Retraining "
            "only the routers (orange) recovers about half of it (2.476, then 2.474), so roughly half of the merge loss is router mismatch that the "
            "routers can fix on their own, and the other half sits in the averaged non-router weights and needs the full model to train.")
EMO_PW_TAKE = ("The squares on their own track the baseline the whole way and end slightly below it (2.383 vs 2.398 at 100%), while the merged "
            "model drifts up to 2.557. Only at 5% is merging a gain (2.424 vs 2.509 piecewise): the four copies of the shared parameters are still "
            "nearly identical, so averaging is free and routing across groups adds experts. From 31% on, averaging diverged shared parameters is "
            "what costs, and the damage is concentrated on group 0, the code group: its own square improves from 1.56 to 1.45 while the merged "
            "model on the same documents worsens from 1.53 to 1.92. The partition and the sub-models are not the problem; averaging is.")
STD_TAKE = ("The standard merge never beats its baseline: 2.482 vs 2.436 at 5%, then flat around 2.46 while the baseline keeps improving to 2.374, "
            "ending 0.09 behind. Unlike EMO it does not degrade with more separate training.")
STD_PW_TAKE = ("For the standard model the squares themselves are the problem: on their own they start far behind (2.614 at 5% vs the start "
            "model's 2.441) and only get back to the start model by 100% (2.448), never near the baseline (2.374), because a quarter of "
            "token-level experts cannot serve a group's documents. Merging costs almost nothing here (2.463 at 100% vs 2.448 piecewise), the "
            "reverse of the EMO case, and is a gain early on while the squares are still weak.")


ROUTER_LR_TAKE = "<p><b>Takeaway.</b> Sticking to the original learning rate is best: 8e-4 and 2e-3 tie, and every other value is worse.</p>"


def squares_results(HELD=HELD, PPL=PPL, SQO=SQO, start="emo_step19074", start_ppl="olmoe3_275m_emo_10b", base_runs=("olmoe3_275m_emo_20b", "olmoe3_275m_emo_20b_filler", "olmoe3_275m_emo_20b_1node"), label="", take_main=EMO_TAKE, take_pw=EMO_PW_TAKE):
    ref_none = _ce(HELD / f"{start}/none")
    ref_ppl = _ppl(ROOT / f"claude_outputs/debug_validation/ppl_validation/{start_ppl}/step19074.json")
    xs, b_h, m_h, b_p, m_p = [], [], [], [], []
    for step, name in MATCH:
        frac = 100 * (step - 19074) / 19074; xs.append(round(frac))
        b_h.append(_ce(HELD / f"baseline_step{step}/none")); m_h.append(_ce(HELD / f"merged_{name}/none"))
        b_p.append(ref_ppl if step == 19074 else next((_ppl(PPL / run / f"step{s}.json") for run in base_runs for s in (step, step - 1) if (PPL / run / f"step{s}.json").exists()), None))  # final ckpt is step38147
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
    # router-only finetune of the 100% merge (block A): a third line that starts at the merged 100% point
    rft = {tag: _ce(HELD / f"merged_{tag}/none") for tag in ("rft", "rft2")}
    rft_ppl = {tag: _ppl(PPL / f"olmoe3_275m_emo_merged_{tag}" / f"step{st}.json") for tag, st in (("rft", 39502), ("rft2", 40456))}
    has_rft = any(v is not None for v in rft.values()) or any(v is not None for v in rft_ppl.values())
    def rline(m, a, b):
        if not has_rft: return []
        y = [None] * (len(xs) - 1) + [m[-1]] + ([a] if has_ft else []) + ([b] if has_ft2 else [])
        return [{"name": "merged squares, router-only finetune", "y": y, "color": "#d97706"}]
    charts = ""
    if have:
        charts = CHART_CSS + line_chart(xs_all, [{"name": "baseline (full model, continued)", "y": ext(b_h, ft["baseline_ft"], ft2["baseline_ft"])}, {"name": "merged squares", "y": ext(m_h, ft["merged_ft"], ft2["merged_ft"])},
                                                *rline(m_h, rft["rft"], rft["rft2"]),
                                                {"name": "start model (step 19,074)", "y": [ref_none] * len(xs_all), "const": True, "dashed": True, "color": "#64748b"}],
                                       title="Held-out CE (20B window, 65M tokens)", xlabels=labels, shade=shade)
        if any(v is not None for v in b_p + m_p):
            charts += line_chart(xs_all, [{"name": "baseline (full model, continued)", "y": ext(b_p, ppl_ft.get("baseline_ft"), ppl_ft.get("baseline_ft2"))}, {"name": "merged squares", "y": ext(m_p, ppl_ft.get("merged_ft"), ppl_ft.get("merged_ft2"))},
                                          *rline(m_p, rft_ppl["rft"], rft_ppl["rft2"]),
                                          {"name": "start model (step 19,074)", "y": [ref_ppl] * len(xs_all), "const": True, "dashed": True, "color": "#64748b"}],
                                 title="v3-small ppl sets, mean CE", xlabels=labels, shade=shade)
    ft_html = ""
    if has_ft:
        ft_html = ("<p><b>Finetuning</b> (shaded): the 100% merged model (Adam moments merged like the weights) and the baseline each trained on the same "
                   "0.5B more tokens (steps 38,548&ndash;39,502) at the same constant LR" + (", then on a further 0.5B (steps 39,502&ndash;40,456; 1B in total)" if has_ft2 else "")
                   + ". The held-out sample is steps 38,148&ndash;38,547 of the stream, so no finetuning token is in it."
                   + (" <b>Router-only finetune</b> (orange): the same tokens and LR from the 100% merge, but only the routed-expert routers "
                      "train (every other parameter at LR 0; verified after each stage by comparing the checkpoints tensor by tensor).</p>" if has_rft else "</p>"))
    if has_rft:
        rows = []
        for lr in ("2e-4", "8e-4", "2e-3", "4e-3", "8e-3", "1.6e-2"):
            tag, run, chk = ("merged_rft", "olmoe3_275m_emo_merged_rft", "check_rft.json") if lr == "8e-4" else (f"merged_rft_lr{lr}", f"olmoe3_275m_emo_merged_rft_lr{lr}", f"check_rft_lr{lr}.json")
            h = _ce(HELD / f"{tag}/none"); pp = _ppl(PPL / run / "step39502.json"); c = _jload(ROOT / "sparse_experts/olmoe3_squares/logs" / chk)
            if h is not None or pp is not None: rows.append((lr, h, pp, None if c is None else c.get("ok")))
        if len(rows) > 1:
            f3 = lambda v: "&ndash;" if v is None else f"{v:.3f}"; best = min((r[1] for r in rows if r[1] is not None), default=None)
            tr = "".join(f"<tr><td>{lr}</td><td>{('<b>%s</b>' % f3(h)) if h == best else f3(h)}</td><td>{f3(pp)}</td><td>{'yes' if ok else ('no' if ok is False else '&ndash;')}</td></tr>" for lr, h, pp, ok in rows)
            ref = (f"<tr><td colspan=4 style='color:#64748b'>reference: merged 100% without finetuning {f3(m_h[-1] if m_h else None)} / {f3(m_p[-1] if m_p else None)}; "
                   f"full finetune +0.5B {f3(ft['merged_ft'])} / {f3(ppl_ft.get('merged_ft'))}; baseline +0.5B {f3(ft['baseline_ft'])} / {f3(ppl_ft.get('baseline_ft'))}</td></tr>")
            ft_html += ("<p><b>Router-only finetune, learning-rate sweep</b> (+0.5B, same start, same tokens; the pretraining LR is 8e-4). Only 2.95M "
                        "of the 2.6B parameters train, but Adam moves each weight by about the LR per step whatever the gradient, so the question is "
                        "how far the routers may move before routing drifts.</p>"
                        "<table style='border-collapse:collapse;font-size:13px'><tr><th style='text-align:left;padding:2px 10px'>LR</th><th style='padding:2px 10px'>held-out CE</th>"
                        "<th style='padding:2px 10px'>v3-small CE</th><th style='padding:2px 10px'>only routers changed</th></tr>" + tr.replace("<td>", "<td style='padding:2px 10px;text-align:center'>") + ref + "</table>" + ROUTER_LR_TAKE)
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
                               {"name": "merged, free routing", "y": [d.get(f"merged_{d['name']}") for d in D]},
                               {"name": "baseline", "y": [d.get("baseline") for d in D], "dashed": True, "color": "#059669"},
                               {"name": "start model", "y": [D[0]["start_full"]] * len(D), "const": True, "dashed": True, "color": "#64748b"}],
                         title="All held-out documents")
        grp = ""
        for g in range(K):
            grp += line_chart(pxs, [{"name": "own square", "y": [d["sub_on_group"][str(g)][str(g)] for d in D]},
                                    {"name": "merged, free routing", "y": [(d.get(f"merged_{d['name']}_by_group") or {}).get(str(g)) for d in D]},
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


def _q4_sweep_rows(T, keys, parse, with_cov):
    rows = []
    for key in keys:
        r = T[key]; h = parse(key)
        dm = r.get("d_soft_mean") or {}; fr = r.get("d_frac_le64") or {}
        ds = [dm.get(str(l)) for l in range(2, 10)]; ds = [d for d in ds if d is not None]; d1 = dm.get("1")
        dtxt = ("16 in every layer" if ds and max(ds) < 16.5 and (d1 or 0) < 16.5 else f"{min(ds):.0f}&ndash;{max(ds):.0f} (L1 {d1:.0f})") if ds else "&mdash;"
        f5 = fr.get("5"); ev = r.get("eval_ce", {}).get("2000"); e2 = sum(ev.values()) / len(ev) if ev else None
        res = [dtxt, f"{100*f5:.0f}%" if f5 is not None else "&mdash;", f(r.get("train_ce_last100"), 3), f(e2, 3), str(r.get("skipped_steps", 0))]
        if h is None:
            hp = ["uniform EMO (control)"] + ["&mdash;"] * (5 if with_cov else 4)
        elif with_cov:
            hp = [f"{h['lambda']:g}", f"{h['cov']:g}", f"{h['lambda']/(h['cov']*496):.3f}", str(h["W"]), f"&times;{h['mult']:g}", f"{h['T']:g}"]
        else:
            hp = [f"{h['lambda']:g}", f"{h['T']:g}", str(h["W"]), f"&times;{h['mult']:g}"]
        rows.append(hp + res)
    return rows


def build_q4():
    body = question_card("q4")
    body += card("info", "Setup shared by everything below",
        "<ul>"
        "<li><b>Predict d.</b> In each MoE layer a small linear head reads the document's mean hidden state and outputs its pool size d, "
        "between 16 (the top-k) and 512 (all experts). Routing then works exactly as in EMO with that pool size: the document's d "
        "highest-scoring experts are the pool, tokens pick their top-16 inside it.</li>"
        "<li><b>Size penalty.</b> The loss gets an extra term &lambda;<sub>d</sub> &middot; (d &minus; 16)/496, averaged over documents, so "
        "every pool is pushed to shrink. Something must push back, otherwise every pool ends at 16; the two sections below are two ways to do that.</li>"
        "<li><b>Warm-up.</b> Training starts with all 512 experts and no penalty; over the first W steps the allowed pool shrinks to 16 and the "
        "penalty ramps up to &lambda;<sub>d</sub>.</li>"
        "<li><b>Head LR.</b> The head trains with a learning-rate multiplier, because at the base rate d barely moves (Adam steps a scalar by "
        "about one learning rate per step).</li>"
        "<li><b>Eval.</b> Route each document with its predicted d.</li>"
        "</ul>")
    tab = LD / "sweep_table.json"
    T = {r["run"].split("sweep_", 1)[1]: r for r in json.load(open(tab))} if tab.exists() else {}
    import re as _re
    def parse(key):
        if key == "control_uniform": return None
        m = _re.match(r"T([\d.]+)_l([\d.]+)_w(\d+)_m([\d.]+)(?:_coveragec([\d.]+))?", key)
        T_, l, w, mult, cov = m.groups()
        return {"T": float(T_), "lambda": float(l), "W": int(w), "mult": float(mult), "cov": float(cov) if cov else None}
    sweep_what = ("One run per row, all from scratch on the same 2000 steps (1.05B tokens) as the first 2000 steps of the 10B arms. "
                  "d = mean predicted pool at step 2000 over layers 2&ndash;9 (layer 1 in brackets); docs &le; 64 = share of documents whose layer-5 "
                  "pool is at most 64 experts; eval CE = mean over the 11 v3-small validation sets at step 2000, routing with the predicted d.")
    # ---- STE ----
    ste_text = (
        "<p><b>Idea.</b> Let the language-model loss itself decide the pool size: if the model predicts better with a few more experts, grow "
        "the pool; if the last experts admitted do nothing, let the penalty shrink it.</p>"
        "<p><b>The obstacle.</b> Membership in the pool is a hard yes/no (rank &le; d or not), and round(d) has no gradient, so the loss cannot "
        "tell d anything directly.</p>"
        "<p><b>The trick (straight-through estimator).</b> Give each expert a soft membership m<sub>e</sub> = sigmoid((d &minus; rank<sub>e</sub> + "
        "0.5) / T): about 1 deep inside the pool, about 0 far outside, sliding between the two over roughly T ranks around d. Multiply each "
        "selected expert's routing weight by m<sub>e</sub> + (hard<sub>e</sub> &minus; m<sub>e</sub>).detach(). In the forward pass this is the hard "
        "membership, always 1 for a selected expert, so predictions are exactly those of the hard pool. In the backward pass its derivative is "
        "that of m<sub>e</sub>, so the loss gradient flows into d through the 16 selected experts' weights, mostly through the ones ranked "
        "closest to d. That gradient is the only thing pushing against the size penalty.</p>"
        "<p><b>Loss</b>: L = L<sub>LM</sub> + EMO's usual auxiliary losses + &lambda;<sub>d</sub> &middot; size, with size = mean over documents of "
        "(d &minus; 16)/496. Knobs: &lambda;<sub>d</sub>, T, warm-up W, head LR. Default T = 2, W = 500, head LR &times;10.</p>")
    body += card("info", "Method 1: STE, let the language-model loss set d", ste_text)
    if T:
        keys = sorted([k for k in T if "coverage" not in k], key=lambda k: (0 if k == "control_uniform" else 1, (parse(k) or {}).get("lambda", 0), (parse(k) or {}).get("T", 0), (parse(k) or {}).get("W", 0), (parse(k) or {}).get("mult", 0)))
        body += section("STE sweep (2000 steps)", sweep_what,
            table(["&lambda;<sub>d</sub>", "T", "warm-up", "head LR", "d at step 2000", "docs &le; 64", "train CE", "eval CE", "skipped steps"], _q4_sweep_rows(T, keys, parse, False)),
            "Every arm collapses to d = 16 in every layer within a few hundred steps, whatever the penalty (even &lambda;<sub>d</sub> = 0), the "
            "temperature, the warm-up or the head LR, at a cost of +0.05 eval CE against the uniform-pool control. The gradient is the problem, not "
            "the tuning: raising d raises the soft membership of all 16 selected experts, the low-ranked ones most, and after the weights are "
            "re-normalised that shifts weight from the document's best experts to its marginal ones. The LM loss almost always prefers the "
            "opposite, so it asks for a smaller d no matter what. The excluded experts, the ones a larger pool would actually add, never enter the "
            "gradient at all.")
    # ---- coverage ----
    cov_text = (
        "<p><b>Why a second signal.</b> The STE gradient only ever sees the experts already in the pool, so it cannot know whether the experts "
        "just outside would help; it can only reshuffle weight among the selected ones, and that reshuffle happens to favour smaller pools. "
        "The counter-force therefore has to come from something that does see the excluded experts.</p>"
        "<p><b>Idea.</b> Use the router's own opinion. Averaging the router's softmax over a document's tokens gives each expert's share of that "
        "document's routing (3% for one expert, 0.5% for another; shares sum to 100%). A pool is too small if it leaves out experts that still "
        "carry a real share, so the loss charges the share left outside the pool: coverage = mean over documents of "
        "&Sigma;<sub>e</sub> share<sub>e</sub> &middot; (1 &minus; m<sub>e</sub>), with the same soft membership m<sub>e</sub> as before (this makes "
        "it differentiable in d; the shares are treated as constants). The language-model loss is left untouched: the hard pool is used with "
        "no straight-through term.</p>"
        "<p><b>Loss</b>: L = L<sub>LM</sub> + EMO's usual auxiliary losses + &lambda;<sub>d</sub> &middot; size + &lambda;<sub>cov</sub> &middot; "
        "coverage. Growing a pool by one expert lowers coverage by that expert's share and raises size by 1/496, so the loss is lowest when the "
        "pool holds exactly the experts whose share exceeds &lambda;<sub>d</sub> / (&lambda;<sub>cov</sub> &middot; 496): a per-document rule "
        "\"keep every expert with at least this much of my routing\". Focused documents get small pools, diffuse ones large pools. The sweep varies "
        "that threshold with &lambda;<sub>cov</sub> = 1 and the other knobs at their defaults.</p>")
    body += card("info", "Method 2: coverage, let the router set d", cov_text)
    if T:
        keys = sorted([k for k in T if "coverage" in k or k == "control_uniform"], key=lambda k: (0 if k == "control_uniform" else 1, (parse(k) or {}).get("lambda", 0)))
        body += section("Coverage sweep (2000 steps)", sweep_what,
            table(["&lambda;<sub>d</sub>", "&lambda;<sub>cov</sub>", "threshold", "warm-up", "head LR", "T", "d at step 2000", "docs &le; 64", "train CE", "eval CE", "skipped steps"], _q4_sweep_rows(T, keys, parse, True)),
            "The threshold sets the pool size as intended: 0.001 gives pools of 215&ndash;256 experts, 0.002 gives 110&ndash;130 with two thirds of "
            "the documents at 64 or fewer, 0.005 gives 23&ndash;50. At 0.002 the eval CE matches the uniform-pool control (3.710 vs 3.705) with "
            "pools a quarter of the size, so that setting is used for the 10B run below.")
    body += learnedd_10b()
    if (ROOT / "sparse_experts/olmoe3_squares_learnedd/groups.json").exists():
        body += VARIANT_CSS + '<div class="variant" id="q4-squares" style="--vc:#0891b2"><h2><span class="vb">S</span>Squares on the learned-pool model</h2>' \
            '<p class="lead">the Q3 pipeline (k = 4 block-groups, documents assigned by routing, four sub-models trained on their own documents of the next 10B, merged, ' \
            'compared with the full model continued; then 0.5B and 1B of post-merge finetuning) applied to the 10B learned-pool checkpoint. The sub-models keep the ' \
            'learned-d head and the coverage signal.</p>' \
            + squares_block("learnedd", "olmoe3_275m_emo_learnedd_10b", "olmoe3_275m_emo_learnedd_10b", "olmoe3_275m_learnedd_20b_1node",
                take_main="Same shape as the uniform-pool model in Q3 A: the merged model beats the baseline only at 5% (2.417 vs 2.453) and then "
                          "drifts up while the baseline keeps improving, ending 0.16 above it (2.549 vs 2.389); 0.5B of finetuning brings it to 2.424.",
                take_pw="The squares on their own track and then beat the baseline (2.365 vs 2.389 at 100%; every group's own square keeps improving), "
                        "so learned pools change nothing about the diagnosis: averaging the diverged shared parameters is what loses.",
                stage1_take="49% of the selections a sub-model's documents make fall inside its own group (52% for the uniform-pool model); the "
                            "learned pools give a partition of the same quality.") + "</div>"
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
            "rows are in the Q1 and Q2 grids. The spikes in its Q1 purity histograms are a quantisation effect: 27% of the held-out "
            "documents get a layer-5 pool of exactly 16 experts, so every token of such a document selects the same 16 experts and its purity "
            "is a multiple of 1/16 (7/16, 8/16, 9/16, &hellip;); no document of the uniform-pool model uses fewer than 65 experts.")
    else:
        out += card("info", "Held-out CE", "<p>Routing passes on the held-out sample (predicted d and full pool) running.</p>")
    return out


EXPLORER_CSS = """<style>
.xp .btn{display:inline-block;padding:4px 10px;margin:2px 6px 2px 0;border:1px solid #cbd5e1;border-radius:6px;background:#fff;cursor:pointer;font-size:13px}
.xp .btn.on{background:#2563eb;color:#fff;border-color:#2563eb}
.xp .meta{font-size:12px;color:#475569;margin:6px 0 10px}
.xp .doc{border:1px solid #e2e8f0;border-radius:6px;padding:8px 10px;margin:8px 0;background:#fff}
.xp .doc .hd{font-size:12px;color:#334155;margin-bottom:4px}.xp .doc .hd b{color:#0f172a}
.xp .doc .tag{display:inline-block;padding:0 6px;border-radius:4px;background:#eef2f7;margin-right:6px;font-size:11px}
.xp .doc pre{white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.35;margin:0;max-height:220px;overflow:hidden;font-family:ui-monospace,Menlo,Consolas,monospace;color:#1e293b}
.xp .doc pre.open{max-height:none}.xp .doc .more{font-size:12px;color:#2563eb;cursor:pointer;margin-top:4px}
.xp table{font-size:12px}
</style>"""
EXPLORER_JS = """<script>(function(){
const D=JSON.parse(document.getElementById('explorer-data').textContent);
const esc=t=>t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function card(d,showPools){const pools=showPools&&d.pools?'<span class="tag">pools L1..L9: '+d.pools.join(' / ')+'</span>':'';
 return '<div class="doc"><div class="hd"><span class="tag">'+esc(d.family)+'</span><b>'+esc(d.source)+'</b> &middot; '+d.tokens+' tokens &middot; in-group share '+d.share.toFixed(2)+' '+pools+'</div>'
 +'<pre>'+esc(d.snippet)+'</pre>'+(d.full.length>d.snippet.length?'<div class="more" data-full="'+esc(d.full).replace(/"/g,'&quot;')+'">show more</div>':'')+'</div>';}
function wire(root){root.querySelectorAll('.more').forEach(m=>m.addEventListener('click',()=>{const pre=m.previousElementSibling;if(pre.classList.contains('open')){pre.textContent=m.dataset.snip;pre.classList.remove('open');m.textContent='show more';}else{m.dataset.snip=pre.textContent;pre.textContent=m.dataset.full;pre.classList.add('open');m.textContent='show less';}}));}
// section 1: documents by sub-model
const s1=document.getElementById('xp-groups'); if(s1&&D.models){const keys=Object.keys(D.models); let mk=keys[0], gi=0;
 const mb=document.getElementById('xp-model'), gb=document.getElementById('xp-group'), body=document.getElementById('xp-groups-body');
 function render(){mb.innerHTML=keys.map(k=>'<span class="btn'+(k===mk?' on':'')+'" data-k="'+k+'">'+esc(D.models[k].label)+'</span>').join('');
  const M=D.models[mk]; gb.innerHTML=M.groups.map((g,i)=>'<span class="btn'+(i===gi?' on':'')+'" data-i="'+i+'">group '+g.group+' &middot; '+(100*g.token_share).toFixed(0)+'% of tokens</span>').join('');
  const g=M.groups[gi]; const fam=Object.entries(g.families).map(([f,v])=>f+' '+(100*v).toFixed(0)+'%').join(', ');
  body.innerHTML='<div class="meta">'+g.n_docs.toLocaleString()+' held-out documents; mean in-group share '+g.mean_share.toFixed(2)+' (assignment rule: '+(M.size_normalized?'selections per expert of the group':'raw count of selections')+'); experts per layer '+Object.entries(g.experts_per_layer).map(([l,n])=>'L'+l+':'+n).join(' ')+'<br>document families: '+fam+'<br>'+g.docs.length+' random documents (at least 64 tokens), most typical first:</div>'+g.docs.map(d=>card(d,mk==='learnedd')).join('');
  mb.querySelectorAll('.btn').forEach(b=>b.addEventListener('click',()=>{mk=b.dataset.k;gi=0;render();})); gb.querySelectorAll('.btn').forEach(b=>b.addEventListener('click',()=>{gi=+b.dataset.i;render();})); wire(body);}
 render();}
// section 2: learned pool size per document
const s2=document.getElementById('xp-pools'); if(s2&&D.buckets){const B=D.buckets; let bi=0; const bb=document.getElementById('xp-bucket'), body=document.getElementById('xp-pools-body');
 function render(){bb.innerHTML=B.buckets.map((b,i)=>'<span class="btn'+(i===bi?' on':'')+'" data-i="'+i+'">'+b.name+' &middot; '+(100*b.frac_docs).toFixed(0)+'% of docs</span>').join('');
  const b=B.buckets[bi]; const fam=Object.entries(b.families).map(([f,v])=>f+' '+(100*v).toFixed(0)+'%').join(', ');
  body.innerHTML='<div class="meta">'+b.n_docs.toLocaleString()+' documents ('+(100*b.frac_docs).toFixed(1)+'% of documents, '+(100*b.frac_tokens).toFixed(1)+'% of tokens); mean length '+(b.mean_tokens||0).toFixed(0)+' tokens, median '+(b.median_tokens||0).toFixed(0)+'<br>document families: '+fam+'<br>'+b.docs.length+' random documents:</div>'+b.docs.map(d=>card(d,true)).join('');
  bb.querySelectorAll('.btn').forEach(x=>x.addEventListener('click',()=>{bi=+x.dataset.i;render();})); wire(body);}
 render();}
})();</script>"""


def build_explorer():
    fp = OUT / "explorer.json"
    if not fp.exists():
        return card("warn", "Explorer", "<p>Explorer data not built yet.</p>")
    D = json.load(open(fp)); payload = open(fp).read().replace("</", "<\\/")
    body = card("info", "What this shows",
        "<p>The held-out 20B-window documents (7,991 instances, 56,547 documents from training-stream steps 38,148&ndash;38,547, unseen by every "
        "model) as the Q3 pipeline partitions them into the four sub-models, for three start models. Each document goes to the group receiving "
        "most of its layer 2&ndash;9 top-16 selections in that model (per expert of the group for the standard model). Text is decoded with the "
        "dolma2 tokenizer; the source label is the training mix's own label for the document's shard.</p>")
    body += card("info", "Documents by sub-model",
        '<div class="xp" id="xp-groups"><div id="xp-model"></div><div id="xp-group"></div><div id="xp-groups-body"></div></div>')
    if D.get("buckets"):
        B = D["buckets"]
        fam_rows = [[fam, f"{v['n']:,}", f(v["mean_pool"], 0), f(v["median_pool"], 0), f"{100*v['frac_le16']:.0f}%", f"{100*v['frac_le64']:.0f}%", f(v["mean_tokens"], 0)] for fam, v in B["by_family"].items()]
        len_rows = [[r, f"{v['n']:,}", f(v["mean_pool"], 0), f"{100*v['frac_le16']:.0f}%", f"{100*v['frac_le64']:.0f}%"] for r, v in B["by_length"].items()]
        body += card("info", "Learned pool size per document (EMO 512e, learned pools)",
            "<p>The learned-pool model's predicted pool size for each held-out document (one prediction per MoE layer; here the mean over "
            f"layers 2&ndash;9, {B['n_docs']:,} documents with at least 64 tokens). Mean pool per layer L1&ndash;L9: {' / '.join(str(v) for v in B['layer_mean'])}.</p>"
            "<p>Pool size is almost entirely a function of document length, not topic: documents of 64&ndash;128 tokens get a mean pool of 17 "
            "(72% exactly 16), 128&ndash;512 tokens 21, 512&ndash;2,048 tokens 164, and longer documents 482. That is what the coverage rule implies: "
            "a short document's tokens touch few experts, so few experts carry more than 0.2% of its routing; a long document spreads its routing "
            "over most of them. Within a length band the families differ little; PDFs get the largest pools because they are the longest documents.</p>"
            "<p><b>By document family</b> (mean and median pool, share of documents whose mean pool is 16 or at most 64):</p>"
            + table(["family", "documents", "mean pool", "median pool", "pool = 16", "pool &le; 64", "mean tokens"], fam_rows)
            + "<p><b>By document length</b>:</p>" + table(["tokens", "documents", "mean pool", "pool = 16", "pool &le; 64"], len_rows)
            + '<p><b>Documents by pool-size bucket</b>:</p><div class="xp" id="xp-pools"><div id="xp-bucket"></div><div id="xp-pools-body"></div></div>')
    else:
        body += card("info", "Learned pool size per document", "<p>Per-document pool pass running.</p>")
    return EXPLORER_CSS + body + f'<script id="explorer-data" type="application/json">{payload}</script>' + EXPLORER_JS


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
        ("explorer", "Explorer", build_explorer()),
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
