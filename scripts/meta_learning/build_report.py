# PARENT: "scripts/modular_extension/build_report.py" (visual framework: CSS/JS/tab structure,
#          card/table/figure helpers kept in sync)
# DESCRIPTION:
#     Builds the meta_learning experiment report at claude_outputs/meta_learning/report.html:
#     goals, the (corrected) two-pass FOMAML algorithm with working-set outer updates, the
#     implementation map, the bf16 pseudo-step quantization finding, the documented negative
#     result of the first launch, the verification gate, and live run status. Figures and run
#     data come from claude_outputs/meta_learning/{figs,report_data.json}, produced by
#     scripts/meta_learning/fetch_report_data.py — rerun that first to refresh.
#
#   python scripts/meta_learning/fetch_report_data.py && python scripts/meta_learning/build_report.py
##############################################################

import base64
import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "claude_outputs/meta_learning"
FIGS = OUT / "figs"

TOKENS_PER_STEP = 1024 * 4096

# --------------------------------------------------------------------------
# HTML helpers (kept in sync with scripts/modular_extension/build_report.py)
# --------------------------------------------------------------------------


def card(kind: str, title: str, body: str) -> str:
    return f'<div class="card {kind}"><h3>{title}</h3>{body}</div>'


def table(headers: list, rows: list) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def img_tag(path: Path, caption: str) -> str:
    if not path.is_file():
        return f'<p class="missing">[missing figure: {path.name}]</p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        "<figure>"
        f'<img loading="lazy" src="data:image/png;base64,{data}" alt="{html.escape(caption)}" '
        'onclick="this.classList.toggle(\'zoom\')" title="click to zoom">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def fig_row(*figs: str) -> str:
    return '<div class="figrow">' + "".join(figs) + "</div>"


# --------------------------------------------------------------------------
# The train-step diagram (inline SVG; accents match the card palette)
# --------------------------------------------------------------------------

STEP_SVG = """
<svg viewBox="0 0 960 300" style="max-width:960px;width:100%;background:#fff;border:1px solid #e2e8f0;border-radius:8px" role="img" aria-label="meta_learning train step diagram">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#64748b"/>
    </marker>
  </defs>
  <style> text { font-family:-apple-system,'Segoe UI',Roboto,sans-serif; fill:#1e293b; } .t{font-size:13px;font-weight:600} .s{font-size:11.5px;fill:#475569} .m{font-size:11px;fill:#64748b;font-style:italic}</style>

  <rect x="18" y="112" width="120" height="64" rx="8" fill="#f1f5f9" stroke="#94a3b8"/>
  <text x="78" y="138" text-anchor="middle" class="t">batch</text>
  <text x="78" y="156" text-anchor="middle" class="s">1024 seqs &middot; 4.2M tok</text>

  <rect x="176" y="96" width="180" height="96" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="266" y="122" text-anchor="middle" class="t">1&#8226; inner pass (selective)</text>
  <text x="266" y="141" text-anchor="middle" class="s">fwd+bwd, each doc routed</text>
  <text x="266" y="157" text-anchor="middle" class="s">only within its top-32 experts</text>
  <text x="266" y="177" text-anchor="middle" class="s">&rarr; g_inner (DP-averaged)</text>

  <rect x="394" y="96" width="180" height="96" rx="8" fill="#f5f3ff" stroke="#7c3aed"/>
  <text x="484" y="122" text-anchor="middle" class="t">2&#8226; pseudo-step</text>
  <text x="484" y="142" text-anchor="middle" class="s">&theta;&prime;_exp = &theta;_exp &minus; &alpha;&middot;g_inner</text>
  <text x="484" y="158" text-anchor="middle" class="s">experts only &middot; SGD &middot; clipped</text>
  <text x="484" y="176" text-anchor="middle" class="s">temporary (undone in 4)</text>

  <rect x="612" y="96" width="196" height="96" rx="8" fill="#ecfdf5" stroke="#059669"/>
  <text x="710" y="122" text-anchor="middle" class="t">3&#8226; outer pass (full)</text>
  <text x="710" y="141" text-anchor="middle" class="s">same tokens, all 128 experts,</text>
  <text x="710" y="157" text-anchor="middle" class="s">at &theta;&prime; &middot; backward MASKED:</text>
  <text x="710" y="176" text-anchor="middle" class="s">expert grads &rarr; working set only</text>

  <rect x="846" y="112" width="100" height="64" rx="8" fill="#f1f5f9" stroke="#94a3b8"/>
  <text x="896" y="138" text-anchor="middle" class="t">4&#8226; AdamW</text>
  <text x="896" y="156" text-anchor="middle" class="s">applied at &theta;</text>

  <line x1="138" y1="144" x2="172" y2="144" stroke="#64748b" stroke-width="1.6" marker-end="url(#arr)"/>
  <line x1="356" y1="144" x2="390" y2="144" stroke="#64748b" stroke-width="1.6" marker-end="url(#arr)"/>
  <line x1="574" y1="144" x2="608" y2="144" stroke="#64748b" stroke-width="1.6" marker-end="url(#arr)"/>
  <line x1="808" y1="144" x2="842" y2="144" stroke="#64748b" stroke-width="1.6" marker-end="url(#arr)"/>

  <path d="M 896 176 C 896 240 484 250 484 196" fill="none" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="6 4" marker-end="url(#arr)"/>
  <text x="690" y="238" text-anchor="middle" class="m">expert weights restored bitwise before the AdamW step</text>

  <text x="266" y="80" text-anchor="middle" class="m">only source of the meta signal</text>
  <text x="710" y="80" text-anchor="middle" class="m">the only REAL update of the step</text>

  <text x="480" y="282" text-anchor="middle" class="s">gradient kept by the AdamW step:&#160;&#160;expert weights &larr; working-set slots of g_outer(&theta;&prime;)&#160;&#160;&middot;&#160;&#160;attention / embeddings / router / norms &larr; full g_outer(&theta;&prime;)</text>
</svg>
"""


# --------------------------------------------------------------------------
# Tab bodies
# --------------------------------------------------------------------------


def _gap(evals: dict) -> float:
    full = [v for k, v in evals.items() if "/lm-full/" in k and not k.endswith("all/CE loss")]
    p32 = [v for k, v in evals.items() if "/lm-pool32/" in k and not k.endswith("all/CE loss")]
    if not full or not p32:
        return float("nan")
    return statistics.mean(p32) - statistics.mean(full)


# Fixed arm order + colors (dataviz reference categorical palette, light mode).
ARM_COLORS = [
    ("vanilla", "#2a78d6"),
    ("sametok λ=0 (degenerate)", "#eb6834"),
    ("sametok_ws", "#1baf7a"),
    ("heldout_ws", "#eda100"),
    ("sametok_ws_lam05", "#e87ba4"),
    ("rpool (canceled)", "#008300"),
    ("heldout_ws_lam05", "#4a3aa7"),
    ("seq_ws", "#e34948"),
]


def build_curves_tab(curves: dict) -> tuple:
    """Interactive W&B curve explorer (uPlot; display pattern per scripts/models_v2). Returns
    (tab_html, payload_json) — payload '' if no curves cached."""
    if not curves:
        return card("results", "Live curves", '<p class="missing">[no curves.json cached]</p>'), ""
    arms = [(a, c) for a, c in ARM_COLORS if a in curves]
    legend = " &middot; ".join(
        f'<span style="color:{c};font-weight:600">{html.escape(a)}</span>'
        f' <span class="note">({curves[a]["state"]}, step {curves[a]["last_step"]})</span>'
        for a, c in arms
    )
    body = f"""
<p>Every arm's W&B history on a shared tokens axis (renamed/crashed runs stitched per arm;
train-side series EMA-smoothed, &alpha;=0.85). Click legend entries to toggle arms; drag to
zoom, double-click to reset. Metrics: <b>train CE</b> (each arm's own training loss — NOT
directly comparable across arms: vanilla's is random-pool, ws arms' is their outer pass,
sequential's is its second step); <b>inner CE</b> (the selective-pass loss at &theta; — leaning
transients show as inner&#8811;outer); <b>eval CE (full)</b> and <b>gap</b> (pool32&minus;full,
the selective-inference metric; sparse points every 500 steps).</p>
<div id="mlcurves">
  <div style="margin:8px 0">
    <label><input type="radio" name="mlmetric" value="ce" checked> train CE</label>&nbsp;&nbsp;
    <label><input type="radio" name="mlmetric" value="ice"> inner CE</label>&nbsp;&nbsp;
    <label><input type="radio" name="mlmetric" value="evalfull"> eval CE (full)</label>&nbsp;&nbsp;
    <label><input type="radio" name="mlmetric" value="gap"> gap (pool32&minus;full)</label>&nbsp;&nbsp;
    <button class="mllog" type="button">toggle log y</button>
  </div>
  <div class="chart" style="min-height:440px"></div>
  <p class="note">{legend}</p>
</div>"""
    payload = json.dumps(
        {
            "arms": [
                {
                    "name": a,
                    "color": c,
                    "ce": [curves[a]["ce_x"], curves[a]["ce_y"]],
                    "ice": [curves[a]["ice_x"], curves[a]["ice_y"]],
                    "evalfull": [curves[a]["eval_x"], curves[a]["eval_full"]],
                    "gap": [curves[a]["eval_x"], curves[a]["eval_gap"]],
                }
                for a, c in arms
            ]
        }
    )
    return card("results", "Live curves (all arms, pulled from W&B)", body), payload


CURVES_JS = """
(function(){
  const DATA = __MLCURVES__;
  const root = document.getElementById('mlcurves');
  if (!root || !DATA.arms) return;
  const el = root.querySelector('.chart');
  let plot = null, metric = 'ce', logy = false;
  function draw(){
    if (plot) { plot.destroy(); plot = null; }
    el.innerHTML = '';
    const armSeries = DATA.arms.filter(a => a[metric] && a[metric][0].length > 0);
    if (!armSeries.length) { el.innerHTML = '<p class="missing">[no data for this metric yet]</p>'; return; }
    const xs = [...new Set([].concat(...armSeries.map(a => a[metric][0])))].sort((p,q)=>p-q);
    const idx = new Map(xs.map((x,i)=>[x,i]));
    const cols = armSeries.map(a => {
      const y = new Array(xs.length).fill(null);
      a[metric][0].forEach((x,i)=>{ y[idx.get(x)] = a[metric][1][i]; });
      return y;
    });
    const w = Math.min(el.clientWidth || 960, 1100);
    const opts = { width: w, height: 430,
      scales: { x: { time: false }, y: { distr: (logy && metric !== 'gap') ? 3 : 1 } },
      cursor: { drag: { x: true, y: true, uni: 10 }, focus: { prox: 30 } },
      axes: [ { label: 'tokens (B)' }, { label: {ce:'train CE',ice:'inner (selective) CE',evalfull:'held-out CE, full pool',gap:'pool32 - full CE (nats)'}[metric] } ],
      series: [ { value: (u,v)=>v==null?'--':(+v).toFixed(2) } ].concat(armSeries.map(a => ({
        label: a.name, stroke: a.color, width: 1.6, spanGaps: true,
        points: { show: metric === 'gap' || metric === 'evalfull', size: 6 },
        value: (u,v)=>v==null?'--':(+v).toFixed(4) }))) };
    plot = new uPlot(opts, [xs].concat(cols), el);
    el.addEventListener('dblclick', () => plot && plot.setScale('x', {min: xs[0], max: xs[xs.length-1]}));
  }
  root.querySelectorAll('input[name=mlmetric]').forEach(r =>
    r.addEventListener('change', e => { metric = e.target.value; draw(); }));
  root.querySelector('.mllog').addEventListener('click', () => { logy = !logy; draw(); });
  draw();
})();
"""


def build_overview(data: dict) -> str:
    goal = card(
        "goal",
        "Goal",
        "<p>Pretrain a 128-expert EMO model (the first 100B-token phase, from scratch) so that "
        "<b>training on only 32 experts effectively improves the full model</b>. Each document "
        "trains only its own 32-expert working set — but toward performing well <i>inside the "
        "full 128-expert model, one selective step ahead of where it currently is</i> "
        "(first-order MAML).</p>"
        "<p>Motivation: the modular_extension <i>k=32 CPT (no extension)</i> result. On a "
        "vanilla-EMO model, post-hoc per-cluster 32-expert CPT <b>worked in the selective view</b> "
        "— right after its own stage, the trained cluster's 32-expert working set improved by "
        "&minus;0.164 nats (~6&times; the 30B plain-CPT baseline's ordinary &minus;0.027) — but "
        "<b>almost none of it transferred to the full model</b>: the full-pool view of the very "
        "same weights showed only &minus;0.022 on the trained cluster (full routing dilutes what "
        "the stage learned across experts it was never co-adapted with), and the pool degraded "
        "broadly elsewhere (+0.08 nats mean), with later stages overwriting earlier ones. Vanilla "
        "EMO is excellent at selective <i>inference</i>; what fails is selective <i>update "
        "transfer</i>. This experiment builds that property into pretraining itself.</p>",
    )
    rows = []
    for name, desc, note in [
        (
            "meta128_vanilla_100b",
            "vanilla EMO-128e baseline (random pools 8\u2013128)",
            "DONE past 20B (paused at 31.5B); 10B/20B fixed ckpts; gap \u22120.04",
        ),
        (
            "meta128_sametok_100b",
            "same-tokens FOMAML, \u03bb=0, ALL-expert outer update",
            "STOPPED \u2014 degenerate: gap +0.71 (see Negative results)",
        ),
        (
            "meta128_sametok_ws",
            "same-tokens FOMAML, working-set outer, \u03bb=0",
            "FINISHED 20B; gap stuck ~+0.31; meta term extinguished by bf16 late in the run",
        ),
        (
            "meta128_heldout_ws",
            "cross-data (heldout) FOMAML, working-set, \u03bb=0",
            "FINISHED 20B; gap ~+0.33; leaning-immune but half the update signal",
        ),
        (
            "meta128_sametok_ws_lam05",
            "same-tokens ws + \u03bb=0.5 anchor (+LB on inner)",
            "TRAINING; gap \u2248+0.01 and full CE ahead of vanilla at matched steps",
        ),
        (
            "meta128_sametok_ws_rpool",
            "ws + randomized outer pool [8,128]",
            "CANCELED per user at ~step 1.1k (deep leaning transient; unproven)",
        ),
        (
            "meta128_heldout_ws_lam05",
            "heldout ws + \u03bb=0.5 anchor",
            "QUEUED \u2014 anchored cross-data meta; both halves carry update signal",
        ),
        (
            "meta128_seq_ws",
            "SEQUENTIAL ablation: committed selective step \u2192 ws-masked full step, same tokens",
            "QUEUED \u2014 isolates what the FOMAML probe adds; NOTE 2\u00d7 optimizer updates per token",
        ),
    ]:
        d = data.get(name, {})
        rows.append(
            [f"<code>{name}</code>", desc, d.get("state", "\u2014"), d.get("step", "\u2014"), note]
        )
    status = card(
        "results",
        "Arms",
        table(["run", "objective", "state", "last step", "notes"], rows)
        + "<p class='note'>All arms: 8 nodes \u00d7 8 H100, <b>20B tokens</b> (4,768 steps; budget cut "
        "2026-08-17 from the original 100B), fixed checkpoints at 10B/20B (steps 2384/4768), "
        "OLMoE-mix-0824, lr 2e-3 flat WSD (warmup 2000), lb 1e-1, pool min=8/max=128/eval=128, "
        "W&amp;B project <code>emo-extension</code> tag <code>meta_learning</code>. Phase 2 = "
        "k=32-CPT-style cluster-wise selective CPT on tokens 20B\u201340B; the document window is "
        "already extracted (18.3M docs / 36.9B doc-tokens in "
        "<code>meta_learning/data/meta128_20B-40B/</code>). The sequential arm counts one trainer "
        "step per token batch but takes two real optimizer updates inside it.</p>",
    )
    metric = card(
        "method",
        "Headline metrics",
        "<ul>"
        "<li><b>Selective-vs-full CE gap</b>: the same held-out ppl mix evaluated twice every 500 "
        "steps — <code>eval/lm-full</code> (eval pool 128) vs <code>eval/lm-pool32</code> (pool "
        "pinned to 32). Measures selective <i>inference</i> competence.</li>"
        "<li><b>Downstream (the real test)</b>: rerun the modular_extension k=32 CPT protocol on "
        "the finished checkpoints — does selective-expert CPT transfer now? Measures selective "
        "<i>update</i> transfer, which the gap alone does not.</li>"
        "<li>Per-step diagnostics: inner CE, pseudo-step delta/weight norm, inner/outer grad "
        "cosine, and the <i>measured</i> bf16 survival of the pseudo-step (see the bf16 tab).</li>"
        "</ul>",
    )
    return goal + status + metric


def build_algorithm() -> str:
    step = card(
        "method",
        "One optimizer step",
        "<p>Every batch is consumed exactly once, by a single optimizer step containing two "
        "forward/backward passes:</p>" + STEP_SVG + "<ol>"
        "<li><b>Inner pass (selective)</b>: all 1024 sequences are forwarded with EMO-style "
        "selective routing — each document restricted to its router top-32 experts (the randpool "
        "router's per-document pool pinned to 32). Backward accumulates across micro-batches and "
        "averages across the 64 GPUs: one gradient for the whole global batch, in which each "
        "expert's slice came only from documents whose working set contains it.</li>"
        "<li><b>Pseudo-step</b>: a single SGD probe on the expert weights only, "
        "<code>&theta;&prime;_exp = &theta;_exp &minus; &alpha;&middot;g_inner</code> "
        "(&alpha;=3e-1, global-norm clip 10). Router, attention, embeddings, norms untouched. "
        "The probe is temporary.</li>"
        "<li><b>Outer pass (full)</b>: the same 1024 sequences again, all 128 experts available, "
        "evaluated at &theta;&prime;. Its gradient is the step's only real update — but in the "
        "backward, <b>each expert's weights receive gradient only from slots whose expert is in "
        "that document's top-32 working set</b>. Non-expert parameters (attention, embeddings, "
        "router, norms) receive the ordinary full gradient.</li>"
        "<li><b>Restore + update</b>: expert weights are restored bitwise to &theta;, then AdamW "
        "consumes the accumulated gradient (first-order MAML: the gradient evaluated at "
        "&theta;&prime; is applied to &theta;).</li>"
        "</ol>"
        "<p>Net effect: <b>no full-routing update exists anywhere</b>. Every expert only ever "
        "trains from documents that select it, toward the full-model loss, one selective step "
        'ahead — "train the original 32 experts so they perform well among all 128, given they '
        'already took a step in the right direction."</p>',
    )
    detach = card(
        "method",
        "Weight-only detach (why early layers lose nothing)",
        "<p>Masking non-working-set experts must not sever backprop to earlier layers. When "
        "backward reaches an expert output <code>y = f(x; W)</code> it computes two independent "
        "chain-rule products: <code>&part;L/&part;W</code> (the expert's weight gradient) and "
        "<code>&part;L/&part;x</code> (the gradient passed through to attention and all earlier "
        "layers). For masked slots we skip only the first: the slot is recomputed under "
        "<code>W.detach()</code> — identical forward value, weights acting as constants (exactly "
        "like a frozen layer), full <code>&part;L/&part;x</code>. This matters most at "
        "initialization, where routing is near-uniform and only ~32/127 &asymp; 25% of a token's "
        "chosen experts fall inside its document's working set — a naive output-detach would have "
        "cut ~75% of the expert backward paths to early layers.</p>"
        "<p>Verified: with the mask on vs off, every non-expert parameter's gradient is "
        "<b>bitwise identical</b> and the forward loss is bitwise unchanged (gate checks M1/M2); "
        "expert gradients change by 15% with 12/96 expert row-blocks exactly zero on the tiny "
        "test config (M3). Implementation: the grouped-GEMM expert computation runs twice on the "
        "outer pass (once live, once under detached weights) with a per-slot select — ~2&times; "
        "expert compute on the outer pass; a zero-extra-FLOPs row-sorting variant is a known "
        "follow-up if throughput matters.</p>",
    )
    heldout = card(
        "method",
        "The two meta arms",
        table(
            ["arm", "inner tokens", "outer tokens", "meta signal", "step cost"],
            [
                [
                    "<code>same_tokens</code>",
                    "all 1024 seqs",
                    "the same 1024 seqs",
                    "a selective step on these docs should make their working sets perform well in the full model on the same data",
                    "~2&times; baseline",
                ],
                [
                    "<code>heldout</code>",
                    "first 512 seqs",
                    "other 512 seqs",
                    "a selective step on some docs should leave OTHER docs' working sets performing well in the full model (cross-data; standard MAML support/query — resists one-step memorization)",
                    "~1.5&times; baseline",
                ],
            ],
        )
        + "<p class='note'>Both arms use working-set outer updates and &lambda;=0 (the inner loss "
        "contributes only through the pseudo-step; a &lambda;&gt;0 blend is implemented as a "
        "contingency knob). The heldout arm's effective outer batch is 512 instances.</p>",
    )
    return step + detach + heldout


def build_implementation() -> str:
    code_map = card(
        "method",
        "Code map",
        table(
            ["file", "role"],
            [
                [
                    "<code>src/olmo_core/train/train_module/transformer/meta_learning.py</code>",
                    "<code>MetaLearningTransformerTrainModule</code> — owns the two-phase train step "
                    "(the trainer only ever calls <code>train_batch/optim_step/zero_grads</code>, so "
                    "the subclass is self-contained). <code>meta_mode=vanilla</code> delegates to the "
                    "stock train module, giving a bit-identical baseline through the same entry script.",
                ],
                [
                    "<code>src/olmo_core/nn/moe/twolevel_batchlb_reducedp_sharedexp_randpool_router.py</code>",
                    "3 per-pass flags on the published EMO router: <code>meta_force_pool</code> (pin "
                    "per-doc pool: 32 inner / keep-all outer; training branch only, eval untouched), "
                    "<code>meta_skip_aux</code> (inner pass skips LB/z losses + metric accumulators "
                    "so nothing double-counts), <code>meta_outer_detach_top_e</code> (builds the "
                    "per-slot working-set mask from the same per-document router-probability ranking "
                    "the inner pool restriction uses; shared experts always trainable).",
                ],
                [
                    "<code>src/olmo_core/nn/moe/{moe,parallel_mlp,mlp}.py</code>",
                    "weight-only detach plumbing via the router's existing <code>_detach_mask</code> "
                    "stash channel (the <code>extension_finetune</code> precedent), ending in "
                    "<code>DroplessMoEMLP.forward(detach_w_rows=…)</code> — the dual grouped-GEMM.",
                ],
                [
                    "<code>src/olmo_core/train/callbacks/pool_pinned_lm_evaluator.py</code>",
                    "pool-pinned ppl evaluator: same validation mix run at eval pool 128 "
                    "(<code>lm-full</code>) and pinned 32 (<code>lm-pool32</code>) &rarr; the gap.",
                ],
                [
                    "<code>src/scripts/train/olmoe-1B-7B_fsl_meta.py</code>",
                    "entry script (randpool-only clone of the parent); meta knobs are dotted "
                    "overrides, e.g. <code>--train_module.meta_mode=heldout</code>.",
                ],
                [
                    "<code>scripts/meta_learning/verify_meta_step.py</code>",
                    "the mechanism-correctness gate (11 checks; see Verification tab).",
                ],
                [
                    "<code>scripts/meta_learning/*.sh</code>",
                    "smoke + launch scripts (one checked-in script per run, per repo convention).",
                ],
            ],
        ),
    )
    fsdp = card(
        "method",
        "FSDP2 mechanics",
        "<ul>"
        "<li><b>Setup</b>: plain FSDP2 (<code>fully_shard</code> per block), bf16 param / fp32 "
        "reduce, torch.compile on, no EP/TP/PP. Expert weights are three fused row-block tensors "
        "per layer (<code>w1/w2/w3</code>, experts = contiguous rows), sharded on dim 0.</li>"
        "<li><b>Pseudo-step</b>: applied in place on the fp32 <i>local shards</i> "
        "(<code>get_local_tensor(p.data).add_(g, alpha=-&alpha;)</code>) — the same mechanism as "
        "an optimizer step; the outer forward's fresh all-gather picks it up. Restore copies the "
        "stashed pre-step shards back (bitwise, assertable via "
        "<code>EMO_META_CHECK_RESTORE=1</code>).</li>"
        "<li><b>Gradient bookkeeping</b>: inner backward completes a normal reduce-scatter cycle; "
        "the expert grads are stashed and all grads zeroed (&lambda;=0), so FSDP2's accumulating "
        "reduce-scatter leaves exactly <code>g_outer(&theta;&prime;)</code> for AdamW. Two full "
        "backward cycles inside one <code>train_batch</code> are just two grad-accumulation "
        "rounds without an optimizer step between them.</li>"
        "<li><b>Memory</b>: the extra state is two fp32 clones of the expert-grad/weight shards "
        "(~1.6 GB/rank); activation memory is unchanged (inner graphs are freed micro-batch by "
        "micro-batch before the outer phase).</li>"
        "</ul>",
    )
    knobs = card(
        "method",
        "Knobs (all --train_module.<knob>=…)",
        table(
            ["knob", "value in the live arms", "meaning"],
            [
                [
                    "<code>meta_mode</code>",
                    "<code>same_tokens</code> / <code>heldout</code>",
                    "vanilla | same_tokens | heldout | outer_only (debug reference)",
                ],
                [
                    "<code>inner_lr</code> (&alpha;)",
                    "3e-1",
                    "pseudo-step SGD size (raw SGD scale, not AdamW-normalized)",
                ],
                [
                    "<code>inner_grad_clip</code>",
                    "10",
                    "global-norm clip of the inner expert grads",
                ],
                [
                    "<code>inner_pool_size</code>",
                    "32",
                    "per-doc pool pinned in the inner pass; also the working-set size for outer masking",
                ],
                [
                    "<code>outer_expert_update</code>",
                    "<code>working_set</code>",
                    "working_set (per-doc top-32 masked, weight-only detach) | all (the degenerate ablation)",
                ],
                [
                    "<code>lambda_inner</code>",
                    "0",
                    "direct weight of L_inner in the real update (0 = pure FOMAML)",
                ],
                [
                    "<code>lb_on_inner</code>",
                    "false",
                    "whether the inner pass carries LB/z aux losses",
                ],
            ],
        ),
    )
    metrics = card(
        "method",
        "Metric dictionary (train/ namespace)",
        table(
            ["metric", "meaning"],
            [
                [
                    "<code>CE loss</code>",
                    "the OUTER pass CE (full routing at &theta;&prime;) — the canonical loss",
                ],
                [
                    "<code>meta inner CE loss</code>",
                    "the selective (top-32) inner-pass CE at &theta;",
                ],
                [
                    "<code>meta pseudo step delta norm</code> / <code>meta delta/weight norm</code>",
                    "&alpha;&middot;&#8214;g_inner&#8214; after clip, absolute and relative to the expert-weight norm",
                ],
                [
                    "<code>meta inner-outer grad cosine (experts)</code>",
                    "cos(g_inner, g_outer) over expert shards — how much the meta term has to teach",
                ],
                [
                    "<code>meta bf16 survival cosine / norm ratio</code>",
                    "MEASURED on the fp32 local shards every 10 steps: cos and norm of "
                    "(bf16(&theta;+&delta;) &minus; bf16(&theta;)) vs the intended &delta; — exactly what "
                    "the outer pass's bf16 all-gather sees of the pseudo-step",
                ],
            ],
        ),
    )
    return code_map + fsdp + knobs + metrics


def build_bf16(data: dict) -> str:
    finding = card(
        "results",
        "Finding: bf16 eats small pseudo-steps",
        "<p>FSDP2 all-gathers parameters in bf16 (~8 mantissa bits &rarr; per-element quantum "
        "&asymp; 0.4% of each weight's own magnitude). An in-place pseudo-step much smaller than "
        "that quantum mostly disappears at the cast — and what survives is dominated by rounding "
        "flips: at the first launch's operating point (&alpha;=3e-2 with clip 1, delta/weight "
        "&asymp; 1.3e-5) the perturbation the outer pass actually saw had ~16&times; the intended "
        "norm but cosine only 0.06 with the intended direction — ~6% signal, 94% noise. The toy "
        "&alpha;-sweep that originally picked 3e-2 was itself partly a quantization artifact.</p>"
        + fig_row(
            img_tag(
                FIGS / "bf16_survival.png",
                "Simulated survival of the pseudo-step direction through the bf16 cast, at the real "
                "run's weight scale (expert-weight norm 2242 over 1.3e10 elements, heavy-tailed "
                "gradient). Dashed lines: the failed first launch's operating point and the "
                "relaunch target.",
            )
        )
        + "<p>Mitigation in the live arms: &alpha;=3e-1 with clip 10 puts delta/weight in the "
        "~1e-4..1e-3 band, and the train module now logs the <b>measured</b> survival "
        "(<code>meta bf16 survival cosine</code>, computed exactly on the fp32 local shards) so "
        "fidelity is observed, not assumed. The exact fix — injecting the delta after the cast "
        "inside the expert forward (the <code>ghost_forward</code> pattern) — is the known "
        "follow-up if measured fidelity proves inadequate.</p>",
    )
    return finding


def build_first_launch(data: dict) -> str:
    v = data.get("meta128_vanilla_100b", {})
    s = data.get("meta128_sametok_100b", {})
    gv = _gap(v.get("evals", {}))
    gs = _gap(s.get("evals", {}))
    body = (
        "<p>The first 100B launch used &lambda;=0 with the outer gradient applied to <b>all</b> "
        "parameters. That degenerates: the inner pass's gradient is used only for the (bf16-"
        "quantized) pseudo-step, so the effective update is <i>always-full-routing</i> training — "
        "the working-set structure vanishes from the update entirely. Train CE looked healthy "
        "(full routing gives lower per-token CE than vanilla's random pools), which is exactly why "
        "train CE is not the metric:</p>"
        + fig_row(
            img_tag(
                FIGS / "ce_curves.png",
                "Train CE. The curves are NOT directly comparable: vanilla's CE is computed under "
                "its random per-doc pools (8–128), the meta arm's under full routing. The spike at "
                "~10.5B is a transient.",
            )
        )
        + "<p>The paired pool-pinned evals told the real story:</p>"
        + fig_row(
            img_tag(
                FIGS / "gap_dumbbells.png",
                "Held-out CE per validation source, evaluated with the full pool (blue) vs pinned "
                "to 32 (orange). Vanilla EMO is BETTER with 32 experts than 128 (its random-pool "
                "training does this); the degenerate meta arm collapsed selective operation on "
                "every source.",
            )
        )
        + f"<p>Mean gap (pool32 &minus; full): <b>vanilla {gv:+.3f}</b> nats at step "
        f"{v.get('step', '?')} vs <b>{gs:+.3f}</b> for the degenerate arm at step "
        f"{s.get('step', '?')} — a ~0.75-nat selective-mode collapse in under 4k steps. Both "
        "meta arms were stopped and rebuilt with working-set outer updates (Algorithm tab); "
        "vanilla was later paused (resumable) to free nodes for the corrected arms.</p>"
        "<p class='note'>Lessons encoded in the redesign: (1) the update, not just the forward, "
        "must carry the working-set structure; (2) pseudo-step fidelity through bf16 must be "
        "measured in-run; (3) selective-inference competence (the gap) and selective-update "
        "transfer (the k=32 CPT protocol) are different quantities — the pilot's go/no-go uses "
        "the gap, the real verdict needs the CPT protocol on finished checkpoints.</p>"
    )
    later = card(
        "results",
        "Later negative results (\u03bb=0 working-set arms, rpool, bf16 extinction)",
        "<ul>"
        "<li><b>\u03bb=0 working-set arms never closed the selective gap.</b> sametok_ws and "
        "heldout_ws trained stably to 20B, but their pool32\u2212full gap sat at <b>+0.31</b> and "
        "<b>+0.33</b> from the first eval to the last, vs vanilla's \u22120.04. Diagnosis: the "
        "working-set masking constrains WHERE gradients land, but at \u03bb=0 the effective "
        "objective never computes a restricted forward \u2014 update sparsity is not "
        "selective-function training. Vanilla's random pools are structured dropout over experts; "
        "removing them from the objective forfeits deletion-robustness. (Full-model CE: sametok_ws "
        "trailed vanilla by ~0.09 at matched steps despite ~2.4\u00d7 compute; heldout_ws by ~0.36 "
        "with half the update signal.)</li>"
        "<li><b>The pseudo-step self-extinguishes under bf16.</b> With \u03b1=3e-1/clip=10 the "
        "measured bf16 survival cosine started at ~0.96 (delta/weight ~1e-3), but as the models "
        "improved their inner gradients shrank: by 20B delta/weight had decayed to ~6e-6 and "
        "survival to ~0.14\u20130.20 \u2014 the meta term effectively vanished for the second half "
        "of both \u03bb=0 ws runs. Any fixed \u03b1 eventually sinks below the bf16 floor; the "
        "principled fix (delta-injection after the cast) remains unimplemented.</li>"
        "<li><b>rpool (randomized outer pool) was canceled unproven.</b> From scratch it entered a "
        "deep pseudo-step-leaning transient (inner CE 27 vs outer 3.5 at step ~900; evals at "
        "\u03b8 ~20 nats). The train-side spread collapsed sharply at ~step 1050 (27\u21923.4), "
        "but the run was canceled before an eval could confirm recovery, to free nodes.</li>"
        "<li><b>Leaning transients are an early-training phenomenon of large probes.</b> Both "
        "same-tokens \u03b1=3e-1 from-scratch starts (sametok_ws first launch, rpool) leaned on "
        "the pseudo-step early (train CE at \u03b8\u2032 fine, everything at \u03b8 bad), and "
        "sametok_ws self-corrected by ~step 1000. Anchored (\u03bb&gt;0) and cross-data (heldout) "
        "objectives are immune by construction.</li>"
        "</ul>",
    )
    levers = card(
        "results",
        "The levers: \u03bb anchor works (early); sequential ablation launched",
        "<ul>"
        "<li><b>sametok_ws_lam05</b> (\u03bb=0.5 + LB on inner): the standout so far \u2014 gap "
        "<b>+0.023 \u2192 +0.010</b> at steps 500/1000 (vs +0.31 for \u03bb=0) AND full-model CE "
        "<i>ahead</i> of vanilla at matched steps (3.99 vs 4.19 @500; 3.36 vs 3.41 @1000). Inner "
        "CE glued to outer CE (no leaning). Caveats: matched steps \u2260 matched compute "
        "(~2.4\u00d7), and its delta/weight is also decaying toward the bf16 floor (~2e-5, "
        "survival 0.46 at step ~1650) \u2014 the \u03bb term, not the probe, may be doing the "
        "work; the sequential ablation tests exactly this.</li>"
        "<li><b>heldout_ws_lam05</b>: anchored cross-data variant \u2014 \u03bb term restores "
        "restricted-forward training and gives the inner half real update signal (fixing "
        "heldout's half-batch weakness); queued.</li>"
        "<li><b>seq_ws (sequential ablation)</b>: per token batch, a COMMITTED ordinary selective "
        "step, then a ws-masked full-routing step on the same tokens \u2014 same masking, same "
        "aux losses, no probe, no evaluation-at-\u03b8\u2032. If it matches sametok_ws_lam05, "
        "the FOMAML probe adds nothing beyond sequential training at this scale. Read with care: "
        "it takes 2\u00d7 real optimizer updates per token (9,536 over 20B) though the step "
        "counter stays token-aligned (4,768).</li>"
        "</ul>",
    )
    return card("results", "First launch: a documented negative result", body) + later + levers


def build_verification() -> str:
    gate = card(
        "results",
        "Mechanism gate (verify_meta_step.py) — 11/11 PASS",
        "<p>Tiny randpool model (2 layers, 16 experts), fp32, compile off, pure-CE loss, synthetic "
        "multi-document batch; run under torchrun with FSDP2 active. All checks re-run after every "
        "mechanism change.</p>"
        + table(
            ["check", "what it proves", "result"],
            [
                [
                    "0 determinism floor",
                    "two identical runs — calibrates all tolerances",
                    "|&Delta;L| = 0 (bitwise)",
                ],
                [
                    "B &alpha;=0 oracle",
                    "same_tokens at &alpha;=0 &equiv; a single full pass (stash/zero/restore is a true no-op)",
                    "loss + grads bitwise equal",
                ],
                [
                    "1 perturbation visibility",
                    "the in-place pseudo-step actually reaches the outer forward through FSDP",
                    "PASS",
                ],
                [
                    "2 directional derivative",
                    "(L(0)&minus;L(&epsilon;))/&epsilon; &asymp; &lang;g_inner, g_outer&rang; — right sign, scale, and row-targeting",
                    "0.76% rel err",
                ],
                [
                    "3 manual reference (&lambda;&isin;{0, 0.5})",
                    "module's final grads &equiv; hand-rolled autograd reference (g_outer(&theta;&prime;) + &lambda;g_inner)",
                    "&le; 5e-8 rel diff",
                ],
                ["H heldout smoke", "split mode runs end-to-end, finite losses", "PASS"],
                [
                    "M1 non-expert grads preserved",
                    "weight-only detach leaves attention/embeddings/router/norm grads untouched (the early-layer backprop property)",
                    "bitwise identical",
                ],
                [
                    "M2 forward unchanged",
                    "the dual-GEMM masked path reproduces identical forward values",
                    "bitwise identical",
                ],
                [
                    "M3 expert grads masked",
                    "confinement is active",
                    "15% grad shift; 12/96 row-blocks exactly zero",
                ],
                ["M4 masked step smoke", "the full relaunch configuration end-to-end", "PASS"],
                [
                    "restore (every run)",
                    "expert weights restored after every step",
                    "bitwise, via EMO_META_CHECK_RESTORE=1",
                ],
            ],
        ),
    )
    smoke = card(
        "method",
        "Smoke + calibration (local A100, real data)",
        "<ul>"
        "<li>All four meta modes &times; 30 steps on OLMoE-mix-0824 with a tiny model: zero errors, "
        "115/115 restore asserts.</li>"
        "<li>&alpha;-sweep (toy scale): 3e-2 best CE with grad cosine ~0.96; 3e-1 clip-saturated — "
        "later reinterpreted through the bf16 lens (see bf16 tab) and superseded by &alpha;=3e-1 + "
        "clip 10 + the measured survival metric.</li>"
        "<li>Compile-on trial: steady 2.55 s/step after warmup (faster than compile-off 3.3), one "
        "warmup-time dynamo eager fallback on the pool-flip frame, no recompile churn &rarr; "
        "compile stays on.</li>"
        "<li>Measured two-pass throughput at 128e scale (first launch): 5.76 vs 4.08 s/step = "
        "1.41&times; baseline before the dual-GEMM; ~2&times; expected with it.</li>"
        "</ul>",
    )
    return gate + smoke


def build_status(data: dict) -> str:
    rows = []
    for name in (
        "meta128_sametok_ws_lam05",
        "meta128_heldout_ws_lam05",
        "meta128_seq_ws",
        "meta128_sametok_ws",
        "meta128_heldout_ws",
        "meta128_vanilla_100b",
    ):
        d = data.get(name, {})
        rows.append(
            [
                f"<code>{name}</code>",
                d.get("state", "queued / no W&B run yet"),
                d.get("step", "—"),
                (
                    f"{d['step'] * TOKENS_PER_STEP / 1e9:.1f}B"
                    if isinstance(d.get("step"), int)
                    else "—"
                ),
            ]
        )
    body = (
        table(["run", "state", "step", "tokens"], rows)
        + "<p>First things checked once the corrected arms produce steps: measured "
        "<code>meta bf16 survival cosine</code> (want &gtrsim;0.5), CE convergence, grad cosine, "
        "throughput vs the ~2&times; expectation, then the first lm-full/lm-pool32 eval pair at "
        "step 500 — where the working-set arms must not reproduce the degenerate collapse.</p>"
        "<p>Then: 100B completions, vanilla resume (auto-resumes from its ephemeral checkpoint), "
        "and the k=32 CPT protocol on finished checkpoints as the decisive update-transfer test. "
        "Known engineering follow-ups: delta-injection after the bf16 cast (exact pseudo-step at "
        "any &alpha;), and the zero-extra-FLOPs row-sorted variant of the weight-only detach.</p>"
    )
    return card("results", "Live status &amp; next steps", body)


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
p { max-width:78ch; }
.card { background:var(--card); border:1px solid var(--line); border-left:4px solid var(--line);
        border-radius:8px; padding:16px 20px; margin:16px 0; }
.card h3 { margin:0 0 8px; font-size:15px; text-transform:uppercase; letter-spacing:0.05em;
           scroll-margin-top:96px; }
.card.goal { border-left-color:#2563eb; } .card.goal h3 { color:#2563eb; }
.card.method { border-left-color:#7c3aed; } .card.method h3 { color:#7c3aed; }
.card.results { border-left-color:#059669; } .card.results h3 { color:#059669; }
table { border-collapse:collapse; margin:12px 0; font-size:14px; width:auto; max-width:100%; }
th, td { border:1px solid var(--line); padding:6px 12px; text-align:left; vertical-align:top; }
th { background:#f1f5f9; }
tbody tr:nth-child(even) { background:#f8fafc; }
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; margin:12px 0; }
.scroll table { margin:0; }
.note { font-size:13px; color:var(--muted); }
code { background:#eef2f7; padding:1px 5px; border-radius:4px; font-size:0.9em; }
ul { max-width:78ch; } li { margin:4px 0; }
ol { max-width:78ch; }
.figrow { display:flex; gap:16px; flex-wrap:wrap; margin:14px 0; }
figure { flex:1 1 380px; min-width:0; max-width:720px; margin:0; }
figure img { width:100%; border:1px solid var(--line); border-radius:6px; cursor:zoom-in; background:#fff; }
figure img.zoom { max-width:none; width:auto; max-height:90vh; position:fixed; inset:0; margin:auto;
                  z-index:100; box-shadow:0 0 0 100vmax rgba(15,23,42,.75); cursor:zoom-out; }
figcaption { font-size:12.5px; color:var(--muted); margin-top:4px; }
.missing { color:#b91c1c; font-size:13px; }
@media (max-width:640px) {
  header, nav, #subnav { padding-left:14px; padding-right:14px; }
  main { padding:16px 14px 60px; }
  nav button { padding:6px 10px; font-size:13px; }
  table { font-size:12.5px; }
  th, td { padding:5px 8px; }
  figure { flex-basis:100%; max-width:100%; }
}
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


def build_phase2() -> str:
    res_path = OUT / "k32cpt_results.json"
    if not res_path.exists():
        return "<p>Phase-2 eval results not yet aggregated (run scripts/meta_learning/aggregate_k32cpt_evals.py).</p>"
    res = json.load(open(res_path))
    n_done = {arm: len(res["arms"][arm]["stages"]) for arm in res["arms"]}

    def _mean(d):
        return sum(d[str(c)] if isinstance(d, dict) and str(c) in d else d[c] for c in range(32)) / 32.0

    rows = []
    for arm in res["arms"]:
        base = {int(k): v for k, v in res["anchors"][arm]["base_20B"].items()}
        upper = res["anchors"][arm].get("upper_40B")
        upper = {int(k): v for k, v in upper.items()} if upper else None
        rows.append([arm,
                     f"{sum(base.values())/32:.4f}",
                     f"{sum(upper.values())/32:.4f}" if upper else "&mdash;",
                     f"{(sum(upper.values())-sum(base.values()))/32:+.4f}" if upper else "&mdash;",
                     f"{n_done[arm]}/32"])

    intro = """
<p>Phase 2 applies the <strong>modular_extension k=32 CPT (no extension) protocol</strong> to each
arm's 20B checkpoint: the arm's own router clusters the (shared) 20B&ndash;40B doc window into k=32
groups; per cluster, its per-layer top-32-of-127 experts are extracted into a 33-expert subset
(surgery), trained on the cluster's proportional share of a 20B token budget with all non-expert
parameters frozen (standard 32-expert training &mdash; pool pinned, no random-pool objective), and
written back. After every stage the full 128-expert pool is evaluated on <em>all 32</em> cluster
held-out sets (25M tokens each) &mdash; the heatmaps below. Reference rows: each arm's 20B base
(the delta reference) and <code>meta128_vanilla/step9537</code>, the same 20B&ndash;40B tokens
consumed as plain EMO pretraining (the upper bound).</p>"""

    anchor_tbl = table(
        ["arm", "20B base mean CE", "40B plain-CPT mean CE", "upper-bound &Delta;", "stages evaluated"],
        rows)

    hm = fig_row(
        img_tag(OUT / "k32cpt_heatmap_vanilla.png",
                "vanilla: full-model CE delta vs its 20B base. Blue diagonal = selective CPT "
                "improves the just-trained cluster in the full-routing view, and columns stay "
                "blue after training (specialization sticks). Bottom row = 40B plain-CPT upper bound."),
        img_tag(OUT / "k32cpt_heatmap_sametok_ws_lam05.png",
                "sametok_ws_lam05: the same protocol hurts the full model nearly everywhere, "
                "including the just-trained cluster (red diagonal), with partial recovery in "
                "later stages."))

    curves_fig = img_tag(OUT / "k32cpt_curves.png",
                         "Summary curves: just-trained delta (specialization), mean over all "
                         "clusters, and mean over previously-trained clusters (forgetting). "
                         "Dashed lines = the 40B plain-CPT reference on each arm's clusters.")

    early = card("bad", "Early readout (in progress &mdash; updates as stages land)", f"""
<p>Through the first {min(n_done.values())} of 32 stages, the comparison runs <strong>against</strong>
the meta hypothesis in the full-model view:</p>
<ul>
<li><strong>vanilla</strong>: the just-trained cluster improves (lag-0 &Delta;CE &minus;0.04&hellip;&minus;0.15
vs the previous pool state) and trained columns stay improved &mdash; selective CPT transfers to full
routing, recovering part of the 40B upper bound's &minus;0.128 mean improvement.</li>
<li><strong>sametok_ws_lam05</strong>: selective CPT initially <em>degrades</em> the full model even on
the cluster it just trained (lag-0 up to +0.28 early), with broad collateral damage that partially heals
by stages 3&ndash;5. This mirrors the arm's phase-1 &ldquo;leaning&rdquo; transient &mdash; selective-mode
updates initially failing to cohere with full routing &mdash; but here on a much larger scale.</li>
</ul>
<p>Caveats: 6/32 stages; the sametok damage trend is shrinking stage-over-stage (own-cluster +0.28 &rarr;
+0.06), so the full sweep decides whether this is a transient or the verdict. The carried Adam moments
also come from each arm's own phase-1 objective, which may mismatch plain selective training more for the
meta arm.</p>""")

    ops = card("info", "Pipeline (all per-arm, resumable)", """
<p><code>convert_20b_to_hf.sh</code> &rarr; <code>launch_embed_docs.sh</code> (router doc-probs, 128
Beaker shards) &rarr; <code>cluster_docs.sh</code> (spherical k-means k=32 on 18.3M docs) &rarr;
<code>cluster_expert_concentration.py</code> (per-layer top-32 selection; cluster mass 0.42&ndash;0.77
vs 0.31 random) &rarr; <code>build_cluster_token_data.sh</code> (20B train + ~0.7B held-out tokens)
&rarr; <code>run_k32cpt_arm.sh</code> (sequential extract/train/writeback, 4-node stages, carry
optimizer, bf16 pool snapshot per stage) &rarr; <code>run_snapshot_evals.sh</code> (8-GPU eval job per
snapshot) &rarr; <code>aggregate_k32cpt_evals.py</code> (this page's figures).</p>""")

    return intro + anchor_tbl + hm + curves_fig + early + ops


def main():
    data_path = OUT / "report_data.json"
    data = json.load(open(data_path)) if data_path.exists() else {}
    curves_path = OUT / "curves.json"
    curves = json.load(open(curves_path)) if curves_path.exists() else {}
    curves_html, curves_payload = build_curves_tab(curves)

    tabs = [
        ("overview", "Overview", build_overview(data)),
        ("curves", "Live curves", curves_html),
        ("phase2", "Phase-2 CPT heatmaps", build_phase2()),
        ("algorithm", "Algorithm", build_algorithm()),
        ("implementation", "Implementation", build_implementation()),
        ("bf16", "bf16 quantization", build_bf16(data)),
        ("negative-results", "Negative results", build_first_launch(data)),
        ("verification", "Verification", build_verification()),
        ("status", "Status", build_status(data)),
    ]
    vendor = ROOT / "scripts/models_v2/vendor"
    uplot_css = (vendor / "uPlot.min.css").read_text() if curves_payload else ""
    uplot_js = (vendor / "uPlot.iife.min.js").read_text() if curves_payload else ""
    curves_js = CURVES_JS.replace("__MLCURVES__", curves_payload) if curves_payload else ""
    nav = "".join(f'<button data-target="{tid}">{name}</button>' for tid, name, _ in tabs)
    sections = "".join(f'<section class="tab" id="{tid}">{body}</section>' for tid, _, body in tabs)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EMO meta_learning: FOMAML pretraining for selective-expert update transfer</title>
<style>{CSS}</style>
<style>{uplot_css}</style>
</head>
<body>
<header>
<a class="home-link" href="/">&larr; all reports</a>
<h1>EMO meta_learning: FOMAML pretraining for selective-expert update transfer</h1>
<p>meta_learning &mdash; pretrain a 128-expert EMO from scratch so that training on a document's
32-expert working set improves the full model &middot; two-pass first-order MAML with working-set
outer updates &middot; motivated by the modular_extension k=32 CPT negative result &middot;
generated by scripts/meta_learning/build_report.py</p>
</header>
<div class="topbar"><nav>{nav}</nav><div id="subnav"></div></div>
<main>{sections}</main>
<script>{uplot_js}</script>
<script>{JS}</script>
<script>{curves_js}</script>
</body>
</html>
"""
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "report.html"
    out.write_text(page)
    print(f"Wrote {out} ({out.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
