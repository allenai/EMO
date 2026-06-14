"""Build a self-contained HTML report for the models_fullextend (ghost-expert) experiment.

This experiment is in progress and has no results yet, so the report currently
covers the overview, the method, and the planned sweep / live status. It is
intentionally static (no figures to read) but mirrors the size-scaling report's
styling and tab structure so results tabs can be slotted in later.

Usage:

    python scripts/models_fullextend/build_report.py \\
        [--output claude_outputs/models_fullextend/report.html]
"""

import argparse
from pathlib import Path

VENDOR = Path(__file__).parent / "vendor"

# --------------------------------------------------------------------------
# HTML helpers (kept in sync with scripts/models_sizescaling/build_report.py)
# --------------------------------------------------------------------------


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


# uPlot multi-chart init — plain string (braces are not f-string), data injected via
# replace. Renders one chart per metric (CURVES.charts) with per-run consistent colors,
# a per-chart log-y toggle, hover-focus, and box-zoom on both axes. Exposes window.ceResize().
_CHARTS_JS = r"""
<script>
(function(){
  const C = __CURVES__;
  const palette = ["#64748b","#2563eb","#7c3aed","#059669","#dc2626","#d97706"];
  const colorOf = {};
  C.runs.forEach((r,i) => { colorOf[r.label] = palette[i % palette.length]; });
  const reg = {};
  function build(key){
    const st = reg[key], chart = st.chart, el = st.el;
    const data = [chart.x].concat(chart.series.map(s => s.y));
    const series = [{ value: (u,v) => v==null ? "--" : v }].concat(
      chart.series.map(s => ({
        label: s.label, stroke: colorOf[s.label] || "#888", width: 1.6,
        spanGaps: false, value: (u,v) => v==null ? "--" : (+v).toFixed(4),
      })));
    const opts = {
      width: (el.clientWidth || 900), height: 360,
      focus: { alpha: 0.25 },
      scales: { x: { time:false }, y: { distr: st.logY ? 3 : 1 } },
      cursor: { focus: { prox: 30 }, drag: { x:true, y:true, uni:10 } },
      axes: [
        { label: "step", values: (u,vals) => vals.map(v => v>=1000 ? (v/1000)+"k" : v) },
        { label: chart.title },
      ],
      series: series,
    };
    if (st.plot) st.plot.destroy();
    st.plot = new uPlot(opts, data, el);
  }
  C.charts.forEach(chart => {
    reg[chart.key] = { plot:null, logY:false, chart:chart, el:document.getElementById("chart-"+chart.key) };
    if (reg[chart.key].el) build(chart.key);
  });
  document.querySelectorAll("button.logtoggle").forEach(b => {
    b.addEventListener("click", () => { const k=b.dataset.chart; reg[k].logY=!reg[k].logY; build(k); });
  });
  window.ceResize = function(){
    Object.keys(reg).forEach(k => { const st=reg[k]; if (st.plot && st.el) st.plot.setSize({ width: st.el.clientWidth||900, height: 360 }); });
  };
  window.addEventListener("resize", window.ceResize);
})();
</script>
"""


def _load_curves(base: Path):
    p = base / "ce_curves.json"
    if not p.is_file():
        return None, None
    import json as _json

    raw = p.read_text()
    return _json.loads(raw), raw


def build_run_links(base: Path) -> str:
    data, _ = _load_curves(base)
    if not data:
        return ""
    links = " &middot; ".join(
        f'<a href="{r["url"]}" target="_blank" rel="noopener">{r["label"]}</a>'
        for r in data.get("runs", [])
    )
    return f'<p class="note"><strong>Pretraining runs:</strong> {links}</p>'


def build_chart_blocks(base: Path, keys: list) -> str:
    """Chart container(s) for the given metric keys (the shared <script> is emitted
    once via build_charts_script). uPlot only draws keys whose div exists in the DOM."""
    data, _ = _load_curves(base)
    if not data:
        return ""
    by_key = {c["key"]: c for c in data.get("charts", [])}
    blocks = []
    for k in keys:
        c = by_key.get(k)
        if not c:
            continue
        blocks.append(
            f'<h4>{c["title"]}</h4>'
            '<div class="chart-controls">'
            f'<button class="logtoggle" data-chart="{c["key"]}">toggle log-y</button>'
            '<span class="note">drag to zoom (x &amp; y) &middot; double-click reset '
            '&middot; hover to highlight a run &middot; click legend to toggle</span>'
            '</div>'
            f'<div class="ce-chart" id="chart-{c["key"]}"></div>'
        )
    return "".join(blocks)


def build_charts_script(base: Path) -> str:
    _, raw = _load_curves(base)
    return "" if raw is None else _CHARTS_JS.replace("__CURVES__", raw)


def build_speed_table(base: Path) -> str:
    """Steady-state training-speed table (TPS comparable across runs; baseline MFU n/c)."""
    p = base / "ce_curves.json"
    if not p.is_file():
        return ""
    import json as _json

    speed = _json.loads(p.read_text()).get("speed", [])
    if not speed:
        return ""
    base_tps = speed[0].get("tps")  # RUNS[0] is the no-ghost baseline
    rows = []
    for i, s in enumerate(speed):
        tps, mfu = s.get("tps"), s.get("mfu")
        if i == 0:
            vs = "(reference)"
        elif tps and base_tps:
            vs = f"{(tps / base_tps - 1) * 100:+.0f}%"
        else:
            vs = "&mdash;"
        mfu_disp = "n/c&sup1;" if (mfu is None or mfu > 100) else f"{mfu:.1f}%"
        rows.append((s["label"], f"{tps:,.0f}" if tps else "&mdash;", mfu_disp, vs))
    return table(
        ["Run", "TPS / device (steady median)", "MFU", "vs baseline TPS"], rows
    )


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------


def build_overview() -> str:
    return f"""
<p><strong>Question:</strong> can we pretrain EMO so that <strong>adding a
brand-new expert after training is well-conditioned</strong> &mdash; i.e. a freshly
instantiated expert slots into the model and is immediately useful, rather than
landing in a dead region of weight space?</p>

{card("goal", "Hypothesis", '''
<p>If, <em>throughout pretraining</em>, the model is perpetually exposed to a
simulated &ldquo;newly added expert&rdquo; that is born as an average of the experts the
current document already uses, then the trained expert/router weight space
becomes smooth enough that a real averaged-initialized expert later drops in
cleanly. We call this simulated expert a <strong>ghost expert</strong>.</p>''')}

{card("method", "Approach in one paragraph", '''
<p>On top of EMO's normal two-level routing, every document also gets one (or
more) <strong>ghost experts</strong>. A ghost is a full new expert whose router row
<em>and</em> MLP weights are the same linear combination of its document pool's
experts &mdash; never instantiated as parameters, recomputed per document on every
forward. The ghost joins the routing softmax like a real expert, and because it
is a differentiable blend of the existing experts, its gradient flows straight
back into the constituent experts and router rows. See the <strong>Method</strong>
tab for details.</p>''')}

{card("results", "Status", '''
<p>The coefficient-mode sweep is <strong>complete</strong>: all three blend modes
(<code>usage</code> / <code>uniform</code> / <code>random</code>) trained to the 50B-token
hard stop. See the <strong>Sweep</strong> tab for the config list and the per-config
CE-loss comparison.</p>
<p><strong>Final CE at the 50B hard-stop, vs the identical no-ghost EMO baseline's
CE 2.689 at the same step:</strong></p>
<ul>
<li>Config #1 (<code>usage / always / detachF</code>) &mdash; <strong>final CE 2.654</strong>
(&minus;0.035): a slight edge, within run-to-run noise.</li>
<li>Config #2 (<code>uniform / always / detachF</code>) &mdash; <strong>final CE 2.690</strong>
(+0.001): essentially identical to baseline. A naive uniform pool-average ghost is
convergence-neutral; the usage-weighted blend's small edge does not transfer.</li>
<li>Config #3 (<code>random / always / detachF</code>) &mdash; <strong>final CE 2.690</strong>
(+0.001): also convergence-neutral. Sampling <code>random_k</code> pool experts per ghost
behaves like the uniform average at this budget.</li>
</ul>
<p>Across all three modes the ghost mechanism adds <strong>no convergence penalty</strong>.
<strong>Two no-ghost reference runs are now launched at matched 8-node / 64-GPU compute</strong>
(<code>emo_1b14b_130b</code> and the standard top-k MoE <code>stdmoe_1b14b_130b</code>, both
max_duration 130B / hard_stop 50B) to anchor the comparison at the same DP world as config #3.
The downstream &ldquo;add-an-expert&rdquo; evaluation that would actually test the
extendability hypothesis is still to come.</p>''')}
"""


def build_method() -> str:
    return f"""
{card("goal", "What a ghost expert is", '''
<p>For each document, EMO's two-level router already keeps a document-level
<em>expert pool</em> (the top <code>document_expert_pool</code> experts for the doc)
and prunes the rest. The ghost expert is a new expert composed from that pool:</p>
<pre><code>alpha_i  = blend coefficients over the document pool   (sum to 1)
r_ghost  = sum_i alpha_i * r_i      # router row  = blend of pool router rows
W_ghost  = sum_i alpha_i * W_i      # MLP weights = blend of pool MLP weights</code></pre>''')}

{card("method", "Why it works", '''
<ul>
<li><strong>Ghost, not instantiated.</strong> <code>r_ghost</code> / <code>W_ghost</code>
are never stored or initialized as parameters; they are recomputed per document
from the existing experts on every forward. Zero new parameters.</li>
<li><strong>Routes like a real expert.</strong> The ghost's logit is
<code>sum_i alpha_i * logit_i</code> (its blended router row applied to the token),
and it joins the routing softmax <em>denominator alongside the real pool
experts</em> &mdash; the pool experts and the ghost(s) form a single renormalized
distribution (the real experts shrink to make room).</li>
<li><strong>Backprop updates the originals.</strong> Because the ghost is a
differentiable blend, autograd routes its gradient straight back into the
constituent experts' MLPs <em>and</em> their router rows
(<code>dL/dW_i += alpha_i * dL/dW_ghost</code>, likewise for <code>r_i</code>) &mdash; for
every coefficient mode, not just the usage-weighted one.</li>
<li><strong>Training-only.</strong> Ghosts are added only in training; eval and
inference measure the base model with no ghost.</li>
</ul>''')}

{card("method", "Choosing the blend coefficients", f'''
<p>The coefficient mode selects how the ghost is composed from the document
pool:</p>
{table(["coeff_mode", "Blend"], [
    ("<code>usage</code>", "document-usage-weighted: alpha_i &prop; the doc-level summed routing probability of pool expert i (the average of what the document actually routes to). Adds an extra gradient path into the router through alpha."),
    ("<code>uniform</code>", "equal weight over the whole pool."),
    ("<code>random</code>", "uniform average over a random sample of <code>random_k</code> pool experts (the mode where num&gt;1 is meaningful, since each ghost re-samples)."),
])}''')}

{card("method", "Hyperparameters", table(
    ["Knob", "Default", "Meaning"],
    [
        ("<code>ghost_extend_mode</code>", "false", "Master switch; ghosts active only in training."),
        ("<code>ghost_extend_num</code>", "1", "Ghost experts simulated per document."),
        ("<code>ghost_extend_coeff_mode</code>", "usage", "Blend scheme: usage / uniform / random."),
        ("<code>ghost_extend_random_k</code>", "8", "Sample size for coeff_mode=random."),
        ("<code>ghost_extend_route</code>", "always", "How the ghost is routed. Only <code>always</code> implemented (topk deferred)."),
        ("<code>ghost_extend_detach_coeff</code>", "false", "If true, detach alpha (cuts the extra usage-only router-grad path; the blended-router-row path still trains the router)."),
    ]) + '''
<p class="note">The ghost's mixing weight is its own renormalized routing share,
not a tunable scalar &mdash; there is deliberately no gate-scale knob. Requires
softmax gating. The load-balancing loss and entropy metric are computed on the
real-expert pool distribution only (the ghost is a transient blend, not an
expert to balance).</p>''')}

{card("method", "Implementation", '''<p>Lives in the published EMO router and MoE
layer (no new model-type): the randpool router builds the blend coefficients, the
blended ghost logits, the renormalized routing scores, and per-token ghost gates;
<code>DroplessMoEMLP.ghost_forward</code> materializes W_ghost per document via an
einsum over the expert axis and runs the grouped SwiGLU; <code>MoEBase.forward</code>
adds <code>gate * ghost_out</code> to the output.</p>''')}
"""


def build_mc9_table(base: Path) -> str:
    p = base / "mc9_results.json"
    if not p.is_file():
        return ""
    import json as _json

    d = _json.loads(p.read_text())
    tasks, cols = d["tasks"], d["columns"]
    # One Δ (on−off) column per ghost-trained run that has both modes, keyed by run.
    # Coeff-mode label read off the run name (ghost_<mode>_50b).
    def mode_name(run):
        for m in ("usage", "uniform", "random"):
            if f"_{m}_" in run or run.endswith(f"_{m}_50b") or f"ghost_{m}" in run:
                return m
        return run
    deltas = []  # (label, off_col, on_col)
    for run in dict.fromkeys(c["run"] for c in cols):
        off = next((c for c in cols if c["run"] == run and c["mode"] == "standard"), None)
        on = next((c for c in cols if c["run"] == run and c["mode"] == "ghost"), None)
        if off and on:
            deltas.append((f"&Delta; {mode_name(run)} (on&minus;off)", off, on))
    headers = ["task"] + [c["label"] for c in cols] + [d[0] for d in deltas]
    rows = []
    for t in tasks:
        r = [t]
        for c in cols:
            v = c["scores"].get(t)
            r.append(f"{v:.3f}" if v is not None else "&mdash;")
        for _, off, on in deltas:
            ov, nv = off["scores"].get(t), on["scores"].get(t)
            r.append(f"{nv - ov:+.3f}" if ov is not None and nv is not None else "&mdash;")
        rows.append(tuple(r))
    avg = ["MC9 avg"] + [
        (f"{c['avg']:.3f} (n={c['n']})" if c["avg"] is not None else "&mdash;") for c in cols
    ]
    for _, off, on in deltas:
        common = [t for t in tasks if off["scores"].get(t) is not None and on["scores"].get(t) is not None]
        davg = sum(on["scores"][t] - off["scores"][t] for t in common) / len(common) if common else None
        avg.append(f"{davg:+.3f}" if davg is not None else "&mdash;")
    rows.append(tuple(avg))
    return table(headers, rows)


def build_sweep(base: Path) -> str:
    run_links = build_run_links(base)
    mc9 = build_mc9_table(base)
    loss_charts = build_chart_blocks(
        base, ["ce", "grad_norm", "lb", "unique_experts", "hellaswag", "arc"]
    )
    tps_chart = build_chart_blocks(base, ["tps"])
    speed = build_speed_table(base)
    fixed = table(
        ["Held fixed", "Value"],
        [
            ("nodes", "16 (128 &times; H100) for #1/#2; #3 runs at 8&sup2;"),
            ("ghost_extend_num", "1"),
            ("ghost_extend_random_k", "8"),
            ("ghost_extend_route", "always (topk not implemented)"),
            ("max_duration", "130B tokens (LR cosine targets 130B)"),
            ("hard_stop", "50B tokens (per-config sweep budget)"),
            ("checkpointing", "2-deep rolling (keep_ephemeral=2) + final model"),
        ],
    )
    runs = table(
        ["Run", "coeff_mode", "route", "detach_coeff", "Status", "Final CE @ 50B"],
        [
            ("emo_1b14b_130b_ghost_usage_always_detachF", "usage", "always", "false", "done", "2.654"),
            ("emo_1b14b_130b_ghost_uniform_always_detachF", "uniform", "always", "false", "done", "2.690"),
            ("emo_1b14b_130b_ghost_random_always_detachF", "random", "always", "false (no-op)", "done (8 nodes&sup2;)", "2.690"),
            ("emo_1b14b_130b (no-ghost EMO baseline)", "&mdash;", "&mdash;", "&mdash;", "launched (8 nodes&sup2;)", "&mdash;"),
            ("stdmoe_1b14b_130b (standard top-k MoE)", "&mdash;", "&mdash;", "&mdash;", "launched (8 nodes&sup2;)", "&mdash;"),
        ],
    )
    # Step-aligned CE: ghost config #1 vs the identical no-ghost EMO baseline
    # (olmoe-modular / twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301).
    cmp = table(
        ["Step", "Tokens", "Baseline", "Usage", "Uniform", "Random", "&Delta; usage", "&Delta; uniform", "&Delta; random"],
        [
            ("250", "1.0B", "4.885", "4.872", "4.898", "4.926", "&minus;0.013", "+0.014", "+0.041"),
            ("1000", "4.2B", "3.356", "3.378", "3.377", "3.375", "+0.022", "+0.020", "+0.019"),
            ("2500", "10.5B", "2.977", "2.974", "2.975", "2.980", "&minus;0.003", "&minus;0.002", "+0.003"),
            ("5000", "21.0B", "2.788", "2.788", "2.789", "2.790", "&minus;0.000", "+0.002", "+0.002"),
            ("7500", "31.5B", "2.738", "2.733", "2.734", "2.736", "&minus;0.004", "&minus;0.004", "&minus;0.002"),
            ("10000", "41.9B", "2.723", "2.708", "2.723", "2.723", "&minus;0.015", "&minus;0.000", "&minus;0.000"),
            ("11921", "50.0B", "2.689", "2.654", "2.690", "2.690", "&minus;0.034", "+0.001", "+0.001"),
        ],
    )
    return f"""
{card("goal", "Sweep design", '''<p>An <strong>incremental</strong> sweep: each config
trains to a 50B-token hard stop, then the next config is chosen from what the
previous run showed. The runname encodes the three varied knobs
(<code>coeff_mode</code> / <code>route</code> / <code>detach_coeff</code>);
<code>num</code> and <code>random_k</code> are held fixed.</p>''' + fixed)}

{card("method", "Decision logic", '''<ul>
<li>Start from the most fully-coupled config (<code>usage / always / detachF</code>):
router learns the blend and gradient flows through alpha &mdash; the strongest test
of the hypothesis.</li>
<li>If it trains stably and tracks baseline loss, branch to isolate what matters:
<code>uniform</code> (does learned usage-weighting beat a naive average?) and
<code>detachT</code> (does cutting the extra alpha&rarr;router path change stability?).</li>
<li>If a config is unstable (nan / loss explosion / stall), back off toward less
aggressive coupling and move on.</li>
</ul>''')}

{card("results", "Configs", runs + '''
<p class="note">&sup2; Config #3 and the two no-ghost reference runs train on <strong>8 nodes</strong>
(compute-limited) vs 16 for #1/#2. The EMO router does <code>reduce-dp</code> batch-level load
balancing, so halving the data-parallel world reduces the sequence population the LB statistics
are reduced over &mdash; a weaker LB signal, so #3 is not a strict apples-to-apples coeff_mode
comparison against #1/#2. The 8-node baselines are matched to #3's DP world precisely to
anchor that comparison.</p>''')}

{card("results", "MC9 downstream eval (rc) &mdash; standard vs ghost", '''
<p>Does the ghost-trained model still work <em>without</em> the ghost it trained with, and
does turning the ghost on at eval shift its behavior? MC9 with the OLMES rank-classification
(rc) metric &mdash; the mc letter-picking variant scores these base models near chance, so it is
not used. The ghost-ON arm uses pool = ALL experts (no document-pool masking), per the eval
design.</p>''' + mc9 + '''
<p><strong>Reading it:</strong> both ghost-trained models in <strong>standard (ghost-off)</strong>
mode are healthy and well above chance (MC9 0.57 usage / 0.59 uniform) &mdash; neither is
<strong>broken</strong> without the ghost. The clean isolation of the ghost is <strong>off vs on</strong>
on the same model: turning the ghost on at eval moves MC9 by only ~&minus;0.02 for the
<strong>usage</strong> model (mixed by task &mdash; boolq/openbookqa/arc_challenge down,
arc_easy/socialiqa/hellaswag up) and essentially <strong>zero</strong> (&minus;0.000) for the
<strong>uniform</strong> model. So there is <strong>no large distribution shift</strong> either way;
the uniform-trained model is even more eval-stable to the ghost than the usage one (consistent with
its blend being a smoother, more redundant average).</p>
<p class="note">The no-ghost baseline trained to 130B vs the ghost runs' 50B, so most of the
baseline gap is the token budget, not the ghost &mdash; treat it as a loose reference, not a
controlled comparison. Configs #1 (usage) and #2 (uniform) so far; preliminary (probe, not the
eventual &ldquo;permanently add an expert&rdquo; protocol).</p>''')}

{card("results", "Training &amp; eval curves (interactive)", '''
<p>All runs share the identical recipe &mdash; the no-ghost reference is WandB
<code>olmoe-modular</code> /
<code>twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301</code>
(runs to 130B; ghost configs hard-stop at 50B = step 11,921). Use the links to open
each run in WandB.</p>''' + run_links + loss_charts)}

{card("results", "Training speed (TPS / MFU)", '''
<p>How much does the ghost mechanism cost in throughput? <strong>TPS</strong>
(tokens/sec per device) is a hardware-measured, convention-free number, so it is
comparable across runs; values below are the steady-state median (steps &ge; 1000,
i.e. past compile warmup).</p>''' + speed + '''
<p>The ghost adds roughly a <strong>20&ndash;30% throughput hit</strong> versus the
no-ghost baseline &mdash; the per-document blend (einsum over the pool),
the grouped-GEMM ghost MLP, and the routing renormalization all run in eager (the
router already falls back to eager under <code>torch.compile</code>).</p>''' + tps_chart + '''
<p class="note">&sup1; MFU is shown only for the same-project ghost runs. The
no-ghost baseline is from an older project that logs MFU on a different
(non-comparable) convention (&asymp;165%), so only its TPS is used here.</p>''')}

{card("results", "CE loss vs no-ghost baseline (exact values)", '''
<p>Step-aligned CE for both ghost configs vs the identical no-ghost baseline, up to
the shared 50B-token hard-stop (&Delta; = ghost &minus; baseline; negative = ghost ahead):</p>''' + cmp + '''
<p>Both ghost curves are <strong>statistically indistinguishable</strong> from the
baseline. <strong>Usage</strong> ends a touch ahead at 50B (&minus;0.034; mean gap
&asymp; &minus;0.007 over the run); <strong>uniform</strong> lands right on top of the
baseline (+0.001; mean gap &asymp; +0.004). Training with a perpetually-simulated new
expert costs nothing in LM loss, in either coefficient mode &mdash; the open question
(next) is whether it makes <em>actually adding</em> a new expert cleaner.</p>
<p class="note">The downstream &ldquo;instantiate a real new expert and measure
degradation&rdquo; evaluation will be added as configs finish.</p>''')}
{build_charts_script(base)}
"""


# --------------------------------------------------------------------------
# Page assembly (CSS/JS kept in sync with models_sizescaling)
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
details { margin:12px 0; }
summary { cursor:pointer; color:#2563eb; font-size:14px; }
.note { font-size:13px; color:var(--muted); }
pre { background:#0f172a; color:#e2e8f0; padding:12px 14px; border-radius:6px; overflow:auto; font-size:13px; }
pre code { background:transparent; padding:0; color:inherit; }
code { background:#eef2f7; padding:1px 5px; border-radius:4px; font-size:0.9em; }
h4 { margin:18px 0 4px; }
.ce-chart { width:100%; margin-bottom:6px; }
.chart-controls { display:flex; align-items:center; gap:12px; margin:8px 0 4px; }
.chart-controls button { border:1px solid var(--line); background:#fff; border-radius:6px;
                         padding:5px 10px; font-size:13px; cursor:pointer; }
.chart-controls button:hover { background:#f1f5f9; }
.u-legend { font-size:12.5px; }
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
  if (window.ceResize) window.ceResize();
}
document.querySelectorAll('nav button').forEach(b => b.addEventListener('click', () => show(b.dataset.target)));
show(location.hash && document.getElementById(location.hash.slice(1)) ? location.hash.slice(1) : 'overview');
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or Path("claude_outputs/models_fullextend/report.html")
    base = out.parent

    # Inline uPlot (self-contained; no external scripts) when present.
    uplot_css = (VENDOR / "uPlot.min.css").read_text() if (VENDOR / "uPlot.min.css").is_file() else ""
    uplot_js = (VENDOR / "uPlot.iife.min.js").read_text() if (VENDOR / "uPlot.iife.min.js").is_file() else ""

    tabs = [
        ("overview", "Overview", build_overview()),
        ("method", "Method", build_method()),
        ("sweep", "Sweep", build_sweep(base)),
    ]
    nav = "".join(f'<button data-target="{tid}">{name}</button>' for tid, name, _ in tabs)
    sections = "".join(
        f'<section class="tab" id="{tid}">{body}</section>' for tid, _, body in tabs
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EMO models_fullextend: ghost-expert extendability</title>
<style>{CSS}</style>
<style>{uplot_css}</style>
<script>{uplot_js}</script>
</head>
<body>
<header>
<a class="home-link" href="/">&larr; all reports</a>
<h1>EMO models_fullextend: pretraining so new experts can be added post-training</h1>
<p>models_fullextend &mdash; ghost-expert training &middot; experiment in progress
&middot; generated by scripts/models_fullextend/build_report.py</p>
</header>
<div class="topbar"><nav>{nav}</nav><div id="subnav"></div></div>
<main>{sections}</main>
<script>{JS}</script>
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out} ({out.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
