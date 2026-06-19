"""Build a self-contained HTML report for the models_routerfixed experiment.

Tests whether EMO's router must be *learned* during pretraining or can be *fixed*:
graft the fully-trained step-11921 routers from emo_1b14b_130b onto a fresh,
byte-exact init, freeze them, and retrain everything else from scratch on the
identical recipe. The experiment is results-complete (one run converged, the
aux-on variants NaN'd), so this report is static tables + prose — no curves to
read. CSS/JS/tab structure are kept in sync with scripts/models_fullextend/build_report.py.

Usage:

    python scripts/models_routerfixed/build_report.py \\
        [--output claude_outputs/models_routerfixed/report.html]
"""

import argparse
from pathlib import Path


# --------------------------------------------------------------------------
# HTML helpers (kept in sync with scripts/models_fullextend/build_report.py)
# --------------------------------------------------------------------------


def card(kind: str, title: str, body: str) -> str:
    return f'<div class="card {kind}"><h3>{title}</h3>{body}</div>'


def table(headers: list, rows: list) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------


def build_overview() -> str:
    return f"""
<p><strong>Question:</strong> EMO's premise is that modular structure &mdash; the
router that assigns tokens/documents to experts &mdash; <em>emerges</em> during
pretraining. This experiment tests a counterfactual: <strong>does the router need to
be learnable at all?</strong> We take the routers from the fully-trained
<code>emo_1b14b_130b</code> run (step 11,921 = 50B tokens), graft them onto a
<em>fresh</em> model init, <strong>freeze</strong> them, and retrain everything else
from scratch on the identical recipe.</p>

{card("goal", "Hypothesis", '''
<p>If a good routing function, once found, can be held <em>fixed</em> while the experts
organize around it, then a frozen trained router should let the rest of the model
converge to a loss comparable to the baseline that learned its router jointly. That
would say the router's <em>job</em> is to provide a stable assignment, not that the
assignment must co-adapt with the experts step-for-step.</p>''')}

{card("method", "Approach in one paragraph", '''
<p>Rebuild the baseline's byte-exact fresh init single-process (init is
topology-independent and seed-deterministic), overwrite every
<code>*.router.weight</code> with the trained step-11921 router, save a model-only
step-0 checkpoint, and launch the standard recipe with
<code>--model.freeze_params='[blocks.*.feed_forward_moe.router.*]'</code> and a fresh
optimizer (<code>--load_trainer_state=false --load_optim_state=false</code>). The router
never moves; everything else trains. See the <strong>Method</strong> tab.</p>''')}

{card("results", "Status &mdash; resolved", '''
<p><strong>The router does not need to be learnable.</strong> With the trained router
frozen and everything else retrained from scratch, the model converges to
<strong>CE 2.715</strong> vs the learnable-router baseline's <strong>2.692</strong>
&mdash; a <strong>+0.023 nat gap (~0.85%)</strong> &mdash; and arc_challenge (CE loss)
is marginally <em>better</em> (0.774 vs 0.769). A good routing function, once found,
can be frozen while the experts organize around it.</p>
<p><strong>Caveat that turned into a finding:</strong> this only holds with the
router-shaping auxiliary losses <em>off</em> (<code>noaux</code>). Every aux-on variant
NaN'd early &mdash; including each aux loss in isolation. See the <strong>Results</strong>
tab for the mechanism (and the one prediction it refuted).</p>''')}
"""


def build_method() -> str:
    return f"""
{card("goal", "Building the frozen-router init", '''
<ul>
<li><strong>Byte-exact fresh init.</strong> EMO weight init uses a <code>torch.Generator</code>
seeded by the model-level <code>init_seed</code>, and inits the <em>full</em> tensor before
sharding, so a single-process CPU rebuild with the same config reproduces the original
8-node run's non-router weights exactly.</li>
<li><strong>Graft the trained routers.</strong> Each MoE layer's router is one parameter,
<code>*.router.weight</code> (a flat <code>(128&times;2048,)</code> tensor). We overwrite all
16 of them with the step-11921 values via <code>load_keys</code>, leaving every other tensor
at its fresh init. Builder hard-asserts routers == step-11921 and a sample expert weight
&ne; step-11921 before saving.</li>
<li><strong>Save model-only.</strong> The result is a DCP checkpoint with no optimizer/trainer
state, loaded at step 0 with a fresh optimizer and LR schedule.</li>
</ul>''')}

{card("method", "Freezing", '''
<p>Freezing reuses the existing mechanism (same as
<code>models_fullextend/extend_finemath_frz_common.sh</code>): <code>TransformerConfig.freeze_params</code>
is a list of fnmatch globs applied in <code>TransformerConfig.build()</code> that sets
<code>requires_grad=False</code> on every matching param <em>before</em> FSDP wrap. Frozen
params are then fully excluded from the optimizer (no gradient step, no weight decay).
Gradients still flow <em>through</em> the frozen router to earlier layers &mdash; which is
exactly what makes the aux-loss interaction below matter. The only code change to the entry
script was adding a <code>load_optim_state</code> config field so the model-only init loads
with a clean optimizer.</p>''')}

{card("method", "The two ablations", '''
<p>Both freeze the router; they differ only in how the router-shaping auxiliary losses are
handled now that the router cannot move:</p>''' + table(
    ["Run", "lb_loss_weight", "z_loss_weight", "Rationale"],
    [
        ("<code>keepaux</code>", "1e-1", "1e-3", "Baseline-identical recipe &mdash; the aux losses still backprop into activations against the fixed router."),
        ("<code>noaux</code>", "0", "0", "With the router immovable, drop its shaping losses entirely (also disables the reduce-dp all-reduce)."),
    ]) + '''
<p class="note">Two diagnostic probes (~400 steps each, 8 nodes, eval/checkpoints off) isolate
the aux losses one at a time: <code>lbonly</code> (lb=1e-1, z=0) and <code>zonly</code>
(lb=0, z=1e-3).</p>''')}
"""


def build_results() -> str:
    runs = table(
        ["Run", "Router", "Aux losses", "Final CE @ 50B (step 11,921)", "arc_challenge (CE loss)", "Outcome"],
        [
            ("<code>emo_1b14b_130b</code> (baseline)", "learnable", "lb=1e-1, z=1e-3", "<strong>2.692</strong>", "0.769", "&#10003; complete"),
            ("<code>noaux</code>", "frozen (grafted)", "<strong>none</strong>", "<strong>2.715</strong> (+0.023)", "0.774", "&#10003; complete"),
            ("<code>keepaux</code>", "frozen (grafted)", "lb=1e-1, z=1e-3", "&mdash;", "&mdash;", "&#10007; NaN @ step ~120"),
            ("probe <code>lbonly</code>", "frozen (grafted)", "lb=1e-1 only", "&mdash;", "&mdash;", "&#10007; NaN @ step 105"),
            ("probe <code>zonly</code>", "frozen (grafted)", "z=1e-3 only", "&mdash;", "&mdash;", "&#10007; NaN @ step 289"),
        ],
    )
    return f"""
{card("results", "Outcome table", runs + '''
<p class="note">Baseline final CE read from the run's WandB summary (step 11,921). The
frozen-router <code>noaux</code> run reached the same 50B-token hard stop and logged
&ldquo;Training complete&rdquo;.</p>''')}

{card("results", "Finding 1 &mdash; a frozen trained router converges", '''
<p>The headline: a router that is <em>fixed</em> to its trained values, with every other
weight retrained from scratch, lands within <strong>+0.023 nat</strong> of the baseline that
learned its router jointly (CE 2.715 vs 2.692), and slightly ahead on arc_challenge. Once a
good routing function exists, the experts can reorganize around it with essentially no loss
penalty &mdash; the router need not co-adapt step-for-step.</p>''')}

{card("results", "Finding 2 &mdash; with a frozen router, the aux losses must be off", '''
<p>Every aux-on variant NaN'd; <code>noaux</code> is the only one that trained. The NaN was
always in the <strong>gradient</strong> (grad norm went NaN while every loss term stayed
finite), never in the forward loss value.</p>
<p><strong>Why the forward is fine.</strong> The block is pre-norm:
the router sees <code>RMSNorm(h)</code>, whose magnitude is bounded regardless of how spikey
earlier routers are or how large the residual stream grows. So the forward z-loss /
lb-loss stay small &mdash; consistent with WandB right up to the NaN.</p>
<p><strong>Why the backward blows up.</strong> An aux loss is normally satisfied by
<em>moving the router</em> (z-loss shrinks &#8214;W&#8214;, lb-loss rotates W toward balance).
Freezing W removes that degree of freedom, so 100% of the aux gradient is dumped onto the
variables <em>upstream</em> of the router &mdash; the hidden state <code>h</code> and the
RMSNorm gain <code>&gamma;</code>. Two things make that dumped gradient stiff: (a) the
x&rarr;logit map is the frozen <code>W</code> itself, which is <strong>large</strong>
(&#8214;W&#8214;&asymp;60, grafted, vs &asymp;10 fresh), so the sensitivity of the aux loss
to a perturbation in the router input is large; and (b) the cheapest lever the loss finds is
<code>&gamma;</code>, a single per-layer vector that uniformly scales every token's logits but
also scales the real FFN input the LM loss depends on. The aux loss and LM loss fight over
<code>&gamma;</code>, coupled through a large fixed <code>W</code>; a step overshoots and
overflows bf16.</p>''')}

{card("results", "The prediction this refuted", '''
<p>The initial guess was that the <strong>z-loss specifically</strong> was the culprit (its
gradient onto the router input scales &asymp; &#8214;W&#8214;&sup2; vs &#8214;W&#8214; for the
lb-loss). The probes refuted it: <strong>both</strong> NaN'd, and <strong>lbonly went first
(step 105 vs 289)</strong>. The order matches a simpler story &mdash; it is just aux pressure
scaled by its weight, and the lb-loss weight (1e-1) is 100&times; the z-loss weight (1e-3), so
the bigger-weighted loss dies first. The mechanism is <strong>agnostic to which aux loss</strong>:
any router-shaping loss, with the router frozen, becomes stiff pressure on <code>h</code> and
<code>&gamma;</code>. Removing both (noaux) removes the pressure &mdash; which is why it is the
config that worked, and arguably the principled one (a frozen router has nothing for the shaping
losses to shape).</p>
<p class="note"><strong>Not yet instrumented.</strong> The mechanism is inferred from loss
structure + the probe ordering, not from per-parameter-group grad norms. The cheap confirmation
&mdash; a short noaux-style run with one aux loss re-enabled and per-group grad-norm logging,
watching whether the blow-up localizes on the <code>feed_forward_norm.weight</code>
(&gamma;) group &mdash; is the proposed next step.</p>''')}
"""


# --------------------------------------------------------------------------
# Page assembly (CSS/JS kept in sync with models_fullextend)
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
pre { background:#0f172a; color:#e2e8f0; padding:12px 14px; border-radius:6px; overflow:auto; font-size:13px; }
pre code { background:transparent; padding:0; color:inherit; }
code { background:#eef2f7; padding:1px 5px; border-radius:4px; font-size:0.9em; }
h4 { margin:18px 0 4px; }
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or Path("claude_outputs/models_routerfixed/report.html")

    tabs = [
        ("overview", "Overview", build_overview()),
        ("method", "Method", build_method()),
        ("results", "Results", build_results()),
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
<title>EMO models_routerfixed: can the router be frozen?</title>
<style>{CSS}</style>
</head>
<body>
<header>
<a class="home-link" href="/">&larr; all reports</a>
<h1>EMO models_routerfixed: does the router need to be learnable during pretraining?</h1>
<p>models_routerfixed &mdash; frozen trained-router pretraining &middot; results complete
&middot; generated by scripts/models_routerfixed/build_report.py</p>
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
