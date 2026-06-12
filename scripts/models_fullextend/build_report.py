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
<p>Experiment in progress &mdash; no results yet. An incremental hyperparameter sweep
is running (16 nodes, hard-stopped at 50B tokens per config); see the
<strong>Sweep</strong> tab for the live config list. This page will gain
loss-trajectory and downstream &ldquo;add-an-expert&rdquo; results as runs complete.</p>''')}
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


def build_sweep() -> str:
    fixed = table(
        ["Held fixed", "Value"],
        [
            ("nodes", "16 (128 &times; H100)"),
            ("ghost_extend_num", "1"),
            ("ghost_extend_random_k", "8"),
            ("ghost_extend_route", "always (topk not implemented)"),
            ("max_duration", "130B tokens (LR cosine targets 130B)"),
            ("hard_stop", "50B tokens (per-config sweep budget)"),
            ("checkpointing", "2-deep rolling (keep_ephemeral=2) + final model"),
        ],
    )
    runs = table(
        ["Run", "coeff_mode", "route", "detach_coeff", "Status"],
        [
            ("emo_1b14b_130b_ghost_usage_always_detachF", "usage", "always", "false", "running"),
            ("&hellip; (uniform / random &times; detachF / detachT)", "&mdash;", "always", "&mdash;", "planned, chosen incrementally"),
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
<p class="note">Loss trajectories and the downstream &ldquo;instantiate a real new
expert and measure degradation&rdquo; evaluation will be added here as runs finish.</p>''')}
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
nav { display:flex; gap:6px; flex-wrap:wrap; padding:10px 28px; background:#1e293b; position:sticky; top:0; z-index:10; }
nav button { border:0; border-radius:6px; padding:7px 14px; font-size:14px; cursor:pointer;
             background:transparent; color:#cbd5e1; }
nav button:hover { background:#334155; }
nav button.active { background:#3b82f6; color:#fff; }
main { max-width:1180px; margin:0 auto; padding:24px 28px 80px; }
section.tab { display:none; }
section.tab.active { display:block; }
.card { background:var(--card); border:1px solid var(--line); border-left:4px solid var(--line);
        border-radius:8px; padding:16px 20px; margin:16px 0; }
.card h3 { margin:0 0 8px; font-size:15px; text-transform:uppercase; letter-spacing:0.05em; }
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
"""

JS = """
function show(id) {
  document.querySelectorAll('section.tab').forEach(s => s.classList.toggle('active', s.id === id));
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('active', b.dataset.target === id));
  history.replaceState(null, '', '#' + id);
}
document.querySelectorAll('nav button').forEach(b => b.addEventListener('click', () => show(b.dataset.target)));
show(location.hash && document.getElementById(location.hash.slice(1)) ? location.hash.slice(1) : 'overview');
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or Path("claude_outputs/models_fullextend/report.html")

    tabs = [
        ("overview", "Overview", build_overview()),
        ("method", "Method", build_method()),
        ("sweep", "Sweep", build_sweep()),
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
</head>
<body>
<header>
<h1>EMO models_fullextend: pretraining so new experts can be added post-training</h1>
<p>models_fullextend &mdash; ghost-expert training &middot; experiment in progress
&middot; generated by scripts/models_fullextend/build_report.py</p>
</header>
<nav>{nav}</nav>
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
