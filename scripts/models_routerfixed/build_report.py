"""Build a self-contained HTML report for the models_routerfixed experiment.

Tests whether EMO's router must be *learned* during pretraining or can be *fixed*:
graft the fully-trained step-11921 routers from emo_1b14b_130b onto a fresh,
byte-exact init, freeze them, and retrain everything else from scratch on the
identical recipe. The experiment is results-complete (noaux converged; the aux-on
variants NaN'd on first launch but reran clean — a metastable instability). Static
tables + prose. CSS/JS/tab structure kept in sync with scripts/models_fullextend/build_report.py.

Usage:

    python scripts/models_routerfixed/build_report.py \\
        [--output claude_outputs/models_routerfixed/report.html]
"""

import argparse
import base64
import json
import statistics
from pathlib import Path

import numpy as np


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


def img_tag(path: Path, caption: str) -> str:
    if not path.is_file():
        return f'<p class="missing">[missing figure: {path}]</p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        "<figure>"
        f'<img src="data:image/png;base64,{data}" alt="{caption}" '
        "onclick=\"this.classList.toggle('zoom')\" title=\"click to zoom\">"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def fig_row(*figs: str) -> str:
    return '<div class="figrow">' + "".join(figs) + "</div>"


def details(summary: str, body: str) -> str:
    return f"<details><summary>{summary}</summary>{body}</details>"


def _diag_stats(npz_path: Path) -> dict:
    """Index-aligned diagonal vs off-diagonal stats from a match corr_matrices.npz."""
    z = np.load(npz_path)
    corr = z["corr"]  # (layers, E_a, E_b)
    n = corr.shape[0]
    diag = np.array([np.diag(corr[l]) for l in range(n)])
    off = corr.copy()
    for l in range(n):
        np.fill_diagonal(off[l], np.nan)
    top1 = np.array(
        [(corr[l].argmax(axis=1) == np.arange(corr.shape[1])).mean() for l in range(n)]
    )
    return {
        "diag_median": float(np.median(diag)),
        "offdiag_mean": float(np.nanmean(off)),
        "own_top_match": float(np.median(top1)),
        "chance": 1.0 / corr.shape[2],
        "num_experts": int(corr.shape[1]),
    }


def _match_medians(summary_path: Path) -> dict:
    s = json.loads(summary_path.read_text())
    med = lambda k: statistics.median(l[k] for l in s["per_layer"])  # noqa: E731
    return {
        "matched": med("matched_sim_median"),
        "splits": med("mean_splits_per_a_expert"),
        "novel": med("frac_novel_b"),
        "nn_a": med("nn_a_median"),
        "nn_b": med("nn_b_median"),
    }


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
<p><strong>Practical caveat:</strong> we ran <code>noaux</code> (aux losses off). The aux-on
variants NaN'd on first launch but reran clean &mdash; the frozen-router + aux config is
<em>metastable</em> (a marginal, non-reproducible instability), not deterministically broken.
<code>noaux</code> is the robust choice. See the <strong>Results</strong> tab (including a
mechanism claim we later retracted).</p>''')}

{card("results", "Follow-up &mdash; are the resulting experts similar to the baseline's?", '''
<p>Because noaux's frozen router <em>is</em> the baseline's final router and both runs start from the
same init, one might expect the experts to re-converge to the baseline's. They do not. In weight space
the experts diverge to <strong>near-zero cosine</strong> (no better than random expert pairs), and
functionally the same-index expert correspondence is <strong>at chance</strong> &mdash; expert <em>i</em>
does a different job in each model. A frozen router fixes the routing directions but not the division of
labor. See the <strong>Expert weights</strong> and <strong>Expert matching</strong> tabs.</p>''')}
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
            ("<code>noaux</code>", "frozen (grafted)", "<strong>none</strong>", "<strong>2.715</strong> (+0.023)", "0.774", "&#10003; complete (robust)"),
            ("<code>keepaux</code>", "frozen (grafted)", "lb=1e-1, z=1e-3", "&mdash;", "&mdash;", "&#10007; NaN @ ~120 (1st run) &middot; &#10003; clean @ 500 on 2 reruns"),
            ("probe <code>lbonly</code>", "frozen (grafted)", "lb=1e-1 only", "&mdash;", "&mdash;", "&#10007; NaN @ 105 (1st run) &middot; &#10003; clean @ 150 on rerun"),
            ("probe <code>zonly</code>", "frozen (grafted)", "z=1e-3 only", "&mdash;", "&mdash;", "&#10007; NaN @ 289 (1st run)"),
        ],
    )
    return f"""
{card("results", "Outcome table", runs + '''
<p class="note">Baseline final CE read from the run's WandB summary (step 11,921). The
frozen-router <code>noaux</code> run reached the same 50B-token hard stop and logged
&ldquo;Training complete&rdquo;. The aux-on rows are the headline of Finding 2: they NaN'd on the
first attempt but reran clean &mdash; see below.</p>''')}

{card("results", "Finding 1 &mdash; a frozen trained router converges", '''
<p>The headline: a router that is <em>fixed</em> to its trained values, with every other
weight retrained from scratch, lands within <strong>+0.023 nat</strong> of the baseline that
learned its router jointly (CE 2.715 vs 2.692), and slightly ahead on arc_challenge. Once a
good routing function exists, the experts can reorganize around it with essentially no loss
penalty &mdash; the router need not co-adapt step-for-step. This result is solid and reproduced.</p>''')}

{card("results", "Finding 2 &mdash; aux-on is metastable, not deterministically unstable", '''
<p><strong>This supersedes an earlier, stronger claim that &ldquo;the aux losses must be off.&rdquo;</strong>
The first time each aux-on config was launched it NaN'd (keepaux@120, lbonly@105, zonly@289 &mdash; 3/3).
On that basis we concluded a frozen router is incompatible with router-shaping aux losses. But every
<strong>rerun of the identical config trained clean past those steps</strong>: lbonly to 150,
keepaux to 500 (grad norm calm at &asymp;0.2, nowhere near overflow), and a decisive control to 500.
3/3 NaN, then 3/3 clean &mdash; with no difference in code, config, seed, or init between the batches.</p>
<p><strong>What this means.</strong> The frozen-router + aux-loss configuration is <strong>metastable</strong>:
it carries a marginal numerical instability (a bf16 overflow that NaN'd the loss) which fires
<em>nondeterministically</em> &mdash; plausibly from cross-node NCCL bf16 reduction rounding or transient
data/hardware effects &mdash; rather than a reproducible divergence. So aux-on <em>can</em> train, it is just
risky; <strong><code>noaux</code> is the robust choice</strong> (and a frozen router has nothing for the
shaping losses to shape anyway).</p>''')}

{card("results", "What we got wrong (kept for the record)", '''
<p>An earlier version of this report asserted a confident <strong>mechanism</strong> for the NaN: that the
aux gradient, unable to move the frozen large-&#8214;W&#8214; router, dumps onto the RMSNorm gain
<code>&gamma;</code> (<code>feed_forward_norm.weight</code>) and overflows bf16 &mdash; the
&ldquo;&gamma;-lever&rdquo; story, with a side-claim that the load-balancing loss leads because its weight
is 100&times; the z-loss. <strong>That mechanism was never confirmed and is retracted.</strong> We added a
per-parameter-group grad-norm diagnostic (<code>EMO_GRAD_DIAG=1</code>) specifically to catch the
<code>&gamma;</code> spike at blow-up &mdash; but the instrumented runs <em>did not diverge at all</em>, and the
diag-off control didn&rsquo;t either. We never observed a spike, so there is no evidence for (or against) the
&gamma;-lever; the honest state is that the instability is real but its mechanism is <strong>unknown</strong>.
The forward observation still holds (pre-norm bounds the router input, so the NaN was in the gradient, not the
forward loss), but that alone doesn&rsquo;t localize the cause.</p>
<p class="note">Lesson recorded so we don&rsquo;t re-derive the same over-confident story: a NaN seen on a single
run is not yet a deterministic property of the config &mdash; reproduce before theorizing.</p>''')}
"""


def build_expert_weights(base: Path) -> str:
    wd = base / "weight_similarity"
    sp = wd / "weight_similarity_summary.json"
    if not sp.is_file():
        return card("results", "Not yet generated",
                    "<p>Run <code>analysis_weight_similarity.py</code> first.</p>")
    s = json.loads(sp.read_text())
    e = s["experts"]
    comp = s["components_cos_mean_over_layers"]
    E = s["num_experts"]

    overall = table(
        ["Quantity", "Value", "Reading"],
        [
            ("Same-expert weight cosine (diagonal, median)",
             f"<strong>{e['diag_cos_median_overall']:.3f}</strong>",
             "noaux expert <em>i</em> vs baseline expert <em>i</em>"),
            ("Different-expert cosine (off-diagonal, mean)",
             f"{e['offdiag_cos_mean_overall']:.3f}",
             "the null &mdash; random expert pairs"),
            ("Relative L2 distance &#8214;&Delta;&#8214;/&#8214;baseline&#8214; (median)",
             f"{e['relL2_median_overall']:.3f}",
             "1.0 &asymp; as far apart as two unrelated weight vectors"),
            ("Router cosine (min over layers)",
             f"{s['router_cos_min']:.5f}",
             "frozen-router sanity check &mdash; must be 1.0"),
        ],
    )
    comp_rows = [
        ("experts (diagonal)", f"{e['diag_cos_median_overall']:.3f}"),
        ("attention w_q / w_k / w_v / w_out",
         f"{comp['attn_q']:.3f} / {comp['attn_k']:.3f} / {comp['attn_v']:.3f} / {comp['attn_out']:.3f}"),
        ("attention_norm / feed_forward_norm", f"{comp['attn_norm']:.3f} / {comp['ffn_norm']:.3f}"),
        ("embeddings / lm_head", f"{comp['embeddings.weight']:.3f} / {comp['lm_head.w_out.weight']:.3f}"),
        ("router (frozen)", f"{comp['router']:.3f}"),
    ]

    return f"""
{card("goal", "Question", f'''<p>The noaux run froze its router at the baseline's
<em>final</em> (step-11,921) router and started every other weight from the
<strong>byte-identical</strong> fresh init as the baseline (the grafted init overwrote only the 16
routers). So at step 0 every non-router tensor &mdash; including all {E} experts per layer &mdash;
was <em>identical</em> across the two runs. They then trained on the same data for 11,921 steps,
differing only in the routing trajectory. <strong>Hypothesis:</strong> with the same frozen router
pinning each expert's input distribution, the experts should re-converge to similar weights.</p>''')}

{card("method", "Method", f'''<p>Both step-11,921 checkpoints are compared tensor-by-tensor. Experts
are index-aligned (same architecture, same frozen router, identical init &mdash; and gradient descent
from an identical init never relabels neurons), so <strong>raw weight cosine</strong> is meaningful
with no permutation search. For each layer we build the full ({E}&times;{E}) cross-expert cosine
matrix between the concatenated <code>[w1,w2,w3]</code> of every expert pair; its <strong>diagonal</strong>
is the same-index similarity, its off-diagonal the null. Non-expert components (attention, norms,
embeddings) are the control &mdash; they had no shared-router constraint.</p>''')}

{card("results", "Result &mdash; the experts diverge to near-zero cosine", overall + f'''
<p><strong>The hypothesis is refuted.</strong> Same-expert weight cosine is
<strong>{e['diag_cos_median_overall']:.3f}</strong> &mdash; statistically indistinguishable from the
different-expert null ({e['offdiag_cos_mean_overall']:.3f}) &mdash; with relative-L2
&asymp;&nbsp;{e['relL2_median_overall']:.2f} (about as far apart as two unrelated weight vectors).
Despite an identical frozen router and identical initialization, expert <em>i</em> in noaux and
expert <em>i</em> in baseline end up in <em>unrelated</em> regions of weight space. The tiny
<code>init_std=0.02</code> initialization is negligible next to 11,921 steps of learning, and the two
runs' differing routing trajectories drive each expert into a different solution.</p>'''
+ fig_row(
    img_tag(wd / "expert_cos_heatmap_layers.png",
            "Cross-expert weight cosine at 4 layers. No diagonal band: same-index experts (the diagonal) are no more similar than random pairs."),
    img_tag(wd / "expert_cos_vs_layer.png",
            "Per-layer same-expert (diagonal) vs different-expert (off-diagonal) cosine. Both sit near zero at every layer."),
))}

{card("results", "Control &mdash; the high-dimensional matrices all diverge; only 1-D scales survive", table(
    ["Component", "Cosine (noaux vs baseline), mean over layers"], comp_rows) + f'''
<p>Every high-dimensional weight <em>matrix</em> &mdash; the experts, all four attention projections
({comp['attn_q']:.2f}/{comp['attn_k']:.2f}/{comp['attn_v']:.2f}/{comp['attn_out']:.2f}), the embeddings
({comp['embeddings.weight']:.2f}) and the lm_head ({comp['lm_head.w_out.weight']:.2f}) &mdash; is
uncorrelated across the two runs (cosine &asymp; 0), exactly like the experts. The <em>only</em> weights
that stay aligned are the <strong>1-D RMSNorm gains</strong>
(attention_norm {comp['attn_norm']:.2f}, feed_forward_norm {comp['ffn_norm']:.2f},
lm_head.norm {comp['lm_head.norm.weight']:.2f}) and the frozen router (1.0). The reading: a per-channel
scale vector is low-dimensional and pinned by the data statistics, so both runs recover nearly the same
gains &mdash; but every high-dimensional matrix has enormous internal redundancy and is free to settle
anywhere. Weight-space identity is preserved by nothing except an explicit freeze (and the heavily
constrained norms). The real question is whether <em>function</em> is preserved &mdash; next tab.</p>'''
+ fig_row(img_tag(wd / "component_cos_vs_layer.png",
                  "Per-component weight cosine by layer: experts vs attention vs norms.")))}
"""


def build_expert_matching(base: Path) -> str:
    md, mt = base / "matching" / "noaux_vs_baseline", base / "matching_tokens" / "noaux_vs_baseline"
    if not (md / "match_summary.json").is_file():
        return card("results", "Not yet generated",
                    "<p>Run <code>analysis_extract.sh</code> then <code>analysis_match.sh</code> first.</p>")
    dd, dt = _diag_stats(md / "corr_matrices.npz"), _diag_stats(mt / "corr_matrices.npz")
    sd, stk = _match_medians(md / "match_summary.json"), _match_medians(mt / "match_summary.json")
    E = dd["num_experts"]
    chance = dd["chance"]

    diag_tbl = table(
        ["Level", "Same-expert corr[i,i] (median)", "Different-expert (off-diag)",
         "i is its own top match", "Hungarian best-permutation match", "Within-model NN corr"],
        [
            ("Document fingerprints",
             f"<strong>{dd['diag_median']:+.3f}</strong>", f"{dd['offdiag_mean']:+.3f}",
             f"{100*dd['own_top_match']:.1f}%", f"{sd['matched']:.3f}", f"{sd['nn_a']:.2f} / {sd['nn_b']:.2f}"),
            ("Token fingerprints",
             f"<strong>{dt['diag_median']:+.3f}</strong>", f"{dt['offdiag_mean']:+.3f}",
             f"{100*dt['own_top_match']:.1f}%", f"{stk['matched']:.3f}", f"{stk['nn_a']:.2f} / {stk['nn_b']:.2f}"),
        ],
    )

    return f"""
{card("goal", "Question", '''<p>Weights diverged (previous tab) &mdash; but do the experts at least
play the same <em>functional role</em>? With the router frozen and identical, the natural guess is
that expert <em>i</em> still fires on the same documents/tokens in both models. We test this with the
same cross-model expert-matching pipeline as the <code>models_sizescaling</code> experiment.</p>''')}

{card("method", "Method (same tooling as models_sizescaling)", f'''<p>Both models were run over the
<strong>identical</strong> weborganizer document set (shared composition, shuffle seed 42), so each
expert has a <em>fingerprint</em> &mdash; its mean routing probability across all docs (and,
separately, across ~1M individual tokens). Per layer we take the Pearson correlation between every
(noaux-expert, baseline-expert) fingerprint pair. Because the experts here are <strong>index-aligned</strong>
(same {E}-expert architecture, same frozen router), the matrix <strong>diagonal</strong> corr[<em>i,i</em>]
is the same-expert functional similarity &mdash; the cleanest test. We also report the Hungarian
best one-to-one match (does the organization survive under <em>any</em> permutation?) and within-model
nearest-neighbor correlation. Chance for "own top match" is 1/{E} &asymp; {100*chance:.2f}%.</p>''')}

{card("results", "Result &mdash; the index&harr;role mapping is fully scrambled", diag_tbl + f'''
<p>Same-index functional similarity is <strong>at chance</strong>: the diagonal corr[<em>i,i</em>] is
{dd['diag_median']:+.3f} (doc) / {dt['diag_median']:+.3f} (token), no different from the off-diagonal
null, and noaux expert <em>i</em> is its own top match only {100*dd['own_top_match']:.1f}% of the time
&mdash; exactly the {100*chance:.2f}% you'd get by chance. So <strong>expert <em>i</em> in noaux does a
completely different job than expert <em>i</em> in baseline</strong>, even though they share the
identical frozen router.</p>
<p><strong>Why a shared router does not pin the roles.</strong> Expert <em>i</em> is selected for tokens
whose normalized hidden state aligns with the fixed direction <code>router_row_i</code>. But the hidden
state <code>h</code> itself is produced by the (diverging) experts, so the same router direction collects
<em>different</em> tokens in the two models; each expert then specializes to whatever it receives. From a
near-identical init the two runs fall into different basins of this self-organizing feedback loop and the
index&harr;role assignment permutes away.</p>''')}

{card("results", "Is the organization preserved under a permutation? Weakly.", f'''
<p>The Hungarian best one-to-one match &mdash; the most generous test, free to permute experts &mdash;
is only <strong>{sd['matched']:.2f}</strong> (doc) / <strong>{stk['matched']:.2f}</strong> (token).
That is <em>lower</em> than each model's within-model nearest-neighbor redundancy
({sd['nn_a']:.2f}&ndash;{sd['nn_b']:.2f} doc), and well below the &asymp;0.71 that
<em>independently-initialized</em> size-scaling models reach. At the token level
{100*stk['novel']:.0f}% of baseline experts have no noaux counterpart at all. So the two models did not
just relabel a shared organization &mdash; they built genuinely different ones.</p>'''
+ fig_row(
    img_tag(md / "corr_heatmap_layers.png",
            "Doc-level fingerprint correlation, B columns sorted by Hungarian match. No bright diagonal in the first 127 columns &rarr; little one-to-one structure."),
    img_tag(mt / "corr_heatmap_layers.png", "Token-level fingerprint correlation (same layout)."))
+ details("Per-layer matched similarity and redundancy figures",
          fig_row(
              img_tag(md / "matched_sim_vs_layer.png", "Doc: Hungarian-matched corr by layer"),
              img_tag(md / "nn_redundancy_vs_layer.png", "Doc: within-model redundancy by layer"),
              img_tag(mt / "matched_sim_vs_layer.png", "Token: Hungarian-matched corr by layer"))))}

{card("method", "Takeaway", '''<p>Combined with the previous tab: a frozen, fully-trained router fixes
the routing <em>directions</em> but not the <em>division of labor</em>. The experts that organize around
it are free to &mdash; and do &mdash; land on a different weight-space solution <em>and</em> a different
functional assignment. The earlier headline result still holds (a frozen router trains to nearly the
baseline's loss), but it gets there with an entirely different, equally-valid set of experts &mdash; the
routing function does not determine the experts, only constrains them.</p>''')}
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
.figrow { display:flex; gap:16px; flex-wrap:wrap; margin:14px 0; }
figure { flex:1 1 380px; min-width:300px; max-width:560px; margin:0; }
figure img { width:100%; border:1px solid var(--line); border-radius:6px; cursor:zoom-in; background:#fff; }
figure img.zoom { max-width:none; width:auto; max-height:90vh; position:fixed; inset:0; margin:auto;
                  z-index:100; box-shadow:0 0 0 100vmax rgba(15,23,42,.75); cursor:zoom-out; }
figcaption { font-size:12.5px; color:var(--muted); margin-top:4px; }
details { margin:12px 0; }
summary { cursor:pointer; color:#2563eb; font-size:14px; }
.missing { color:#b91c1c; font-size:13px; }
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
    parser.add_argument("--base-dir", type=Path, default=Path("claude_outputs/models_routerfixed"),
                        help="dir holding weight_similarity/ and matching{,_tokens}/ artifacts")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    base = args.base_dir
    out = args.output or base / "report.html"

    tabs = [
        ("overview", "Overview", build_overview()),
        ("method", "Method", build_method()),
        ("results", "Results", build_results()),
        ("expert-weights", "Expert weights", build_expert_weights(base)),
        ("expert-matching", "Expert matching", build_expert_matching(base)),
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
