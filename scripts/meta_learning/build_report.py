# PARENT: "scripts/modular_extension/build_report.py" (visual framework: CSS/JS/tab structure,
#          card/table/figure helpers kept in sync)
# DESCRIPTION:
#     Builds the meta_learning experiment report at claude_outputs/meta_learning/report.html.
#     Structured as three sections — (1) Overview, (2) Implementation, (3) Results — kept concise.
#     Only working-set arms are reported; the original non-working-set FOMAML runs (an incorrect
#     implementation: outer gradient applied to ALL experts) are intentionally excluded.
#     Phase-1 numbers are the final 20B evals pulled from W&B (hardcoded below); phase-2 figures
#     come from claude_outputs/meta_learning/{k32cpt_*.png,k32cpt_results.json}; the live-curve
#     explorer reads curves.json (refresh with scripts/meta_learning/fetch_report_data.py).
#
#   python scripts/meta_learning/build_report.py   # then scripts/publish_reports.sh to deploy
##############################################################

import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "claude_outputs/meta_learning"
FIGS = OUT / "figs"

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
# Final phase-1 evals (20B, step 4769), pulled from W&B project emo-extension.
# gap = mean_source(lm-pool32 CE) - mean_source(lm-full CE); negative = selective
# inference is at least as good as full. vanilla shown at its latest (31B) step.
# --------------------------------------------------------------------------
PHASE1 = [
    # (arm, objective, full CE, gap, note)
    ("vanilla", "baseline EMO-128e (random pools 8&ndash;128)", "2.67*", "&minus;0.041",
     "reference; *at 31B (more tokens)"),
    ("sametok_ws", "working-set FOMAML, &lambda;=0", "2.799", "+0.366",
     "gap never closed"),
    ("heldout_ws", "cross-data working-set FOMAML, &lambda;=0", "3.109", "+0.326",
     "gap never closed"),
    ("sametok_ws_lam05", "working-set FOMAML + &lambda;=0.5 anchor", "2.745", "&minus;0.012",
     "gap solved; best full CE"),
    ("heldout_ws_lam05", "cross-data ws FOMAML + &lambda;=0.5 anchor", "2.764", "&minus;0.012",
     "gap solved"),
    ("seq_ws", "sequential ablation (committed selective step, no probe)", "2.750", "&minus;0.027",
     "matches the probe arms"),
    ("sametok_ws_lam05_adam", "Adam-preconditioned pseudo-step, scale band [1,32]", "3.053",
     "&minus;0.023", "full CE regressed +0.30"),
]

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
  <text x="484" y="158" text-anchor="middle" class="s">experts only &middot; SGD or Adam-precond</text>
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

# Fixed arm order + colors for the live-curve explorer (working-set arms only).
ARM_COLORS = [
    ("vanilla", "#2a78d6"),
    ("sametok_ws", "#1baf7a"),
    ("heldout_ws", "#eda100"),
    ("sametok_ws_lam05", "#e87ba4"),
    ("heldout_ws_lam05", "#4a3aa7"),
    ("seq_ws", "#e34948"),
]


# --------------------------------------------------------------------------
# (1) OVERVIEW
# --------------------------------------------------------------------------


def build_overview() -> str:
    goal = card(
        "goal",
        "Goal",
        "<p>Pretrain a 128-expert EMO model from scratch so that <b>training on only a document's "
        "32-expert working set also improves the full 128-expert model</b>. Each step takes one "
        "selective (32-expert) gradient and asks the full model to be better one selective step "
        "ahead (first-order MAML).</p>"
        "<p><b>Motivation.</b> In modular_extension, post-hoc <i>k=32 selective CPT</i> on a vanilla "
        "EMO model improved the trained cluster in the <i>selective</i> view (&minus;0.164 nats) "
        "but <b>almost none of it transferred to the full model</b> (&minus;0.022), and it degraded "
        "the pool elsewhere. Vanilla EMO is great at selective <i>inference</i>; what fails is "
        "selective <i>update transfer</i>. This experiment tries to build that property into "
        "pretraining.</p>",
    )
    two_q = card(
        "goal",
        "Two questions (don't conflate them)",
        "<ul>"
        "<li><b>Selective inference</b> &mdash; is the model as good with 32 experts as with 128? "
        "Measured by the <b>gap</b> = CE(pool 32) &minus; CE(full 128) on a held-out mix "
        "(negative = selective is as good). This is the phase-1 headline.</li>"
        "<li><b>Selective update transfer (the real test)</b> &mdash; does a 32-expert CPT step on "
        "the finished model improve the <i>full</i> model? Measured by re-running the k=32 CPT "
        "protocol (phase 2). The gap does <i>not</i> answer this.</li>"
        "</ul>",
    )
    rows = [
        [f"<code>meta128_{a}</code>", obj, fce, gap, note] for (a, obj, fce, gap, note) in PHASE1
    ]
    arms = card(
        "results",
        "Arms at a glance (20B / step 4769)",
        table(["run", "objective", "full CE", "gap", "one-line result"], rows)
        + "<p class='note'>Shared config: 128 experts (1 shared), top-k 8, lr 2e-3 flat WSD "
        "(warmup 2000), lb 1e-1, OLMoE-mix-0824, 20B tokens (4,768 steps), fixed ckpts at 10B/20B, "
        "W&amp;B project <code>emo-extension</code> tag <code>meta_learning</code>. "
        "<b>Only working-set arms are shown</b> &mdash; the original non-working-set FOMAML runs "
        "(outer gradient applied to all experts) were an incorrect implementation and are "
        "excluded. The <code>_adam</code> arm and its 4-node re-runs (scale [1,1] / [1,4]) are the "
        "current redesign (see Results).</p>",
    )
    return goal + two_q + arms


# --------------------------------------------------------------------------
# (2) IMPLEMENTATION
# --------------------------------------------------------------------------


def build_implementation() -> str:
    step = card(
        "method",
        "One optimizer step (two-pass first-order MAML)",
        "<p>Every batch is consumed once, by a single optimizer step with two forward/backward "
        "passes:</p>" + STEP_SVG + "<ol>"
        "<li><b>Inner pass (selective)</b>: all sequences forwarded with each document restricted "
        "to its router top-32 experts. Backward gives one DP-averaged gradient <code>g_inner</code> "
        "in which each expert's slice comes only from documents that select it.</li>"
        "<li><b>Pseudo-step</b>: a temporary probe on the expert weights only, "
        "<code>&theta;&prime;_exp = &theta;_exp &minus; &alpha;&middot;g_inner</code>. Router, "
        "attention, embeddings, norms untouched.</li>"
        "<li><b>Outer pass (full)</b>: the same tokens with all 128 experts at &theta;&prime;. Its "
        "gradient is the step's only real update, but <b>each expert receives gradient only from "
        "documents whose top-32 working set contains it</b> (non-expert params get the full "
        "gradient).</li>"
        "<li><b>Restore + update</b>: expert weights restored bitwise to &theta;, then AdamW "
        "consumes the accumulated gradient (evaluated at &theta;&prime;, applied at &theta;).</li>"
        "</ol>"
        "<p>Net effect: no full-routing update ever exists &mdash; every expert only trains from "
        "documents that select it, toward the full-model loss one selective step ahead.</p>",
    )
    pseudo = card(
        "method",
        "Pseudo-step variants (what the redesign changes)",
        table(
            ["variant", "&theta;&prime; is&hellip;", "step size", "why"],
            [
                ["<code>inner_optim=sgd</code> (original)",
                 "&theta; &minus; &alpha;&middot;g_inner (raw SGD)",
                 "fixed &alpha; (3e-1), clip 10",
                 "simple; but tiny deltas are quantized away by the bf16 all-gather, and raw-SGD "
                 "scale &ne; the real AdamW step the model actually takes"],
                ["<code>inner_optim=adam</code> (redesign)",
                 "the AdamW step the real optimizer WOULD take on g_inner, from the live expert "
                 "moments (read-only; moments consumed unchanged by the outer step)",
                 "<code>inner_lr_mode=match_lr</code>: base = live scheduler lr; per-step scale "
                 "band <code>[min,max]</code>",
                 "&theta;&prime; is the true &ldquo;next selective step&rdquo;, immune to bf16 "
                 "self-extinction (Adam delta ~lr/coord regardless of gradient scale)"],
            ],
        )
        + "<p class='note'>The magnitude band samples a per-step log-uniform displacement scale "
        "(the update-magnitude analog of vanilla's random pools). Knobs: "
        "<code>inner_optim</code>, <code>inner_lr_mode</code>, "
        "<code>inner_lr_scale_min/max</code>, <code>inner_pool_size=32</code>, "
        "<code>outer_expert_update=working_set</code>, <code>lambda_inner</code> (direct weight of "
        "L_inner in the real update), <code>lb_on_inner</code>.</p>",
    )
    detach = card(
        "method",
        "Working-set detach (early layers lose nothing)",
        "<p>Masking non-working-set experts must not sever backprop to earlier layers. At an "
        "expert output <code>y=f(x;W)</code>, backward computes <code>&part;L/&part;W</code> "
        "(the weight gradient) and <code>&part;L/&part;x</code> (passed to earlier layers) "
        "independently. Masked slots are recomputed under <code>W.detach()</code>: identical "
        "forward value, weights act as constants (like a frozen layer), full <code>&part;L/&part;x"
        "</code> preserved. Verified bitwise (gate checks M1/M2); expert grads confined (M3).</p>",
    )
    mech = card(
        "method",
        "Code map &amp; mechanics",
        table(
            ["file", "role"],
            [
                ["<code>train_module/transformer/meta_learning.py</code>",
                 "the two-phase train step; <code>meta_mode=vanilla</code> delegates to the stock "
                 "module for a bit-identical baseline. Owns <code>_apply_pseudo_step</code> "
                 "(sgd | adam)."],
                ["<code>nn/moe/&hellip;_randpool_router.py</code>",
                 "per-pass flags: <code>meta_force_pool</code> (pin pool 32 inner / keep-all "
                 "outer), <code>meta_skip_aux</code>, <code>meta_outer_detach_top_e</code> "
                 "(working-set mask)."],
                ["<code>train/callbacks/pool_pinned_lm_evaluator.py</code>",
                 "runs the held-out mix at pool 128 (<code>lm-full</code>) and pinned 32 "
                 "(<code>lm-pool32</code>) &rarr; the gap."],
                ["<code>src/scripts/train/olmoe-1B-7B_fsl_meta.py</code>",
                 "entry script; meta knobs are dotted <code>--train_module.&lt;knob&gt;</code> "
                 "overrides."],
                ["<code>scripts/meta_learning/model_scripts/verify_meta_step.py</code>",
                 "mechanism-correctness gate &mdash; <b>15/15 pass</b> (&alpha;=0 &equiv; single "
                 "full pass bitwise; directional-derivative 0.76%; manual autograd reference "
                 "&le;5e-8 at &lambda;&isin;{0,0.5}; Adam delta matches hand-rolled AdamW; expert "
                 "moments bitwise unchanged across the step; bitwise restore)."],
            ],
        )
        + "<p class='note'>FSDP2 (bf16 param / fp32 reduce, compile on). The pseudo-step is applied "
        "in place on the fp32 local shards; the outer all-gather picks it up; restore copies the "
        "stashed pre-step shards back. Two backward cycles per step = two grad-accumulation rounds "
        "without an optimizer step. Extra memory ~2 fp32 expert-shard clones (~1.6 GB/rank at 8 "
        "nodes). 4-node runs need <code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code> + "
        "<code>rank_microbatch_size=8192</code> (the 8-node arm peaked at 96% reserved memory).</p>",
    )
    phase2 = card(
        "method",
        "Phase-2 protocol (the update-transfer test)",
        "<p>On each arm's 20B checkpoint, re-run the modular_extension <b>k=32 CPT (no extension)</b> "
        "protocol: the arm's own router clusters the shared 20B&ndash;40B doc window into 32 groups; "
        "per cluster its per-layer top-32-of-127 experts are extracted into a 33-expert subset, "
        "trained on the cluster's token share (standard 32-expert training, non-expert params "
        "frozen, optimizer carried), and written back. After every stage the <b>full 128-expert "
        "pool</b> is evaluated on all 32 cluster held-outs &rarr; the heatmaps in Results. "
        "Upper bound = <code>meta128_vanilla/step9537</code> (the same 20B&ndash;40B tokens as plain "
        "EMO pretraining). A finer <b>step-probe</b> evaluates the full model after each of the "
        "first 10 single CPT steps.</p>",
    )
    return step + pseudo + detach + mech + phase2


# --------------------------------------------------------------------------
# (3) RESULTS
# --------------------------------------------------------------------------


def build_phase1_result() -> str:
    rows = [
        [f"<code>meta128_{a}</code>", fce, gap, note] for (a, _obj, fce, gap, note) in PHASE1
    ]
    return card(
        "results",
        "Result 1 &mdash; Phase-1 selective-inference gap",
        table(["arm", "full CE", "gap (pool32&minus;full)", "note"], rows)
        + "<p><b>What it tested.</b> Whether working-set FOMAML pretraining makes the 128-expert "
        "model usable with only 32 experts (gap &le; vanilla's &minus;0.04).</p>"
        "<p><b>Conclusion.</b> The <b>&lambda;=0 arms fail</b> (gap +0.37 / +0.33): masking WHERE "
        "gradients land is not selective-function training &mdash; with no restricted-forward loss "
        "in the objective, deletion-robustness is forfeited. Adding a <b>&lambda;=0.5 anchor</b> "
        "(a real restricted-forward loss) closes the gap to vanilla's level (&minus;0.012) while "
        "keeping full CE competitive (2.745). But the <b>sequential ablation</b> (<code>seq_ws</code>, "
        "a committed selective step and no probe) matches the probe arms (&minus;0.027, 2.750) "
        "&mdash; so the FOMAML probe adds little over ordinary sequential training <i>for this "
        "metric</i>.</p>"
        "<p class='note'><b>What to try next:</b> the gap is the wrong headline &mdash; it is "
        "solved by &lambda;-anchoring and does not distinguish the probe from sequential training. "
        "Move the verdict to phase-2 update transfer (Result 2).</p>",
    )


def build_phase2_result() -> str:
    res_path = OUT / "k32cpt_results.json"
    anchor_tbl = ""
    if res_path.exists():
        res = json.load(open(res_path))
        rows = []
        for arm in res["arms"]:
            base = {int(k): v for k, v in res["anchors"][arm]["base_20B"].items()}
            upper = res["anchors"][arm].get("upper_40B")
            upper = {int(k): v for k, v in upper.items()} if upper else None
            n_done = len(res["arms"][arm]["stages"])
            rows.append([
                arm,
                f"{sum(base.values())/32:.4f}",
                f"{sum(upper.values())/32:.4f}" if upper else "&mdash;",
                f"{(sum(upper.values())-sum(base.values()))/32:+.4f}" if upper else "&mdash;",
                f"{n_done}/32",
            ])
        anchor_tbl = table(
            ["arm", "20B base mean CE", "40B plain-CPT (upper) CE", "upper-bound &Delta;",
             "stages"], rows)

    hm = fig_row(
        img_tag(OUT / "k32cpt_heatmap_vanilla.png",
                "vanilla: full-model CE delta vs its 20B base. Blue diagonal = a selective CPT "
                "stage improves the just-trained cluster in the FULL-routing view, and columns "
                "stay blue (specialization sticks). Bottom row = 40B plain-CPT upper bound."),
        img_tag(OUT / "k32cpt_heatmap_sametok_ws_lam05.png",
                "sametok_ws_lam05: the same protocol HURTS the full model almost everywhere, "
                "including the just-trained cluster (red diagonal)."))

    probe_tbl = table(
        ["after", "own cluster-0 heldout", "&Delta; vs base", "mean over 32 heldouts", "&Delta;"],
        [
            ["base", "2.985", "&mdash;", "2.795", "&mdash;"],
            ["step 1", "3.064", "<b>+0.079</b>", "2.816", "+0.020"],
            ["step 2", "3.050", "+0.064", "2.815", "+0.020"],
            ["step 3 (partial)", "3.038", "+0.052", "2.819", "+0.021"],
        ])

    return card(
        "results",
        "Result 2 &mdash; Phase-2 selective-update transfer (the real test)",
        "<p><b>What it tested.</b> Does a 32-expert CPT step on the finished model improve the "
        "<i>full</i> model on the trained cluster?</p>"
        + (anchor_tbl if anchor_tbl else "")
        + hm
        + "<p><b>Step-probe</b> (full-128e CE after each single CPT step on cluster 0, "
        "<code>sametok_ws_lam05</code>):</p>"
        + probe_tbl
        + "<p><b>Conclusion.</b> Despite closing the inference gap, the meta arm's selective "
        "<i>updates still do not transfer</i>: on <code>vanilla</code> a selective CPT stage "
        "improves the full model on its own cluster (blue diagonal), but on "
        "<code>sametok_ws_lam05</code> it makes the full model <b>worse</b> even on the cluster it "
        "just trained (red diagonal). The step-probe shows the damage is <b>immediate</b> "
        "(+0.079 at the very first committed step, partial recovery after), plus a small persistent "
        "+0.02 hit across all clusters &mdash; so it is not a slow drift or a non-locality effect. "
        "The core hypothesis is <b>not yet validated</b>.</p>"
        "<p class='note'><b>What to try next:</b> the raw-SGD probe's &theta;&prime; has the wrong "
        "shape and magnitude vs the real AdamW step (and self-extinguishes under bf16). Replace it "
        "with an Adam-preconditioned, schedule-matched pseudo-step (Result 3).</p>",
    )


def build_adam_result() -> str:
    return card(
        "results",
        "Result 3 &mdash; Adam-preconditioned pseudo-step redesign",
        "<p><b>What it tested.</b> Make &theta;&prime; the true next AdamW selective step "
        "(read-only from live moments) at the live scheduler lr, trained across a magnitude band "
        "&mdash; fixing the SGD probe's shape/scale mismatch and bf16 self-extinction.</p>"
        + table(
            ["arm", "full CE", "gap", "vs SGD arm"],
            [
                ["<code>sametok_ws_lam05</code> (SGD probe)", "2.745", "&minus;0.012", "&mdash;"],
                ["<code>&hellip;_adam</code>, band [1,32]", "<b>3.053</b>", "&minus;0.023",
                 "full CE +0.30, gap improved"],
            ],
        )
        + "<p><b>Conclusion.</b> The [1,32] band <b>regressed full CE by +0.30</b> while moving the "
        "gap toward vanilla (&minus;0.023). Up-to-32&times; committed-step displacements evaluate "
        "the outer loss at a &theta;&prime; ~50% of the weight norm away &mdash; closer to heavy "
        "noise injection than meta-learning &mdash; damaging base quality. The band was too "
        "aggressive.</p>"
        "<p class='note'><b>What to try next (running):</b> "
        "<code>_adam_scale1</code> (band [1,1] &mdash; isolate Adam+match_lr with no magnitude "
        "randomization) and <code>_adam_band4</code> (gentle [1,4]). If either recovers full CE "
        "toward 2.75 while keeping the improved gap, carry it into a phase-2 CPT transfer test.</p>",
    )


def build_next() -> str:
    return card(
        "goal",
        "Next set of experiments",
        "<ol>"
        "<li><b>scale1 / band4</b> (4-node, running): does Adam-preconditioning + match_lr, with a "
        "no / gentle magnitude band, recover full CE while keeping the improved selective "
        "behavior?</li>"
        "<li><b>Finish the step-probe grid</b>: 3 clusters &times; 10 steps (currently only "
        "cluster 0, steps 1&ndash;3) &mdash; confirm the immediate-damage pattern is general.</li>"
        "<li><b>Phase-2 CPT on the best adam variant</b>: the decisive transfer test &mdash; does "
        "the redesigned pseudo-step finally make selective CPT improve the full model?</li>"
        "<li><b>Heldout-mode adam variant</b>: also removes the same-tokens confound (same-doc "
        "outer pass partially rewards one-step memorization).</li>"
        "</ol>",
    )


def build_results(curves_html: str) -> str:
    return (
        build_phase1_result()
        + build_phase2_result()
        + build_adam_result()
        + build_next()
        + card("results", "Live training curves (working-set arms, from W&B)", curves_html)
    )


# --------------------------------------------------------------------------
# Live-curve explorer (uPlot); returns (inner_html, payload_json).
# --------------------------------------------------------------------------


def build_curves_inner(curves: dict) -> tuple:
    if not curves:
        return '<p class="missing">[no curves.json cached]</p>', ""
    arms = [(a, c) for a, c in ARM_COLORS if a in curves]
    if not arms:
        return '<p class="missing">[no working-set arms in curves.json]</p>', ""
    legend = " &middot; ".join(
        f'<span style="color:{c};font-weight:600">{html.escape(a)}</span>'
        f' <span class="note">({curves[a]["state"]}, step {curves[a]["last_step"]})</span>'
        for a, c in arms
    )
    body = f"""
<p>Each arm's W&B history on a shared tokens axis (train-side series EMA-smoothed). Metrics:
<b>train CE</b> (each arm's own training loss &mdash; not directly comparable across arms),
<b>inner CE</b> (selective-pass loss), <b>eval CE (full)</b> and <b>gap</b> (pool32&minus;full,
every 500 steps). Click legend to toggle; drag to zoom, double-click to reset.</p>
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
    return body, payload


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
p { max-width:82ch; }
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
ul { max-width:82ch; } li { margin:4px 0; }
ol { max-width:82ch; }
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


def main():
    curves_path = OUT / "curves.json"
    curves = json.load(open(curves_path)) if curves_path.exists() else {}
    curves_inner, curves_payload = build_curves_inner(curves)

    tabs = [
        ("overview", "Overview", build_overview()),
        ("implementation", "Implementation", build_implementation()),
        ("results", "Results", build_results(curves_inner)),
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
<p>Pretrain a 128-expert EMO from scratch so that training on a document's 32-expert working set
improves the full model &middot; two-pass first-order MAML with working-set outer updates &middot;
motivated by the modular_extension k=32 CPT negative result</p>
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
