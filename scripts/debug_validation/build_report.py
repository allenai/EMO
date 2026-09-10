# PARENT: "scripts/olmoe3_routing/build_report.py" (visual framework via scripts/meta_learning/build_report.py)
# DESCRIPTION:
#     Builds claude_outputs/debug_validation/report.html: v3-small perplexity validation (11 sets) of
#     the four OLMoE3-ladder 275M 512-expert arms trained in this repo - standard router, EMO with a
#     uniform per-document pool in [16, 512], EMO with Beta(2,1)- and Beta(4,1)-skewed pools - across
#     training, plus their pretraining CE loss. Figures from scripts/debug_validation/plot_validation.py,
#     tables from claude_outputs/debug_validation/ppl_validation/*.json (rerun plot_validation.py first
#     when the data changes; this script re-runs it if the figures are missing).
#
#   python scripts/debug_validation/build_report.py
##############################################################
import html
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "claude_outputs/debug_validation"
PPL = OUT / "ppl_validation"
FIGS = OUT / "figs"
_spec = importlib.util.spec_from_file_location(
    "ml_report", ROOT / "scripts/meta_learning/build_report.py"
)
_ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ml)
card, table, img_tag, fig_row, CSS, JS = (
    _ml.card,
    _ml.table,
    _ml.img_tag,
    _ml.fig_row,
    _ml.CSS,
    _ml.JS,
)

SETS = [
    "c4_en",
    "dolma_books",
    "dolma_common-crawl",
    "dolma_pes2o",
    "dolma_reddit",
    "dolma_stack",
    "dolma_wiki",
    "ice",
    "m2d2_s2orc",
    "pile",
    "wikitext_103",
]
SHORT = {s: s.replace("dolma_", "").replace("common-crawl", "cc") for s in SETS}
ARMS = [
    ("standard", "standard router", "olmoe3_275m_10b"),
    ("uniform", "EMO, uniform pool [16, 512]", "olmoe3_275m_emo_10b"),
    ("beta2", "EMO, Beta(2,1) pool", "olmoe3_275m_emo_beta2_10b"),
    ("beta4", "EMO, Beta(4,1) pool", "olmoe3_275m_emo_beta4_10b"),
]
CKPTS = [5000, 10000, 15000, 19000, 19074]
TOK = 524_288


def fig(name, caption):
    return img_tag(FIGS / name, caption)


def load_series():
    summary = json.load(open(PPL / "summary.json"))
    inloop = json.load(open(PPL / "inloop_beta_arms.json"))
    ser = {}
    for arm, _, run in ARMS:
        if run in summary:
            ser[arm] = {
                int(st): {s: d["per_set"][f"{s}-validation"]["CE loss"] for s in SETS}
                for st, d in summary[run].items()
            }
        else:
            ser[arm] = {int(st): d for st, d in inloop[run].items() if len(d) == len(SETS)}
    return ser


def mean(d):
    return sum(d[s] for s in SETS) / len(SETS)


def build_overview(ser):
    setup = (
        "<p>Four OLMoE3-ladder 275M models (d640, 10 layers, 512 routed experts + 1 shared, top-16, LatentMoE, KDA + NoPE "
        "attention; 276.7M active / 2.61B total), all trained here for 10B tokens of Dolma 3.5 with the ladder's WSD trunk "
        "(2,000-step warmup, constant peak LR 8e-4, no decay), the same data order, 4 jupiter nodes. They differ only in the "
        "router: <b>standard</b> softmax top-16; <b>EMO</b> with a per-document expert pool drawn <b>uniformly</b> from [16, 512] "
        "(the ladder's EMO setting); EMO with the pool size <b>d = round(16 + 496 &middot; u<sup>1/&alpha;</sup>)</b>, u ~ U(0,1), "
        "i.e. a Beta(&alpha;, 1) fraction of the range, for <b>&alpha; = 2</b> (mean pool 347) and <b>&alpha; = 4</b> (mean 413; "
        "uniform = 264). Eval routing always uses all 512 experts.</p>"
        "<p>Validation: OLMo-core's <code>v3-small-ppl-validation</code> mix, 11 sets (c4_en, dolma books / common-crawl / pes2o / "
        "reddit / stack / wiki, ice, m2d2_s2orc, pile, wikitext_103), 18,523 documents = one padded 8,192-token instance each, "
        "per-set mean token CE exactly as OLMo-core's in-loop LM evaluator computes it. The standard and uniform-EMO runs were "
        "trained without in-loop evals, so their five permanent checkpoints (2.6B / 5.2B / 7.9B / 10.0B / 10.0B tokens) were "
        "evaluated offline with <code>scripts/debug_validation/eval_ppl_validation.py</code> (reuses the in-loop evaluator; "
        "cross-checked on a Beta(4,1) checkpoint: agrees with that run's in-loop numbers within 0.0006 CE on every set). The Beta "
        "arms ran the in-loop eval every 1,000 steps.</p>"
    )
    rows = []
    for st in CKPTS:
        rows.append(
            [f"{st} ({st * TOK / 1e9:.2f}B)"]
            + [f"{mean(ser[a][st]):.3f}" if st in ser[a] else "&mdash;" for a, _, _ in ARMS]
            + [
                f"{mean(ser[a][st]) - mean(ser['standard'][st]):+.3f}"
                for a in ("uniform", "beta2", "beta4")
            ]
        )
    tbl = table(
        ["step (tokens)"]
        + [lab for _, lab, _ in ARMS]
        + ["&Delta; uniform", "&Delta; Beta(2,1)", "&Delta; Beta(4,1)"],
        rows,
    )
    findings = (
        "<ul>"
        "<li><b>Skewing the document pool toward large pools closes EMO's full-routing gap, monotonically in the skew.</b> "
        "Mean over the 11 sets at step 19,000: standard 2.973, uniform EMO 2.996 (+0.022), Beta(2,1) 2.982 (+0.009), "
        "Beta(4,1) 2.971 (&minus;0.003). The ordering standard &asymp; Beta(4,1) &lt; Beta(2,1) &lt; uniform holds at every "
        "checkpoint from 2.6B tokens on and on every set (per-set Beta(4,1) &minus; standard within &plusmn;0.016).</li>"
        "<li><b>Uniform EMO's cost sits on web-like text.</b> +0.03 to +0.05 on c4, common-crawl, reddit and m2d2_s2orc; only "
        "~+0.01 on pes2o, stack, books. The gap does not shrink with tokens.</li>"
        "<li><b>Pretraining loss tells the same story, with a caveat.</b> Smoothed train CE (EMO minus standard) ends at +0.009 / "
        "&minus;0.004 / &minus;0.008 for uniform / Beta(2,1) / Beta(4,1). The train CE of an EMO arm is computed under its sampled "
        "training pools (restricted routing), so it is not the same quantity as the standard router's train CE; the validation "
        "numbers (full routing, held-out) are the like-for-like comparison.</li>"
        "<li><b>Not measured here:</b> what the Beta arms give up in selective (small-pool) routing, which is EMO's purpose. "
        "Uniform training sees pools of 16&ndash;64 in ~10% of documents, Beta(2,1) in ~1%, Beta(4,1) in ~0%.</li>"
        "</ul>"
    )
    return (
        card("info", "Setup", setup)
        + card("results", "Findings", findings)
        + card("info", "Mean validation CE over the 11 sets", tbl)
        + fig_row(
            fig(
                "val_mean.png",
                "Mean CE over the 11 validation sets vs training tokens (from 2.1B tokens; the earlier steep descent hides the arm differences).",
            ),
            fig(
                "val_delta.png",
                "Mean CE, each EMO arm minus the standard router, at the five permanent checkpoints.",
            ),
        )
    )


def build_sets(ser):
    body = (
        "<p>One plot per validation set. Standard router and uniform EMO: offline eval of the five permanent checkpoints; "
        "Beta arms: in-loop eval every 1,000 steps (from step 4,000). End labels give the final (step 19,074) value.</p>"
    )
    figs = [fig(f"val_{s}.png", f"{s}: CE loss vs training tokens.") for s in SETS]
    for i in range(0, len(figs), 2):
        body += fig_row(*figs[i : i + 2])
    rows = []
    for a, lab, _ in ARMS:
        d = ser[a][19000]
        rows.append([lab] + [f"{d[s]:.3f}" for s in SETS] + [f"{mean(d):.3f}"])
    base = ser["standard"][19000]
    for a, lab, _ in ARMS[1:]:
        d = ser[a][19000]
        rows.append(
            [f"{lab} &minus; standard"]
            + [f"{d[s] - base[s]:+.3f}" for s in SETS]
            + [f"{mean(d) - mean(base):+.3f}"]
        )
    body += card(
        "info",
        "Per-set CE at step 19,000 (last periodic eval, 9.96B tokens)",
        table(["arm"] + [SHORT[s] for s in SETS] + ["mean"], rows),
    )
    return body


def build_train():
    curves = json.load(open(PPL / "train_ce_curves.json"))
    rows = [
        [
            lab,
            curves[a]["run"],
            f"{curves[a]['train_ce'][-1][1]:.3f}",
            f"{sum(c for _, c in curves[a]['train_ce'][-100:]) / 100:.3f}",
        ]
        for a, lab, _ in ARMS
    ]
    body = (
        "<p><code>train/CE loss</code> from W&B for the four runs (same data order, so per-step noise is common-mode). "
        "For the EMO arms this loss is computed under the sampled per-document training pools, i.e. with restricted routing, "
        "so a lower value partly reflects larger pools rather than a better model; the validation tab is the like-for-like view.</p>"
    )
    body += fig_row(
        fig("train_ce.png", "Pretraining CE loss, 1,000-step rolling mean, full run."),
        fig("train_ce_zoom.png", "Same, last 6B tokens."),
    )
    body += fig_row(
        fig(
            "train_ce_delta.png",
            "EMO arm minus standard router, 1,000-step rolling mean (warmup transient before 1B tokens cropped).",
        )
    )
    body += card(
        "info",
        "Final train CE",
        table(["arm", "W&B run", "last logged step", "mean of last 1,000 steps"], rows),
    )
    return body


def build_launch():
    readme = (ROOT / "scripts/debug_validation/README.md").read_text()
    launched = readme.split("## Launched", 1)[1].split("## Results", 1)[0].strip().splitlines()
    rows = [
        [c.strip() for c in ln.strip("|").split("|")]
        for ln in launched
        if ln.startswith("| `") or ln.startswith("| ppl") or ln.startswith("| offline")
    ]
    rows = [[html.escape(c).replace("`", "") for c in r] for r in rows]
    for r in rows:
        r[1] = (
            r[1]
            .replace("https://beaker.org/ex/", '<a href="https://beaker.org/ex/')
            .replace(" (", '">link</a> (', 1)
            if "beaker.org" in r[1]
            else r[1]
        )
    body = card(
        "info",
        "Beaker launches (2026-09-09)",
        table(["run", "Beaker", "commit", "submitted", "status"], rows),
    )
    body += card(
        "info",
        "Files",
        (
            "<ul><li><code>scripts/debug_validation/eval_ppl_validation.py</code> &mdash; offline evaluator (padded FSL dataset + LMEvaluator "
            "from the pinned submodule; bf16 weights; serial data prep).</li>"
            "<li><code>scripts/debug_validation/launch_ppl_validation.sh</code>, <code>aggregate_ppl_validation.py</code>, "
            "<code>plot_validation.py</code>, <code>build_report.py</code>.</li>"
            "<li><code>scripts/debug_validation/model_scripts/olmoe3_275m_emo_beta{2,4}_10b.sh</code> &mdash; the Beta arms "
            "(<code>OLMOE3_EMO_POOL_DIST=beta:&alpha;</code>, <code>OLMOE3_PPL_EVAL_INTERVAL=1000</code> in "
            "<code>scripts/sparse_experts/olmoe3_275m.py</code>; the pinned OLMo-core submodule is unchanged).</li>"
            "<li>Results: <code>claude_outputs/debug_validation/ppl_validation/</code> (per-checkpoint JSON, summary.json, "
            "inloop_beta_arms.json, train_ce_curves.json, table_*.md).</li></ul>"
        ),
    )
    return body


def main():
    if not (FIGS / "val_mean.png").exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/debug_validation/plot_validation.py")], check=True
        )
    ser = load_series()
    tabs = [
        ("overview", "1 &middot; Overview", build_overview(ser)),
        ("sets", "2 &middot; Validation sets", build_sets(ser)),
        ("train", "3 &middot; Pretraining loss", build_train()),
        ("launch", "4 &middot; Launches &amp; files", build_launch()),
    ]
    nav = "".join(f'<button data-target="{tid}">{name}</button>' for tid, name, _ in tabs)
    sections = "".join(f'<section class="tab" id="{tid}">{body}</section>' for tid, _, body in tabs)
    title = "debug_validation: v3-small ppl validation of the OLMoE3 512e arms (standard vs EMO uniform / Beta(2,1) / Beta(4,1) pools)"
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><header><a class="home-link" href="/">&larr; all reports</a><h1>{html.escape(title)}</h1>
<p>Four 512-expert OLMoE3-ladder 275M runs, 10B tokens each, same data &middot; per-set CE on the 11 v3-small validation sets across training
&middot; pretraining CE loss &middot; the Beta arms skew each document's training pool toward large pools</p></header>
<div class="topbar"><nav>{nav}</nav><div id="subnav"></div></div>
<main>{sections}</main><script>{JS}</script></body></html>"""
    (OUT / "report.html").write_text(page)
    print(f"wrote {OUT / 'report.html'} ({(OUT / 'report.html').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
