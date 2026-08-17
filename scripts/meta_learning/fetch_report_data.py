# PARENT: none (report data/figure builder for scripts/meta_learning/build_report.py)
# DESCRIPTION:
#     Pulls W&B histories + eval summaries for the meta_learning runs, runs the bf16
#     pseudo-step survival simulation, and renders the report figures into
#     claude_outputs/meta_learning/figs/*.png plus report_data.json. Idempotent; rerun to
#     refresh the report's data before `python scripts/meta_learning/build_report.py`.
#
#   python scripts/meta_learning/fetch_report_data.py
##############################################################

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

OUT = pathlib.Path("claude_outputs/meta_learning")
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# Reference categorical palette (dataviz skill, light mode, fixed order).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#1a1a1a", "#52514e", "#e5e5e2"
TOKENS_PER_STEP = 1024 * 4096

RUNS = {
    "meta128_vanilla_100b": "vanilla EMO-128e (baseline)",
    "meta128_sametok_100b": "same-tokens, λ=0 all-experts (stopped)",
    "meta128_sametok_ws": "same-tokens, working-set λ=0",
    "meta128_heldout_ws": "heldout, working-set λ=0",
    "meta128_sametok_ws_lam05": "same-tokens ws + λ=0.5",
    "meta128_heldout_ws_lam05": "heldout ws + λ=0.5",
    "meta128_seq_ws": "sequential ablation",
}


def style_ax(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)


def fetch():
    api = wandb.Api(timeout=60)
    data = {}
    for name in RUNS:
        try:
            runs = sorted(
                api.runs("ryanyxw/emo-extension", {"display_name": name}),
                key=lambda x: x.created_at,
            )
        except Exception:
            runs = []
        if not runs:
            continue
        r = runs[-1]
        hist = [
            h
            for h in r.scan_history(keys=["_step", "train/CE loss"], page_size=2000)
            if h.get("train/CE loss") is not None
        ]
        evals = {
            k: v
            for k, v in r.summary.items()
            if k.startswith("eval/") and k.endswith("CE loss") and isinstance(v, (int, float))
        }
        extras = {}
        for k in (
            "train/meta bf16 survival cosine",
            "train/meta bf16 survival norm ratio",
            "train/meta delta/weight norm",
            "train/meta inner-outer grad cosine (experts)",
        ):
            if k in r.summary:
                extras[k] = r.summary[k]
        data[name] = {
            "state": r.state,
            "step": r.summary.get("_step", 0),
            "ce": [[h["_step"], h["train/CE loss"]] for h in hist],
            "evals": evals,
            "extras": extras,
        }
    return data


def fig_ce(data):
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    colors = {
        "meta128_vanilla_100b": BLUE,
        "meta128_sametok_100b": ORANGE,
        "meta128_sametok_ws": AQUA,
    }
    for name, c in colors.items():
        if name not in data:
            continue
        pts = [(s, v) for s, v in data[name]["ce"] if s >= 20]
        if not pts:
            continue
        xs = [s * TOKENS_PER_STEP / 1e9 for s, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, color=c, linewidth=2, zorder=3)
        ax.annotate(
            RUNS[name].split(",")[0].split(" (")[0],
            (xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            va="center",
        )
    style_ax(ax)
    ax.set_xlabel("tokens (B)", color=INK2, fontsize=9)
    ax.set_ylabel("train CE (nats)", color=INK2, fontsize=9)
    ax.set_title(
        "Train CE — note: NOT comparable across arms (vanilla trains under random pools;\n"
        "the meta arms' CE is their full-routing outer pass)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(FIGS / "ce_curves.png", facecolor="white")
    plt.close(fig)


def fig_gap(data):
    """Per-label dumbbells: full-pool CE vs pool-32 CE, one panel per run with eval data."""
    panels = [
        (n, RUNS[n])
        for n in (
            "meta128_vanilla_100b",
            "meta128_sametok_100b",
            "meta128_sametok_ws",
            "meta128_sametok_ws_lam05",
        )
        if n in data and any("/lm-full/" in k for k in data[n]["evals"])
    ]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2), dpi=150, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, title) in zip(axes, panels):
        evals = data[name]["evals"]
        labels = sorted(
            k.split("/")[2] for k in evals if "/lm-full/" in k and k.split("/")[2] != "all"
        )
        ys = np.arange(len(labels))
        full = [evals.get(f"eval/lm-full/{lb}/CE loss") for lb in labels]
        p32 = [evals.get(f"eval/lm-pool32/{lb}/CE loss") for lb in labels]
        for y, f, p in zip(ys, full, p32):
            if f is None or p is None:
                continue
            ax.plot([f, p], [y, y], color=GRID, linewidth=1.5, zorder=2)
        ax.scatter(full, ys, color=BLUE, s=28, zorder=3, label="full pool (128)")
        ax.scatter(p32, ys, color=ORANGE, s=28, zorder=3, label="pool pinned to 32")
        ax.set_yticks(ys)
        ax.set_yticklabels([lb.replace("-validation", "") for lb in labels], fontsize=8)
        gap = np.mean([p - f for f, p in zip(full, p32) if f is not None and p is not None])
        ax.set_title(
            f"{title.split(' (')[0]}\nstep {data[name]['step']}, mean gap {gap:+.3f}",
            color=INK,
            fontsize=9.5,
            loc="left",
        )
        style_ax(ax)
        ax.set_xlabel("held-out CE (nats)", color=INK2, fontsize=9)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle(
        "Selective-vs-full CE per validation source (dumbbell = same data, two eval pools)",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "gap_dumbbells.png", facecolor="white")
    plt.close(fig)


def fig_bf16():
    """Simulated bf16 survival of the pseudo-step vs its relative size (heavy-tailed grads),
    at the real run's weight scale (expert-weight global norm 2242 over 1.29e10 elements)."""
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N = 8_000_000
    sigma_w = 2242 / (1.29e10**0.5)
    w = torch.randn(N, device=dev) * sigma_w
    scale = (N / 1.29e10) ** 0.5
    ratios, cosines = [], []
    g = torch.distributions.StudentT(3).sample((N,)).to(dev)
    g = g / g.norm()
    for mult in np.logspace(-1, 3.2, 18):
        delta = g * 0.03 * scale * mult
        survived = (w + delta).bfloat16().float() - w.bfloat16().float()
        cos = torch.nn.functional.cosine_similarity(survived, delta, dim=0).item()
        ratios.append(0.03 * mult / 2242)  # delta/weight norm ratio
        cosines.append(cos)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
    ax.plot(ratios, cosines, color=BLUE, linewidth=2, zorder=3)
    ax.set_xscale("log")
    for x, lab in [
        (1.34e-5, "as first launched\n(α=3e-2, clip=1)"),
        (4e-4, "relaunch target\n(α=3e-1, clip=10)"),
    ]:
        ax.axvline(x, color=ORANGE, linewidth=1.2, linestyle="--", zorder=2)
        ax.annotate(lab, (x, 0.8), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK)
    style_ax(ax)
    ax.set_xlabel("pseudo-step size, ‖δ‖ / ‖W‖", color=INK2, fontsize=9)
    ax.set_ylabel("cos(survived, intended δ)", color=INK2, fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "How much of the pseudo-step survives the bf16 all-gather (simulation, heavy-tailed grads)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(FIGS / "bf16_survival.png", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interactive curves payload (uPlot explorer; display pattern per models_v2)
# ---------------------------------------------------------------------------

# Arm -> W&B run names in chronological order (renames/crashes split some arms
# across runs; histories are stitched, newer run wins on overlapping steps).
CURVE_ARMS = {
    "vanilla": ["meta128_vanilla_100b"],
    "sametok λ=0 (degenerate)": ["meta128_sametok_100b"],
    "sametok_ws": ["meta128_sametok_ws_100b", "meta128_sametok_ws"],
    "heldout_ws": ["meta128_heldout_ws_100b", "meta128_heldout_ws"],
    "sametok_ws_lam05": ["meta128_sametok_ws_lam05"],
    "rpool (canceled)": ["meta128_sametok_ws_rpool"],
    "heldout_ws_lam05": ["meta128_heldout_ws_lam05"],
    "seq_ws": ["meta128_seq_ws"],
}
EMA = 0.85


def _smooth(pairs):
    out, m = [], None
    for st, v in pairs:
        m = v if m is None else EMA * m + (1 - EMA) * v
        out.append((st, round(m, 4)))
    return out


def fetch_curves():
    api = wandb.Api(timeout=60)
    arms = {}
    for arm, names in CURVE_ARMS.items():
        ce, ice, evals = {}, {}, {}
        state, last = None, 0
        for name in names:
            runs = sorted(
                api.runs("ryanyxw/emo-extension", {"display_name": name}),
                key=lambda x: x.created_at,
            )
            if not runs:
                continue
            r = runs[-1]
            state, last = r.state, max(last, r.summary.get("_step", 0) or 0)
            # NOTE: history(keys=[...]) only returns rows where ALL keys are non-null, so
            # fetch each metric separately (vanilla has no inner-CE metric at all).
            for key, dest in (("train/CE loss", ce), ("train/meta inner CE loss", ice)):
                for row in r.history(keys=[key], samples=800, pandas=False):
                    st, v = row.get("_step"), row.get(key)
                    if st is not None and v is not None:
                        dest[int(st)] = float(v)
            kf = [
                k
                for k in r.summary.keys()
                if k.startswith("eval/lm-full/") and k.endswith("CE loss")
            ]
            kp = [
                k
                for k in r.summary.keys()
                if k.startswith("eval/lm-pool32/") and k.endswith("CE loss")
            ]
            for row in r.scan_history(keys=["_step"] + kf + kp, page_size=1000):
                f = [v for k, v in row.items() if k in kf and isinstance(v, (int, float))]
                pp = [v for k, v in row.items() if k in kp and isinstance(v, (int, float))]
                if f and pp:
                    import statistics as _st

                    evals[int(row["_step"])] = (round(_st.mean(f), 4), round(_st.mean(pp), 4))
        if not ce:
            continue
        sce = _smooth(sorted(ce.items()))
        sice = _smooth(sorted(ice.items())) if ice else []
        arms[arm] = {
            "state": state,
            "last_step": last,
            "ce_x": [round(st * TOKENS_PER_STEP / 1e9, 4) for st, _ in sce],
            "ce_y": [v for _, v in sce],
            "ice_x": [round(st * TOKENS_PER_STEP / 1e9, 4) for st, _ in sice],
            "ice_y": [v for _, v in sice],
            "eval_x": [round(st * TOKENS_PER_STEP / 1e9, 4) for st in sorted(evals)],
            "eval_full": [evals[st][0] for st in sorted(evals)],
            "eval_p32": [evals[st][1] for st in sorted(evals)],
            "eval_gap": [round(evals[st][1] - evals[st][0], 4) for st in sorted(evals)],
        }
        print(f"curves: {arm} ({state}, last step {last}, {len(sce)} ce pts, {len(evals)} evals)")
    (OUT / "curves.json").write_text(json.dumps(arms))


def main():
    data = fetch()
    (OUT / "report_data.json").write_text(json.dumps(data, indent=1, default=str))
    fig_ce(data)
    fig_gap(data)
    fig_bf16()
    fetch_curves()
    print("figs:", sorted(p.name for p in FIGS.glob("*.png")))
    for n, d in data.items():
        print(n, d["state"], "step", d["step"], "evals:", len(d["evals"]), d["extras"])


if __name__ == "__main__":
    main()
