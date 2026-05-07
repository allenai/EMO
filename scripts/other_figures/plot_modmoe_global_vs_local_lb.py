#!/usr/bin/env python3
"""EMO-only grad-norm: local vs global load balancing, first 5000 steps.

Single-panel sibling of ``plot_global_vs_local_lb.py`` that keeps only the
two EMO (formerly MOSE) runs and clips the x-axis to the early
training regime where the global vs local LB difference is most visible.

Color theme matches the rest of the paper figures (EMO magenta + a
warm orange contrast, with a green-tinted legend frame).

Reads : claude_outputs/other_figures/global_vs_local_lb.csv
Writes: claude_outputs/other_figures/modmoe_global_vs_local_lb.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = REPO_ROOT / "claude_outputs" / "other_figures" / "global_vs_local_lb.csv"
OUT_PATH = REPO_ROOT / "claude_outputs" / "other_figures" / "modmoe_global_vs_local_lb.pdf"

METRIC_SUFFIX = " - optim/total grad norm"
MAX_STEP = 3000

LOCAL = "local load balance"
GLOBAL = "global load balance"

# Same wandb runs as plot_global_vs_local_lb.py for the MOSE / EMO panel.
RUNS = {
    LOCAL: "twolevelbatchlb-32_1b14b_lr-4e-3_lb-1e-1_0119",
    GLOBAL: "twolevelbatchlbreducedp512-32_1b14b_lr-4e-3_lb-1e-1_0119",
}

# Palette from the paper figures: EMO magenta and Dense-baseline orange.
PALETTE = {
    LOCAL: "#E78532",  # warm orange
    GLOBAL: "#B8327C",  # EMO magenta
}


def load_long(df: pd.DataFrame, runs: dict[str, str], max_step: int) -> pd.DataFrame:
    frames = []
    for label, run in runs.items():
        col = f"Name: {run}{METRIC_SUFFIX}"
        y = pd.to_numeric(df[col], errors="coerce")
        mask = y.notna() & (df["Step"] <= max_step)
        frames.append(
            pd.DataFrame(
                {
                    "step": df.loc[mask, "Step"].values,
                    "grad_norm": y[mask].values,
                    "load balance": label,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")

    df = pd.read_csv(CSV_PATH)
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")

    long_df = load_long(df, RUNS, MAX_STEP)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    sns.lineplot(
        data=long_df,
        x="step",
        y="grad_norm",
        hue="load balance",
        hue_order=[LOCAL, GLOBAL],
        palette=PALETTE,
        linewidth=2.5,
        ax=ax,
    )

    ax.set_title("EMO: global vs. local load balancing", fontweight="bold")
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient norm")
    ax.set_xlim(0, MAX_STEP)

    # Drop the auto seaborn legend; rebuild it with the paper-style tinted box.
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    handles = [
        mlines.Line2D(
            [],
            [],
            color=PALETTE[label],
            linewidth=2.8,
            label=label,
        )
        for label in (LOCAL, GLOBAL)
    ]
    leg = fig.legend(
        handles,
        [h.get_label() for h in handles],
        title="EMO",
        ncol=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        handletextpad=0.6,
        columnspacing=1.2,
        borderpad=0.7,
        labelspacing=0.4,
        handlelength=2.4,
        facecolor="#FBE7EF",
        edgecolor="#B8327C",  # keep the EMO-themed frame
    )
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_color("#3F1052")
    leg.get_frame().set_linewidth(1.1)

    fig.tight_layout(rect=(0, 0.13, 1, 1.0))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
