#!/usr/bin/env python3
"""Grad-norm curves: local load balancing only.

Sibling of ``plot_global_vs_local_lb.py`` that drops the global-LB curves and
only shows the local-LB run for each architecture.

Left panel : Standard MoE (local LB)
Right panel: MOSE (local LB)

Reads : claude_outputs/other_figures/global_vs_local_lb.csv
Writes: claude_outputs/other_figures/local_lb_only.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = REPO_ROOT / "claude_outputs" / "other_figures" / "global_vs_local_lb.csv"
OUT_DIR = REPO_ROOT / "claude_outputs" / "other_figures"

METRIC_SUFFIX = " - optim/total grad norm"
LABEL = "local load balance"

PANELS = [
    ("Standard MoE", "moe_1b14b_lr-4e-3_lb-1e-1_0117"),
    ("MOSE", "twolevelbatchlb-32_1b14b_lr-4e-3_lb-1e-1_0119"),
]


def load_long(df: pd.DataFrame, run: str) -> pd.DataFrame:
    col = f"Name: {run}{METRIC_SUFFIX}"
    y = pd.to_numeric(df[col], errors="coerce")
    mask = y.notna()
    return pd.DataFrame(
        {
            "step": df.loc[mask, "Step"].values,
            "grad_norm": y[mask].values,
            "load balance": LABEL,
        }
    )


def main() -> None:
    sns.set_theme()

    df = pd.read_csv(CSV_PATH)
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")

    palette = {LABEL: "tab:blue"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (title, run) in zip(axes, PANELS):
        long_df = load_long(df, run)
        sns.lineplot(
            data=long_df,
            x="step",
            y="grad_norm",
            color=palette[LABEL],
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.set_ylabel("gradient norm")
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "local_lb_only.png"
    pdf = OUT_DIR / "local_lb_only.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
