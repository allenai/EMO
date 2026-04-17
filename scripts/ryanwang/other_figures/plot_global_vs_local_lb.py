#!/usr/bin/env python3
"""Grad-norm curves: global vs local load balancing.

Left panel : Standard MoE
Right panel: MOSE (two-level batch LB)

Blue line = local load balance, red line = global load balance.

Reads : claude_outputs/other_figures/global_vs_local_lb.csv
Writes: claude_outputs/other_figures/global_vs_local_lb.{png,pdf}
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
LOCAL = "local load balance"
GLOBAL = "global load balance"

PANELS = [
    (
        "Standard MoE",
        {
            LOCAL: "moe_1b14b_lr-4e-3_lb-1e-1_0117",
            GLOBAL: "moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211",
        },
    ),
    (
        "MOSE",
        {
            LOCAL: "twolevelbatchlb-32_1b14b_lr-4e-3_lb-1e-1_0119",
            GLOBAL: "twolevelbatchlbreducedp512-32_1b14b_lr-4e-3_lb-1e-1_0119",
        },
    ),
]


def load_long(df: pd.DataFrame, runs: dict[str, str]) -> pd.DataFrame:
    frames = []
    for label, run in runs.items():
        col = f"Name: {run}{METRIC_SUFFIX}"
        y = pd.to_numeric(df[col], errors="coerce")
        mask = y.notna()
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
    sns.set_theme()

    df = pd.read_csv(CSV_PATH)
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")

    palette = {LOCAL: "tab:blue", GLOBAL: "tab:red"}
    hue_order = [LOCAL, GLOBAL]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (title, runs) in zip(axes, PANELS):
        long_df = load_long(df, runs)
        sns.lineplot(
            data=long_df,
            x="step",
            y="grad_norm",
            hue="load balance",
            hue_order=hue_order,
            palette=palette,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.set_ylabel("gradient norm")

    fig.suptitle("Gradient norm: global vs. local load balancing")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "global_vs_local_lb.png"
    pdf = OUT_DIR / "global_vs_local_lb.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
