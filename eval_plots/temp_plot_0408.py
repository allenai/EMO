#!/usr/bin/env python3
"""Generate a 2x3 grid of plots for selected MMLU categories:
  Top row:    mmlu_biology,  mmlu_business, mmlu_computer_science
  Bottom row: mmlu_history,  mmlu_physics,  mmlu_other
All using acc_per_byte and left-aligned (relative) checkpoint indices.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============================================================================
# CONFIGURATION — mirrors plot_presentation_0319.py
# ============================================================================

MODEL_SPECS = {
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308step30995-hf": {
        "label": "Standard MoE",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keep-8)"},
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keep-32)"},
            {
                "suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise",
                "label": "(keep-128)",
            },
        ],
    },
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301step30995-hf": {
        "label": "MOSE",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keep-8)"},
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keep-32)"},
            {
                "suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise",
                "label": "(keep-128)",
            },
        ],
    },
    # "moereducedp512sharedexp1_1b4b_lr-4e-3_lb-1e-1_0308step30995-hf": {
    #     "label": "Standard MoE (32 experts)",
    #     "baseline": False,
    #     "variants": [
    #         {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": " "},
    #     ],
    # },
    # "dense_1b_lr-4e-3_0213step30995-hf": {
    #     "label": "Dense (8-expert Equivalent)",
    #     "baseline": False,
    #     "variants": [
    #         {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": " "},
    #     ],
    # },
    # "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313step238419-hf": {
    #     "label": "specialized moe 1T",
    #     "baseline": False,
    #     "variants": [
    #         {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 8)"},
    #         {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
    #         {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32)"},
    #     ],
    # },
    # "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419step250339-hf": {
    #     "label": "specialized moe 1T + anneal",
    #     "baseline": False,
    #     "variants": [
    #         {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 8)"},
    #         {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
    #         {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32)"},
    #     ],
    # },
}

MODEL_LABELS = {model: spec["label"] for model, spec in MODEL_SPECS.items() if spec.get("label")}

MMLU_SUBTASKS_NO_OTHER = [
    "mmlu_biology",
    "mmlu_business",
    "mmlu_chemistry",
    "mmlu_computer_science",
    "mmlu_culture",
    "mmlu_economics",
    "mmlu_engineering",
    "mmlu_geography",
    "mmlu_health",
    "mmlu_history",
    "mmlu_law",
    "mmlu_math",
    "mmlu_philosophy_cat",
    "mmlu_physics",
    "mmlu_politics",
    "mmlu_psychology",
]

_MODEL_BASE_COLORS: Dict[str, object] = {
    "Standard MoE": (0.1216, 0.4667, 0.7059),  # blue
    "Standard MoE (32 experts)": (1.0000, 0.4980, 0.0549),  # orange
    "Dense": (0.1725, 0.6275, 0.1725),  # green
    "MOSE": (0.8902, 0.4667, 0.7608),  # pink
}

_VARIANT_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
_VARIANT_MARKERS = ["o", "s", "^", "D", "v", "X"]
_VARIANT_ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55]

_MODEL_VARIANT_STYLE: Dict[str, Dict[str, object]] = {}
for _model_name, _spec in MODEL_SPECS.items():
    _base_label = _spec.get("label", _model_name)
    _variants = _spec.get("variants", [])
    if not _variants:
        _MODEL_VARIANT_STYLE[_base_label] = {
            "alpha": 1.0,
            "linestyle": "-",
            "marker": "o",
        }
    for _vi, _v in enumerate(_variants):
        _full_label = _base_label + " " + _v["label"]
        _idx = min(_vi, len(_VARIANT_ALPHAS) - 1)
        _MODEL_VARIANT_STYLE[_full_label] = {
            "alpha": _VARIANT_ALPHAS[_idx],
            "linestyle": _VARIANT_LINESTYLES[_idx],
            "marker": _VARIANT_MARKERS[_idx],
        }

_ALL_VARIANT_SUFFIXES: List[str] = sorted(
    {v["suffix"] for spec in MODEL_SPECS.values() for v in spec.get("variants", [])},
    key=len,
    reverse=True,
)

LEGEND_WRAP_WIDTH = 30
METRIC_KEY = "acc_per_byte"


# ============================================================================
# Helpers (copied from plot_presentation_0319.py)
# ============================================================================


def _get_model_variants(model_name: str) -> List[Tuple[str, str]]:
    spec = MODEL_SPECS.get(model_name)
    if spec is not None:
        return [(v["suffix"], v["label"]) for v in spec.get("variants", [])]
    return []


def _wrap_label(label: str) -> str:
    return textwrap.fill(label, width=LEGEND_WRAP_WIDTH, break_on_hyphens=True)


def _get_color(label: str) -> object:
    for base_label, color in sorted(
        _MODEL_BASE_COLORS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if label.startswith(base_label):
            return color
    return "gray"


def _get_vstyle(label: str) -> Dict[str, object]:
    return _MODEL_VARIANT_STYLE.get(label, {"alpha": 1.0, "linestyle": "-", "marker": "o"})


def read_metrics(metrics_path: Path) -> Optional[Dict[str, object]]:
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _scan_checkpoints(
    task_dir: Path,
    metric_key: str,
    model_key: str,
    model_label: str,
    task_run: str,
    rows: List[Dict[str, object]],
) -> None:
    results_dir = task_dir / "results"
    if not results_dir.is_dir():
        return
    for checkpoint_dir in sorted(results_dir.glob("checkpoint-*")):
        if not checkpoint_dir.is_dir():
            continue
        try:
            step = int(checkpoint_dir.name.replace("checkpoint-", ""))
        except ValueError:
            continue
        metrics_files = sorted(checkpoint_dir.glob("task-*-metrics.json"))
        if not metrics_files:
            continue
        metrics = read_metrics(metrics_files[0])
        if metrics is None:
            continue
        metric_values = metrics.get("metrics")
        if not isinstance(metric_values, dict):
            continue
        value = metric_values.get(metric_key)
        if value is None:
            continue
        rows.append(
            {
                "model": model_key,
                "model_label": model_label,
                "task_run": task_run,
                "checkpoint": step,
                "metric_value": value,
            }
        )


def collect_records(
    root: Path,
    model_names: Sequence[str],
    task_runs: Sequence[str],
    metric_key: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_name in model_names:
        model_dir = root / model_name
        if not model_dir.is_dir():
            continue
        model_label = MODEL_LABELS.get(model_name, model_name)
        for task_run in task_runs:
            for suffix, label_mod in _get_model_variants(model_name):
                variant_dir = model_dir / (task_run + suffix)
                if not variant_dir.is_dir():
                    continue
                _scan_checkpoints(
                    variant_dir,
                    metric_key,
                    model_key=model_name + suffix,
                    model_label=model_label + " " + label_mod,
                    task_run=task_run,
                    rows=rows,
                )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["task_run", "model", "checkpoint"])
    # Convert absolute checkpoint steps to relative indices per model
    df["checkpoint"] = df.groupby(["model", "task_run"]).cumcount()
    return df


def collect_mmlu_avg_no_other(
    root: Path,
    model_names: Sequence[str],
    metric_key: str,
) -> pd.DataFrame:
    subtasks = MMLU_SUBTASKS_NO_OTHER
    df = collect_records(root, model_names, subtasks, metric_key)
    if df.empty:
        return df
    df = df[df["task_run"].isin(subtasks)]
    if df.empty:
        return df

    # Only keep models with all subtasks
    complete = []
    for mk in df["model"].unique():
        present = set(df.loc[df["model"] == mk, "task_run"].unique())
        if all(s in present for s in subtasks):
            complete.append(mk)
    df = df[df["model"].isin(complete)]
    if df.empty:
        return df

    # Left-align checkpoints
    df = df.sort_values(["model", "task_run", "checkpoint"])
    df["checkpoint_rel"] = df.groupby(["model", "task_run"]).cumcount()

    # Macro average across subtasks at each relative checkpoint
    avg_df = df.groupby(["model", "model_label", "checkpoint_rel"], as_index=False).agg(
        metric_value=("metric_value", "mean")
    )
    avg_df = avg_df.rename(columns={"checkpoint_rel": "checkpoint"})
    avg_df = avg_df[avg_df["checkpoint"] <= 6]
    return avg_df


# ============================================================================
# Plotting
# ============================================================================


def _plot_on_ax(ax, df, title):
    """Plot all model lines on a given axes."""
    for model_label in sorted(df["model_label"].unique()):
        mdf = df[df["model_label"] == model_label].sort_values("checkpoint")
        color = _get_color(model_label)
        vstyle = _get_vstyle(model_label)
        ax.plot(
            mdf["checkpoint"],
            mdf["metric_value"],
            marker=vstyle["marker"],
            linestyle=vstyle["linestyle"],
            linewidth=2,
            markersize=9,
            color=color,
            alpha=vstyle["alpha"],
            label=_wrap_label(model_label),
        )
    ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=13)


def main():
    root = Path(__file__).resolve().parent.parent / "prune_evals_0313"
    model_names = list(MODEL_SPECS.keys())

    # Tasks to plot in 2x3 grid order (left-to-right, top-to-bottom)
    PANEL_TASKS = [
        ("mmlu_biology", "MMLU Biology"),
        ("mmlu_business", "MMLU Business"),
        ("mmlu_computer_science", "MMLU Computer Science"),
        ("mmlu_history", "MMLU History"),
        ("mmlu_physics", "MMLU Physics"),
        ("mmlu_other", "MMLU Other"),
    ]

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    last_ax_with_data = None
    for ax, (task, title) in zip(axes_flat, PANEL_TASKS):
        df = collect_records(root, model_names, [task], METRIC_KEY)
        if not df.empty:
            _plot_on_ax(ax, df, title)
            last_ax_with_data = ax
        else:
            ax.set_title(f"{title} — no data", fontsize=16)

    # Shared axis labels
    fig.supxlabel("Checkpoint Number", fontsize=16)
    fig.supylabel(METRIC_KEY, fontsize=16)

    # Single shared legend to the right of the figure
    if last_ax_with_data is not None:
        handles, labels = last_ax_with_data.get_legend_handles_labels()
        seen = set()
        unique_handles, unique_labels = [], []
        for h, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                unique_handles.append(h)
                unique_labels.append(label)

        fig.legend(
            unique_handles,
            unique_labels,
            title="Model",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=11,
            title_fontsize=12,
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08, left=0.06)
    out_path = Path(__file__).resolve().parent / "temp_plot_0408.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
