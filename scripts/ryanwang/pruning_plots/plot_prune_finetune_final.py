#!/usr/bin/env python3
"""Plot prune_evals_final metric curves for the merged-task suite.

Slimmed-down sibling of eval_plots/plot_presentation_0319.py, sharing the
same model/task universe as scripts/ryanwang/pruning_plots/get_table_scores_prune_evals_final.py.

For every (metric, task) it plots a curve per (model, variant) across the
finetuning checkpoints, plus left-aligned group-average curves for the MC9,
MMLU-merged, and MMLU-Pro-merged groups.

Reads from   : <repo>/prune_evals_final/
Writes into  : <repo>/claude_outputs/prune_plots/<output-subdir>/<metric>/
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from get_table_scores_prune_evals_final import (
    GEN5_ALL_TASKS,
    GEN5_TASKS,
    GSM8K_TASKS,
    MC9_TASKS,
    MMLU_MERGED_TASKS,
    MMLU_PRO_MERGED_TASKS,
    MODEL_SPECS,
    REPO_ROOT,
    TASK_SPECS,
    parse_csv_arg,
    read_metrics,
    sanitize_filename,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PRUNE_EVALS_ROOT = REPO_ROOT / "prune_evals_final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "prune_plots"
DEFAULT_OUTPUT_SUBDIR = "prune_finetune_plots_final"

MODEL_LABELS: Dict[str, str] = {
    name: spec.get("label", name) for name, spec in MODEL_SPECS.items()
}

# Models to include in plots. Comment/uncomment to control what's plotted.
PLOT_MODELS = [
    "dense_1b_lr-4e-3_0213step30995-hf",
    "moereducedp512sharedexp1_1b4b_lr-4e-3_lb-1e-1_0308step30995-hf",
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308step30995-hf",
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301step30995-hf",
    # "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419step250339-hf",
    # "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_twolevel_randpool-8-128_from_step238419step250339-hf",
    # "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419step250339-hf",
]

# Hardcoded base colors so they don't shift when models are added/removed.
# Color matching is longest-prefix-first in _color_for, so longer labels
# (e.g. "moe 1T + anneal") win over shorter ones (e.g. "moe").
_MODEL_BASE_COLORS: Dict[str, object] = {
    "moe 1T + anneal":             (0.1216, 0.4667, 0.7059),  # tab10 blue
    "moe 1T + twolevel anneal":    (0.5490, 0.3373, 0.2941),  # tab10 brown
    "specialized moe 1T + anneal": (0.8392, 0.1529, 0.1569),  # tab10 red
    "moe":                                                  (0.1216, 0.4667, 0.7059),  # tab10 blue
    "moe_small":                                            (1.0000, 0.4980, 0.0549),  # tab10 orange
    "dense":                                                (0.1725, 0.6275, 0.1725),  # tab10 green
    "specialized moe + globallb + 1shardexp + randpool":    (0.8902, 0.4667, 0.7608),  # tab10 pink
}

_VARIANT_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
_VARIANT_MARKERS    = ["o", "s", "^", "D", "v", "X"]
_VARIANT_ALPHAS     = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55]

_MODEL_VARIANT_STYLE: Dict[str, Dict[str, object]] = {}
for _model_name, _spec in MODEL_SPECS.items():
    _base_label = _spec.get("label", _model_name)
    for _vi, _v in enumerate(_spec.get("variants", [])):
        _full = f"{_base_label} {_v['label']}".strip()
        _idx = min(_vi, len(_VARIANT_ALPHAS) - 1)
        _MODEL_VARIANT_STYLE[_full] = {
            "alpha":     _VARIANT_ALPHAS[_idx],
            "linestyle": _VARIANT_LINESTYLES[_idx],
            "marker":    _VARIANT_MARKERS[_idx],
        }

LEGEND_WRAP_WIDTH = 35

GROUPS: List[Tuple[str, List[str]]] = [
    ("mc9_avg",             MC9_TASKS),
    ("gen5_avg",            GEN5_TASKS),
    ("mmlu_merged_avg",     MMLU_MERGED_TASKS),
    ("mmlu_pro_merged_avg", MMLU_PRO_MERGED_TASKS),
]

# ============================================================================
# Helpers
# ============================================================================


def _get_model_variants(model_name: str) -> List[Tuple[str, str]]:
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        return []
    return [(v["suffix"], v["label"]) for v in spec.get("variants", [])]


def _wrap_label(label: str) -> str:
    return textwrap.fill(label, width=LEGEND_WRAP_WIDTH, break_on_hyphens=True)


def _color_for(label: str) -> object:
    for base_label, color in sorted(
        _MODEL_BASE_COLORS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if label.startswith(base_label):
            return color
    return "gray"


def _scan_checkpoints(
    task_dir: Path,
    metric_key: str,
    model_key: str,
    model_label: str,
    task_run: str,
    rows: List[Dict[str, object]],
) -> None:
    """Scan checkpoint dirs under *task_dir*/results and append rows."""
    results_dir = task_dir / "results"
    if not results_dir.is_dir():
        return

    for checkpoint_dir in sorted(results_dir.glob("checkpoint-*")):
        if not checkpoint_dir.is_dir():
            continue

        try:
            step = int(checkpoint_dir.name.replace("checkpoint-", ""))
        except ValueError:
            print(f"[WARN] Unexpected checkpoint dir: {checkpoint_dir}")
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
        try:
            value = float(value)
        except (TypeError, ValueError):
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
    prune_evals_root: Path,
    model_names: Sequence[str],
    task_runs: Sequence[str],
    metric_key: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_name in model_names:
        model_dir = prune_evals_root / model_name
        if not model_dir.is_dir():
            print(f"[WARN] Missing model dir: {model_dir}")
            continue
        model_label = MODEL_LABELS.get(model_name, model_name)

        for task_run in task_runs:
            for suffix, label_mod in _get_model_variants(model_name):
                variant_task_dir = model_dir / (task_run + suffix)
                if not variant_task_dir.is_dir():
                    continue
                _scan_checkpoints(
                    variant_task_dir,
                    metric_key,
                    model_key=model_name + suffix,
                    model_label=f"{model_label} {label_mod}".strip(),
                    task_run=task_run,
                    rows=rows,
                )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["task_run", "model", "checkpoint"])


def collect_group_avg(
    prune_evals_root: Path,
    model_names: Sequence[str],
    metric_key: str,
    subtasks: Sequence[str],
    avg_name: str,
) -> pd.DataFrame:
    """Per-(model, subtask) checkpoints are aligned to a 0-based relative index,
    then averaged across subtasks per relative checkpoint."""
    df = collect_records(prune_evals_root, model_names, subtasks, metric_key)
    if df.empty:
        return pd.DataFrame()

    df = df[df["task_run"].isin(subtasks)]
    if df.empty:
        return pd.DataFrame()

    # Drop models that are missing any subtask entirely.
    complete_models: List[str] = []
    for model_key in sorted(df["model"].unique()):
        model_label = df.loc[df["model"] == model_key, "model_label"].iloc[0]
        present = set(df.loc[df["model"] == model_key, "task_run"].unique())
        missing = [t for t in subtasks if t not in present]
        if missing:
            print(
                f"[WARN] Excluding model {model_label!r} from {avg_name}: "
                f"missing {len(missing)}/{len(subtasks)} sub-task(s) for "
                f"metric={metric_key!r}: {missing}"
            )
        else:
            complete_models.append(model_key)
    df = df[df["model"].isin(complete_models)]
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["model", "task_run", "checkpoint"])
    df["checkpoint_rel"] = df.groupby(["model", "task_run"]).cumcount()

    avg_df = (
        df.groupby(["model", "model_label", "checkpoint_rel"], as_index=False)
          .agg(metric_value=("metric_value", "mean"))
          .rename(columns={"checkpoint_rel": "checkpoint"})
    )
    avg_df["task_run"] = avg_name
    return avg_df


# ============================================================================
# Plotting
# ============================================================================


def plot_curves(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    style: str,
    show: bool,
) -> None:
    if df.empty:
        print(f"[WARN] No data to plot for {title}")
        return

    sns.set_theme(style=style)
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    for model_label in sorted(df["model_label"].unique()):
        model_df = df[df["model_label"] == model_label].sort_values("checkpoint")
        vstyle = _MODEL_VARIANT_STYLE.get(
            model_label,
            {"alpha": 1.0, "linestyle": "-", "marker": "o"},
        )
        ax.plot(
            model_df["checkpoint"],
            model_df["metric_value"],
            marker=vstyle["marker"],
            linestyle=vstyle["linestyle"],
            linewidth=2,
            markersize=6,
            color=_color_for(model_label),
            alpha=vstyle["alpha"],
            label=_wrap_label(model_label),
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"[INFO] Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot prune_evals_final metric curves across checkpoints."
    )
    parser.add_argument("--prune-evals-root", type=Path, default=DEFAULT_PRUNE_EVALS_ROOT)
    parser.add_argument("--output-dir",       type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-subdir",    default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument(
        "--metric-key",
        default=None,
        help="Comma-separated metric keys (overrides TASK_SPECS for all tasks).",
    )
    parser.add_argument("--style", default="darkgrid", help="Seaborn style.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    parser.add_argument("--models", default=None, help="Comma-separated model names to plot.")
    parser.add_argument("--tasks",  default=None, help="Comma-separated task names to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_models = parse_csv_arg(args.models) or list(PLOT_MODELS)
    selected_tasks = parse_csv_arg(args.tasks) or list(TASK_SPECS.keys())

    if not selected_models:
        raise RuntimeError("No models selected.")
    if not selected_tasks:
        raise RuntimeError("No tasks selected.")

    metric_override = parse_csv_arg(args.metric_key)

    metric_to_tasks: Dict[str, List[str]] = {}
    for task_run in selected_tasks:
        metrics = metric_override or TASK_SPECS.get(task_run)
        if not metrics:
            print(f"[WARN] No metrics configured for task: {task_run}")
            continue
        for metric_key in metrics:
            metric_to_tasks.setdefault(metric_key, []).append(task_run)

    if not metric_to_tasks:
        raise RuntimeError("No metrics selected. Check TASK_SPECS or --metric-key.")

    base_output_dir = (args.output_dir / args.output_subdir).resolve()

    for metric_key in sorted(metric_to_tasks):
        tasks_for_metric = metric_to_tasks[metric_key]
        metric_dir = base_output_dir / sanitize_filename(metric_key)

        # --- per-task plots ---
        df = collect_records(
            args.prune_evals_root, selected_models, tasks_for_metric, metric_key
        )
        if df.empty:
            print(f"[WARN] No data for metric {metric_key!r}; skipping per-task plots.")
        else:
            for task_run in tasks_for_metric:
                task_df = df[df["task_run"] == task_run]
                if task_df.empty:
                    print(f"[WARN] No data for task={task_run!r}, metric={metric_key!r}")
                    continue
                output_file = (
                    metric_dir / f"{sanitize_filename(task_run)}_{sanitize_filename(metric_key)}.png"
                )
                plot_curves(
                    task_df,
                    title=f"{task_run} ({metric_key})",
                    xlabel="Checkpoint",
                    ylabel=metric_key,
                    output_path=output_file,
                    style=args.style,
                    show=args.show,
                )

        # --- group-average plots (left-aligned) ---
        for avg_name, group_tasks in GROUPS:
            relevant = [t for t in group_tasks if t in tasks_for_metric]
            if not relevant:
                continue
            avg_df = collect_group_avg(
                args.prune_evals_root,
                selected_models,
                metric_key,
                subtasks=relevant,
                avg_name=avg_name,
            )
            if avg_df.empty:
                print(f"[WARN] No data for {avg_name} (metric={metric_key!r}); skipping.")
                continue
            output_file = metric_dir / f"{avg_name}_{sanitize_filename(metric_key)}.png"
            plot_curves(
                avg_df,
                title=f"{avg_name} ({metric_key}, left-aligned)",
                xlabel="Relative checkpoint (starts at 0)",
                ylabel=metric_key,
                output_path=output_file,
                style=args.style,
                show=args.show,
            )


if __name__ == "__main__":
    main()
