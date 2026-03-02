#!/usr/bin/env python3
"""Plot metrics from prune_evals checkpoints for selected models/tasks."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============================================================================
# CONFIGURATION - edit these lists to control what gets plotted
# ============================================================================

# Auto-discover models/tasks on disk. Set to False to use the static lists below.
AUTO_DISCOVER = True

# Current prune_evals inventory.
# Key: model directory name.
# Value: dict with:
#   "label"    — display name for plots/tables
#   "variants" — list of {"suffix": ..., "label": ...} dicts specifying which
#                task-directory suffixes to scan and their legend labels.
#                Use [] for models with no pruning (dense, small MoE).
MODEL_SPECS = {
    "moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211step30995-hf": {
        "label": "moe_reduce",
        "variants": [
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1", "label": "(keepk 32)"},
        ],
    },
    "moereducedp256_1b4b_lr-4e-3_lb-1e-1_0212step30995-hf": {
        "label": "moe_1b4b_reduce",
        "variants": [
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1", "label": "(keepk 32)"},
        ],
    },
    "dense_1b_lr-4e-3_0213step30995-hf": {
        "label": "dense-lr4e-3",
        "variants": [
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1", "label": "(keepk 32)"},
        ],
    },

    "twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211step30995-hf": {
        "label": "twolevelbatchlbreducedp512sharedexp1-lr4e-3-lb1e-1",
        "variants": [
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32, layerwise)"},
        ],
    },
    # "twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-2_0213step30995-hf": {
    #     "label": "twolevelbatchlbreducedp512sharedexp1-lr4e-3-lb1e-2",
    #     "variants": [
    #         {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32, layerwise)"},
    #     ],
    # },
    # "twolevelbatchlbreducedp512sharedexp4c2-32_1b14b_lr-4e-3_lb-1e-2_sharelb-1e-2_0214step30995-hf": {
    #     "label": "twolevelbatchlbreducedp512sharedexp4c2-lr4e-3-lb1e-2",
    #     "variants": [
    #         {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1", "label": "(keepk 32)"},
    #     ],
    # },

    # deprecated
    # "twolevelbatchlb-32_1b14b_lr-4e-3_lb-1e-1_0119step30995-hf": {"label": "twolevelbatchlb-lr4e-3-lb1e-1", "variants": [{"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1", "label": "(keepk 32)"}]},
    # "dense_1b_olmoe-mix_prenorm_noqknorm_1123step30995-hf": {"label": "dense", "variants": []},
    # "moe_1b4b_32experts_1224step30995-hf": {"label": "moe_1b4b", "variants": []},

}
AVAILABLE_MODELS = list(MODEL_SPECS)

# Task list + per-task metrics.
# Key: task run name. Value: list of metric keys to plot for that task.
TASK_SPECS = {
    "arc_challenge": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score"
    ],
    "arc_easy": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score"
    ],
    "boolq": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score"
    ],
    "coqa_0shot": [
        "recall",
        "f1",
        "primary_score"
    ],
    "csqa": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score"
    ],
    "gsm8k_generation_0shot": [
        "exact_match",
        "primary_score"
    ],
    "gsm8k_perplexity_0shot": [
        "bits_per_byte",
        "primary_score"
    ],
    "hellaswag": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score"
    ],
    "openbookqa": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score"
    ],
    "piqa": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score"
    ],
    "socialiqa": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score"
    ],
    "squad_0shot": [
        "recall",
        "f1",
        "primary_score"
    ],
    "winogrande": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score"
    ],

    "mmlu_biology": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_business": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_chemistry": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_computer_science": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_culture": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_economics": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_engineering": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_geography": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_health": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_history": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_law": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_math": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_other": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_philosophy_cat": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_physics": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_politics": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_psychology": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],

    # Virtual aggregated task: macro average across all MMLU categories.
    "mmlu_avg": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],

}
# MMLU sub-tasks whose metrics are averaged for the "mmlu_avg" virtual task.
MMLU_SUBTASKS = [t for t in TASK_SPECS if t.startswith("mmlu_") and t != "mmlu_avg"]

AVAILABLE_TASK_RUNS = list(TASK_SPECS)

# Select which models/tasks to plot.
# Tip: set to a subset of AVAILABLE_* lists for quick filtering.
SELECTED_MODELS = list(AVAILABLE_MODELS)
SELECTED_TASK_RUNS = list(AVAILABLE_TASK_RUNS)

# Optional display names for models (legend labels).
# Key: model directory name. Value: label to show in plots.
MODEL_LABELS = {
    model: spec["label"]
    for model, spec in MODEL_SPECS.items()
    if spec.get("label")
}

DEFAULT_OUTPUT_SUBDIR = "prune_eval_plots_0302"

# Collect all known variant suffixes from MODEL_SPECS for auto-discovery.
_ALL_VARIANT_SUFFIXES: List[str] = sorted(
    {
        v["suffix"]
        for spec in MODEL_SPECS.values()
        for v in spec.get("variants", [])
    },
    key=len,
    reverse=True,  # longest first for greedy stripping
)


def _get_model_variants(model_name: str) -> List[Tuple[str, str]]:
    """Return (suffix, label) pairs for the variants to scan for a model."""
    spec = MODEL_SPECS.get(model_name)
    if spec is not None:
        return [(v["suffix"], v["label"]) for v in spec.get("variants", [])]
    return []

# ============================================================================
# END CONFIGURATION
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot prune_evals metrics across checkpoints."
    )
    parser.add_argument(
        "--prune-evals-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "prune_evals",
        help="Path to prune_evals directory (defaults to repo_root/prune_evals).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Subdirectory inside output-dir for generated plots.",
    )
    parser.add_argument(
        "--metric-key",
        default=None,
        help=(
            "Comma-separated metric keys to plot (overrides TASK_SPECS for all tasks). "
            "Example: acc_per_byte,softloss_corr."
        ),
    )
    parser.add_argument(
        "--style",
        default="darkgrid",
        help="Seaborn style to apply before plotting.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to plot (overrides SELECTED_MODELS).",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task run names to plot (overrides SELECTED_TASK_RUNS).",
    )
    return parser.parse_args()


def _is_variant_task(task_name: str) -> bool:
    return any(task_name.endswith(suffix) for suffix in _ALL_VARIANT_SUFFIXES)


def _strip_variant_suffix(task_name: str) -> str:
    """Strip the longest matching variant suffix to get the base task name."""
    for suffix in _ALL_VARIANT_SUFFIXES:  # already sorted longest-first
        if task_name.endswith(suffix):
            return task_name[: -len(suffix)]
    return task_name


LEGEND_WRAP_WIDTH = 30


def _wrap_label(label: str) -> str:
    """Wrap long legend labels across multiple lines."""
    return textwrap.fill(label, width=LEGEND_WRAP_WIDTH, break_on_hyphens=True)


def discover_catalog(prune_evals_root: Path) -> Tuple[List[str], List[str]]:
    models = sorted([p.name for p in prune_evals_root.iterdir() if p.is_dir()])
    task_runs = sorted(
        {
            _strip_variant_suffix(t.name)
            for model_dir in prune_evals_root.iterdir()
            if model_dir.is_dir()
            for t in model_dir.iterdir()
            if t.is_dir()
        }
    )
    return models, task_runs


def sanitize_filename(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def read_metrics(metrics_path: Path) -> Optional[Dict[str, object]]:
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Failed to parse {metrics_path}: {exc}")
        return None
    if not isinstance(metrics, dict):
        print(f"[WARN] Unexpected metrics format in {metrics_path}")
        return None
    return metrics


def extract_task_label(metrics: Dict[str, object], metrics_path: Path) -> str:
    task_name = metrics.get("task_name")
    if isinstance(task_name, str) and task_name:
        return task_name
    task_config = metrics.get("task_config")
    if isinstance(task_config, dict):
        config_task = task_config.get("task_name")
        if isinstance(config_task, str) and config_task:
            return config_task
    stem = metrics_path.name
    if stem.startswith("task-") and stem.endswith("-metrics.json"):
        stem = stem[len("task-") : -len("-metrics.json")]
    return stem


def _scan_checkpoints(
    task_dir: Path,
    metric_key: str,
    model_key: str,
    model_label: str,
    task_run: str,
    rows: List[Dict[str, object]],
    task_labels: Dict[str, str],
) -> None:
    """Scan checkpoint dirs under *task_dir*/results and append rows."""
    results_dir = task_dir / "results"
    if not results_dir.is_dir():
        return

    for checkpoint_dir in sorted(results_dir.glob("checkpoint-*")):
        if not checkpoint_dir.is_dir():
            continue

        step_str = checkpoint_dir.name.replace("checkpoint-", "")
        try:
            step = int(step_str)
        except ValueError:
            print(f"[WARN] Unexpected checkpoint dir: {checkpoint_dir}")
            continue

        metrics_files = sorted(checkpoint_dir.glob("task-*-metrics.json"))
        if not metrics_files:
            print(f"[WARN] No metrics files in {checkpoint_dir}")
            continue

        metrics_path = metrics_files[0]
        metrics = read_metrics(metrics_path)
        if metrics is None:
            continue

        metric_values = metrics.get("metrics")
        if not isinstance(metric_values, dict):
            print(f"[WARN] Missing metrics dict in {metrics_path}")
            continue

        metric_value = metric_values.get(metric_key)
        if metric_value is None:
            print(f"[WARN] Missing {metric_key!r} in {metrics_path}")
            continue

        task_label = extract_task_label(metrics, metrics_path)
        task_labels.setdefault(task_run, task_label)

        rows.append(
            {
                "model": model_key,
                "model_label": model_label,
                "task_run": task_run,
                "task_label": task_label,
                "checkpoint": step,
                "metric_value": metric_value,
            }
        )


def collect_records(
    prune_evals_root: Path,
    model_names: Sequence[str],
    task_runs: Sequence[str],
    metric_key: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rows: List[Dict[str, object]] = []
    task_labels: Dict[str, str] = {}

    for model_name in model_names:
        model_dir = prune_evals_root / model_name
        if not model_dir.is_dir():
            print(f"[WARN] Missing model dir: {model_dir}")
            continue
        model_label = MODEL_LABELS.get(model_name, model_name)

        for task_run in task_runs:
            _scan_checkpoints(
                model_dir / task_run, metric_key,
                model_key=model_name, model_label=model_label,
                task_run=task_run, rows=rows, task_labels=task_labels,
            )

            for suffix, label_mod in _get_model_variants(model_name):
                variant_task_dir = model_dir / (task_run + suffix)
                if not variant_task_dir.is_dir():
                    continue
                _scan_checkpoints(
                    variant_task_dir, metric_key,
                    model_key=model_name + suffix,
                    model_label=model_label + " " + label_mod,
                    task_run=task_run, rows=rows, task_labels=task_labels,
                )

    if not rows:
        raise RuntimeError("No metrics loaded. Check paths and selections.")

    df = pd.DataFrame(rows)
    return df.sort_values(["task_run", "model", "checkpoint"]), task_labels


def collect_mmlu_avg_records(
    prune_evals_root: Path,
    model_names: Sequence[str],
    metric_key: str,
) -> Dict[int, pd.DataFrame]:
    """Collect MMLU sub-task records and return macro averages grouped by
    checkpoint count.

    Instead of truncating all sub-tasks to the minimum checkpoint count,
    sub-tasks are grouped by how many checkpoints they have (per model).
    Each group is averaged independently, yielding one DataFrame per
    distinct checkpoint count.

    Returns a dict mapping checkpoint count -> averaged DataFrame (same
    schema as ``collect_records`` output, with ``task_run`` set to
    ``"mmlu_avg_<N>ckpts"``).  An empty dict is returned when no data is
    available.
    """
    try:
        df, _ = collect_records(
            prune_evals_root, model_names, MMLU_SUBTASKS, metric_key
        )
    except RuntimeError:
        return {}

    df = df[df["task_run"].isin(MMLU_SUBTASKS)]
    if df.empty:
        return {}

    # Exclude models missing any MMLU sub-task to avoid misleading partial averages.
    # model keys in the df include variant suffixes, so check all unique keys.
    complete_models = []
    for model_key in sorted(df["model"].unique()):
        model_label = df.loc[df["model"] == model_key, "model_label"].iloc[0]
        model_tasks = set(
            df.loc[df["model"] == model_key, "task_run"].unique()
        )
        missing = [t for t in MMLU_SUBTASKS if t not in model_tasks]
        if missing:
            print(
                f"[WARN] Excluding model {model_label!r} from mmlu_avg: "
                f"missing {len(missing)}/{len(MMLU_SUBTASKS)} MMLU "
                f"sub-task(s) for metric={metric_key!r}: {missing}"
            )
        else:
            complete_models.append(model_key)

    df = df[df["model"].isin(complete_models)]
    if df.empty:
        return {}

    # Assign a 1-based ordinal checkpoint index per (model, subtask).
    df = df.sort_values(["model", "task_run", "checkpoint"])
    df["ckpt_idx"] = df.groupby(["model", "task_run"]).cumcount() + 1

    # Checkpoint count per (model, subtask).
    ckpt_counts = (
        df.groupby(["model", "task_run"])["ckpt_idx"]
        .max()
        .rename("n_ckpts")
        .reset_index()
    )
    df = df.merge(ckpt_counts, on=["model", "task_run"])

    result: Dict[int, pd.DataFrame] = {}
    for n in sorted(df["n_ckpts"].unique()):
        group_df = df[df["n_ckpts"] == n]

        for model_key in sorted(group_df["model"].unique()):
            mlabel = group_df.loc[
                group_df["model"] == model_key, "model_label"
            ].iloc[0]
            subtasks = sorted(
                group_df.loc[group_df["model"] == model_key, "task_run"].unique()
            )
            print(
                f"[INFO] mmlu_avg ({n} ckpts, metric={metric_key!r}): "
                f"model {mlabel!r} averages over {len(subtasks)} sub-task(s)"
            )

        avg_df = (
            group_df.groupby(
                ["model", "model_label", "ckpt_idx"], as_index=False
            ).agg(metric_value=("metric_value", "mean"))
        )
        avg_df = avg_df.rename(columns={"ckpt_idx": "checkpoint"})
        avg_df["task_run"] = f"mmlu_avg_{n}ckpts"
        avg_df["task_label"] = f"MMLU avg ({n} ckpts)"

        result[n] = avg_df

    return result


def plot_mmlu_avg_subplots(
    ckpt_groups: Dict[int, pd.DataFrame],
    output_path: Path,
    style: str,
    show: bool,
    metric_key: str,
) -> None:
    """Plot MMLU macro-avg results as side-by-side subplots, one per checkpoint count."""
    if not ckpt_groups:
        print("[WARN] No MMLU avg data to plot.")
        return

    sorted_counts = sorted(ckpt_groups)
    n_groups = len(sorted_counts)

    all_labels = sorted(
        {
            label
            for avg_df in ckpt_groups.values()
            for label in avg_df["model_label"].unique()
        }
    )

    sns.set_theme(style=style)
    fig, axes = plt.subplots(
        1, n_groups, figsize=(8 * n_groups, 5), squeeze=False
    )

    palette = sns.color_palette("colorblind", n_colors=len(all_labels))
    color_map = {label: palette[i] for i, label in enumerate(all_labels)}

    for ax_idx, n_ckpts in enumerate(sorted_counts):
        ax = axes[0][ax_idx]
        avg_df = ckpt_groups[n_ckpts]

        for model_label in sorted(avg_df["model_label"].unique()):
            model_df = avg_df[
                avg_df["model_label"] == model_label
            ].sort_values("checkpoint")
            ax.plot(
                model_df["checkpoint"],
                model_df["metric_value"],
                marker="o",
                linewidth=2,
                markersize=6,
                color=color_map[model_label],
                label=_wrap_label(model_label),
            )

        ax.set_title(f"MMLU avg ({n_ckpts} checkpoints)")
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel(metric_key)

    # Deduplicated legend shared across all subplots.
    seen: Dict[str, object] = {}
    for ax_row in axes:
        for ax in ax_row:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(label, handle)
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        title="Model",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"[INFO] Saved MMLU avg subplot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_task(
    df: pd.DataFrame,
    task_run: str,
    task_label: str,
    output_path: Path,
    style: str,
    show: bool,
    metric_key: str,
    expected_models: Optional[Sequence[str]] = None,
    task_metrics: Optional[Sequence[str]] = None,
) -> None:
    task_df = df[df["task_run"] == task_run]
    if task_df.empty:
        print(f"[WARN] No data for task run: {task_run}")
        return

    # Warn when any explicitly configured variant is missing for this
    # task/metric. Base model keys are not treated as independent entities.
    if expected_models is not None:
        models_present = set(task_df["model"].unique())
        metrics_info = list(task_metrics) if task_metrics is not None else [metric_key]
        for model in expected_models:
            model_label = MODEL_LABELS.get(model, model)
            variants = _get_model_variants(model)
            for suffix, label_mod in variants:
                variant_key = model + suffix
                if variant_key in models_present:
                    continue
                variant_label = model_label + " " + label_mod
                print(
                    f"[WARN] Model {variant_label!r} has no data for "
                    f"task={task_run!r}, metric={metric_key!r}, "
                    f"task_metrics={metrics_info!r}"
                )

    sns.set_theme(style=style)
    plt.figure(figsize=(8, 5))

    ax = plt.gca()
    palette = sns.color_palette("colorblind", n_colors=task_df["model_label"].nunique())
    color_map = {
        model: palette[idx]
        for idx, model in enumerate(sorted(task_df["model_label"].unique()))
    }

    for model_label in sorted(task_df["model_label"].unique()):
        model_df = task_df[task_df["model_label"] == model_label].sort_values("checkpoint")
        ax.plot(
            model_df["checkpoint"],
            model_df["metric_value"],
            marker="o",
            linewidth=2,
            markersize=6,
            color=color_map[model_label],
            label=_wrap_label(model_label),
        )

    ax.set_title(f"{task_label} ({task_run})")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(metric_key)
    ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"[INFO] Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def parse_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()

    if AUTO_DISCOVER:
        discovered_models, discovered_tasks = discover_catalog(args.prune_evals_root)
        available_models = discovered_models
        available_tasks = discovered_tasks
    else:
        available_models = list(AVAILABLE_MODELS)
        available_tasks = list(AVAILABLE_TASK_RUNS)

    selected_models = parse_csv_arg(args.models) or list(SELECTED_MODELS)
    selected_tasks = parse_csv_arg(args.tasks) or list(SELECTED_TASK_RUNS)

    # Virtual tasks (e.g. mmlu_avg) don't exist on disk; always keep them.
    VIRTUAL_TASKS = {"mmlu_avg"}
    model_set = [m for m in selected_models if m in available_models]
    task_set = [t for t in selected_tasks if t in available_tasks or t in VIRTUAL_TASKS]

    if not model_set:
        raise RuntimeError("No valid models selected.")
    if not task_set:
        raise RuntimeError("No valid task runs selected.")

    metric_override = parse_csv_arg(args.metric_key)
    metric_to_tasks: Dict[str, List[str]] = {}
    for task_run in task_set:
        metrics = metric_override or TASK_SPECS.get(task_run)
        if not metrics:
            print(f"[WARN] No metrics configured for task: {task_run}")
            continue
        for metric_key in metrics:
            metric_to_tasks.setdefault(metric_key, []).append(task_run)

    if not metric_to_tasks:
        raise RuntimeError("No metrics selected. Check TASK_SPECS or --metric-key.")

    base_output_dir = (args.output_dir / args.output_subdir).resolve()
    for metric_key, tasks_for_metric in metric_to_tasks.items():
        # Separate the virtual mmlu_avg task from regular on-disk tasks.
        regular_tasks = [t for t in tasks_for_metric if t != "mmlu_avg"]
        has_mmlu_avg = "mmlu_avg" in tasks_for_metric

        # --- regular per-task plots ---
        if regular_tasks:
            df, task_labels = collect_records(
                args.prune_evals_root,
                model_set,
                regular_tasks,
                metric_key,
            )
            metric_output_dir = base_output_dir / sanitize_filename(metric_key)
            for task_run in regular_tasks:
                label = task_labels.get(task_run, task_run)
                output_file = (
                    metric_output_dir
                    / f"{sanitize_filename(task_run)}_{sanitize_filename(metric_key)}.png"
                )
                plot_task(
                    df,
                    task_run,
                    label,
                    output_file,
                    args.style,
                    args.show,
                    metric_key,
                    expected_models=model_set,
                    task_metrics=metric_override or TASK_SPECS.get(task_run),
                )

        # --- mmlu_avg (macro average across MMLU categories) ---
        if has_mmlu_avg:
            ckpt_groups = collect_mmlu_avg_records(
                args.prune_evals_root, model_set, metric_key
            )
            if not ckpt_groups:
                print(f"[WARN] No MMLU data for metric {metric_key!r}; skipping mmlu_avg.")
            else:
                metric_output_dir = base_output_dir / sanitize_filename(metric_key)
                output_file = (
                    metric_output_dir
                    / f"mmlu_avg_{sanitize_filename(metric_key)}.png"
                )
                plot_mmlu_avg_subplots(
                    ckpt_groups,
                    output_file,
                    args.style,
                    args.show,
                    metric_key,
                )


if __name__ == "__main__":
    main()
