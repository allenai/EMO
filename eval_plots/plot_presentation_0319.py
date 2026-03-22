#!/usr/bin/env python3
"""Plot metrics from prune_evals checkpoints for selected models/tasks. see if adding per-subject sampling changes the results much"""

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
    "dense_1b_lr-4e-3_0213step30995-hf": {
        "label": "dense",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": " "},
        ],
    },
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308step30995-hf": {
        "label": "moe",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 8)"},
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32)"},
            {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
        ],
    },
    "moereducedp512sharedexp1_1b4b_lr-4e-3_lb-1e-1_0308step30995-hf": {
        "label": "moe_small",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": " "},
        ],
    },

    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301step30995-hf": {
        "label": "specialized moe + globallb + 1shardexp + randpool",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 8)"},
            # {"suffix": "_keepk_16_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 16)"},
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32)"},
            # {"suffix": "_keepk_64_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 64)"},
            # {"suffix": "_keepk_96_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 96)"},
            # {"suffix": "_keepk_120_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 120)"},
            {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
        ],
    },

    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313step238419-hf": {
        "label": "specialized moe 1T",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 8)"},
            {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32)"},
        ]
    },


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

    # Virtual aggregated tasks: macro average across MMLU categories.
    "mmlu_avg": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],
    "mmlu_avg_no_other": [
        "softloss_corr",
        "acc_per_byte",
        "primary_score",
    ],

}
# MMLU sub-tasks whose metrics are averaged for the "mmlu_avg" virtual task.
MMLU_SUBTASKS = [t for t in TASK_SPECS if t.startswith("mmlu_") and t not in ("mmlu_avg", "mmlu_avg_no_other")]
MMLU_SUBTASKS_NO_OTHER = [t for t in MMLU_SUBTASKS if t != "mmlu_other"]

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

DEFAULT_OUTPUT_SUBDIR = "presentation_0319_plots"

# Hardcoded color assignments so colors never change regardless of which
# models are commented in/out.  Colors are from matplotlib's tab10 palette.
_MODEL_BASE_COLORS: Dict[str, object] = {
    "moe":                                                  (0.1216, 0.4667, 0.7059),  # tab10 blue
    "moe_small":                                            (1.0000, 0.4980, 0.0549),  # tab10 orange
    # different shaed of orange
    "moe_small_sharedexp":                                   (1.0000, 0.6471, 0.0000),  # tab10 orange (darker)
    "dense":                                                (0.1725, 0.6275, 0.1725),  # tab10 green
    "specialized moe":                                      (0.8392, 0.1529, 0.1569),  # tab10 red
    "specialized moe + globallb":                           (0.5804, 0.4039, 0.7412),  # tab10 purple
    "specialized moe + globallb + 1sharedexp":              (0.5490, 0.3373, 0.2941),  # tab10 brown
    "specialized moe + globallb + 1shardexp + randpool":    (0.8902, 0.4667, 0.7608),  # tab10 pink
}

# Per-variant alpha: first variant is fully opaque, subsequent ones fade.
_VARIANT_ALPHAS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 1.0]
_MODEL_VARIANT_ALPHA: Dict[str, float] = {}
for _model_name, _spec in MODEL_SPECS.items():
    _base_label = _spec.get("label", _model_name)
    _variants = _spec.get("variants", [])
    if not _variants:
        _MODEL_VARIANT_ALPHA[_base_label] = 1.0
    for _vi, _v in enumerate(_variants):
        _full_label = _base_label + " " + _v["label"]
        _MODEL_VARIANT_ALPHA[_full_label] = _VARIANT_ALPHAS[min(_vi, len(_VARIANT_ALPHAS) - 1)]

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
        default=Path(__file__).resolve().parent.parent / "prune_evals_0313",
        help="Path to prune_evals directory (defaults to repo_root/prune_evals_0313).",
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
            if t.is_dir() and t.name != "original_model"
        }
    )
    return models, task_runs


def _load_baseline_metric(
    prune_evals_root: Path,
    model_name: str,
    task_run: str,
    metric_key: str,
) -> Optional[float]:
    """Load the baseline (unpruned, unfinetuned) metric for a model+task.

    Looks in ``<model>/original_model/<task>/results/checkpoint-0/``.
    """
    ckpt_dir = (
        prune_evals_root / model_name / "original_model" / task_run
        / "results" / "checkpoint-0"
    )
    if not ckpt_dir.is_dir():
        return None

    metrics_files = sorted(ckpt_dir.glob("task-*-metrics.json"))
    if not metrics_files:
        return None

    metrics = read_metrics(metrics_files[0])
    if metrics is None:
        return None

    metric_values = metrics.get("metrics")
    if not isinstance(metric_values, dict):
        return None

    value = metric_values.get(metric_key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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
    subtasks: Optional[List[str]] = None,
    avg_name: str = "mmlu_avg",
) -> pd.DataFrame:
    """Collect MMLU sub-task records and return one left-aligned macro average.

    For each (model, subtask), checkpoints are sorted and re-indexed to a
    zero-based relative index so all curves are aligned at checkpoint 0.
    Macro averaging is then computed across available sub-tasks at each
    relative checkpoint index.

    Parameters
    ----------
    subtasks : list of str, optional
        Which MMLU subtasks to average over. Defaults to ``MMLU_SUBTASKS``.
    avg_name : str
        Name used for ``task_run`` column and plot title.

    Returns a DataFrame with the same schema as ``collect_records`` output,
    with ``task_run`` set to *avg_name* and ``checkpoint`` as the
    left-aligned relative index. An empty DataFrame is returned when no data
    is available.
    """
    if subtasks is None:
        subtasks = MMLU_SUBTASKS

    try:
        df, _ = collect_records(
            prune_evals_root, model_names, subtasks, metric_key
        )
    except RuntimeError:
        return pd.DataFrame()

    df = df[df["task_run"].isin(subtasks)]
    if df.empty:
        return pd.DataFrame()

    # Exclude models missing any MMLU sub-task to avoid misleading partial averages.
    # model keys in the df include variant suffixes, so check all unique keys.
    complete_models = []
    for model_key in sorted(df["model"].unique()):
        model_label = df.loc[df["model"] == model_key, "model_label"].iloc[0]
        model_tasks = set(
            df.loc[df["model"] == model_key, "task_run"].unique()
        )
        missing = [t for t in subtasks if t not in model_tasks]
        if missing:
            print(
                f"[WARN] Excluding model {model_label!r} from {avg_name}: "
                f"missing {len(missing)}/{len(subtasks)} MMLU "
                f"sub-task(s) for metric={metric_key!r}: {missing}"
            )
        else:
            complete_models.append(model_key)

    df = df[df["model"].isin(complete_models)]
    if df.empty:
        return pd.DataFrame()

    # Assign a 0-based relative checkpoint index per (model, subtask).
    df = df.sort_values(["model", "task_run", "checkpoint"])
    df["checkpoint_rel"] = df.groupby(["model", "task_run"]).cumcount()

    # Log coverage per model after alignment.
    for model_key in sorted(df["model"].unique()):
        mlabel = df.loc[df["model"] == model_key, "model_label"].iloc[0]
        model_subtasks = sorted(
            df.loc[df["model"] == model_key, "task_run"].unique()
        )
        max_rel_ckpt = int(
            df.loc[df["model"] == model_key, "checkpoint_rel"].max()
        )
        print(
            f"[INFO] {avg_name} aligned (metric={metric_key!r}): "
            f"model {mlabel!r} averages over {len(model_subtasks)} sub-task(s), "
            f"max_relative_checkpoint={max_rel_ckpt}"
        )

    avg_df = (
        df.groupby(
            ["model", "model_label", "checkpoint_rel"], as_index=False
        ).agg(metric_value=("metric_value", "mean"))
    )
    avg_df = avg_df.rename(columns={"checkpoint_rel": "checkpoint"})
    avg_df = avg_df[avg_df["checkpoint"] <= 6]
    avg_df["task_run"] = avg_name
    avg_df["task_label"] = f"{avg_name} (left-aligned @ checkpoint 0)"
    return avg_df


def plot_mmlu_avg(
    avg_df: pd.DataFrame,
    output_path: Path,
    style: str,
    show: bool,
    metric_key: str,
    baselines: Optional[Dict[str, float]] = None,
) -> None:
    """Plot one left-aligned MMLU macro-average curve per model."""
    if avg_df.empty:
        print("[WARN] No MMLU avg data to plot.")
        return

    all_labels = sorted(avg_df["model_label"].unique())

    sns.set_theme(style=style)
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    # Use fixed colors: match each plotted label to its base model color.
    color_map: Dict[str, object] = {}
    for plotted_label in all_labels:
        matched = False
        for base_label, color in sorted(_MODEL_BASE_COLORS.items(), key=lambda x: len(x[0]), reverse=True):
            if plotted_label.startswith(base_label):
                color_map[plotted_label] = color
                matched = True
                break
        if not matched:
            color_map[plotted_label] = "gray"

    for model_label in sorted(avg_df["model_label"].unique()):
        model_df = avg_df[
            avg_df["model_label"] == model_label
        ].sort_values("checkpoint")
        alpha = _MODEL_VARIANT_ALPHA.get(model_label, 1.0)
        ax.plot(
            model_df["checkpoint"],
            model_df["metric_value"],
            marker="o",
            linewidth=2,
            markersize=6,
            color=color_map[model_label],
            alpha=alpha,
            label=_wrap_label(model_label),
        )

    # Draw baseline horizontal lines
    if baselines:
        for model_name, baseline_val in baselines.items():
            model_label = MODEL_LABELS.get(model_name, model_name)
            baseline_color = _MODEL_BASE_COLORS.get(model_label, "gray")
            ax.axhline(
                y=baseline_val,
                color=baseline_color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                label=_wrap_label(f"{model_label} (original)"),
            )

    ax.set_title("MMLU avg (left-aligned checkpoints)")
    ax.set_xlabel("Relative checkpoint (starts at 0)")
    ax.set_ylabel(metric_key)
    ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"[INFO] Saved MMLU avg plot to {output_path}")

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
    baselines: Optional[Dict[str, float]] = None,
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

    # Use fixed colors: match each plotted label to its base model color.
    color_map: Dict[str, object] = {}
    for plotted_label in sorted(task_df["model_label"].unique()):
        matched = False
        for base_label, color in sorted(_MODEL_BASE_COLORS.items(), key=lambda x: len(x[0]), reverse=True):
            if plotted_label.startswith(base_label):
                color_map[plotted_label] = color
                matched = True
                break
        if not matched:
            color_map[plotted_label] = "gray"

    for model_label in sorted(task_df["model_label"].unique()):
        model_df = task_df[task_df["model_label"] == model_label].sort_values("checkpoint")
        alpha = _MODEL_VARIANT_ALPHA.get(model_label, 1.0)
        ax.plot(
            model_df["checkpoint"],
            model_df["metric_value"],
            marker="o",
            linewidth=2,
            markersize=6,
            color=color_map[model_label],
            alpha=alpha,
            label=_wrap_label(model_label),
        )

    # Draw baseline horizontal lines
    if baselines:
        for model_name, baseline_val in baselines.items():
            model_label = MODEL_LABELS.get(model_name, model_name)
            baseline_color = _MODEL_BASE_COLORS.get(model_label, "gray")
            ax.axhline(
                y=baseline_val,
                color=baseline_color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                label=_wrap_label(f"{model_label} (original)"),
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
    VIRTUAL_TASKS = {"mmlu_avg", "mmlu_avg_no_other"}
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
        # Separate virtual mmlu_avg tasks from regular on-disk tasks.
        regular_tasks = [t for t in tasks_for_metric if t not in VIRTUAL_TASKS]
        has_mmlu_avg = "mmlu_avg" in tasks_for_metric
        has_mmlu_avg_no_other = "mmlu_avg_no_other" in tasks_for_metric

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
                # Gather baselines for this task
                task_baselines: Dict[str, float] = {}
                for model_name in model_set:
                    spec = MODEL_SPECS.get(model_name)
                    if spec is None or not spec.get("baseline"):
                        continue
                    val = _load_baseline_metric(
                        args.prune_evals_root, model_name, task_run, metric_key
                    )
                    if val is not None:
                        task_baselines[model_name] = val
                    else:
                        model_label = MODEL_LABELS.get(model_name, model_name)
                        print(
                            f"[WARN] Baseline data missing for model {model_label!r}, "
                            f"task={task_run!r}, metric={metric_key!r}"
                        )

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
                    baselines=task_baselines if task_baselines else None,
                )

        # --- mmlu_avg (macro average across MMLU categories) ---
        if has_mmlu_avg:
            avg_df = collect_mmlu_avg_records(
                args.prune_evals_root, model_set, metric_key
            )
            # Gather MMLU avg baselines
            mmlu_baselines: Dict[str, float] = {}
            for model_name in model_set:
                spec = MODEL_SPECS.get(model_name)
                if spec is None or not spec.get("baseline"):
                    continue
                vals = []
                missing = []
                for subtask in MMLU_SUBTASKS:
                    v = _load_baseline_metric(
                        args.prune_evals_root, model_name, subtask, metric_key
                    )
                    if v is not None:
                        vals.append(v)
                    else:
                        missing.append(subtask)
                model_label = MODEL_LABELS.get(model_name, model_name)
                if missing:
                    print(
                        f"[WARN] Partial MMLU baseline for model {model_label!r}, "
                        f"metric={metric_key!r}: missing {len(missing)}/{len(MMLU_SUBTASKS)} "
                        f"sub-task(s): {missing}"
                    )
                if len(vals) == len(MMLU_SUBTASKS):
                    mmlu_baselines[model_name] = sum(vals) / len(vals)

            if avg_df.empty:
                print(f"[WARN] No MMLU data for metric {metric_key!r}; skipping mmlu_avg.")
            else:
                metric_output_dir = base_output_dir / sanitize_filename(metric_key)
                output_file = (
                    metric_output_dir
                    / f"mmlu_avg_{sanitize_filename(metric_key)}.png"
                )
                plot_mmlu_avg(
                    avg_df,
                    output_file,
                    args.style,
                    args.show,
                    metric_key,
                    baselines=mmlu_baselines if mmlu_baselines else None,
                )

        # --- mmlu_avg_no_other (excludes mmlu_other) ---
        if has_mmlu_avg_no_other:
            avg_no_other_df = collect_mmlu_avg_records(
                args.prune_evals_root, model_set, metric_key,
                subtasks=MMLU_SUBTASKS_NO_OTHER,
                avg_name="mmlu_avg_no_other",
            )
            # Gather baselines for mmlu_avg_no_other
            mmlu_no_other_baselines: Dict[str, float] = {}
            for model_name in model_set:
                spec = MODEL_SPECS.get(model_name)
                if spec is None or not spec.get("baseline"):
                    continue
                vals = []
                missing = []
                for subtask in MMLU_SUBTASKS_NO_OTHER:
                    v = _load_baseline_metric(
                        args.prune_evals_root, model_name, subtask, metric_key
                    )
                    if v is not None:
                        vals.append(v)
                    else:
                        missing.append(subtask)
                model_label = MODEL_LABELS.get(model_name, model_name)
                if missing:
                    print(
                        f"[WARN] Partial MMLU (no other) baseline for model {model_label!r}, "
                        f"metric={metric_key!r}: missing {len(missing)}/{len(MMLU_SUBTASKS_NO_OTHER)} "
                        f"sub-task(s): {missing}"
                    )
                if len(vals) == len(MMLU_SUBTASKS_NO_OTHER):
                    mmlu_no_other_baselines[model_name] = sum(vals) / len(vals)

            if avg_no_other_df.empty:
                print(f"[WARN] No MMLU data for metric {metric_key!r}; skipping mmlu_avg_no_other.")
            else:
                metric_output_dir = base_output_dir / sanitize_filename(metric_key)
                output_file = (
                    metric_output_dir
                    / f"mmlu_avg_no_other_{sanitize_filename(metric_key)}.png"
                )
                plot_mmlu_avg(
                    avg_no_other_df,
                    output_file,
                    args.style,
                    args.show,
                    metric_key,
                    baselines=mmlu_no_other_baselines if mmlu_no_other_baselines else None,
                )


if __name__ == "__main__":
    main()
