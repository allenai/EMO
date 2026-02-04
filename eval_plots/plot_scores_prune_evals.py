#!/usr/bin/env python3
"""Plot metrics from prune_evals checkpoints for selected models/tasks."""

from __future__ import annotations

import argparse
import json
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

# Current prune_evals inventory (auto-generated at script creation time).
AVAILABLE_MODELS = [
    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123step30995-hf",
    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121step30995-hf",
    "dense_1b_olmoe-mix_prenorm_noqknorm_1123step30995-hf",
    "moe_1b4b_32experts_1224step30995-hf",
]

AVAILABLE_TASK_RUNS = [
    # "arc_challenge_keepk_32_bs-32_lr-5e-5_epoch-1",
    # "arc_easy_keepk_32_bs-32_lr-5e-5_epoch-1",
    # "boolq_keepk_32_bs-32_lr-5e-5_epoch-1",
    "coqa_0shot_keepk_32_bs-32_lr-5e-5_epoch-1",
    # "csqa_keepk_32_bs-32_lr-5e-5_epoch-1",
    # "gsm8k_generation_0shot_keepk_32_bs-32_lr-5e-5_epoch-1",
    # "hellaswag_keepk_32_bs-32_lr-5e-5_epoch-1",
    # "openbookqa_keepk_32_bs-32_lr-5e-5_epoch-1",
    # "piqa_keepk_32_bs-32_lr-5e-5_epoch-1",
    # "socialiqa_keepk_32_bs-32_lr-5e-5_epoch-1",
    # "squad_0shot_keepk_32_bs-32_lr-5e-5_epoch-1",
    # "winogrande_keepk_32_bs-32_lr-5e-5_epoch-1",
]

# Select which models/tasks to plot.
# Tip: set to a subset of AVAILABLE_* lists for quick filtering.
SELECTED_MODELS = list(AVAILABLE_MODELS)
SELECTED_TASK_RUNS = list(AVAILABLE_TASK_RUNS)

# Optional display names for models (legend labels).
# Key: model directory name. Value: label to show in plots.
MODEL_LABELS = {
    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123step30995-hf": "moe",
    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121step30995-hf": "twolevelbatchlb",
    "dense_1b_olmoe-mix_prenorm_noqknorm_1123step30995-hf": "dense",
    "moe_1b4b_32experts_1224step30995-hf": "moe_1b4b",
}

# Metric to plot (override with --metric-key if desired).
METRIC_KEY = "recall"

DEFAULT_OUTPUT_SUBDIR = "prune_eval_plots"

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
        default=METRIC_KEY,
        help="Metric key to plot (e.g., acc_per_byte).",
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


def discover_catalog(prune_evals_root: Path) -> Tuple[List[str], List[str]]:
    models = sorted([p.name for p in prune_evals_root.iterdir() if p.is_dir()])
    task_runs = sorted(
        {
            t.name
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
            task_dir = model_dir / task_run / "results"
            if not task_dir.is_dir():
                continue

            for checkpoint_dir in sorted(task_dir.glob("checkpoint-*")):
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
                        "model": model_name,
                        "model_label": model_label,
                        "task_run": task_run,
                        "task_label": task_label,
                        "checkpoint": step,
                        "metric_value": metric_value,
                    }
                )

    if not rows:
        raise RuntimeError("No metrics loaded. Check paths and selections.")

    df = pd.DataFrame(rows)
    return df.sort_values(["task_run", "model", "checkpoint"]), task_labels


def plot_task(
    df: pd.DataFrame,
    task_run: str,
    task_label: str,
    output_path: Path,
    style: str,
    show: bool,
    metric_key: str,
) -> None:
    task_df = df[df["task_run"] == task_run]
    if task_df.empty:
        print(f"[WARN] No data for task run: {task_run}")
        return

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
            label=model_label,
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

    model_set = [m for m in selected_models if m in available_models]
    task_set = [t for t in selected_tasks if t in available_tasks]

    if not model_set:
        raise RuntimeError("No valid models selected.")
    if not task_set:
        raise RuntimeError("No valid task runs selected.")

    df, task_labels = collect_records(
        args.prune_evals_root,
        model_set,
        task_set,
        args.metric_key,
    )

    base_output_dir = (args.output_dir / args.output_subdir).resolve()
    for task_run in task_set:
        label = task_labels.get(task_run, task_run)
        output_file = (
            base_output_dir
            / f"{sanitize_filename(task_run)}_{sanitize_filename(args.metric_key)}.png"
        )
        plot_task(
            df,
            task_run,
            label,
            output_file,
            args.style,
            args.show,
            args.metric_key,
        )


if __name__ == "__main__":
    main()
