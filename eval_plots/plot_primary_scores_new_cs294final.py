#!/usr/bin/env python3
"""Plot primary_score over training steps for all nine tasks in a single 3x3 grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns

# ============================================================================
# CONFIGURATION - Full control over model runs and visual settings
# ============================================================================

# Base model identifier (used in output directory naming)
MAIN_MODEL = "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121_step30995"

# Task-specific configuration.
TASKS: List[str] = [
    "arc_challenge:rc_test",
    "arc_easy:rc_test",
    "boolq:rc_test",
    "csqa:rc_test",
    "hellaswag:rc_test",
    "openbookqa:rc_test",
    "piqa:rc_test",
    "socialiqa:rc_test",
    "winogrande:rc_test",
]

# Relative steps per task to compare across both model families.
TASK_STEPS = {
    "arc_easy:rc_test":       [0, 42, 84, 126, 168, 210],
    "arc_challenge:rc_test":  [0, 20, 40, 60, 80, 102],
    "boolq:rc_test":          [0, 157, 314, 471, 628, 789],
    "csqa:rc_test":           [0, 163, 326, 489, 652, 819],
    "hellaswag:rc_test":      [0, 729, 1458, 2187, 2916, 3645],
    "openbookqa:rc_test":     [0, 92, 184, 276, 368, 462],
    "piqa:rc_test":           [0, 283, 566, 849, 1132, 1416],
    "socialiqa:rc_test":      [0, 607, 1214, 1821, 2428, 3036],
    "winogrande:rc_test":     [0, 738, 1476, 2214, 2952, 3693],
}

# Family color mapping - define base colors for each model family
# Colors are specified as hex strings (e.g., "#988ED5")
FAMILY_COLORS: Dict[str, str] = {
    "dense": "#E24A33",        # Red/Orange
    "moe": "#348ABD",          # Blue
    "twolevelbatchlb train32/128": "#988ED5",  # Purple
    "twolevelbatchlb train32/128 lr high": "#FFB5B8",  # Light Red/Pink
    "twolevelbatchlb train8/64": "#8EBA42",    # Green
    "twolevelsamplingnolb": "#777777",  # Gray
    "mutualinfo": "#FBC15E",   # Yellow
}

# Model run configurations
# Each entry defines a model run that will be plotted as a line
# Template must contain {step} and {task_core} placeholders
MODEL_RUNS: List[Dict[str, Any]] = [
    {
        "label": "moe keepk32",
        "template": (
            "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123_step30995_"
            "task-{task_core}_rc_validation_keepk32_newdefault_lr-4e-5_finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "moe",
        "marker": "o",              # matplotlib marker style (o=circle, s=square, ^=triangle, etc.)
        "brightness": 1.0,          # Brightness multiplier (1.0 = no change, >1.0 = lighter, <1.0 = darker)
        "linewidth": 2,             # Line width
        "markersize": 9,            # Marker size
    },
    {
        "label": "twolevel keepk32",
        "template": (
            f"{MAIN_MODEL}_"
            "task-{task_core}_rc_validation_keepk32_newdefault_lr-4e-5_finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "twolevelbatchlb train32/128",
        "marker": "o",
        "brightness": 1.0,
        "linewidth": 2,
        "markersize": 9,
    },
    # {
    #     "label": "twolevel keepk8",
    #     "template": (
    #         f"{MAIN_MODEL}_"
    #         "task-{task_core}_rc_validation_keepk8_newdefault_lr-4e-5_finetune-task-{task_core}_rc_train_step{step}-hf"
    #     ),
    #     "family": "twolevelbatchlb train32/128",
    #     "marker": "o",
    #     "brightness": 0.7,
    #     "linewidth": 2,
    #     "markersize": 9,
    # },
    {
        "label": "mutualinfo keepk32",
        "template": (
            f"mutualinfo_1b14b_cond-1e-2_uncond-1e-2_1205_step30995_"
            "task-{task_core}_rc_validation_keepk32_newdefault_lr-4e-5_finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "mutualinfo",
        "marker": "o",
        "brightness": 1.0,
        "linewidth": 2,
        "markersize": 9,
    },
    {
        "label": "mutualinfo keepk8",
        "template": (
            f"mutualinfo_1b14b_cond-1e-2_uncond-1e-2_1205_step30995_"
            "task-{task_core}_rc_validation_keepk8_newdefault_lr-4e-5_finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "mutualinfo",
        "marker": "o",
        "brightness": 0.7,
        "linewidth": 2,
        "markersize": 9,
    },
    {
        "label": "dense finetuned",
        "template": (
            "dense_1b_olmoe-mix_prenorm_noqknorm_1123_step30995_newdefault_lr-4e-5_"
            "finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "dense",
        "marker": "v",              # Triangle down for dense
        "brightness": 1.0,
        "linewidth": 2,
        "markersize": 9,
    },
]

# Baseline model configurations
# These are plotted as horizontal dashed lines
BASELINE_RUNS: List[Dict[str, Any]] = [
    {
        "label": "moe full",
        "template": "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123_step30995-hf",
        "family": "moe",
        "linestyle": "--",          # Line style for baseline
        "linewidth": 1.2,
    },
    {
        "label": "twolevel full",
        "template": f"{MAIN_MODEL}-hf",
        "family": "twolevelbatchlb",
        "linestyle": "--",
        "linewidth": 1.2,
    },
    {
        "label": "dense full",
        "template": "dense_1b_olmoe-mix_prenorm_noqknorm_1123_step30995-hf",
        "family": "dense",
        "linestyle": "--",
        "linewidth": 1.2,
    },
]

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Build dictionaries from configuration for backward compatibility
MODEL_GROUPS: Dict[str, str] = {run["label"]: run["template"] for run in MODEL_RUNS}
BASELINE_MODELS: Dict[str, str] = {run["label"]: run["template"] for run in BASELINE_RUNS}
GROUP_FAMILY: Dict[str, str] = {run["label"]: run["family"] for run in MODEL_RUNS}
BASELINE_FAMILY: Dict[str, str] = {run["label"]: run["family"] for run in BASELINE_RUNS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 3x3 grid of seaborn line plots comparing primary_score across "
            "training steps for all nine tasks."
        )
    )
    parser.add_argument(
        "--evals-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "evals",
        help="Path to the evals directory (defaults to repo_root/evals).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to save the generated plot images.",
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
    return parser.parse_args()


def task_core_from_name(task_name: str) -> str:
    return task_name.split(":", 1)[0]


def collect_primary_scores(
        evals_root: Path,
        model_template: str,
        steps: Iterable[int],
        task_name: str,
        task_core: str,
        metrics_filename: str,
) -> tuple[List[Dict[str, float]], str | None]:
    """Load primary_score values for a model template across the provided steps."""
    records: List[Dict[str, float]] = []
    primary_metric_name: str | None = None

    for step in steps:
        try:
            formatted_path = model_template.format(step=step, task_core=task_core)
        except KeyError as exc:
            raise KeyError(
                f"Template {model_template!r} missing placeholder {exc}."
            ) from exc

        model_dir = evals_root / formatted_path
        metrics_path = model_dir / metrics_filename

        if not metrics_path.exists():
            print(f"[WARN] Missing metrics for {model_dir}")
            print(f"[DEBUG] Looking for: {metrics_path}")
            continue

        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Failed to parse JSON at {metrics_path}: {exc}")
            continue

        if not isinstance(metrics, dict):
            print(f"[WARN] Unexpected metrics format in {metrics_path}")
            continue

        task_config = metrics.get("task_config", {})
        if task_config and isinstance(task_config, dict):
            primary_metric_name = primary_metric_name or task_config.get("primary_metric")

        detected_task = metrics.get("task_name")
        if detected_task and detected_task != task_name:
            print(
                f"[WARN] Task mismatch detected in {metrics_path}: "
                f"{detected_task} vs {task_name}."
            )

        score = None
        metric_values = metrics.get("metrics")
        if isinstance(metric_values, dict):
            score = metric_values.get("primary_score")

        if score is None:
            print(
                f"[WARN] No 'primary_score' found in metrics for {metrics_path}"
            )
            continue

        records.append({"step": step, "primary_score": score})
        print(f"[DEBUG] Found score {score} for step {step} in {model_dir}")

    return records, primary_metric_name


def build_dataframe(
        evals_root: Path,
        task_name: str,
        metrics_filename: str,
        steps: Iterable[int],
) -> tuple[pd.DataFrame, str | None]:
    """Build a tidy DataFrame with columns [model_group, step, primary_score]."""
    rows: List[Dict[str, object]] = []
    primary_metric_name: str | None = None
    task_core = task_core_from_name(task_name)

    for group_label, template in MODEL_GROUPS.items():
        records, group_metric = collect_primary_scores(
            evals_root,
            template,
            steps,
            task_name,
            task_core,
            metrics_filename,
        )
        if group_metric and primary_metric_name and group_metric != primary_metric_name:
            print(
                f"[WARN] Primary metric mismatch: {group_metric} vs {primary_metric_name}. "
                "Using the first encountered metric."
            )
        primary_metric_name = primary_metric_name or group_metric

        for record in records:
            record["model_group"] = group_label
            rows.append(record)

    if not rows:
        raise RuntimeError(
            "No evaluation metrics were loaded. Check that the evals directory "
            "and model templates are correct."
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["model_group", "step"]), primary_metric_name


def load_baseline_scores(
        evals_root: Path,
        task_name: str,
        primary_metric_name: str | None,
        metrics_filename: str,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    for label, dir_name in BASELINE_MODELS.items():
        metrics_path = evals_root / dir_name / metrics_filename
        if not metrics_path.exists():
            print(f"[WARN] Missing baseline metrics for {metrics_path}")
            continue

        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Failed to parse JSON at {metrics_path}: {exc}")
            continue

        if not isinstance(metrics, dict):
            print(f"[WARN] Unexpected metrics format in {metrics_path}")
            continue

        entry_task = metrics.get("task_name")
        task_config = metrics.get("task_config", {})
        entry_primary_metric = None
        if isinstance(task_config, dict):
            entry_task = entry_task or task_config.get("task_name")
            entry_primary_metric = task_config.get("primary_metric")

        task_match = entry_task == task_name
        metric_match = (
                primary_metric_name is None
                or entry_primary_metric == primary_metric_name
        )

        metric_values = metrics.get("metrics")
        score = None
        if task_match and metric_match and isinstance(metric_values, dict):
            score = metric_values.get("primary_score")

        if score is None:
            warn_parts = [
                f"No primary_score found for baseline '{label}'",
                f"(task={task_name!r}, metric={primary_metric_name!r})",
                f"in {metrics_path}",
            ]
            print(f"[WARN] {' '.join(warn_parts)}")
            continue

        scores[label] = score

    return scores


def adjust_color_brightness(hex_color: str, brightness_factor: float) -> str:
    """Adjust the brightness of a hex color.
    
    Args:
        hex_color: Hex color string (e.g., "#988ED5")
        brightness_factor: Factor to adjust brightness (>1.0 = lighter, <1.0 = darker)
    
    Returns:
        Adjusted hex color string
    """
    rgb = mcolors.hex2color(hex_color)
    hls = colorsys.rgb_to_hls(*rgb)
    # Adjust lightness while keeping hue and saturation
    new_lightness = max(0.0, min(1.0, hls[1] * brightness_factor))
    new_rgb = colorsys.hls_to_rgb(hls[0], new_lightness, hls[2])
    return mcolors.rgb2hex(new_rgb)


def plot_single_subplot(
        ax: plt.Axes,
        df: pd.DataFrame,
        task_name: str,
        primary_metric_name: str | None,
        baseline_scores: Dict[str, float],
        run_configs: Dict[str, Dict[str, Any]],
        baseline_configs: Dict[str, Dict[str, Any]],
        group_palette: Dict[str, str],
        family_colors: Dict[str, str],
) -> None:
    """Plot a single task's data on the given axes."""
    # Plot each group using configuration
    for group in sorted(df["model_group"].unique()):
        group_data = df[df["model_group"] == group].sort_values("step")
        
        # Get configuration for this group
        if group not in run_configs:
            # Fallback configuration
            marker = "o"
            linewidth = 2
            markersize = 9
        else:
            config = run_configs[group]
            marker = config.get("marker", "o")
            linewidth = config.get("linewidth", 2)
            markersize = config.get("markersize", 9)

        ax.plot(
            group_data["step"],
            group_data["primary_score"],
            marker=marker,
            color=group_palette[group],
            label=group,
            linewidth=linewidth,
            markersize=markersize,
        )

    # Plot baseline models using configuration
    for label, score in baseline_scores.items():
        if score is None:
            continue
        
        if label not in baseline_configs:
            # Fallback configuration
            family = BASELINE_FAMILY.get(label, "unknown")
            color = family_colors.get(family, "#000000")
            linestyle = "--"
            linewidth = 1.2
        else:
            config = baseline_configs[label]
            family = config["family"]
            color = family_colors[family]
            linestyle = config.get("linestyle", "--")
            linewidth = config.get("linewidth", 1.2)
        
        ax.axhline(
            score,
            linestyle=linestyle,
            linewidth=linewidth,
            color=color,
            label=label,
        )

    # Set labels and title
    metric_label = primary_metric_name or "Primary Score"
    ax.set_title(task_name, fontsize=22)
    ax.set_xlabel("Training Step", fontsize=20)
    ax.set_ylabel(metric_label, fontsize=20)
    ax.tick_params(labelsize=18)


def plot_all_tasks_grid(
        output: Path,
        style: str,
        show: bool,
        evals_root: Path,
) -> None:
    """Create a 3x3 grid plot with all nine tasks."""
    sns.set_theme(style=style)
    
    # Create figure with 3x3 subplots - increased size for bigger plots
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    fig.suptitle("Primary Score vs. Training Step Across All Tasks, ModuleFormer", fontsize=28, y=0.98)
    
    # Set up color mapping and configuration (shared across all subplots)
    unique_families = sorted(
        set(GROUP_FAMILY.values()) | set(BASELINE_FAMILY.values())
    )
    base_palette = sns.color_palette("colorblind", n_colors=len(unique_families))
    family_colors = {
        family: FAMILY_COLORS.get(family, base_palette[idx])
        for idx, family in enumerate(unique_families)
    }
    
    run_configs = {run["label"]: run for run in MODEL_RUNS}
    baseline_configs = {run["label"]: run for run in BASELINE_RUNS}
    
    # Create color palette for each model group using configured brightness
    group_palette = {}
    all_groups = set()
    for run in MODEL_RUNS:
        all_groups.add(run["label"])
    for group in all_groups:
        if group not in run_configs:
            family = GROUP_FAMILY.get(group, "unknown")
            base_color = family_colors.get(family, "#000000")
            group_palette[group] = base_color
        else:
            config = run_configs[group]
            family = config["family"]
            base_color = family_colors[family]
            brightness = config.get("brightness", 1.0)
            group_palette[group] = adjust_color_brightness(base_color, brightness)
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()
    
    # Collect handles and labels for shared legend (from first successful plot)
    handles_list = []
    labels_list = []
    legend_collected = False
    
    # Process each task
    for idx, task_name in enumerate(TASKS):
        if idx >= 9:
            break  # Safety check
        
        ax = axes_flat[idx]
        metrics_filename = f"task-{task_name.replace(':', '_')}-metrics.json"
        steps = TASK_STEPS.get(task_name)
        
        if not steps:
            print(f"[WARN] No step configuration for {task_name}; skipping.")
            ax.text(0.5, 0.5, f"No data for\n{task_name}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=20)
            ax.set_title(task_name, fontsize=22)
            continue
        
        try:
            df, primary_metric_name = build_dataframe(
                evals_root,
                task_name,
                metrics_filename,
                steps,
            )
        except RuntimeError as exc:
            print(f"[WARN] Skipping {task_name}: {exc}")
            ax.text(0.5, 0.5, f"Error loading\n{task_name}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=20)
            ax.set_title(task_name, fontsize=22)
            continue

        baseline_scores = load_baseline_scores(
            evals_root,
            task_name,
            primary_metric_name,
            metrics_filename,
        )
        
        # Plot on this subplot
        plot_single_subplot(
            ax,
            df,
            task_name,
            primary_metric_name,
            baseline_scores,
            run_configs,
            baseline_configs,
            group_palette,
            family_colors,
        )
        
        # Collect handles and labels from the first successful subplot for legend
        # matplotlib tracks handles even without explicit legend creation
        if not legend_collected:
            # Get all handles and labels from the axes
            handles, labels = ax.get_legend_handles_labels()
            # Remove duplicates while preserving order
            seen_labels = set()
            for handle, label in zip(handles, labels):
                if label not in seen_labels:
                    handles_list.append(handle)
                    labels_list.append(label)
                    seen_labels.add(label)
            legend_collected = True
    
    # Create shared legend at the bottom of the figure
    if handles_list and labels_list:
        fig.legend(
            handles_list,
            labels_list,
            title="Model Group",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=4,
            fontsize=18,
            title_fontsize=20,
            frameon=True,
        )
    else:
        # Fallback: create legend from first subplot if handles weren't collected
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles and labels:
            fig.legend(
                handles,
                labels,
                title="Model Group",
                loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=4,
                fontsize=18,
                title_fontsize=20,
                frameon=True,
            )
    
    # Adjust spacing: reduce wspace (column spacing), keep hspace reasonable
    # Increased top and bottom margins to accommodate title and legend
    plt.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.92,  # Reduced from 0.96 to give more space for title
        bottom=0.10,  # Increased from 0.08 to give more space for legend
        wspace=0.25,  # Reduced column spacing
        hspace=0.35,  # Row spacing
    )
    
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches='tight', dpi=300)
    print(f"[INFO] Saved plot to {output}")

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        args.output_dir
        / f"{MAIN_MODEL}_cs294final"
        / "all_tasks_primary_score_comparison.png"
    )

    plot_all_tasks_grid(
        output_file,
        args.style,
        args.show,
        args.evals_root,
    )


if __name__ == "__main__":
    main()

