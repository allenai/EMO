#!/usr/bin/env python3
"""Plot primary_score over training steps for selected evaluation runs."""

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
TASK_STEPS: Dict[str, List[int]] = {
    "arc_challenge:rc_test": [0, 41, 82, 123, 164, 207],
    "arc_easy:rc_test": [0, 84, 168, 252, 336, 420],
    "boolq:rc_test": [0, 315, 630, 945, 1260, 1578],
    "csqa:rc_test": [0, 327, 654, 981, 1308, 1638],
    "hellaswag:rc_test": [0, 1458, 2916, 4374, 5832, 7293],
    "openbookqa:rc_test": [0, 185, 370, 555, 740, 927],
    "piqa:rc_test": [0, 566, 1132, 1698, 2264, 2832],
    "socialiqa:rc_test": [0, 1215, 2430, 3645, 4860, 6075],
    "winogrande:rc_test": [0, 1477, 2954, 4431, 5908, 7386],
}

# Family color mapping - define base colors for each model family
# Colors are specified as hex strings (e.g., "#988ED5")
FAMILY_COLORS: Dict[str, str] = {
    "dense": "#E24A33",        # Red/Orange
    "moe": "#348ABD",          # Blue
    "twolevelbatchlb": "#988ED5",  # Purple
    # Add more families as needed
}

# Model run configurations
# Each entry defines a model run that will be plotted as a line
# Template must contain {step} and {task_core} placeholders
MODEL_RUNS: List[Dict[str, Any]] = [
    {
        "label": "moe keepk32",
        "template": (
            "moe_1b14b_128experts_olmoe-mix_130B_1117_step30995_"
            "task-{task_core}_rc_validation_keepk32_finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "moe",
        "marker": "o",              # matplotlib marker style (o=circle, s=square, ^=triangle, etc.)
        "brightness": 1.0,          # Brightness multiplier (1.0 = no change, >1.0 = lighter, <1.0 = darker)
        "linewidth": 2,             # Line width
        "markersize": 9,            # Marker size
    },
    {
        "label": "twolevelbatchlb keepk32",
        "template": (
            f"{MAIN_MODEL}_"
            "task-{task_core}_rc_validation_keepk32_finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "twolevelbatchlb",
        "marker": "o",
        "brightness": 1.0,
        "linewidth": 2,
        "markersize": 9,
    },
    {
        "label": "lr-3e-5_warmup-0.2",
        "template": (
            f"{MAIN_MODEL}_"
            "task-{task_core}_rc_validation_keepk32_lr-3e-5_warmup-0.2_finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "twolevelbatchlb",
        "marker": "s",              # Square marker for ablation
        "brightness": 0.8,          # Slightly darker
        "linewidth": 2,
        "markersize": 9,
    },
    {
        "label": "lr-7e-5_warmup-0.2",
        "template": (
            f"{MAIN_MODEL}_"
            "task-{task_core}_rc_validation_keepk32_lr-7e-5_warmup-0.2_finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "twolevelbatchlb",
        "marker": "s",              # Square marker for ablation
        "brightness": 0.6,          # Slightly darker
        "linewidth": 2,
        "markersize": 9,
    },
    {
        "label": "lr-9e-5_warmup-0.2",
        "template": (
            f"{MAIN_MODEL}_"
            "task-{task_core}_rc_validation_keepk32_lr-9e-5_warmup-0.2_finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "twolevelbatchlb",
        "marker": "s",              # Square marker for ablation
        "brightness": 0.4,          # Slightly darker
        "linewidth": 2,
        "markersize": 9,
    },
    {
        "label": "lr-7e-5_warmup-0.1",
        "template": (
            f"{MAIN_MODEL}_"
            "task-{task_core}_rc_validation_keepk32_lr-7e-5_warmup-0.1_finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "twolevelbatchlb",
        "marker": "s",              # Square marker for ablation
        "brightness": 0.2,          # Slightly darker
        "linewidth": 2,
        "markersize": 9,
    },
    {
        "label": "dense finetuned",
        "template": (
            "dense_1b_olmoe-mix_1119_step30995_"
            "finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "dense",
        "marker": "v",              # Triangle down for dense
        "brightness": 1.0,
        "linewidth": 2,
        "markersize": 9,
    },
    {
        "label": "dense finetuned_newoptim",
        "template": (
            "dense_1b_olmoe-mix_1119_step30995_noloadoptim_"
            "finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "dense",
        "marker": "v",              # Triangle down for dense
        "brightness": 0.6,
        "linewidth": 2,
        "markersize": 9,
    },
]

# Baseline model configurations
# These are plotted as horizontal dashed lines
BASELINE_RUNS: List[Dict[str, Any]] = [
    {
        "label": "moe full",
        "template": "moe_1b14b_128experts_olmoe-mix_130B_1117_step30995-hf",
        "family": "moe",
        "linestyle": "--",          # Line style for baseline
        "linewidth": 1.2,
    },
    {
        "label": "twolevelbatchlb full",
        "template": f"{MAIN_MODEL}-hf",
        "family": "twolevelbatchlb",
        "linestyle": "--",
        "linewidth": 1.2,
    },
    {
        "label": "dense full",
        "template": "dense_1b_olmoe-mix_1119_step30995-hf",
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
            "Generate a seaborn line plot comparing primary_score across "
            "training steps for two model families."
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


def plot_primary_scores(
        df: pd.DataFrame,
        output: Path,
        style: str,
        show: bool,
        task_name: str | None,
        primary_metric_name: str | None,
        baseline_scores: Dict[str, float],
) -> None:
    sns.set_theme(style=style)
    plt.figure(figsize=(8, 5))

    # Get unique families and set up color mapping
    unique_families = sorted(
        set(GROUP_FAMILY.values()) | set(BASELINE_FAMILY.values())
    )
    # Fallback to colorblind palette for any families not in custom mapping
    base_palette = sns.color_palette("colorblind", n_colors=len(unique_families))
    family_colors = {
        family: FAMILY_COLORS.get(family, base_palette[idx])
        for idx, family in enumerate(unique_families)
    }
    
    # Build configuration lookup dictionaries for quick access
    run_configs = {run["label"]: run for run in MODEL_RUNS}
    baseline_configs = {run["label"]: run for run in BASELINE_RUNS}
    
    # Create color palette for each model group using configured brightness
    group_palette = {}
    for group in df["model_group"].unique():
        if group not in run_configs:
            # Fallback for groups not in configuration
            family = GROUP_FAMILY.get(group, "unknown")
            base_color = family_colors.get(family, "#000000")
            group_palette[group] = base_color
        else:
            config = run_configs[group]
            family = config["family"]
            base_color = family_colors[family]
            brightness = config.get("brightness", 1.0)
            group_palette[group] = adjust_color_brightness(base_color, brightness)

    ax = plt.gca()

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

    metric_label = primary_metric_name or "Primary Score"
    if task_name:
        ax.set_title(task_name)
    else:
        ax.set_title("Primary Score vs. Training Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(metric_label)

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
        
        line = ax.axhline(
            score,
            linestyle=linestyle,
            linewidth=linewidth,
            color=color,
        )
        line.set_label(label)

    ax.legend(title="Model Group", loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    print(f"[INFO] Saved plot to {output}")

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for task_name in TASKS:
        metrics_filename = f"task-{task_name.replace(':', '_')}-metrics.json"
        steps = TASK_STEPS.get(task_name)
        if not steps:
            print(f"[WARN] No step configuration for {task_name}; skipping.")
            continue
        try:
            df, primary_metric_name = build_dataframe(
                args.evals_root,
                task_name,
                metrics_filename,
                steps,
            )
        except RuntimeError as exc:
            print(f"[WARN] Skipping {task_name}: {exc}")
            continue

        baseline_scores = load_baseline_scores(
            args.evals_root,
            task_name,
            primary_metric_name,
            metrics_filename,
        )

        output_file = (
                args.output_dir
                / f"{MAIN_MODEL}_full"
                / f"{task_name.replace(':', '_')}_primary_score_comparison.png"
        )

        plot_primary_scores(
            df,
            output_file,
            args.style,
            args.show,
            task_name,
            primary_metric_name,
            baseline_scores,
        )


if __name__ == "__main__":
    main()

