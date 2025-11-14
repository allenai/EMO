#!/usr/bin/env python3
"""Plot primary_score over training steps for selected evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configuration -----------------------------------------------------------------

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

# Mapping of legend label -> directory template. The template must contain a
# `{step}` placeholder that will be formatted with each step in `STEPS`.
MODEL_GROUPS: Dict[str, str] = {
    "moe keepk32": (
        "moe_1b7b_128experts_olmoe-mix_130B_1103_step30995_"
        "task-{task_core}_rc_train_0shot_finetune_random-keepk32_step{step}-hf"
    ),
    "twolevel keepk32": (
        "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110_step30995_"
        "task-{task_core}_rc_train_0shot_finetune_random-keepk32_step{step}-hf"
    ),
}

BASELINE_MODELS: Dict[str, str] = {
    "moe full": (
        "moe_1b7b_128experts_olmoe-mix_130B_1103_step30995-hf"
    ),
    "twolevel full": (
        "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110_step30995-hf"
    ),
}

GROUP_FAMILY: Dict[str, str] = {
    "moe keepk32": "moe",
    "twolevel keepk32": "twolevel",
}

BASELINE_FAMILY: Dict[str, str] = {
    "moe full": "moe",
    "twolevel full": "twolevel",
}

GROUP_FAMILY: Dict[str, str] = {
    "moe keepk32": "moe",
    "twolevel keepk32": "twolevel",
}

BASELINE_FAMILY: Dict[str, str] = {
    "moe full": "moe",
    "twolevel full": "twolevel",
}


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

    unique_families = sorted(
        set(GROUP_FAMILY.values()) | set(BASELINE_FAMILY.values())
    )
    base_palette = sns.color_palette("colorblind", n_colors=len(unique_families))
    family_colors = {family: base_palette[idx] for idx, family in enumerate(unique_families)}
    group_palette = {group: family_colors[GROUP_FAMILY[group]] for group in df["model_group"].unique()}

    ax = sns.lineplot(
        data=df,
        x="step",
        y="primary_score",
        hue="model_group",
        marker="o",
        palette=group_palette,
    )

    metric_label = primary_metric_name or "Primary Score"
    if task_name:
        ax.set_title(task_name)
    else:
        ax.set_title("Primary Score vs. Training Step, Random Experts")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(metric_label)

    for idx, (label, score) in enumerate(baseline_scores.items()):
        if score is None:
            continue
        family = BASELINE_FAMILY.get(label)
        color = family_colors.get(family, f"C{idx + len(GROUP_FAMILY)}")
        line = ax.axhline(
            score,
            linestyle="--",
            linewidth=1.2,
            color=color,
        )
        line.set_label(label)

    ax.legend(title="Model Group", loc="best")

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
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
            / f"{task_name.replace(':', '_')}_primary_score_random_expert_comparison.png"
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

