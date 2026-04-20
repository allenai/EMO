#!/usr/bin/env python3
"""Plot evaluation metrics over training steps for selected runs."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============================================================================
# CONFIGURATION - Full control over model runs and visual settings
# ============================================================================

# Task configuration now keeps task names, steps, and optional metric key together.
# Only override metric_key when you want something other than "primary_score".
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    # "arc_easy:rc_test": {
    #     "steps": [0, 42, 84, 126, 168, 210],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "arc_challenge:rc_test": {
    #     "steps": [0, 20, 40, 60, 80, 102],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "boolq:rc_test": {
    #     "steps": [0, 157, 314, 471, 628, 789],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "csqa:rc_test": {
    #     "steps": [0, 163, 326, 489, 652, 819],
    #     "metric_key": "bits_per_byte_corr",
    # },
    "hellaswag:rc_test": {
        "steps": [0, 729, 1458, 2187, 2916, 3645],
        "metric_key": "softloss_per_char_corr",
    },
    # "openbookqa:rc_test": {
    #     "steps": [0, 92, 184, 276, 368, 462],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "piqa:rc_test": {
    #     "steps": [0, 283, 566, 849, 1132, 1416],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "socialiqa:rc_test": {
    #     "steps": [0, 607, 1214, 1821, 2428, 3036],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "winogrande:rc_test": {
    #     "steps": [0, 738, 1476, 2214, 2952, 3693],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "gsm8k_generation:train_0shot": {
    #     "steps": [0, 121, 242, 363, 484, 606],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "synthea:rc_train_0shot": {
    #     "steps": [0, 161, 322, 483, 644, 807],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "coqa:train_0shot": {
    #     "steps": [0, 28, 56, 84, 112, 144],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "squad:train_0shot": {
    #     "steps": [0, 1623, 3246, 4869, 6492, 8118],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_abstract_algebra:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    # },
    # "mmlu_anatomy:rc_test": {
    #     "steps": [0, 6, 12, 18, 24, 30],
    # },
    # "mmlu_astronomy:rc_test": {
    #     "steps": [0, 6, 12, 18, 24, 33],
    # },
    # "mmlu_business_ethics:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    # },
    # "mmlu_clinical_knowledge:rc_test": {
    #     "steps": [0, 11, 22, 33, 44, 57],
    # },
    # "mmlu_college_biology:rc_test": {
    #     "steps": [0, 6, 12, 18, 24, 30],
    # },
    # "mmlu_college_chemistry:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    # },
    # "mmlu_college_computer_science:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    # },
    # "mmlu_college_mathematics:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    # },
    # "mmlu_college_medicine:rc_test": {
    #     "steps": [0, 7, 14, 21, 28, 36],
    # },
    # "mmlu_college_physics:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    # },
    # "mmlu_computer_security:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    # },
    # "mmlu_conceptual_physics:rc_test": {
    #     "steps": [0, 10, 20, 30, 40, 51],
    # },
    # "mmlu_econometrics:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 20, 24],
    # },
    # "mmlu_electrical_engineering:rc_test": {
    #     "steps": [0, 6, 12, 18, 24, 30],
    # },
    # "mmlu_elementary_mathematics:rc_test": {
    #     "steps": [0, 16, 32, 48, 64, 84],
    # },
    # "mmlu_formal_logic:rc_test": {
    #     "steps": [0, 5, 10, 15, 20, 27],
    # },
    # "mmlu_global_facts:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    # },
    # "mmlu_high_school_biology:rc_test": {
    #     "steps": [0, 13, 26, 39, 52, 69],
    # },
    # "mmlu_high_school_chemistry:rc_test": {
    #     "steps": [0, 9, 18, 27, 36, 45],
    # },
    # "mmlu_high_school_computer_science:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    # },
    # "mmlu_high_school_european_history:rc_test": {
    #     "steps": [0, 7, 14, 21, 28, 36],
    # },
    # "mmlu_high_school_geography:rc_test": {
    #     "steps": [0, 8, 16, 24, 32, 42],
    # },
    # "mmlu_high_school_government_and_politics:rc_test": {
    #     "steps": [0],
    # },
    # "mmlu_high_school_macroeconomics:rc_test": {
    #     "steps": [0, 17, 34, 51, 68, 87],
    # },
    # "mmlu_high_school_mathematics:rc_test": {
    #     "steps": [0, 12, 24, 36, 48, 60],
    # },
    # "mmlu_high_school_microeconomics:rc_test": {
    #     "steps": [0, 10, 20, 30, 40, 51],
    # },
    # "mmlu_high_school_physics:rc_test": {
    #     "steps": [0, 6, 12, 18, 24, 33],
    # },
    # "mmlu_high_school_psychology:rc_test": {
    #     "steps": [0, 24, 48, 72, 96, 120],
    # },
    # "mmlu_high_school_statistics:rc_test": {
    #     "steps": [0, 9, 18, 27, 36, 48],
    # },
    # "mmlu_high_school_us_history:rc_test": {
    #     "steps": [0, 9, 18, 27, 36, 45],
    # },
    # "mmlu_high_school_world_history:rc_test": {
    #     "steps": [0, 10, 20, 30, 40, 51],
    # },
    # "mmlu_human_aging:rc_test": {
    #     "steps": [0, 9, 18, 27, 36, 48],
    # },
    # "mmlu_human_sexuality:rc_test": {
    #     "steps": [0, 5, 10, 15, 20, 27],
    # },
    # "mmlu_international_law:rc_test": {
    #     "steps": [0, 5, 10, 15, 20, 27],
    # },
    # "mmlu_jurisprudence:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 20, 24],
    # },
    # "mmlu_logical_fallacies:rc_test": {
    #     "steps": [0, 7, 14, 21, 28, 36],
    # },
    # "mmlu_machine_learning:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 20, 24],
    # },
    # "mmlu_management:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    # },
    # "mmlu_marketing:rc_test": {
    #     "steps": [0, 10, 20, 30, 40, 51],
    # },
    # "mmlu_medical_genetics:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    # },
    # "mmlu_miscellaneous:rc_test": {
    #     "steps": [0, 34, 68, 102, 136, 174],
    # },
    # "mmlu_moral_disputes:rc_test": {
    #     "steps": [0, 15, 30, 45, 60, 75],
    # },
    # "mmlu_moral_scenarios:rc_test": {
    #     "steps": [0, 40, 80, 120, 160, 201],
    # },
    # "mmlu_nutrition:rc_test": {
    #     "steps": [0, 13, 26, 39, 52, 66],
    # },
    # "mmlu_philosophy:rc_test": {
    #     "steps": [0, 13, 26, 39, 52, 69],
    # },
    # "mmlu_prehistory:rc_test": {
    #     "steps": [0, 14, 28, 42, 56, 72],
    # },
    # "mmlu_professional_accounting:rc_test": {
    #     "steps": [0, 12, 24, 36, 48, 63],
    # },
    # "mmlu_professional_law:rc_test": {
    #     "steps": [0, 69, 138, 207, 276, 345],
    # },
    # "mmlu_professional_medicine:rc_test": {
    #     "steps": [0, 12, 24, 36, 48, 60],
    # },
    # "mmlu_professional_psychology:rc_test": {
    #     "steps": [0, 27, 54, 81, 108, 135],
    # },
    # "mmlu_public_relations:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 20, 24],
    # },
    # "mmlu_security_studies:rc_test": {
    #     "steps": [0, 10, 20, 30, 40, 54],
    # },
    # "mmlu_sociology:rc_test": {
    #     "steps": [0, 9, 18, 27, 36, 45],
    # },
    # "mmlu_us_foreign_policy:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    # },
    # "mmlu_virology:rc_test": {
    #     "steps": [0, 7, 14, 21, 28, 36],
    # },
    # "mmlu_world_religions:rc_test": {
    #     "steps": [0, 7, 14, 21, 28, 36],
    # },
    # "mmlu_biology:rc_test": {
    #     "steps": [0, 10, 20, 30, 40, 51],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_business:rc_test": {
    #     "steps": [0, 9, 18, 27, 36, 48],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_chemistry:rc_test": {
    #     "steps": [0, 6, 12, 18, 24, 33],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_computer_science:rc_test": {
    #     "steps": [0, 9, 18, 27, 36, 45],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_culture:rc_test": {
    #     "steps": [0, 7, 14, 21, 28, 36],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_economics:rc_test": {
    #     "steps": [0, 16, 32, 48, 64, 81],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_engineering:rc_test": {
    #     "steps": [0, 3, 6, 9, 12, 15],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_geography:rc_test": {
    #     "steps": [0, 4, 8, 12, 16, 21],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_health:rc_test": {
    #     "steps": [0, 36, 72, 108, 144, 183],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_history:rc_test": {
    #     "steps": [0, 20, 40, 60, 80, 102],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_law:rc_test": {
    #     "steps": [0, 39, 78, 117, 156, 198],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_math:rc_test": {
    #     "steps": [0, 23, 46, 69, 92, 117],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_other:rc_test": {
    #     "steps": [0, 25, 50, 75, 100, 129],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_philosophy_cat:rc_test": {
    #     "steps": [0, 45, 90, 135, 180, 225],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_physics:rc_test": {
    #     "steps": [0, 14, 28, 42, 56, 72],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_politics:rc_test": {
    #     "steps": [0, 14, 28, 42, 56, 72],
    #     "metric_key": "bits_per_byte_corr",
    # },
    # "mmlu_psychology:rc_test": {
    #     "steps": [0, 25, 50, 75, 100, 129],
    #     "metric_key": "bits_per_byte_corr",
    # },
}


# Family color mapping - define base colors for each model family
# Colors are specified as hex strings (e.g., "#988ED5")
FAMILY_COLORS: Dict[str, str] = {
    "dense": "#E24A33",  # Red/Orange
    "moe": "#348ABD",  # Blue
    "moe 1b4b": "#00BFA5",  # Teal for clearer distinction from moe blue
    "twolevelbatchlb train32/128": "#988ED5",  # Purple
    "twolevelbatchlb train32/128 lr high": "#FFB5B8",  # Light Red/Pink
    "twolevelbatchlb train8/64": "#8EBA42",  # Green
    "twolevelsamplingnolb": "#777777",  # Gray
    "mutualinfo": "#FBC15E",  # Yellow
}

DEFAULT_OUTPUT_SUBDIR = "primary_score_plots"

# Model run configurations
# Each entry defines a model run that will be plotted as a line
# Template must contain {step} and {task_core} placeholders
MODEL_RUNS: List[Dict[str, Any]] = [
    {
        "label": "moe keepk32",
        "template": (
            "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123_step30995_"
            "task-{task_core}{validation_suffix}_keepk32_newdefault_{lr_suffix}_finetune-task-{task_core}{task_suffix}_step{step}-hf"
        ),
        "family": "moe",
        "marker": "o",  # matplotlib marker style (o=circle, s=square, ^=triangle, etc.)
        "brightness": 1.0,  # Brightness multiplier (1.0 = no change, >1.0 = lighter, <1.0 = darker)
        "linewidth": 2,  # Line width
        "markersize": 9,  # Marker size
    },
    {
        "label": "twolevelbatchlb train32/128 keepk32",
        "template": (
            "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121_step30995_"
            "task-{task_core}{validation_suffix}_keepk32_newdefault_{lr_suffix}_finetune-task-{task_core}{task_suffix}_step{step}-hf"
        ),
        "family": "twolevelbatchlb train32/128",
        "marker": "o",
        "brightness": 1.0,
        "linewidth": 2,
        "markersize": 9,
    },
    {
        "label": "twolevelbatchlb train32/320 keepk32",
        "template": (
            "twolevelbatchlb-32_1b35b_320experts_lb-1e-1_1216_step30995_"
            "task-{task_core}{validation_suffix}_keepk32_newdefault_{lr_suffix}_finetune-task-{task_core}{task_suffix}_step{step}-hf"
        ),
        "family": "twolevelbatchlb train32/128",
        "marker": "o",
        "brightness": 0.5,
        "linewidth": 2,
        "markersize": 9,
    },
    {
        "label": "mutualinfo keepk32",
        "template": (
            "mutualinfo_1b14b_cond-1e-2_uncond-1e-2_1205_step30995_"
            "task-{task_core}_rc_validation_keepk32_newdefault_lr-4e-5_finetune-task-{task_core}_rc_train_step{step}-hf"
        ),
        "family": "mutualinfo",
        "marker": "o",
        "brightness": 1.0,
        "linewidth": 2,
        "markersize": 9,
    },
    {
        "label": "dense finetuned",
        "template": (
            "dense_1b_olmoe-mix_prenorm_noqknorm_1123_step30995_newdefault_{lr_suffix}_"
            "finetune-task-{task_core}{task_suffix}_step{step}-hf"
        ),
        "family": "dense",
        "marker": "v",  # Triangle down for dense
        "brightness": 1.0,
        "linewidth": 2,
        "markersize": 9,
    },
    {
        "label": "moe 1b4b",
        "template": (
            "moe_1b4b_32experts_1224_step30995_newdefault_{lr_suffix}_"
            "finetune-task-{task_core}{task_suffix}_step{step}-hf"
        ),
        "family": "moe 1b4b",
        "marker": "v",  # Triangle down for dense
        "brightness": 0.3,
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
        "linestyle": "--",  # Line style for baseline
        "linewidth": 1.2,
    },
    {
        "label": "moe 1b4b",
        "template": "moe_1b4b_32experts_1224_step30995-hf",
        "family": "moe 1b4b",
        "linestyle": "--",  # Line style for baseline
        "linewidth": 1.2,
    },
    {
        "label": "twolevelbatchlb full",
        "template": "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121_step30995-hf",
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
            "Generate seaborn line plots comparing evaluation metrics across "
            "training steps for selected model groups."
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
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Name of the subdirectory inside output-dir for generated plots.",
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
    """Extract task core from task name for use in model paths."""
    # Special cases for tasks with different naming in model paths
    if task_name == "synthea:rc_train_0shot":
        return "synthea_rc"
    return task_name.split(":", 1)[0]


def get_task_suffix(task_name: str) -> str:
    """Get the suffix pattern for task in model paths (e.g., '_rc_train', '_train_0shot')."""
    if task_name == "gsm8k_generation:train_0shot":
        return "_train_0shot"
    elif task_name == "synthea:rc_train_0shot":
        return "_train_0shot"
    elif task_name == "coqa:train_0shot":
        return "_train_0shot"
    elif task_name == "squad:train_0shot":
        return "_train_0shot"
    else:
        # Default pattern for other tasks
        return "_rc_train"


def get_validation_suffix(task_name: str) -> str:
    """Get the validation suffix pattern for task in model paths (e.g., '_rc_validation', '_validation_0shot')."""
    if task_name == "gsm8k_generation:train_0shot":
        return "_validation_0shot"
    elif task_name == "synthea:rc_train_0shot":
        # Note: task_core is "synthea_rc", so we only need "_validation_0shot" (not "_rc_validation_0shot")
        return "_validation_0shot"
    elif task_name == "coqa:train_0shot":
        return "_validation_0shot"
    elif task_name == "squad:train_0shot":
        return "_validation_0shot"
    else:
        # Default pattern for other tasks
        return "_rc_validation"


def get_lr_suffix(task_name: str) -> str:
    """Get the learning rate suffix for task in model paths (e.g., 'lr-4e-5', 'lr-4e-6_bs-128')."""
    if task_name == "synthea:rc_train_0shot":
        return "lr-4e-6_bs-128"
    elif task_name == "coqa:train_0shot":
        return "lr-4e-5_bs-128"
    elif "mmlu" in task_name:
        return "lr-4e-5_bs-16"
    else:
        # Default learning rate for other tasks
        return "lr-4e-5"


def get_metrics_task_name(task_name: str) -> str:
    """Get the task name to use for metrics filename (may differ from directory task name)."""
    # For gsm8k_generation, the directory uses train_0shot but metrics file uses test_0shot
    if task_name == "gsm8k_generation:train_0shot":
        return "gsm8k_generation:test_0shot"
    # For synthea, the directory uses rc_train_0shot but metrics file uses rc_test_0shot
    elif task_name == "synthea:rc_train_0shot":
        return "synthea:rc_test_0shot"
    # For coqa, the directory uses train_0shot but metrics file uses test_0shot
    elif task_name == "coqa:train_0shot":
        return "coqa:test_0shot"
    # For squad, the directory uses train_0shot but metrics file uses test_0shot
    elif task_name == "squad:train_0shot":
        return "squad:test_0shot"
    # For other tasks, use the task name as-is
    return task_name


def collect_metric_scores(
    evals_root: Path,
    model_template: str,
    steps: Iterable[int],
    task_name: str,
    task_core: str,
    metrics_filename: str,
    metric_key: str,
) -> List[Dict[str, float]]:
    """Load metric values for a model template across the provided steps."""
    records: List[Dict[str, float]] = []
    task_suffix = get_task_suffix(task_name)
    validation_suffix = get_validation_suffix(task_name)
    lr_suffix = get_lr_suffix(task_name)

    for step in steps:
        try:
            # Replace {task_suffix}, {validation_suffix}, and {lr_suffix} placeholders if present
            template_to_format = model_template.replace("{task_suffix}", task_suffix)
            template_to_format = template_to_format.replace(
                "{validation_suffix}", validation_suffix
            )
            template_to_format = template_to_format.replace("{lr_suffix}", lr_suffix)
            formatted_path = template_to_format.format(step=step, task_core=task_core)
        except KeyError as exc:
            raise KeyError(f"Template {model_template!r} missing placeholder {exc}.") from exc

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

        detected_task = metrics.get("task_name")
        metrics_task_name = get_metrics_task_name(task_name)
        if detected_task and detected_task != metrics_task_name:
            print(
                f"[WARN] Task mismatch detected in {metrics_path}: "
                f"{detected_task} vs {metrics_task_name}."
            )

        metric_value = None
        metric_values = metrics.get("metrics")
        if isinstance(metric_values, dict):
            metric_value = metric_values.get(metric_key)

        if metric_value is None:
            print(f"[WARN] No '{metric_key}' found in metrics for {metrics_path}")
            continue

        records.append({"step": step, "metric_value": metric_value})
        print(f"[DEBUG] Found {metric_key}={metric_value} for step {step} in {model_dir}")

    return records


def build_dataframe(
    evals_root: Path,
    task_name: str,
    metrics_filename: str,
    steps: Iterable[int],
    metric_key: str,
) -> pd.DataFrame:
    """Build a tidy DataFrame with columns [model_group, step, metric_value]."""
    rows: List[Dict[str, object]] = []
    task_core = task_core_from_name(task_name)

    for group_label, template in MODEL_GROUPS.items():
        # Check if this model run is restricted to specific tasks or excluded from this task
        run_config = next((r for r in MODEL_RUNS if r["label"] == group_label), None)
        if run_config:
            if "tasks" in run_config and task_name not in run_config["tasks"]:
                continue  # Skip this model run for this task
            if "exclude_tasks" in run_config and task_name in run_config["exclude_tasks"]:
                continue  # Skip this model run for excluded tasks

        records = collect_metric_scores(
            evals_root,
            template,
            steps,
            task_name,
            task_core,
            metrics_filename,
            metric_key,
        )

        for record in records:
            record["model_group"] = group_label
            rows.append(record)

    if not rows:
        raise RuntimeError(
            "No evaluation metrics were loaded. Check that the evals directory "
            "and model templates are correct."
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["model_group", "step"])


def load_baseline_scores(
    evals_root: Path,
    task_name: str,
    metrics_filename: str,
    metric_key: str,
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
        if isinstance(task_config, dict):
            entry_task = entry_task or task_config.get("task_name")

        metrics_task_name = get_metrics_task_name(task_name)
        task_match = entry_task == metrics_task_name

        metric_values = metrics.get("metrics")
        score = None
        if task_match and isinstance(metric_values, dict):
            score = metric_values.get(metric_key)

        if score is None:
            warn_parts = [
                f"No {metric_key!r} found for baseline '{label}'",
                f"(task={metrics_task_name!r})",
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


def plot_metric_scores(
    df: pd.DataFrame,
    output: Path,
    style: str,
    show: bool,
    task_name: str | None,
    metric_name: str,
    baseline_scores: Dict[str, float],
) -> None:
    sns.set_theme(style=style)
    plt.figure(figsize=(8, 5))

    # Get unique families and set up color mapping
    unique_families = sorted(set(GROUP_FAMILY.values()) | set(BASELINE_FAMILY.values()))
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
            group_data["metric_value"],
            marker=marker,
            color=group_palette[group],
            label=group,
            linewidth=linewidth,
            markersize=markersize,
        )

    if task_name:
        ax.set_title(task_name)
    else:
        ax.set_title(f"{metric_name} vs. Training Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(metric_name)

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
    plt.savefig(output, bbox_inches="tight")
    print(f"[INFO] Saved plot to {output}")

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    args = parse_args()
    base_output_dir = (args.output_dir / args.output_subdir).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for task_name, task_config in TASK_CONFIGS.items():
        metrics_task_name = get_metrics_task_name(task_name)
        metrics_filename = f"task-{metrics_task_name.replace(':', '_')}-metrics.json"
        steps = task_config.get("steps")
        metric_key = task_config.get("metric_key", "primary_score")
        if not steps:
            print(f"[WARN] No step configuration for {task_name}; skipping.")
            continue
        try:
            df = build_dataframe(
                args.evals_root,
                task_name,
                metrics_filename,
                steps,
                metric_key,
            )
        except RuntimeError as exc:
            print(f"[WARN] Skipping {task_name}: {exc}")
            continue

        baseline_scores = load_baseline_scores(
            args.evals_root,
            task_name,
            metrics_filename,
            metric_key,
        )

        output_file = base_output_dir / f"{task_name.replace(':', '_')}_{metric_key}_comparison.png"

        plot_metric_scores(
            df,
            output_file,
            args.style,
            args.show,
            task_name,
            metric_key,
            baseline_scores,
        )


if __name__ == "__main__":
    main()
