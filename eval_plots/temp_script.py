#!/usr/bin/env python3
"""Calculate average difference between twolevel keepk32 and moe keepk32 at step 0 and last step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

# Configuration from the main plotting script
MAIN_MODEL = "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121_step30995"

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

# Model templates
MOE_TEMPLATE = (
    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123_step30995_"
    "task-{task_core}_rc_validation_keepk32_newdefault_lr-4e-5_finetune-task-{task_core}_rc_train_step{step}-hf"
)

TWOLEVEL_TEMPLATE = (
    f"{MAIN_MODEL}_"
    "task-{task_core}_rc_validation_keepk32_newdefault_lr-4e-5_finetune-task-{task_core}_rc_train_step{step}-hf"
)


def task_core_from_name(task_name: str) -> str:
    return task_name.split(":", 1)[0]


def get_primary_score(
    evals_root: Path,
    model_template: str,
    step: int,
    task_name: str,
    task_core: str,
    metrics_filename: str,
) -> float | None:
    """Get primary_score for a specific model, task, and step."""
    try:
        formatted_path = model_template.format(step=step, task_core=task_core)
    except KeyError as exc:
        print(f"[ERROR] Template {model_template!r} missing placeholder {exc}.")
        return None

    model_dir = evals_root / formatted_path
    metrics_path = model_dir / metrics_filename

    if not metrics_path.exists():
        print(f"[WARN] Missing metrics for {model_dir}")
        return None

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[WARN] Failed to parse JSON at {metrics_path}: {exc}")
        return None

    if not isinstance(metrics, dict):
        print(f"[WARN] Unexpected metrics format in {metrics_path}")
        return None

    metric_values = metrics.get("metrics")
    if isinstance(metric_values, dict):
        score = metric_values.get("primary_score")
        if score is not None:
            return float(score)

    print(f"[WARN] No 'primary_score' found in metrics for {metrics_path}")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate average difference between twolevel keepk32 and moe keepk32 "
            "at step 0 and last step across all tasks."
        )
    )
    parser.add_argument(
        "--evals-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "evals",
        help="Path to the evals directory (defaults to repo_root/evals).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    step0_differences = []
    last_step_differences = []

    print("=" * 80)
    print("Calculating differences: twolevel keepk32 - moe keepk32")
    print("=" * 80)

    for task_name in TASKS:
        task_core = task_core_from_name(task_name)
        metrics_filename = f"task-{task_name.replace(':', '_')}-metrics.json"
        steps = TASK_STEPS.get(task_name)

        if not steps:
            print(f"[WARN] No step configuration for {task_name}; skipping.")
            continue

        step0 = steps[0]
        last_step = steps[-1]

        print(f"\nTask: {task_name}")
        print(f"  Steps: {step0} (start) and {last_step} (end)")

        # Get moe scores
        moe_step0 = get_primary_score(
            args.evals_root, MOE_TEMPLATE, step0, task_name, task_core, metrics_filename
        )
        moe_last = get_primary_score(
            args.evals_root, MOE_TEMPLATE, last_step, task_name, task_core, metrics_filename
        )

        # Get twolevel scores
        twolevel_step0 = get_primary_score(
            args.evals_root, TWOLEVEL_TEMPLATE, step0, task_name, task_core, metrics_filename
        )
        twolevel_last = get_primary_score(
            args.evals_root, TWOLEVEL_TEMPLATE, last_step, task_name, task_core, metrics_filename
        )

        # Calculate differences
        if moe_step0 is not None and twolevel_step0 is not None:
            diff_step0 = twolevel_step0 - moe_step0
            step0_differences.append(diff_step0)
            print(f"  Step 0: twolevel={twolevel_step0:.4f}, moe={moe_step0:.4f}, diff={diff_step0:.4f}")
        else:
            print(f"  Step 0: Missing data (twolevel={twolevel_step0}, moe={moe_step0})")

        if moe_last is not None and twolevel_last is not None:
            diff_last = twolevel_last - moe_last
            last_step_differences.append(diff_last)
            print(f"  Last step: twolevel={twolevel_last:.4f}, moe={moe_last:.4f}, diff={diff_last:.4f}")
        else:
            print(f"  Last step: Missing data (twolevel={twolevel_last}, moe={moe_last})")

    # Calculate averages
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if step0_differences:
        avg_step0 = sum(step0_differences) / len(step0_differences)
        print(f"\nAverage difference at Step 0 (start of finetuning):")
        print(f"  {avg_step0:.6f}")
        print(f"  (Based on {len(step0_differences)} tasks)")
    else:
        print("\n[ERROR] No valid step 0 differences found!")

    if last_step_differences:
        avg_last = sum(last_step_differences) / len(last_step_differences)
        print(f"\nAverage difference at Last Step (end of finetuning):")
        print(f"  {avg_last:.6f}")
        print(f"  (Based on {len(last_step_differences)} tasks)")
    else:
        print("\n[ERROR] No valid last step differences found!")

    print("\n" + "=" * 80)
    print("Interpretation:")
    print("  Positive values mean twolevel keepk32 is better than moe keepk32")
    print("  Negative values mean moe keepk32 is better than twolevel keepk32")
    print("=" * 80)


if __name__ == "__main__":
    main()

