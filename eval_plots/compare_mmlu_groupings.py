#!/usr/bin/env python3
"""Compare MMLU human-17 vs cluster-16 groupings using per-subject eval results.

Reads from a local prune_evals directory (same structure as get_table_scores_prune_evals_0319.py).
Discovers mmlu_* and mmlu_cluster_* task directories, reads per-subject metrics
from per_subject/ subdirectories, and computes:
  - avg: mean of per-category scores (where each category score = mean of its subjects)
  - avg_micro: mean of all per-subject scores directly

Also reports variants excluding subjects from mmlu_other.

Usage:
    python eval_plots/compare_mmlu_groupings.py
    python eval_plots/compare_mmlu_groupings.py --prune-evals-root /path/to/prune_evals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301step30995-hf"
VARIANT_SUFFIX = "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise"
METRIC = "acc_per_char"

# Subjects to exclude for _no_other variants
OTHER_SUBJECTS = {"global_facts", "miscellaneous", "professional_accounting"}


# ============================================================================
# Helpers (reused from get_table_scores_prune_evals_0319.py)
# ============================================================================


def read_metrics(metrics_path: Path) -> Optional[dict]:
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def find_largest_checkpoint(results_dir: Path) -> Optional[Path]:
    best_step = -1
    best_dir = None
    for ckpt_dir in results_dir.glob("checkpoint-*"):
        if not ckpt_dir.is_dir():
            continue
        try:
            step = int(ckpt_dir.name.replace("checkpoint-", ""))
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best_dir = ckpt_dir
    return best_dir


# ============================================================================
# Discovery and collection
# ============================================================================


def discover_mmlu_tasks(model_dir: Path, variant_suffix: str) -> Tuple[Dict, Dict]:
    """Discover mmlu category and cluster task directories.

    Returns:
        human_tasks: {category_name: task_dir_path} for mmlu_* (non-cluster) tasks
        cluster_tasks: {category_name: task_dir_path} for mmlu_cluster_* tasks
    """
    human_tasks = {}
    cluster_tasks = {}

    for task_dir in sorted(model_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        name = task_dir.name
        if not name.startswith("mmlu_"):
            continue
        if not name.endswith(variant_suffix):
            continue

        # Strip variant suffix to get task name
        task_name = name[: -len(variant_suffix)]

        # Skip individual subject tasks (they don't have per_subject/ dirs)
        # We only want category-level tasks
        results_dir = task_dir / "results"
        if not results_dir.is_dir():
            continue
        ckpt = find_largest_checkpoint(results_dir)
        if ckpt is None:
            continue
        per_subj_dir = ckpt / "per_subject"
        if not per_subj_dir.is_dir():
            continue

        if task_name.startswith("mmlu_cluster_"):
            cluster_tasks[task_name] = task_dir
        else:
            human_tasks[task_name] = task_dir

    return human_tasks, cluster_tasks


def collect_per_subject_scores(
    task_dirs: Dict[str, Path],
    metric_key: str,
) -> Tuple[Dict[str, Dict[str, Tuple[float, int]]], Dict[str, float]]:
    """Read per-subject metrics from per_subject/ directories.

    Returns:
        per_cat: {task_name: {subject: (score, n)}}
        all_subjects: {subject: (score, n)} flattened across all categories
    """
    per_cat = {}
    all_subjects = {}

    for task_name, task_dir in sorted(task_dirs.items()):
        results_dir = task_dir / "results"
        ckpt = find_largest_checkpoint(results_dir)
        if ckpt is None:
            continue

        per_subj_dir = ckpt / "per_subject"
        if not per_subj_dir.is_dir():
            continue

        cat_scores = {}
        for subj_dir in sorted(per_subj_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            subject = subj_dir.name
            metrics_files = sorted(subj_dir.glob("task-*-metrics.json"))
            if not metrics_files:
                continue
            data = read_metrics(metrics_files[0])
            if data is None:
                continue
            val = data.get("metrics", {}).get(metric_key)
            n = data.get("num_instances")
            if val is not None and n is not None:
                cat_scores[subject] = (float(val), int(n))
                all_subjects[subject] = (float(val), int(n))

        per_cat[task_name] = cat_scores

    return per_cat, all_subjects


def compute_averages(
    per_cat: Dict[str, Dict[str, Tuple[float, int]]],
    all_subjects: Dict[str, Tuple[float, int]],
    exclude_subjects: set = frozenset(),
) -> Tuple[Optional[float], Optional[float]]:
    """Compute avg and avg_micro.

    avg: mean of per-category means (each category = mean of its subject scores)
    avg_micro: mean of all subject scores directly
    """
    # avg: mean of category means
    cat_means = []
    for task_name, cat_scores in per_cat.items():
        scores = [v[0] for s, v in cat_scores.items() if s not in exclude_subjects]
        if scores:
            cat_means.append(sum(scores) / len(scores))
    avg = sum(cat_means) / len(cat_means) if cat_means else None

    # avg_micro: mean of all subject scores
    subj_scores = [v[0] for s, v in all_subjects.items() if s not in exclude_subjects]
    avg_micro = sum(subj_scores) / len(subj_scores) if subj_scores else None

    return avg, avg_micro


# ============================================================================
# Main
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Compare MMLU human-17 vs cluster-16 groupings")
    parser.add_argument(
        "--prune-evals-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "prune_evals_0313",
    )
    parser.add_argument("--metric", default=METRIC)
    return parser.parse_args()


def main():
    args = parse_args()
    metric_key = args.metric
    root = args.prune_evals_root
    model_dir = root / MODEL

    print(f"Model: {MODEL}")
    print(f"Variant: {VARIANT_SUFFIX}")
    print(f"Metric: {metric_key}")
    print(f"Root: {root}")
    print()

    if not model_dir.is_dir():
        print(f"[ERROR] Model directory not found: {model_dir}")
        return

    # Discover task directories
    human_tasks, cluster_tasks = discover_mmlu_tasks(model_dir, VARIANT_SUFFIX)
    print(
        f"Found {len(human_tasks)} human category tasks, {len(cluster_tasks)} cluster category tasks\n"
    )

    # Collect per-subject scores
    h_per_cat, h_all = collect_per_subject_scores(human_tasks, metric_key)
    c_per_cat, c_all = collect_per_subject_scores(cluster_tasks, metric_key)

    print(f"Human-17:   {len(h_all)} subjects across {len(h_per_cat)} categories")
    print(f"Cluster-16: {len(c_all)} subjects across {len(c_per_cat)} categories\n")

    # Compute averages
    h_avg, h_micro = compute_averages(h_per_cat, h_all)
    c_avg, c_micro = compute_averages(c_per_cat, c_all)

    h_avg_no, h_micro_no = compute_averages(h_per_cat, h_all, OTHER_SUBJECTS)
    c_avg_no, c_micro_no = compute_averages(c_per_cat, c_all, OTHER_SUBJECTS)

    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    def fmt_diff(a, b):
        return f"{b - a:+.4f}" if (a is not None and b is not None) else "N/A"

    print("=" * 70)
    print(f"{'':>30} {'Human-17':>12} {'Cluster-16':>12} {'Diff':>12}")
    print("-" * 70)
    print(
        f"{'avg (mean of cat means)':>30} {fmt(h_avg):>12} {fmt(c_avg):>12} {fmt_diff(h_avg, c_avg):>12}"
    )
    print(
        f"{'avg_micro (mean of subjects)':>30} {fmt(h_micro):>12} {fmt(c_micro):>12} {fmt_diff(h_micro, c_micro):>12}"
    )
    print(
        f"{'avg_no_other':>30} {fmt(h_avg_no):>12} {fmt(c_avg_no):>12} {fmt_diff(h_avg_no, c_avg_no):>12}"
    )
    print(
        f"{'avg_micro_no_other':>30} {fmt(h_micro_no):>12} {fmt(c_micro_no):>12} {fmt_diff(h_micro_no, c_micro_no):>12}"
    )
    print("=" * 70)

    # Per-subject detail
    all_subjs = sorted(set(h_all.keys()) | set(c_all.keys()))
    print(f"\n{'Subject':<42} {'H Score':>8} {'C Score':>8} {'Diff':>8}")
    print("-" * 70)
    for subj in all_subjs:
        h = h_all.get(subj)
        c = c_all.get(subj)
        h_val = f"{h[0]:.4f}" if h else "N/A"
        c_val = f"{c[0]:.4f}" if c else "N/A"
        diff = f"{c[0] - h[0]:+.4f}" if (h and c) else "N/A"
        print(f"{subj:<42} {h_val:>8} {c_val:>8} {diff:>8}")


if __name__ == "__main__":
    main()
