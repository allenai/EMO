#!/usr/bin/env python3
"""Compare MMLU human-17 vs cluster-16 groupings using per-subject eval results.

Reads from a local prune_evals directory (same structure as get_table_scores_prune_evals_0319.py).
Computes two types of averages for each grouping:
  - avg: mean of per-category scores (where each category score = mean of its subjects)
  - avg_micro: mean of all 57 per-subject scores directly

Also reports variants excluding mmlu_other subjects.

Usage:
    python eval_plots/compare_mmlu_groupings.py
    python eval_plots/compare_mmlu_groupings.py --prune-evals-root /path/to/prune_evals
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.offline_evals.tasks.splits_mmlu import MMLU_CATEGORIES, MMLU_CLUSTER_CATEGORIES

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301step30995-hf"
VARIANT_SUFFIX = "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise"
METRIC = "acc_per_char"

# Subjects in the human "other" category (to exclude for _no_other variants)
OTHER_SUBJECTS = set(MMLU_CATEGORIES["other"])

# Build reverse mappings
SUBJECT_TO_HUMAN_CAT = {}
for cat, subjects in MMLU_CATEGORIES.items():
    for s in subjects:
        SUBJECT_TO_HUMAN_CAT[s] = cat

SUBJECT_TO_CLUSTER_CAT = {}
for cat, subjects in MMLU_CLUSTER_CATEGORIES.items():
    for s in subjects:
        SUBJECT_TO_CLUSTER_CAT[s] = cat

ALL_SUBJECTS = sorted(SUBJECT_TO_HUMAN_CAT.keys())


# ============================================================================
# Helpers
# ============================================================================

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


def read_subject_metric(ckpt_dir: Path, subject: str, metric_key: str) -> Optional[Tuple[float, int]]:
    """Read (metric_value, num_instances) for a subject from per_subject/ dir."""
    metrics_files = sorted((ckpt_dir / "per_subject" / subject).glob("task-*-metrics.json"))
    if not metrics_files:
        return None
    try:
        with metrics_files[0].open() as f:
            data = json.load(f)
        val = data["metrics"][metric_key]
        n = data["num_instances"]
        return (float(val), int(n))
    except (KeyError, json.JSONDecodeError, OSError):
        return None


def collect_subject_scores(
    prune_evals_root: Path,
    categories_dict: Dict[str, List[str]],
    metric_key: str,
) -> Dict[str, Tuple[float, int]]:
    """Collect per-subject scores for a grouping scheme.

    Returns: {subject: (score, num_instances)} for all subjects found.
    """
    results = {}
    for cat_name, subjects in categories_dict.items():
        task_dir = prune_evals_root / MODEL / f"mmlu_{cat_name}{VARIANT_SUFFIX}" / "results"
        if not task_dir.is_dir():
            print(f"  [WARN] Missing: {task_dir.name}")
            continue
        ckpt_dir = find_largest_checkpoint(task_dir)
        if ckpt_dir is None:
            print(f"  [WARN] No checkpoints in {task_dir}")
            continue
        for subj in subjects:
            result = read_subject_metric(ckpt_dir, subj, metric_key)
            if result is not None:
                results[subj] = result
            else:
                print(f"  [WARN] Missing per_subject/{subj} in {ckpt_dir.name} for mmlu_{cat_name}")
    return results


def compute_averages(
    scores: Dict[str, Tuple[float, int]],
    categories_dict: Dict[str, List[str]],
    exclude_subjects: set = frozenset(),
) -> Tuple[Optional[float], Optional[float]]:
    """Compute avg (mean of category means) and avg_micro (mean of all subjects).

    Returns (avg, avg_micro) or (None, None) if no data.
    """
    # avg: mean of per-category averages
    cat_avgs = []
    for cat_name, subjects in categories_dict.items():
        cat_scores = [scores[s][0] for s in subjects if s in scores and s not in exclude_subjects]
        if cat_scores:
            cat_avgs.append(sum(cat_scores) / len(cat_scores))
    avg = sum(cat_avgs) / len(cat_avgs) if cat_avgs else None

    # avg_micro: mean of all subject scores directly
    all_scores = [scores[s][0] for s in scores if s not in exclude_subjects]
    avg_micro = sum(all_scores) / len(all_scores) if all_scores else None

    return avg, avg_micro


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Compare MMLU human-17 vs cluster-16 groupings")
    parser.add_argument("--prune-evals-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "prune_evals_0313")
    parser.add_argument("--metric", default=METRIC)
    return parser.parse_args()


def main():
    args = parse_args()
    metric_key = args.metric
    root = args.prune_evals_root

    print(f"Model: {MODEL}")
    print(f"Variant: {VARIANT_SUFFIX}")
    print(f"Metric: {metric_key}")
    print(f"Root: {root}")
    print()

    # Collect per-subject scores under each grouping
    print("Collecting human-17 per-subject scores...")
    human_scores = collect_subject_scores(root, MMLU_CATEGORIES, metric_key)
    print(f"  Found {len(human_scores)}/{len(ALL_SUBJECTS)} subjects\n")

    print("Collecting cluster-16 per-subject scores...")
    cluster_scores = collect_subject_scores(root, MMLU_CLUSTER_CATEGORIES, metric_key)
    print(f"  Found {len(cluster_scores)}/{len(ALL_SUBJECTS)} subjects\n")

    # Compute averages
    h_avg, h_micro = compute_averages(human_scores, MMLU_CATEGORIES)
    c_avg, c_micro = compute_averages(cluster_scores, MMLU_CLUSTER_CATEGORIES)

    h_avg_no, h_micro_no = compute_averages(human_scores, MMLU_CATEGORIES, OTHER_SUBJECTS)
    c_avg_no, c_micro_no = compute_averages(cluster_scores, MMLU_CLUSTER_CATEGORIES, OTHER_SUBJECTS)

    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    print("=" * 70)
    print(f"{'':>30} {'Human-17':>12} {'Cluster-16':>12} {'Diff':>12}")
    print("-" * 70)
    print(f"{'avg (mean of cat means)':>30} {fmt(h_avg):>12} {fmt(c_avg):>12} {fmt(c_avg - h_avg) if h_avg and c_avg else 'N/A':>12}")
    print(f"{'avg_micro (mean of subjects)':>30} {fmt(h_micro):>12} {fmt(c_micro):>12} {fmt(c_micro - h_micro) if h_micro and c_micro else 'N/A':>12}")
    print(f"{'avg_no_other':>30} {fmt(h_avg_no):>12} {fmt(c_avg_no):>12} {fmt(c_avg_no - h_avg_no) if h_avg_no and c_avg_no else 'N/A':>12}")
    print(f"{'avg_micro_no_other':>30} {fmt(h_micro_no):>12} {fmt(c_micro_no):>12} {fmt(c_micro_no - h_micro_no) if h_micro_no and c_micro_no else 'N/A':>12}")
    print("=" * 70)

    # Also print per-subject detail
    print(f"\n{'Subject':<42} {'Human Cat':<20} {'H Score':>8} {'Cluster Cat':<30} {'C Score':>8} {'Diff':>8}")
    print("-" * 130)
    for subj in ALL_SUBJECTS:
        h = human_scores.get(subj)
        c = cluster_scores.get(subj)
        h_val = f"{h[0]:.4f}" if h else "N/A"
        c_val = f"{c[0]:.4f}" if c else "N/A"
        diff = f"{c[0] - h[0]:+.4f}" if (h and c) else "N/A"
        print(f"{subj:<42} {SUBJECT_TO_HUMAN_CAT[subj]:<20} {h_val:>8} {SUBJECT_TO_CLUSTER_CAT[subj]:<30} {c_val:>8} {diff:>8}")


if __name__ == "__main__":
    main()
