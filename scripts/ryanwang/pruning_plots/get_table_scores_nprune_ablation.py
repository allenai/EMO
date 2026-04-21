#!/usr/bin/env python3
"""Generate CSV tables for the nprune (validation data quantity) ablation.

Compares expert selection quality when using different amounts of validation
data for layerwise pruning.  Two models are compared (Reg. MoE vs \methodname)
across keepk values and nprune settings.

Reads from   : <repo>/prune_evals_final/
Writes into  : <repo>/claude_outputs/prune_plots/<output-subdir>/<metric>/

Table layout (per metric):
    rows    = (model, task)   e.g. "moe 1T + anneal / mmlu_merged"
    columns = (keepk, nprune) e.g. "keepk_8 (100)", "keepk_8 (All)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]

# --- Models ----------------------------------------------------------------

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419step250339-hf": {
        "label": "moe 1T + anneal",
    },
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419step250339-hf": {
        "label": "specialized moe 1T + anneal",
    },
}

# --- KeepK values ----------------------------------------------------------

KEEPK_VALUES = [8, 16, 32, 64, 128]
KEEPK_SUFFIX_TEMPLATE = "_keepk_{k}_bs-32_lr-5e-5_epoch-1"

# --- Nprune variants -------------------------------------------------------
# Each entry: (tag shown in column headers, suffix appended after prunemode)
# "All" means no nprune suffix (uses all available validation data).

NPRUNE_VARIANTS: List[Tuple[str, str]] = [
    ("1", "_prunemode-layerwise_nprune-1"),
    ("5", "_prunemode-layerwise_nprune-5"),
    ("10", "_prunemode-layerwise_nprune-10"),
    ("100", "_prunemode-layerwise_nprune-100"),
    ("All", "_prunemode-layerwise"),
    ("Random", "_prunemode-random"),
]

# --- Tasks and metrics -----------------------------------------------------

MMLU_MERGED_TASKS = [
    "mmlu_merged_biology",
    "mmlu_merged_business",
    "mmlu_merged_chemistry",
    "mmlu_merged_computer_science",
    "mmlu_merged_culture",
    "mmlu_merged_economics",
    "mmlu_merged_engineering",
    "mmlu_merged_geography",
    "mmlu_merged_health",
    "mmlu_merged_history",
    "mmlu_merged_law",
    "mmlu_merged_math",
    "mmlu_merged_other",
    "mmlu_merged_philosophy_cat",
    "mmlu_merged_physics",
    "mmlu_merged_politics",
    "mmlu_merged_psychology",
]

MMLU_PRO_MERGED_TASKS = [
    "mmlu_pro_merged_math",
    "mmlu_pro_merged_health",
    "mmlu_pro_merged_physics",
    "mmlu_pro_merged_business",
    "mmlu_pro_merged_biology",
    "mmlu_pro_merged_chemistry",
    "mmlu_pro_merged_computer_science",
    "mmlu_pro_merged_economics",
    "mmlu_pro_merged_engineering",
    "mmlu_pro_merged_philosophy",
    "mmlu_pro_merged_other",
    "mmlu_pro_merged_history",
    "mmlu_pro_merged_psychology",
    "mmlu_pro_merged_law",
]

GSM8K_TASKS = [
    "gsm8k_generation_8shot_merged",
]

# Aggregate task groups: (display_name, subtask_list, exclude_from_avg, metric)
TASK_GROUPS = [
    ("mmlu_merged", MMLU_MERGED_TASKS, ["mmlu_merged_other"], "acc_raw"),
    ("mmlu_pro_merged", MMLU_PRO_MERGED_TASKS, ["mmlu_pro_merged_other"], "acc_raw"),
    ("gsm8k", GSM8K_TASKS, [], "exact_match"),
]

DEFAULT_PRUNE_EVALS_ROOT = REPO_ROOT / "prune_evals_final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "prune_plots"
DEFAULT_OUTPUT_SUBDIR = "nprune_ablation_tables"

# ============================================================================
# END CONFIGURATION
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate nprune ablation tables from prune_evals_final."
    )
    parser.add_argument("--prune-evals-root", type=Path, default=DEFAULT_PRUNE_EVALS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument(
        "--checkpoint-mode",
        default="last",
        choices=["last", "first"],
        help="Which checkpoint to read: 'last' (finetuned) or 'first' (ckpt-0).",
    )
    parser.add_argument("--format", default="csv", choices=["csv", "tsv", "markdown"])
    return parser.parse_args()


def _select_checkpoint(results_dir: Path, mode: str) -> Optional[Path]:
    """Return a checkpoint-* subdirectory based on `mode`."""
    best_step: Optional[int] = None
    best_dir: Optional[Path] = None
    for ckpt_dir in results_dir.glob("checkpoint-*"):
        if not ckpt_dir.is_dir():
            continue
        try:
            step = int(ckpt_dir.name.replace("checkpoint-", ""))
        except ValueError:
            continue
        if (
            best_step is None
            or (mode == "last" and step > best_step)
            or (mode == "first" and step < best_step)
        ):
            best_step = step
            best_dir = ckpt_dir
    return best_dir


def _read_metric(task_dir: Path, metric_key: str, checkpoint_mode: str) -> Optional[float]:
    """Read a metric from task_dir/results/checkpoint-*/task-*-metrics.json."""
    results_dir = task_dir / "results"
    if not results_dir.is_dir():
        return None
    ckpt_dir = _select_checkpoint(results_dir, checkpoint_mode)
    if ckpt_dir is None:
        return None
    metrics_files = sorted(ckpt_dir.glob("task-*-metrics.json"))
    if not metrics_files:
        return None
    try:
        with metrics_files[0].open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    metric_values = data.get("metrics")
    if not isinstance(metric_values, dict):
        return None
    value = metric_values.get(metric_key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_nprune_table(
    prune_evals_root: Path,
    checkpoint_mode: str,
) -> pd.DataFrame:
    """Build the nprune ablation table.

    Rows:   (model_label, task_group_name) — e.g. "moe 1T + anneal / mmlu_merged"
    Columns: "keepk_{k} ({nprune_tag})" — e.g. "keepk_8 (100)", "keepk_8 (All)"
    """
    rows: Dict[str, Dict[str, Optional[float]]] = {}

    for model_name, spec in MODEL_SPECS.items():
        model_dir = prune_evals_root / model_name
        if not model_dir.is_dir():
            print(f"[WARN] Model dir missing: {model_dir}")
            continue
        model_label = spec["label"]

        for group_name, subtasks, exclude, metric_key in TASK_GROUPS:
            row_label = f"{model_label} / {group_name}"
            active_subtasks = [t for t in subtasks if t not in exclude]

            for k in KEEPK_VALUES:
                keepk_suffix = KEEPK_SUFFIX_TEMPLATE.format(k=k)

                for nprune_tag, nprune_suffix in NPRUNE_VARIANTS:
                    col = f"keepk_{k} ({nprune_tag})"

                    # For aggregate tasks (MMLU, MMLU Pro): average over subtasks.
                    # For single tasks (GSM8K): read directly.
                    values = []
                    for subtask in active_subtasks:
                        task_dir = model_dir / (subtask + keepk_suffix + nprune_suffix)
                        val = _read_metric(task_dir, metric_key, checkpoint_mode)
                        if val is not None:
                            values.append(val)

                    if values and len(values) == len(active_subtasks):
                        avg = sum(values) / len(values)
                        rows.setdefault(row_label, {})[col] = avg
                    elif values:
                        print(
                            f"[WARN] {row_label} {col}: only {len(values)}/{len(active_subtasks)} "
                            f"subtasks found — setting to NaN"
                        )
                        rows.setdefault(row_label, {})[col] = None

    if not rows:
        return pd.DataFrame()

    # Enforce row order: models × task groups
    ordered_labels = []
    for spec in MODEL_SPECS.values():
        for group_name, _, _, _ in TASK_GROUPS:
            ordered_labels.append(f"{spec['label']} / {group_name}")

    # Enforce column order: keepk × nprune
    ordered_cols = []
    for k in KEEPK_VALUES:
        for nprune_tag, _ in NPRUNE_VARIANTS:
            ordered_cols.append(f"keepk_{k} ({nprune_tag})")

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(index=ordered_labels)
    df.index.name = "model / task"
    df = df.reindex(columns=[c for c in ordered_cols if c in df.columns])
    return df


def main() -> None:
    args = parse_args()

    df = collect_nprune_table(args.prune_evals_root, args.checkpoint_mode)
    if df.empty:
        print("[ERROR] No data collected.")
        return

    out_dir = (args.output_dir / args.output_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "csv":
        out_path = out_dir / "nprune_ablation.csv"
        df.to_csv(out_path, float_format="%.4f")
    elif args.format == "tsv":
        out_path = out_dir / "nprune_ablation.tsv"
        df.to_csv(out_path, sep="\t", float_format="%.4f")
    elif args.format == "markdown":
        out_path = out_dir / "nprune_ablation.md"
        out_path.write_text(df.to_markdown(floatfmt=".4f") + "\n", encoding="utf-8")

    print(f"[INFO] Saved {out_path}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print()


if __name__ == "__main__":
    main()
