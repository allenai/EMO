#!/usr/bin/env python3
"""Generate CSV/TSV tables of final-checkpoint metrics from prune_evals.

For each metric, produces a table where rows = models, columns = tasks,
and values = the metric at the largest checkpoint.  Mirrors the config
structure of plot_scores_prune_evals.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# ============================================================================
# CONFIGURATION - keep in sync with plot_scores_prune_evals.py
# ============================================================================

AUTO_DISCOVER = True

MODEL_SPECS = {
    "moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211step30995-hf": "moe_reduce",
    "moereducedp256_1b4b_lr-4e-3_lb-1e-1_0212step30995-hf": "moe_1b4b_reduce",
    "dense_1b_lr-4e-3_0213step30995-hf": "dense-lr4e-3",

    "twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211step30995-hf": "twolevelbatchlbreducedp512sharedexp1-lr4e-3-lb1e-1",
    "twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-2_0213step30995-hf": "twolevelbatchlbreducedp512sharedexp1-lr4e-3-lb1e-2",
}
AVAILABLE_MODELS = list(MODEL_SPECS)

TASK_SPECS = {
    "arc_challenge": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "arc_easy": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "boolq": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "coqa_0shot": [
        "recall", "f1", "primary_score",
    ],
    "csqa": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "gsm8k_generation_0shot": [
        "exact_match", "primary_score",
    ],
    "gsm8k_perplexity_0shot": [
        "bits_per_byte", "primary_score",
    ],
    "hellaswag": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "openbookqa": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "piqa": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "socialiqa": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "squad_0shot": [
        "recall", "f1", "primary_score",
    ],
    "winogrande": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_biology": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_business": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_chemistry": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_computer_science": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_culture": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_economics": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_engineering": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_geography": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_health": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_history": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_law": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_math": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_other": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_philosophy_cat": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_physics": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_politics": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_psychology": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
}
MMLU_SUBTASKS = [t for t in TASK_SPECS if t.startswith("mmlu_")]

AVAILABLE_TASK_RUNS = list(TASK_SPECS)

SELECTED_MODELS = list(AVAILABLE_MODELS)
SELECTED_TASK_RUNS = list(AVAILABLE_TASK_RUNS)

MODEL_LABELS = {model: label for model, label in MODEL_SPECS.items() if label}

DEFAULT_OUTPUT_SUBDIR = "prune_eval_tables"

PRUNE_MODE_VARIANTS: Dict[str, str] = {
    # keepk 32 variants
    "_keepk_32_bs-32_lr-5e-5_epoch-1": "(keepk 32)",
    "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise": "(keepk 32, layerwise)",
    "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise_variable_first2_unpruned": "(keepk 32, lw first2 unpruned)",
    # To add more keepk values, add entries like:
    # "_keepk_16_bs-32_lr-5e-5_epoch-1": "(keepk 16)",
    # "_keepk_16_bs-32_lr-5e-5_epoch-1_prunemode-layerwise": "(keepk 16, layerwise)",
    # "_keepk_64_bs-32_lr-5e-5_epoch-1": "(keepk 64)",
}

# ============================================================================
# END CONFIGURATION
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metric tables from prune_evals checkpoints."
    )
    parser.add_argument(
        "--prune-evals-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "prune_evals",
        help="Path to prune_evals directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to save generated tables.",
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Subdirectory inside output-dir for generated tables.",
    )
    parser.add_argument(
        "--metric-key",
        default=None,
        help="Comma-separated metric keys (overrides TASK_SPECS for all tasks).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to include.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task run names to include.",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "tsv", "markdown"],
        help="Output format (default: csv).",
    )
    return parser.parse_args()


def _is_variant_task(task_name: str) -> bool:
    return any(task_name.endswith(suffix) for suffix in PRUNE_MODE_VARIANTS)


def _strip_variant_suffix(task_name: str) -> str:
    """Strip the longest matching PRUNE_MODE_VARIANTS suffix to get the base task name."""
    for suffix in sorted(PRUNE_MODE_VARIANTS, key=len, reverse=True):
        if task_name.endswith(suffix):
            return task_name[: -len(suffix)]
    return task_name


def discover_catalog(prune_evals_root: Path) -> Tuple[List[str], List[str]]:
    models = sorted([p.name for p in prune_evals_root.iterdir() if p.is_dir()])
    task_runs = sorted(
        {
            _strip_variant_suffix(t.name)
            for model_dir in prune_evals_root.iterdir()
            if model_dir.is_dir()
            for t in model_dir.iterdir()
            if t.is_dir()
        }
    )
    return models, task_runs


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


def _find_largest_checkpoint(results_dir: Path) -> Optional[Path]:
    """Return the checkpoint-* subdirectory with the highest step number."""
    best_step = -1
    best_dir: Optional[Path] = None
    for ckpt_dir in results_dir.glob("checkpoint-*"):
        if not ckpt_dir.is_dir():
            continue
        step_str = ckpt_dir.name.replace("checkpoint-", "")
        try:
            step = int(step_str)
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best_dir = ckpt_dir
    return best_dir


def _read_final_metric(
    task_dir: Path, metric_key: str
) -> Optional[float]:
    """Read a single metric value from the largest checkpoint in task_dir/results."""
    results_dir = task_dir / "results"
    if not results_dir.is_dir():
        return None

    ckpt_dir = _find_largest_checkpoint(results_dir)
    if ckpt_dir is None:
        return None

    metrics_files = sorted(ckpt_dir.glob("task-*-metrics.json"))
    if not metrics_files:
        return None

    data = read_metrics(metrics_files[0])
    if data is None:
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


def collect_table(
    prune_evals_root: Path,
    model_names: Sequence[str],
    task_runs: Sequence[str],
    metric_key: str,
) -> pd.DataFrame:
    """Build a models x tasks table of final-checkpoint metric values.

    Also includes variant (e.g. layerwise) rows automatically.
    """
    rows: Dict[str, Dict[str, Optional[float]]] = {}

    for model_name in model_names:
        model_dir = prune_evals_root / model_name
        if not model_dir.is_dir():
            continue
        model_label = MODEL_LABELS.get(model_name, model_name)

        for task_run in task_runs:
            val = _read_final_metric(model_dir / task_run, metric_key)
            if val is not None:
                rows.setdefault(model_label, {})[task_run] = val

            for suffix, label_mod in PRUNE_MODE_VARIANTS.items():
                variant_dir = model_dir / (task_run + suffix)
                if not variant_dir.is_dir():
                    continue
                variant_label = model_label + " " + label_mod
                val = _read_final_metric(variant_dir, metric_key)
                if val is not None:
                    rows.setdefault(variant_label, {})[task_run] = val

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "model"
    df = df.reindex(columns=task_runs)
    return df


MMLU_EXCLUDE_SETS: Dict[str, List[str]] = {
    "mmlu_avg_no_other": [
        "mmlu_other",
    ],
    "mmlu_avg_no_other_hist_phil": [
        "mmlu_other",
        "mmlu_history",
        "mmlu_philosophy_cat",
    ],
}


def add_mmlu_avg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add mmlu_avg and filtered variants, placed at the front."""
    mmlu_cols = [c for c in df.columns if c in MMLU_SUBTASKS]
    if not mmlu_cols:
        return df

    df["mmlu_avg"] = df[mmlu_cols].mean(axis=1)

    for col_name, excluded in MMLU_EXCLUDE_SETS.items():
        filtered = [c for c in mmlu_cols if c not in excluded]
        if filtered and len(filtered) < len(mmlu_cols):
            df[col_name] = df[filtered].mean(axis=1)

    avg_cols = ["mmlu_avg"] + [
        c for c in MMLU_EXCLUDE_SETS if c in df.columns
    ]
    other_cols = [c for c in df.columns if c not in avg_cols]
    df = df[avg_cols + other_cols]
    return df


def shorten_task_name(task_run: str) -> str:
    """Strip the common keepk/bs/lr/epoch suffix for compact column headers."""
    for suffix_start in ("_keepk_",):
        idx = task_run.find(suffix_start)
        if idx != -1:
            return task_run[:idx]
    return task_run


def sanitize_filename(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def parse_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()

    if AUTO_DISCOVER:
        available_models, available_tasks = discover_catalog(args.prune_evals_root)
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

    metric_override = parse_csv_arg(args.metric_key)

    all_metrics: set[str] = set()
    for task_run in task_set:
        metrics = metric_override or TASK_SPECS.get(task_run)
        if metrics:
            all_metrics.update(metrics)

    if not all_metrics:
        raise RuntimeError("No metrics selected. Check TASK_SPECS or --metric-key.")

    base_output_dir = (args.output_dir / args.output_subdir).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for metric_key in sorted(all_metrics):
        relevant_tasks = [
            t for t in task_set
            if metric_key in (metric_override or TASK_SPECS.get(t, []))
        ]
        if not relevant_tasks:
            continue

        df = collect_table(
            args.prune_evals_root, model_set, relevant_tasks, metric_key
        )
        if df.empty:
            print(f"[WARN] No data for metric {metric_key!r}; skipping.")
            continue

        df = add_mmlu_avg_columns(df)

        df = df.rename(columns={c: shorten_task_name(c) for c in df.columns})

        safe_metric = sanitize_filename(metric_key)
        if args.format == "csv":
            out_path = base_output_dir / f"{safe_metric}.csv"
            df.to_csv(out_path, float_format="%.4f")
        elif args.format == "tsv":
            out_path = base_output_dir / f"{safe_metric}.tsv"
            df.to_csv(out_path, sep="\t", float_format="%.4f")
        elif args.format == "markdown":
            out_path = base_output_dir / f"{safe_metric}.md"
            out_path.write_text(
                df.to_markdown(floatfmt=".4f") + "\n", encoding="utf-8"
            )

        print(f"[INFO] Saved {out_path}")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))
        print()


if __name__ == "__main__":
    main()
