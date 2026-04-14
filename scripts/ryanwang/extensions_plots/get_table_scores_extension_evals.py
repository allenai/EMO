#!/usr/bin/env python3
"""Generate CSV tables of metrics from extension_evals_0414.

Unlike prune_evals (which has checkpoint subdirs and variant suffixes), this
dataset has one metrics file per (model, task) with no finetuning checkpoints.

Reads from   : <repo>/extension_evals_0414/
Writes into  : <repo>/claude_outputs/extensions_plots/<output-subdir>/<metric>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]

MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419_ct-m8_lb0step2385-hf": {
        "label": "spec moe ct-m8 lb0",
    },
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419_ct-math_8step2385-hf": {
        "label": "spec moe ct-math 8",
    },
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238_ct-m8_lb0_wdstep2385-hf": {
        "label": "spec moe ct-m8 lb0 wd",
    },
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step23_ct-m8_lb0_frzstep2385-hf": {
        "label": "spec moe ct-m8 lb0 frz",
    },
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step23_ct-math_8-128step2385-hf": {
        "label": "spec moe ct-math 8-128",
    },
}

# MC9 tasks (used for mc9_avg aggregate)
MC9_TASKS = [
    "arc_easyrc_testolmes",
    "arc_challengerc_testolmes",
    "boolqrc_testolmes",
    "hellaswagrc_testolmes",
    "csqarc_testolmes",
    "openbookqarc_testolmes",
    "piqarc_testolmes",
    "socialiqarc_testolmes",
    "winogranderc_testolmes",
]

# Other tasks shown alongside mc9 in the aggregate table
OTHER_TASKS = [
    "gsm8k_generation_0shottestolmes",
    "gsm8kolmes",
    "minerva_math_500olmes",
    "basic_skillsolmes",
    "codex_humaneval3shotbpbnone",
    "mbpp3shotbpbnone",
    "squadolmes",
    "triviaqaolmes",
]

ALL_TASKS = MC9_TASKS + OTHER_TASKS

# Per-task metric lists.
MC_METRICS = ["acc_per_byte", "acc_raw", "primary_score"]
GEN_METRICS = ["exact_match", "primary_score"]
CODE_METRICS = ["acc_per_byte", "primary_score"]
QA_METRICS = ["exact_match", "f1", "recall", "primary_score"]

TASK_SPECS: Dict[str, List[str]] = {
    # MC9
    "arc_easyrc_testolmes":             MC_METRICS,
    "arc_challengerc_testolmes":        MC_METRICS,
    "boolqrc_testolmes":                MC_METRICS,
    "hellaswagrc_testolmes":            MC_METRICS,
    "csqarc_testolmes":                 MC_METRICS,
    "openbookqarc_testolmes":           MC_METRICS,
    "piqarc_testolmes":                 MC_METRICS,
    "socialiqarc_testolmes":            MC_METRICS,
    "winogranderc_testolmes":           MC_METRICS,
    # Generation
    "gsm8k_generation_0shottestolmes":  GEN_METRICS,
    "gsm8kolmes":                       GEN_METRICS,
    "minerva_math_500olmes":            ["exact_match", "exact_match_flex", "primary_score"],
    "basic_skillsolmes":                MC_METRICS,
    # Code
    "codex_humaneval3shotbpbnone":      CODE_METRICS,
    "mbpp3shotbpbnone":                 CODE_METRICS,
    # QA
    "squadolmes":                       QA_METRICS,
    "triviaqaolmes":                    QA_METRICS,
}

DEFAULT_EVALS_ROOT = REPO_ROOT / "extension_evals_0414"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "extensions_plots"
DEFAULT_OUTPUT_SUBDIR = "extension_eval_tables"

# ============================================================================
# END CONFIGURATION
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metric tables from extension_evals_0414."
    )
    parser.add_argument("--evals-root", type=Path, default=DEFAULT_EVALS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--metric-key", default=None,
                        help="Comma-separated metric keys (overrides TASK_SPECS).")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model names to include.")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated task names to include.")
    parser.add_argument("--format", default="csv", choices=["csv", "tsv", "markdown"])
    return parser.parse_args()


def parse_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


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


def _read_metric(task_dir: Path, metric_key: str) -> Optional[float]:
    """Read a metric from a task dir (no checkpoint subdirs — files are directly here)."""
    if not task_dir.is_dir():
        return None

    metrics_files = sorted(task_dir.glob("task-*-metrics.json"))
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


def sanitize_filename(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def collect_table(
    evals_root: Path,
    model_names: Sequence[str],
    task_runs: Sequence[str],
    metric_key: str,
) -> pd.DataFrame:
    """Build a models × tasks table."""
    rows: Dict[str, Dict[str, Optional[float]]] = {}

    for model_name in model_names:
        model_dir = evals_root / model_name
        if not model_dir.is_dir():
            print(f"[WARN] Model dir missing: {model_dir}")
            continue
        model_label = MODEL_SPECS.get(model_name, {}).get("label", model_name)

        for task_run in task_runs:
            task_dir = model_dir / task_run
            val = _read_metric(task_dir, metric_key)
            if val is not None:
                rows.setdefault(model_label, {})[task_run] = val

    if not rows:
        return pd.DataFrame()

    # Enforce deterministic row order from MODEL_SPECS.
    ordered_labels: List[str] = []
    for model_name in model_names:
        spec = MODEL_SPECS.get(model_name)
        if spec is None:
            continue
        label = spec.get("label", model_name)
        ordered_labels.append(label)
        rows.setdefault(label, {})
    for label in rows:
        if label not in ordered_labels:
            ordered_labels.append(label)

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(index=ordered_labels)
    df.index.name = "model"
    df = df.reindex(columns=task_runs)
    return df


def add_mc9_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Prepend mc9_avg column."""
    present = [c for c in MC9_TASKS if c in df.columns]
    if not present:
        return df

    for model_name in df.index:
        missing = [c for c in present if pd.isna(df.loc[model_name, c])]
        if missing:
            print(
                f"[WARN] Model {model_name!r} missing {len(missing)}/{len(present)} "
                f"mc9_avg sub-task(s): {missing} — mc9_avg will be NaN"
            )
    df["mc9_avg"] = df[present].mean(axis=1, skipna=False)

    other_cols = [c for c in df.columns if c != "mc9_avg"]
    return df[["mc9_avg"] + other_cols]


def main() -> None:
    args = parse_args()

    selected_models = parse_csv_arg(args.models) or list(MODEL_SPECS.keys())
    selected_tasks = parse_csv_arg(args.tasks) or list(TASK_SPECS.keys())

    if not selected_models:
        raise RuntimeError("No models selected.")
    if not selected_tasks:
        raise RuntimeError("No tasks selected.")

    metric_override = parse_csv_arg(args.metric_key)

    all_metrics: set[str] = set()
    for task_run in selected_tasks:
        metrics = metric_override or TASK_SPECS.get(task_run)
        if metrics:
            all_metrics.update(metrics)

    if not all_metrics:
        raise RuntimeError("No metrics selected.")

    base_output_dir = (args.output_dir / args.output_subdir).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for metric_key in sorted(all_metrics):
        relevant_tasks = [
            t for t in selected_tasks
            if metric_key in (metric_override or TASK_SPECS.get(t, []))
        ]
        if not relevant_tasks:
            continue

        df = collect_table(
            args.evals_root, selected_models, relevant_tasks, metric_key
        )
        if df.empty:
            print(f"[WARN] No data for metric {metric_key!r}; skipping.")
            continue

        df = add_mc9_avg(df)

        safe_metric = sanitize_filename(metric_key)
        metric_dir = base_output_dir / safe_metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        # Single aggregate table: mc9_avg + all MC9 tasks + other tasks
        agg_cols = [c for c in ["mc9_avg"] + MC9_TASKS + OTHER_TASKS if c in df.columns]

        slices = {
            "aggregate": agg_cols,
        }

        for slice_name, cols in slices.items():
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            slice_df = df[cols]
            if slice_df.dropna(how="all").empty:
                continue

            if args.format == "csv":
                out_path = metric_dir / f"{slice_name}.csv"
                slice_df.to_csv(out_path, float_format="%.4f")
            elif args.format == "tsv":
                out_path = metric_dir / f"{slice_name}.tsv"
                slice_df.to_csv(out_path, sep="\t", float_format="%.4f")
            elif args.format == "markdown":
                out_path = metric_dir / f"{slice_name}.md"
                out_path.write_text(
                    slice_df.to_markdown(floatfmt=".4f") + "\n", encoding="utf-8"
                )

            print(f"[INFO] Saved {out_path}")
            print(slice_df.to_string(float_format=lambda x: f"{x:.4f}"))
            print()


if __name__ == "__main__":
    main()
