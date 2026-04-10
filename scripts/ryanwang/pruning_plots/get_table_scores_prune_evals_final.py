#!/usr/bin/env python3
"""Generate CSV/TSV tables of final-checkpoint metrics from prune_evals_final.

Slimmed-down sibling of eval_plots/get_table_scores_prune_evals_0319.py,
targeting the *_merged task suite for the two annealed 1T models.

Reads from   : <repo>/prune_evals_final/
Writes into  : <repo>/claude_outputs/prune_plots/<output-subdir>/<metric>/

For each metric a table is produced with rows = (model, variant) and
columns = tasks, where values come from the largest checkpoint of each
finetuning run.
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

DEFAULT_VARIANTS = [
    {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise",   "label": "(keepk 8)"},
    {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise",  "label": "(keepk 32)"},
    {"suffix": "_keepk_64_bs-32_lr-5e-5_epoch-1_prunemode-layerwise",  "label": "(keepk 64)"},
    {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
]

# Variant lists for the older 200B-token (step30995) checkpoints, mirroring
# the configuration in eval_plots/get_table_scores_prune_evals_0319.py.
DENSE_VARIANTS = [
    {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": " "},
]
MOE_200B_VARIANTS = [
    {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise",   "label": "(keepk 8)"},
    {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise",  "label": "(keepk 32)"},
    {"suffix": "_keepk_64_bs-32_lr-5e-5_epoch-1_prunemode-layerwise",  "label": "(keepk 64)"},
    {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
]
MOE_SMALL_VARIANTS = [
    {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": " "},
]
SPEC_MOE_200B_VARIANTS = MOE_200B_VARIANTS

MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "dense_1b_lr-4e-3_0213step30995-hf": {
        "label": "dense",
        "variants": DENSE_VARIANTS,
    },
    "moereducedp512sharedexp1_1b4b_lr-4e-3_lb-1e-1_0308step30995-hf": {
        "label": "moe_small",
        "variants": MOE_SMALL_VARIANTS,
    },
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308step30995-hf": {
        "label": "moe",
        "variants": MOE_200B_VARIANTS,
    },
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301step30995-hf": {
        "label": "specialized moe + globallb + 1shardexp + randpool",
        "variants": SPEC_MOE_200B_VARIANTS,
    },
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419step250339-hf": {
        "label": "moe 1T + anneal",
        "variants": DEFAULT_VARIANTS,
    },
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_twolevel_randpool-8-128_from_step238419step250339-hf": {
        "label": "moe 1T + twolevel anneal",
        "variants": DEFAULT_VARIANTS,
    },
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419step250339-hf": {
        "label": "specialized moe 1T + anneal",
        "variants": DEFAULT_VARIANTS,
    },
}

MC9_TASKS = [
    "arc_easy_merged",
    "arc_challenge_merged",
    "boolq_merged",
    "hellaswag_merged",
    "csqa_merged",
    "openbookqa_merged",
    "piqa_merged",
    "socialiqa_merged",
    "winogrande_merged",
]

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
    "gsm8k_generation_0shot_merged",
    "gsm8k_generation_8shot_merged",
]
GSM8K_METRICS = ["exact_match", "primary_score"]

DEFAULT_METRICS = ["softloss_corr", "acc_per_byte", "acc_raw", "primary_score"]

TASK_SPECS: Dict[str, List[str]] = {
    t: list(DEFAULT_METRICS)
    for t in MC9_TASKS + MMLU_MERGED_TASKS + MMLU_PRO_MERGED_TASKS
}
TASK_SPECS.update({t: list(GSM8K_METRICS) for t in GSM8K_TASKS})

DEFAULT_PRUNE_EVALS_ROOT = REPO_ROOT / "prune_evals_final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "prune_plots"
DEFAULT_OUTPUT_SUBDIR = "prune_eval_tables_final"
DEFAULT_OUTPUT_SUBDIR_CKPT0 = "prune_eval_tables_final_ckpt0"

# Which checkpoint to read per (output-subdir, mode) pair.
# "last"  -> highest-numbered checkpoint-* (final finetuning state)
# "first" -> checkpoint-0 (pre-finetuning, immediately after pruning)
CHECKPOINT_MODES: List[Tuple[str, str]] = [
    (DEFAULT_OUTPUT_SUBDIR, "last"),
    (DEFAULT_OUTPUT_SUBDIR_CKPT0, "first"),
]

# ============================================================================
# END CONFIGURATION
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metric tables from prune_evals_final checkpoints."
    )
    parser.add_argument(
        "--prune-evals-root",
        type=Path,
        default=DEFAULT_PRUNE_EVALS_ROOT,
        help="Path to prune_evals_final directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated tables.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help=(
            "Subdirectory inside output-dir for generated tables. "
            "If unset, the script writes one subdir per checkpoint mode "
            f"(default: {DEFAULT_OUTPUT_SUBDIR} for last + "
            f"{DEFAULT_OUTPUT_SUBDIR_CKPT0} for first)."
        ),
    )
    parser.add_argument(
        "--checkpoint-mode",
        default=None,
        choices=["last", "first", "both"],
        help=(
            "Which finetuning checkpoint to read per task. "
            "'last' = highest-numbered (final), 'first' = checkpoint-0 "
            "(pre-finetuning), 'both' = emit both into separate subdirs. "
            "Defaults to 'both' unless --output-subdir is given."
        ),
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
        help="Comma-separated task names to include.",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "tsv", "markdown"],
        help="Output format (default: csv).",
    )
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


def _select_checkpoint(results_dir: Path, mode: str) -> Optional[Path]:
    """Return a checkpoint-* subdirectory based on `mode`.

    mode == "last":  highest-numbered checkpoint
    mode == "first": lowest-numbered checkpoint (typically checkpoint-0)
    """
    best_step: Optional[int] = None
    best_dir: Optional[Path] = None
    for ckpt_dir in results_dir.glob("checkpoint-*"):
        if not ckpt_dir.is_dir():
            continue
        try:
            step = int(ckpt_dir.name.replace("checkpoint-", ""))
        except ValueError:
            continue
        if best_step is None or (mode == "last" and step > best_step) or (mode == "first" and step < best_step):
            best_step = step
            best_dir = ckpt_dir
    return best_dir


def _read_final_metric(task_dir: Path, metric_key: str, mode: str) -> Optional[float]:
    """Read a single metric value from a checkpoint in task_dir/results.

    `mode` selects which checkpoint to read (see _select_checkpoint).
    """
    results_dir = task_dir / "results"
    if not results_dir.is_dir():
        return None

    ckpt_dir = _select_checkpoint(results_dir, mode)
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


def _get_model_variants(model_name: str) -> List[Tuple[str, str]]:
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        return []
    return [(v["suffix"], v["label"]) for v in spec.get("variants", [])]


def collect_table(
    prune_evals_root: Path,
    model_names: Sequence[str],
    task_runs: Sequence[str],
    metric_key: str,
    checkpoint_mode: str,
) -> pd.DataFrame:
    """Build a (model, variant) x task table of checkpoint metric values."""
    rows: Dict[str, Dict[str, Optional[float]]] = {}

    for model_name in model_names:
        model_dir = prune_evals_root / model_name
        if not model_dir.is_dir():
            print(f"[WARN] Model dir missing: {model_dir}")
            continue
        model_label = MODEL_SPECS.get(model_name, {}).get("label", model_name)

        for suffix, label_mod in _get_model_variants(model_name):
            variant_label = f"{model_label} {label_mod}".strip()
            for task_run in task_runs:
                variant_dir = model_dir / (task_run + suffix)
                if not variant_dir.is_dir():
                    continue
                val = _read_final_metric(variant_dir, metric_key, checkpoint_mode)
                if val is not None:
                    rows.setdefault(variant_label, {})[task_run] = val

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "model"
    df = df.reindex(columns=task_runs)
    return df


def add_group_avg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Prepend mc9_avg, mmlu_merged_avg(_no_other), mmlu_pro_merged_avg columns."""
    avg_cols_added: List[str] = []

    # (avg_col_name, full_task_list, exclude_set)
    groups: List[Tuple[str, List[str], List[str]]] = [
        ("mc9_avg",                      MC9_TASKS,             []),
        ("mmlu_merged_avg_no_other",     MMLU_MERGED_TASKS,     ["mmlu_merged_other"]),
        ("mmlu_pro_merged_avg_no_other", MMLU_PRO_MERGED_TASKS, ["mmlu_pro_merged_other"]),
    ]
    for avg_name, group_tasks, exclude in groups:
        present = [c for c in group_tasks if c in df.columns and c not in exclude]
        if not present:
            continue
        for model_name in df.index:
            missing = [c for c in present if pd.isna(df.loc[model_name, c])]
            if missing:
                print(
                    f"[WARN] Model {model_name!r} missing {len(missing)}/{len(present)} "
                    f"{avg_name} sub-task(s): {missing} — {avg_name} will be NaN"
                )
        df[avg_name] = df[present].mean(axis=1, skipna=False)
        avg_cols_added.append(avg_name)

    if not avg_cols_added:
        return df

    other_cols = [c for c in df.columns if c not in avg_cols_added]
    return df[avg_cols_added + other_cols]


def sanitize_filename(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


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
        raise RuntimeError("No metrics selected. Check TASK_SPECS or --metric-key.")

    # Resolve which (output-subdir, checkpoint-mode) pairs to emit.
    if args.output_subdir is not None:
        mode = args.checkpoint_mode or "last"
        if mode == "both":
            raise RuntimeError(
                "--checkpoint-mode=both is incompatible with an explicit --output-subdir."
            )
        runs: List[Tuple[str, str]] = [(args.output_subdir, mode)]
    else:
        mode = args.checkpoint_mode or "both"
        if mode == "both":
            runs = list(CHECKPOINT_MODES)
        else:
            subdir = next(s for s, m in CHECKPOINT_MODES if m == mode)
            runs = [(subdir, mode)]

    for output_subdir, checkpoint_mode in runs:
        run_one(
            args=args,
            selected_models=selected_models,
            selected_tasks=selected_tasks,
            metric_override=metric_override,
            all_metrics=all_metrics,
            output_subdir=output_subdir,
            checkpoint_mode=checkpoint_mode,
        )


def run_one(
    *,
    args: argparse.Namespace,
    selected_models: List[str],
    selected_tasks: List[str],
    metric_override: Optional[List[str]],
    all_metrics: set,
    output_subdir: str,
    checkpoint_mode: str,
) -> None:
    base_output_dir = (args.output_dir / output_subdir).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Generating tables for checkpoint_mode={checkpoint_mode!r} → {base_output_dir} ===\n")

    for metric_key in sorted(all_metrics):
        relevant_tasks = [
            t for t in selected_tasks
            if metric_key in (metric_override or TASK_SPECS.get(t, []))
        ]
        if not relevant_tasks:
            continue

        df = collect_table(
            args.prune_evals_root, selected_models, relevant_tasks, metric_key, checkpoint_mode
        )
        if df.empty:
            print(f"[WARN] No data for metric {metric_key!r}; skipping.")
            continue

        df = add_group_avg_columns(df)

        safe_metric = sanitize_filename(metric_key)
        metric_dir = base_output_dir / safe_metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        # --- Define output slices ---
        agg_cols = [c for c in [
            "mc9_avg",
            "mmlu_merged_avg_no_other",
            "mmlu_pro_merged_avg_no_other",
        ] if c in df.columns]

        mc9_cols = [c for c in ["mc9_avg"] + MC9_TASKS if c in df.columns]
        mmlu_cols = [c for c in ["mmlu_merged_avg_no_other"] + MMLU_MERGED_TASKS if c in df.columns]
        mmlu_pro_cols = [c for c in ["mmlu_pro_merged_avg_no_other"] + MMLU_PRO_MERGED_TASKS if c in df.columns]
        gsm8k_cols = [c for c in GSM8K_TASKS if c in df.columns]

        slices = {
            "aggregate": agg_cols,
            "mc9": mc9_cols,
            "mmlu_merged": mmlu_cols,
            "mmlu_pro_merged": mmlu_pro_cols,
            "gsm8k": gsm8k_cols,
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
