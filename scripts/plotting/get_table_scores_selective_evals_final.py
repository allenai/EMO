#!/usr/bin/env python3
"""Generate CSV/TSV tables of final-checkpoint metrics from selective_evals_final.

Reads from   : <repo>/selective_evals_final/
Writes into  : <repo>/plots/<output-subdir>/<metric>/

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

# Selective modes: each variant (keepk value) is scanned for all suffixes.
# Results appear as paired columns: "task (lw)", "task (ep)", and "task (rd)"
# per base task. Tag names are kept as lw/ep/rd because the downstream
# paper_table_numbers/ and paper_figure_codes/ scripts hardcode them.
#   lw = layerwise (validation-data-based) selection
#   ep = Easy-EP   (validation-data-based) selection
#   rd = random    (no validation data used)
PRUNEMODE_SUFFIXES: Dict[str, str] = {
    "lw": "_selectivemode-layerwise",
    "ep": "_selectivemode-easy_ep",
    "rd": "_selectivemode-random",
}

# Variant definitions: just the keepk part (prunemode is added automatically).
# keepk 9 is only populated for specialized moe 1T + anneal so far — other
# models will simply have empty cells for that column.
KEEPK_VARIANTS_ALL = [
    {"keepk_suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1", "label": "(keepk 8)"},
    {"keepk_suffix": "_keepk_9_bs-32_lr-5e-5_epoch-1", "label": "(keepk 9)"},
    {"keepk_suffix": "_keepk_16_bs-32_lr-5e-5_epoch-1", "label": "(keepk 16)"},
    {"keepk_suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1", "label": "(keepk 32)"},
    {"keepk_suffix": "_keepk_64_bs-32_lr-5e-5_epoch-1", "label": "(keepk 64)"},
    {"keepk_suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1", "label": "(keepk 128)"},
]

KEEPK_VARIANTS_32_ONLY = [
    {"keepk_suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1", "label": " "},
]

# For external baselines whose task dirs are plain "{task}" (no keepk / prunemode).
KEEPK_VARIANTS_PLAIN = [
    {"keepk_suffix": "", "label": " "},
]


# Build "variants" in the old format for backward compat (used by plots, row ordering).
# These use the layerwise suffix only — the ep lookup is done in collect_table.
def _build_legacy_variants(keepk_variants):
    return [
        {"suffix": v["keepk_suffix"] + PRUNEMODE_SUFFIXES["lw"], "label": v["label"]}
        for v in keepk_variants
    ]


# Keys are the directory names that launch_selective_hf.sh writes under
# selective_evals_final/. For HF Hub models, this is the model id with the
# org slash stripped: "allenai/Emo_1b14b_1T" -> "allenaiEmo_1b14b_1T".
#
# Labels are deliberately preserved from the legacy local-path setup
# ("moe 1T + anneal", "specialized moe 1T + anneal", "moe_small", "dense",
# "olmoe_1b_7b") because downstream paper_table_numbers/ scripts hardcode
# them as row keys.
MODEL_SPECS: Dict[str, Dict[str, object]] = {
    # 130B-token ablation models ----------------------------------------------
    "allenaiDense_1b_130B": {
        "label": "dense",
        "keepk_variants": KEEPK_VARIANTS_32_ONLY,
        "variants": _build_legacy_variants(KEEPK_VARIANTS_32_ONLY),
    },
    "allenaiStdMoE_1b4b_130B": {
        "label": "moe_small",
        "keepk_variants": KEEPK_VARIANTS_32_ONLY,
        "variants": _build_legacy_variants(KEEPK_VARIANTS_32_ONLY),
    },
    "allenaiStdMoE_1b14b_130B": {
        "label": "moe 130B",
        "keepk_variants": KEEPK_VARIANTS_ALL,
        "variants": _build_legacy_variants(KEEPK_VARIANTS_ALL),
    },
    "allenaiEmo_1b14b_130B": {
        "label": "specialized moe 130B",
        "keepk_variants": KEEPK_VARIANTS_ALL,
        "variants": _build_legacy_variants(KEEPK_VARIANTS_ALL),
    },
    # Main release (1T tokens) -----------------------------------------------
    "allenaiStdMoE_1b14b_1T": {
        "label": "moe 1T + anneal",
        "keepk_variants": KEEPK_VARIANTS_ALL,
        "variants": _build_legacy_variants(KEEPK_VARIANTS_ALL),
    },
    "allenaiEmo_1b14b_1T": {
        "label": "specialized moe 1T + anneal",
        "keepk_variants": KEEPK_VARIANTS_ALL,
        "variants": _build_legacy_variants(KEEPK_VARIANTS_ALL),
    },
    # Midtraining ablations --------------------------------------------------
    "allenaiStdMoE_1b14b_1T_Preanneal": {
        "label": "moe 1T (preanneal)",
        "keepk_variants": KEEPK_VARIANTS_ALL,
        "variants": _build_legacy_variants(KEEPK_VARIANTS_ALL),
    },
    "allenaiStdMoE_1b14b_1T_EmoAnnealed": {
        "label": "moe 1T + emo-anneal",
        "keepk_variants": KEEPK_VARIANTS_ALL,
        "variants": _build_legacy_variants(KEEPK_VARIANTS_ALL),
    },
    # External baseline: AllenAI OLMoE-1B-7B-0924.
    # Task dirs are plain "{task}" (no keepk suffix, no selectivemode suffix).
    # Only (lw) column is populated — (ep) has no data for this model.
    "allenaiOLMoE-1B-7B-0924": {
        "label": "olmoe_1b_7b",
        "keepk_variants": KEEPK_VARIANTS_PLAIN,
        "variants": [{"suffix": "", "label": " "}],
        "prunemode_override": {"lw": ""},
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

# Gen5: 5-task generation average.
# squad_0shot_merged is shown as a reference column but excluded from the average.
GEN5_TASKS = [
    "squad_merged",
    "coqa_merged",
    "naturalqs_merged",
    "triviaqa_merged",
    "drop_merged",
]
GEN5_REFERENCE_TASKS = [
    "squad_0shot_merged",
]
GEN5_ALL_TASKS = GEN5_TASKS + GEN5_REFERENCE_TASKS
GEN5_METRICS = ["f1", "exact_match", "recall", "primary_score"]

DEFAULT_METRICS = ["softloss_corr", "acc_per_byte", "acc_raw", "primary_score"]

TASK_SPECS: Dict[str, List[str]] = {
    t: list(DEFAULT_METRICS) for t in MC9_TASKS + MMLU_MERGED_TASKS + MMLU_PRO_MERGED_TASKS
}
TASK_SPECS.update({t: list(GSM8K_METRICS) for t in GSM8K_TASKS})
TASK_SPECS.update({t: list(GEN5_METRICS) for t in GEN5_ALL_TASKS})

DEFAULT_SELECTIVE_EVALS_ROOT = REPO_ROOT / "selective_evals_final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "plots"
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
        description="Generate metric tables from selective_evals_final checkpoints."
    )
    parser.add_argument(
        "--selective-evals-root",
        type=Path,
        default=DEFAULT_SELECTIVE_EVALS_ROOT,
        help="Path to selective_evals_final directory.",
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
        if (
            best_step is None
            or (mode == "last" and step > best_step)
            or (mode == "first" and step < best_step)
        ):
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
    selective_evals_root: Path,
    model_names: Sequence[str],
    task_runs: Sequence[str],
    metric_key: str,
    checkpoint_mode: str,
) -> pd.DataFrame:
    """Build a (model, keepk) × (task, prunemode) table.

    Rows = model + keepk variant label (e.g. "moe 1T + anneal (keepk 32)").
    Columns = paired per task: "task (lw)" and "task (ep)" for each base task.
    """
    rows: Dict[str, Dict[str, Optional[float]]] = {}

    for model_name in model_names:
        model_dir = selective_evals_root / model_name
        if not model_dir.is_dir():
            print(f"[WARN] Model dir missing: {model_dir}")
            continue
        spec = MODEL_SPECS.get(model_name, {})
        model_label = spec.get("label", model_name)
        keepk_variants = spec.get("keepk_variants", [])
        pm_suffixes = spec.get("prunemode_override", PRUNEMODE_SUFFIXES)

        for kv in keepk_variants:
            variant_label = f"{model_label} {kv['label']}".strip()
            for task_run in task_runs:
                for pm_tag, pm_suffix in pm_suffixes.items():
                    col_name = f"{task_run} ({pm_tag})"
                    variant_dir = model_dir / (task_run + kv["keepk_suffix"] + pm_suffix)
                    if not variant_dir.is_dir():
                        continue
                    val = _read_final_metric(variant_dir, metric_key, checkpoint_mode)
                    if val is not None:
                        rows.setdefault(variant_label, {})[col_name] = val

    if not rows:
        return pd.DataFrame()

    # Enforce deterministic row order: MODEL_SPECS order × keepk variant order.
    ordered_labels: List[str] = []
    for model_name in model_names:
        spec = MODEL_SPECS.get(model_name)
        if spec is None:
            continue
        model_label = spec.get("label", model_name)
        for kv in spec.get("keepk_variants", []):
            label = f"{model_label} {kv['label']}".strip()
            ordered_labels.append(label)
            rows.setdefault(label, {})
    for label in rows:
        if label not in ordered_labels:
            ordered_labels.append(label)

    # Build column order: for each base task, interleave (lw) then (ep).
    ordered_cols: List[str] = []
    for task_run in task_runs:
        for pm_tag in PRUNEMODE_SUFFIXES:
            ordered_cols.append(f"{task_run} ({pm_tag})")

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(index=ordered_labels)
    df.index.name = "model"
    # Only keep columns that actually exist in the data.
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    df = df.reindex(columns=ordered_cols)
    return df


def add_group_avg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Prepend group averages, computed separately for each prunemode (lw, ep)."""
    avg_cols_added: List[str] = []

    # (avg_base_name, full_task_list, exclude_set)
    groups: List[Tuple[str, List[str], List[str]]] = [
        ("mc9_avg", MC9_TASKS, []),
        ("gen5_avg", GEN5_TASKS, []),
        ("mmlu_merged_avg_no_other", MMLU_MERGED_TASKS, ["mmlu_merged_other"]),
        ("mmlu_pro_merged_avg_no_other", MMLU_PRO_MERGED_TASKS, ["mmlu_pro_merged_other"]),
    ]
    for avg_base, group_tasks, exclude in groups:
        for pm_tag in PRUNEMODE_SUFFIXES:
            avg_name = f"{avg_base} ({pm_tag})"
            present = [
                f"{t} ({pm_tag})"
                for t in group_tasks
                if f"{t} ({pm_tag})" in df.columns and t not in exclude
            ]
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
    print(
        f"\n=== Generating tables for checkpoint_mode={checkpoint_mode!r} → {base_output_dir} ===\n"
    )

    for metric_key in sorted(all_metrics):
        relevant_tasks = [
            t for t in selected_tasks if metric_key in (metric_override or TASK_SPECS.get(t, []))
        ]
        if not relevant_tasks:
            continue

        df = collect_table(
            args.selective_evals_root, selected_models, relevant_tasks, metric_key, checkpoint_mode
        )
        if df.empty:
            print(f"[WARN] No data for metric {metric_key!r}; skipping.")
            continue

        df = add_group_avg_columns(df)

        safe_metric = sanitize_filename(metric_key)
        metric_dir = base_output_dir / safe_metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        # --- Helper to expand task lists into paired (lw)/(ep) columns ---
        def _pm_cols(tasks: List[str]) -> List[str]:
            """Expand base task names to paired (lw)/(ep) columns that exist."""
            cols = []
            for t in tasks:
                for pm_tag in PRUNEMODE_SUFFIXES:
                    c = f"{t} ({pm_tag})"
                    if c in df.columns:
                        cols.append(c)
            return cols

        def _avg_cols(avg_base: str) -> List[str]:
            """Get the average columns for each prunemode."""
            return [
                f"{avg_base} ({pm})"
                for pm in PRUNEMODE_SUFFIXES
                if f"{avg_base} ({pm})" in df.columns
            ]

        # --- Define output slices ---
        agg_cols = (
            _avg_cols("mc9_avg")
            + _avg_cols("gen5_avg")
            + _avg_cols("mmlu_merged_avg_no_other")
            + _avg_cols("mmlu_pro_merged_avg_no_other")
        )

        mc9_cols = _avg_cols("mc9_avg") + _pm_cols(MC9_TASKS)
        gen5_cols = _avg_cols("gen5_avg") + _pm_cols(GEN5_TASKS) + _pm_cols(GEN5_REFERENCE_TASKS)
        mmlu_cols = _avg_cols("mmlu_merged_avg_no_other") + _pm_cols(MMLU_MERGED_TASKS)
        mmlu_pro_cols = _avg_cols("mmlu_pro_merged_avg_no_other") + _pm_cols(MMLU_PRO_MERGED_TASKS)
        gsm8k_cols = _pm_cols(GSM8K_TASKS)

        slices = {
            "aggregate": agg_cols,
            "mc9": mc9_cols,
            "gen5": gen5_cols,
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
                out_path.write_text(slice_df.to_markdown(floatfmt=".4f") + "\n", encoding="utf-8")

            print(f"[INFO] Saved {out_path}")
            print(slice_df.to_string(float_format=lambda x: f"{x:.4f}"))
            print()


if __name__ == "__main__":
    main()
