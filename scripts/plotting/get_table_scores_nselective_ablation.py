#!/usr/bin/env python3
"""Generate CSV tables for the nselective (validation data quantity) ablation.

Compares expert selection quality when using different amounts of validation
data for layerwise selection.  Two models are compared (Reg. MoE vs \methodname)
across keepk values and nselective settings.

Reads from   : <repo>/selective_evals_final/
Writes into  : <repo>/plots/<output-subdir>/<metric>/

Table layout (per metric):
    rows    = (model, task)   e.g. "moe 1T + anneal / mmlu_merged"
    columns = (keepk, nprune) e.g. "keepk_8 (100)", "keepk_8 (All)"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]

# --- Models ----------------------------------------------------------------

# Keys are the directory names that launch_selective_validation_hf.sh writes
# under selective_evals_final/. For HF Hub models, this is the model id with
# the org slash stripped: "allenai/Emo_1b14b_1T" -> "allenaiEmo_1b14b_1T".
#
# Labels are preserved from the legacy local-path setup ("moe 1T + anneal",
# "specialized moe 1T + anneal") because downstream paper_table_numbers/
# scripts hardcode them as row keys.
MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "allenaiStdMoE_1b14b_1T": {
        "label": "moe 1T + anneal",
    },
    "allenaiEmo_1b14b_1T": {
        "label": "specialized moe 1T + anneal",
    },
}

# --- KeepK values ----------------------------------------------------------

KEEPK_VALUES = [8, 16, 32, 64, 128]
KEEPK_SUFFIX_TEMPLATE = "_keepk_{k}_bs-32_lr-5e-5_epoch-1"

# --- Pruning-calibration seed retries -------------------------------------
# When a small calibration set (e.g. nprune=1) might pick an unlucky example,
# we run additional pruning passes with different seeds. Their dirs are tagged
# with `_pseed-{N}` injected after the `_nselective-N` token (or after
# `_selectivemode-...` if there is no nprune token). For each cell, we compute the
# mean across all seed variants that exist locally; cells whose retries are
# absent (most non-GSM8K cells) just use seed-0 and are unchanged.
PSEED_RETRIES: List[int] = [1, 2]

# --- Prunemode × nprune variants -------------------------------------------
# Each prunemode (Router / Easy-EP) produces its own row per (model, task).
# Inside a prunemode, each entry is (tag shown in columns, suffix on the dir).
#
# Row labels in the output CSV look like:
#   "moe 1T + anneal (Router)  / mmlu_merged"
#   "moe 1T + anneal (Easy-EP) / mmlu_merged"
# Column labels look like: "keepk_8 (1)", "keepk_8 (All)", "keepk_8 (Random)".
#
# "Random" is a separate prunemode (``prunemode-random``) — it does not depend
# on validation data, so it shows up with identical values in both the Router
# and Easy-EP row groups.

PRUNEMODE_VARIANTS: Dict[str, List[Tuple[str, str]]] = {
    "Router": [
        ("Random", "_selectivemode-random"),
        ("1", "_selectivemode-layerwise_nselective-1"),
        ("5", "_selectivemode-layerwise_nselective-5"),
        ("10", "_selectivemode-layerwise_nselective-10"),
        ("100", "_selectivemode-layerwise_nselective-100"),
        ("All", "_selectivemode-layerwise"),
    ],
    "Easy-EP": [
        ("Random", "_selectivemode-random"),
        ("1", "_selectivemode-easy_ep_nselective-1"),
        ("5", "_selectivemode-easy_ep_nselective-5"),
        ("10", "_selectivemode-easy_ep_nselective-10"),
        ("100", "_selectivemode-easy_ep_nselective-100"),
        ("All", "_selectivemode-easy_ep"),
    ],
    # Router using 0-shot demonstrations for both pruning-calibration and eval.
    # 2026-04-24: dir naming changed from the single `_0shot` suffix to two
    # orthogonal tokens `_pshots-0_eshots-0` (pruning shots / eval shots). The
    # old `_0shot` stems are empty after the rename, so we read from the new
    # tokens directly. Only run for FlexMoE so far — Reg. MoE rows drop out.
    "Router (0-shot)": [
        ("1", "_selectivemode-layerwise_nselective-1_pshots-0_eshots-0"),
        ("5", "_selectivemode-layerwise_nselective-5_pshots-0_eshots-0"),
        ("10", "_selectivemode-layerwise_nselective-10_pshots-0_eshots-0"),
        ("100", "_selectivemode-layerwise_nselective-100_pshots-0_eshots-0"),
        ("All", "_selectivemode-layerwise_pshots-0_eshots-0"),
    ],
    # Easy-EP with both pruning-calibration and eval at 0-shot. Same naming
    # update applies. Partial results only (Reg. MoE nprune-1 at a subset of
    # keepks); missing cells remain blank; fully-empty rows get dropped.
    "Easy-EP (0-shot)": [
        ("1", "_selectivemode-easy_ep_nselective-1_pshots-0_eshots-0"),
        ("5", "_selectivemode-easy_ep_nselective-5_pshots-0_eshots-0"),
        ("10", "_selectivemode-easy_ep_nselective-10_pshots-0_eshots-0"),
        ("100", "_selectivemode-easy_ep_nselective-100_pshots-0_eshots-0"),
        ("All", "_selectivemode-easy_ep_pshots-0_eshots-0"),
    ],
    # Router with task-default pruning shots but 0-shot eval. The `_pshots-*`
    # token is omitted (= task default), only `_eshots-0` is set. Lets us
    # isolate the effect of eval-time shots while keeping pruning calibration
    # at the task default.
    "Router (e0)": [
        ("1", "_selectivemode-layerwise_nselective-1_eshots-0"),
        ("5", "_selectivemode-layerwise_nselective-5_eshots-0"),
        ("10", "_selectivemode-layerwise_nselective-10_eshots-0"),
        ("100", "_selectivemode-layerwise_nselective-100_eshots-0"),
        ("All", "_selectivemode-layerwise_eshots-0"),
    ],
}

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

DEFAULT_SELECTIVE_EVALS_ROOT = REPO_ROOT / "selective_evals_final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "plots"
DEFAULT_OUTPUT_SUBDIR = "nprune_ablation_tables"

# ============================================================================
# END CONFIGURATION
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate nselective ablation tables from selective_evals_final."
    )
    parser.add_argument("--selective-evals-root", type=Path, default=DEFAULT_SELECTIVE_EVALS_ROOT)
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


def _inject_pseed(suffix: str, seed: int) -> str:
    """Inject `_pseed-{seed}` into a prunemode suffix.

    Insertion point: immediately after `_nselective-N` if present, else immediately
    after `_selectivemode-XXX`. Returns the suffix unchanged when seed == 0.
    """
    if seed == 0:
        return suffix
    m = re.search(r"_nselective-\d+", suffix)
    if m:
        return suffix[: m.end()] + f"_pseed-{seed}" + suffix[m.end():]
    m = re.search(r"_selectivemode-[a-z_]+", suffix)
    if m:
        return suffix[: m.end()] + f"_pseed-{seed}" + suffix[m.end():]
    return suffix


def _read_metric_seed_avg(
    model_dir: Path,
    relative: str,
    nprune_suffix: str,
    metric_key: str,
    checkpoint_mode: str,
) -> Optional[float]:
    """Mean of the metric across seed-0 + PSEED_RETRIES dirs that exist.

    For cells without any retry dirs on disk (most cells), this is just the
    seed-0 value, so existing tables are unchanged.
    """
    vals: List[float] = []
    for seed in [0, *PSEED_RETRIES]:
        suf = _inject_pseed(nprune_suffix, seed)
        d = model_dir / (relative + suf)
        v = _read_metric(d, metric_key, checkpoint_mode)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def collect_nprune_table(
    selective_evals_root: Path,
    checkpoint_mode: str,
) -> pd.DataFrame:
    """Build the nprune ablation table.

    Rows:   (model_label, prunemode, task_group_name)
            e.g. "moe 1T + anneal (Router) / mmlu_merged"
    Columns: "keepk_{k} ({nprune_tag})" — e.g. "keepk_8 (100)", "keepk_8 (All)"
    """
    rows: Dict[str, Dict[str, Optional[float]]] = {}

    # Stable column ordering: derive from the first prunemode's variant list.
    any_prunemode = next(iter(PRUNEMODE_VARIANTS))
    column_tag_order = [tag for tag, _ in PRUNEMODE_VARIANTS[any_prunemode]]

    for model_name, spec in MODEL_SPECS.items():
        model_dir = selective_evals_root / model_name
        if not model_dir.is_dir():
            print(f"[WARN] Model dir missing: {model_dir}")
            continue
        model_label = spec["label"]

        for prunemode, variants in PRUNEMODE_VARIANTS.items():
            for group_name, subtasks, exclude, metric_key in TASK_GROUPS:
                row_label = f"{model_label} ({prunemode}) / {group_name}"
                active_subtasks = [t for t in subtasks if t not in exclude]

                for k in KEEPK_VALUES:
                    keepk_suffix = KEEPK_SUFFIX_TEMPLATE.format(k=k)

                    for nprune_tag, nprune_suffix in variants:
                        col = f"keepk_{k} ({nprune_tag})"

                        # For aggregate tasks (MMLU, MMLU Pro): average over subtasks.
                        # For single tasks (GSM8K): read directly.
                        # Inside each subtask, we also average across pruning-
                        # seed retries (`_pseed-1`, `_pseed-2`) when those dirs
                        # exist — cells without retries fall back to seed-0
                        # alone (unchanged behavior).
                        values = []
                        for subtask in active_subtasks:
                            relative = subtask + keepk_suffix
                            val = _read_metric_seed_avg(
                                model_dir, relative, nprune_suffix,
                                metric_key, checkpoint_mode,
                            )
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

    # Enforce row order: models × prunemodes × task groups
    ordered_labels = []
    for spec in MODEL_SPECS.values():
        for prunemode in PRUNEMODE_VARIANTS:
            for group_name, _, _, _ in TASK_GROUPS:
                ordered_labels.append(f"{spec['label']} ({prunemode}) / {group_name}")

    # Enforce column order: keepk × nprune (tag order from the first prunemode)
    ordered_cols = []
    for k in KEEPK_VALUES:
        for tag in column_tag_order:
            ordered_cols.append(f"keepk_{k} ({tag})")

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(index=ordered_labels)
    df.index.name = "model / task"
    df = df.reindex(columns=[c for c in ordered_cols if c in df.columns])
    # Drop rows with no data at all (happens e.g. for Reg. MoE × 0-shot, which
    # hasn't been run). Row ordering on the remaining rows is preserved.
    df = df.dropna(axis=0, how="all")
    return df


def main() -> None:
    args = parse_args()

    df = collect_nprune_table(args.selective_evals_root, args.checkpoint_mode)
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
