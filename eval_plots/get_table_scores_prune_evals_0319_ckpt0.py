#!/usr/bin/env python3
"""Generate CSV/TSV tables of checkpoint-0 metrics from prune_evals.

For each metric, produces a table where rows = models, columns = tasks,
and values = the metric at checkpoint-0 (the first finetuning checkpoint).
Baseline-only models (e.g. OLMoE) read from original_model/<task>/results/checkpoint-0/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

AUTO_DISCOVER = True

MODEL_SPECS = {
    "allenaiOLMoE-1B-7B-0924": {
        "label": "OLMoE-1B-7B",
        "baseline": True,
        "variants": [],
    },
    "allenaiOLMoE-1B-7B-0924_step240000-tokens1006B": {
        "label": "OLMoE-1B-7B (1T)",
        "baseline": True,
        "variants": [],
    },
    "dense_1b_lr-4e-3_0213step30995-hf": {
        "label": "dense",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": " "},
        ],
    },
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308step30995-hf": {
        "label": "moe",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 8)"},
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32)"},
            {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
        ],
    },
    "moereducedp512sharedexp1_1b4b_lr-4e-3_lb-1e-1_0308step30995-hf": {
        "label": "moe_small",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": " "},
        ],
    },

    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301step30995-hf": {
        "label": "specialized moe + globallb + 1shardexp + randpool",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 8)"},
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32)"},
            {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
        ],
    },

    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313step238419-hf": {
        "label": "specialized moe 1T",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 8)"},
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32)"},
            {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
        ]
    },

    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419step250339-hf": {
        "label": "specialized moe 1T + anneal",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 8)"},
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32)"},
            {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
        ]
    },
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322step238419-hf": {
        "label": "moe 1T",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 8)"},
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32)"},
            {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
        ]
    },

    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419step250339-hf": {
        "label": "moe 1T + anneal",
        "baseline": False,
        "variants": [
            {"suffix": "_keepk_8_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 8)"},
            {"suffix": "_keepk_32_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 32)"},
            {"suffix": "_keepk_128_bs-32_lr-5e-5_epoch-1_prunemode-layerwise", "label": "(keepk 128)"},
        ]
    },


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
    "mmlu_pro_biology": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_business": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_chemistry": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_computer_science": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_economics": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_engineering": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_health": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_history": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_law": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_math": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_other": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_philosophy": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_physics": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_psychology": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_biology": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_business": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_chemistry": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_computer_science": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_economics": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_engineering": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_health": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_history": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_law": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_math": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_other": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_philosophy": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_physics": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "mmlu_pro_merged_psychology": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    "gsm8k_generation_8shot": [
        "exact_match", "primary_score",
    ],
    # HellaSwag merged (baseline: single model on all data)
    "hellaswag_merged": [
        "softloss_corr", "acc_per_byte", "primary_score",
    ],
    # HellaSwag cluster-merged variants (all k values)
    **{
        f"hellaswag_k{k}_cluster_merged_{c}": ["softloss_corr", "acc_per_byte", "primary_score"]
        for k in [6, 8, 10, 16]
        for c in range(k)
    },
    # Legacy k=6 aliases
    **{
        f"hellaswag_cluster_merged_{c}": ["softloss_corr", "acc_per_byte", "primary_score"]
        for c in range(6)
    },
}
MMLU_SUBTASKS = [t for t in TASK_SPECS if t.startswith("mmlu_") and not t.startswith("mmlu_pro_")]
MMLU_PRO_SUBTASKS = [t for t in TASK_SPECS if t.startswith("mmlu_pro_") and not t.startswith("mmlu_pro_merged_")]
MMLU_PRO_MERGED_SUBTASKS = [t for t in TASK_SPECS if t.startswith("mmlu_pro_merged_")]
HELLASWAG_CLUSTER_SUBTASKS_BY_K = {
    k: [f"hellaswag_k{k}_cluster_merged_{c}" for c in range(k)]
    for k in [6, 8, 10, 16]
}
HELLASWAG_CLUSTER_SUBTASKS = [f"hellaswag_cluster_merged_{c}" for c in range(6)]
HELLASWAG_CLUSTER_TEST_SIZES = {
    "hellaswag_k6_cluster_merged_0": 1044, "hellaswag_k6_cluster_merged_1": 2999,
    "hellaswag_k6_cluster_merged_2": 1596, "hellaswag_k6_cluster_merged_3": 1529,
    "hellaswag_k6_cluster_merged_4": 1080, "hellaswag_k6_cluster_merged_5": 1794,
    "hellaswag_k8_cluster_merged_0": 870, "hellaswag_k8_cluster_merged_1": 1307,
    "hellaswag_k8_cluster_merged_2": 1534, "hellaswag_k8_cluster_merged_3": 369,
    "hellaswag_k8_cluster_merged_4": 1007, "hellaswag_k8_cluster_merged_5": 1921,
    "hellaswag_k8_cluster_merged_6": 1369, "hellaswag_k8_cluster_merged_7": 1665,
    "hellaswag_k10_cluster_merged_0": 1901, "hellaswag_k10_cluster_merged_1": 1372,
    "hellaswag_k10_cluster_merged_2": 1179, "hellaswag_k10_cluster_merged_3": 1631,
    "hellaswag_k10_cluster_merged_4": 1286, "hellaswag_k10_cluster_merged_5": 953,
    "hellaswag_k10_cluster_merged_6": 289, "hellaswag_k10_cluster_merged_7": 540,
    "hellaswag_k10_cluster_merged_8": 341, "hellaswag_k10_cluster_merged_9": 550,
    "hellaswag_k16_cluster_merged_0": 580, "hellaswag_k16_cluster_merged_1": 238,
    "hellaswag_k16_cluster_merged_2": 1382, "hellaswag_k16_cluster_merged_3": 319,
    "hellaswag_k16_cluster_merged_4": 500, "hellaswag_k16_cluster_merged_5": 902,
    "hellaswag_k16_cluster_merged_6": 582, "hellaswag_k16_cluster_merged_7": 764,
    "hellaswag_k16_cluster_merged_8": 307, "hellaswag_k16_cluster_merged_9": 126,
    "hellaswag_k16_cluster_merged_10": 1450, "hellaswag_k16_cluster_merged_11": 926,
    "hellaswag_k16_cluster_merged_12": 495, "hellaswag_k16_cluster_merged_13": 200,
    "hellaswag_k16_cluster_merged_14": 437, "hellaswag_k16_cluster_merged_15": 834,
    "hellaswag_cluster_merged_0": 1044, "hellaswag_cluster_merged_1": 2999,
    "hellaswag_cluster_merged_2": 1596, "hellaswag_cluster_merged_3": 1529,
    "hellaswag_cluster_merged_4": 1080, "hellaswag_cluster_merged_5": 1794,
}

AVAILABLE_TASK_RUNS = list(TASK_SPECS)

SELECTED_MODELS = list(AVAILABLE_MODELS)
SELECTED_TASK_RUNS = list(AVAILABLE_TASK_RUNS)

MODEL_LABELS = {
    model: spec["label"]
    for model, spec in MODEL_SPECS.items()
    if spec.get("label")
}

DEFAULT_OUTPUT_SUBDIR = "prune_eval_tables_0319_ckpt0"

# Collect all known variant suffixes from MODEL_SPECS for auto-discovery.
_ALL_VARIANT_SUFFIXES: List[str] = sorted(
    {
        v["suffix"]
        for spec in MODEL_SPECS.values()
        for v in spec.get("variants", [])
    },
    key=len,
    reverse=True,  # longest first for greedy stripping
)


def _get_model_variants(model_name: str) -> List[Tuple[str, str]]:
    """Return (suffix, label) pairs for the variants to scan for a model."""
    spec = MODEL_SPECS.get(model_name)
    if spec is not None:
        return [(v["suffix"], v["label"]) for v in spec.get("variants", [])]
    return []

# ============================================================================
# END CONFIGURATION
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metric tables from prune_evals checkpoint-0."
    )
    parser.add_argument(
        "--prune-evals-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "prune_evals_0313",
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
    return any(task_name.endswith(suffix) for suffix in _ALL_VARIANT_SUFFIXES)


def _strip_variant_suffix(task_name: str) -> str:
    """Strip the longest matching variant suffix to get the base task name."""
    for suffix in _ALL_VARIANT_SUFFIXES:  # already sorted longest-first
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
            if t.is_dir() and t.name != "original_model"
        }
    )
    return models, task_runs


def _load_baseline_metric(
    prune_evals_root: Path,
    model_name: str,
    task_run: str,
    metric_key: str,
) -> Optional[float]:
    """Load the baseline (unpruned, unfinetuned) metric for a model+task.

    Looks in ``<model>/original_model/<task>/results/checkpoint-0/``.
    """
    ckpt_dir = (
        prune_evals_root / model_name / "original_model" / task_run
        / "results" / "checkpoint-0"
    )
    if not ckpt_dir.is_dir():
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
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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


def _read_checkpoint0_metric(
    task_dir: Path, metric_key: str
) -> Optional[float]:
    """Read a single metric value from checkpoint-0 in task_dir/results."""
    ckpt_dir = task_dir / "results" / "checkpoint-0"
    if not ckpt_dir.is_dir():
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
    """Build a models x tasks table of checkpoint-0 metric values.

    Baseline-only models read from original_model/<task>/results/checkpoint-0/.
    Variant models read from <task><suffix>/results/checkpoint-0/.
    """
    rows: Dict[str, Dict[str, Optional[float]]] = {}

    for model_name in model_names:
        model_dir = prune_evals_root / model_name
        if not model_dir.is_dir():
            continue
        model_label = MODEL_LABELS.get(model_name, model_name)

        spec = MODEL_SPECS.get(model_name)

        # Baseline-only models: read from original_model path
        if spec is not None and spec.get("baseline"):
            for task_run in task_runs:
                val = _load_baseline_metric(
                    prune_evals_root, model_name, task_run, metric_key
                )
                if val is not None:
                    rows.setdefault(model_label, {})[task_run] = val
                else:
                    print(
                        f"[WARN] Baseline data missing for model {model_label!r}, "
                        f"task={task_run!r}, metric={metric_key!r}"
                    )
            continue

        # Non-baseline models: read checkpoint-0 from variant task dirs
        for task_run in task_runs:
            for suffix, label_mod in _get_model_variants(model_name):
                variant_dir = model_dir / (task_run + suffix)
                if not variant_dir.is_dir():
                    continue
                variant_label = model_label + " " + label_mod
                val = _read_checkpoint0_metric(variant_dir, metric_key)
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
    """Add mmlu_avg, mmlu_pro_avg, and filtered variants, placed at the front.

    Models missing any sub-task within a group get NaN for that group's average
    to avoid misleading partial averages.
    """
    avg_cols_added: List[str] = []

    # --- MMLU averages ---
    mmlu_cols = [c for c in df.columns if c in MMLU_SUBTASKS]
    if mmlu_cols:
        for model_name in df.index:
            missing = [c for c in mmlu_cols if pd.isna(df.loc[model_name, c])]
            if missing:
                print(
                    f"[WARN] Model {model_name!r} is missing {len(missing)}/{len(mmlu_cols)} "
                    f"MMLU sub-task(s): {missing} — mmlu_avg will be NaN"
                )

        df["mmlu_avg"] = df[mmlu_cols].mean(axis=1, skipna=False)
        avg_cols_added.append("mmlu_avg")

        for col_name, excluded in MMLU_EXCLUDE_SETS.items():
            filtered = [c for c in mmlu_cols if c not in excluded]
            if filtered and len(filtered) < len(mmlu_cols):
                df[col_name] = df[filtered].mean(axis=1, skipna=False)
                avg_cols_added.append(col_name)

    # --- MMLU-Pro averages ---
    mmlu_pro_cols = [c for c in df.columns if c in MMLU_PRO_SUBTASKS]
    if mmlu_pro_cols:
        for model_name in df.index:
            missing = [c for c in mmlu_pro_cols if pd.isna(df.loc[model_name, c])]
            if missing:
                print(
                    f"[WARN] Model {model_name!r} is missing {len(missing)}/{len(mmlu_pro_cols)} "
                    f"MMLU-Pro sub-task(s): {missing} — mmlu_pro_avg will be NaN"
                )

        df["mmlu_pro_avg"] = df[mmlu_pro_cols].mean(axis=1, skipna=False)
        avg_cols_added.append("mmlu_pro_avg")

        pro_no_other = [c for c in mmlu_pro_cols if c != "mmlu_pro_other"]
        if pro_no_other and len(pro_no_other) < len(mmlu_pro_cols):
            df["mmlu_pro_avg_no_other"] = df[pro_no_other].mean(axis=1, skipna=False)
            avg_cols_added.append("mmlu_pro_avg_no_other")

    # --- MMLU-Pro-Merged averages ---
    mmlu_pro_merged_cols = [c for c in df.columns if c in MMLU_PRO_MERGED_SUBTASKS]
    if mmlu_pro_merged_cols:
        for model_name in df.index:
            missing = [c for c in mmlu_pro_merged_cols if pd.isna(df.loc[model_name, c])]
            if missing:
                print(
                    f"[WARN] Model {model_name!r} is missing {len(missing)}/{len(mmlu_pro_merged_cols)} "
                    f"MMLU-Pro-Merged sub-task(s): {missing} — mmlu_pro_merged_avg will be NaN"
                )

        df["mmlu_pro_merged_avg"] = df[mmlu_pro_merged_cols].mean(axis=1, skipna=False)
        avg_cols_added.append("mmlu_pro_merged_avg")

        merged_no_other = [c for c in mmlu_pro_merged_cols if c != "mmlu_pro_merged_other"]
        if merged_no_other and len(merged_no_other) < len(mmlu_pro_merged_cols):
            df["mmlu_pro_merged_avg_no_other"] = df[merged_no_other].mean(axis=1, skipna=False)
            avg_cols_added.append("mmlu_pro_merged_avg_no_other")

    # --- HellaSwag cluster averages (weighted by test set size per cluster) ---
    for k_val, k_subtasks in HELLASWAG_CLUSTER_SUBTASKS_BY_K.items():
        k_cols = [c for c in df.columns if c in k_subtasks]
        if not k_cols:
            continue
        avg_name = f"hellaswag_k{k_val}_cluster_avg"
        for model_name in df.index:
            missing = [c for c in k_cols if pd.isna(df.loc[model_name, c])]
            if missing:
                print(
                    f"[WARN] Model {model_name!r} is missing {len(missing)}/{len(k_cols)} "
                    f"HellaSwag k={k_val} cluster sub-task(s): {missing} — {avg_name} will be NaN"
                )
        weights = np.array([HELLASWAG_CLUSTER_TEST_SIZES[c] for c in k_cols], dtype=float)
        weights /= weights.sum()
        cluster_vals = df[k_cols].values
        has_nan = np.isnan(cluster_vals).any(axis=1)
        weighted = (cluster_vals * weights[None, :]).sum(axis=1)
        weighted[has_nan] = np.nan
        df[avg_name] = weighted
        avg_cols_added.append(avg_name)

    # Legacy k=6 alias
    hellaswag_legacy_cols = [c for c in df.columns if c in HELLASWAG_CLUSTER_SUBTASKS]
    if hellaswag_legacy_cols:
        weights = np.array([HELLASWAG_CLUSTER_TEST_SIZES[c] for c in hellaswag_legacy_cols], dtype=float)
        weights /= weights.sum()
        cluster_vals = df[hellaswag_legacy_cols].values
        has_nan = np.isnan(cluster_vals).any(axis=1)
        weighted = (cluster_vals * weights[None, :]).sum(axis=1)
        weighted[has_nan] = np.nan
        df["hellaswag_cluster_avg"] = weighted
        avg_cols_added.append("hellaswag_cluster_avg")

    if not avg_cols_added:
        return df

    other_cols = [c for c in df.columns if c not in avg_cols_added]
    df = df[avg_cols_added + other_cols]
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

    # Baseline-only models may not appear in auto-discovery (no variant task dirs).
    # Add them if they exist on disk.
    for model_name in selected_models:
        spec = MODEL_SPECS.get(model_name)
        if spec is not None and spec.get("baseline"):
            model_dir = args.prune_evals_root / model_name
            if model_dir.is_dir() and model_name not in available_models:
                available_models.append(model_name)

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
        metric_dir = base_output_dir / safe_metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        # --- Define output slices ---
        MC9_TASKS = [
            "arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag",
            "openbookqa", "piqa", "socialiqa", "winogrande",
        ]
        mc9_cols = [c for c in MC9_TASKS if c in df.columns]
        if mc9_cols:
            df["mc9_avg"] = df[mc9_cols].mean(axis=1, skipna=False)

        # 1. Aggregate overview
        agg_cols = [c for c in [
            "mc9_avg",
            "mmlu_avg", "mmlu_avg_no_other", "mmlu_avg_no_other_hist_phil",
            "mmlu_pro_avg", "mmlu_pro_avg_no_other",
            "mmlu_pro_merged_avg", "mmlu_pro_merged_avg_no_other",
        ] if c in df.columns]
        other_tasks = [c for c in df.columns
                       if c not in agg_cols
                       and not c.startswith("mmlu_")
                       and not c.startswith("hellaswag_")
                       and c not in MC9_TASKS]
        agg_cols += other_tasks

        # 2. MC9 detail
        mc9_cols_all = [c for c in ["mc9_avg"] + MC9_TASKS if c in df.columns]

        # 3. MMLU detail
        mmlu_avg_cols = [c for c in df.columns if c.startswith("mmlu_avg")]
        mmlu_topic_cols = [c for c in df.columns
                          if c.startswith("mmlu_") and not c.startswith("mmlu_pro")
                          and not c.startswith("mmlu_avg")]
        mmlu_cols_all = mmlu_avg_cols + mmlu_topic_cols

        # 4. MMLU-Pro detail (not n* or merged)
        mmlu_pro_avg_cols = [c for c in df.columns if c.startswith("mmlu_pro_avg")]
        mmlu_pro_topic_cols = [c for c in df.columns
                               if c.startswith("mmlu_pro_") and not c.startswith("mmlu_pro_merged")
                               and not c.startswith("mmlu_pro_avg")]
        mmlu_pro_cols_all = mmlu_pro_avg_cols + mmlu_pro_topic_cols

        # 5. MMLU-Pro-Merged detail (not n*)
        mmlu_pm_avg_cols = [c for c in df.columns
                            if c.startswith("mmlu_pro_merged_avg")]
        mmlu_pm_topic_cols = [c for c in df.columns
                              if c.startswith("mmlu_pro_merged_")
                              and not c.startswith("mmlu_pro_merged_avg")
                              and not any(c.startswith(f"mmlu_pro_merged_n{n}") for n in [50, 100, 200])]
        mmlu_pm_cols_all = mmlu_pm_avg_cols + mmlu_pm_topic_cols

        # 6. HellaSwag clusters
        hella_lead = [c for c in ["hellaswag_merged", "hellaswag_cluster_avg"] if c in df.columns]
        hella_k_avgs = [c for c in [
            "hellaswag_k8_cluster_avg", "hellaswag_k10_cluster_avg", "hellaswag_k16_cluster_avg",
        ] if c in df.columns]
        hella_cols_all = hella_lead + hella_k_avgs

        # 7. MMLU-Pro-Merged N-val
        nval_cols_all = [c for c in [
            "mmlu_pro_merged_avg",
            "mmlu_pro_merged_n50_avg",
            "mmlu_pro_merged_n100_avg",
            "mmlu_pro_merged_n200_avg",
            "mmlu_pro_merged_avg_no_other",
            "mmlu_pro_merged_n50_avg_no_other",
            "mmlu_pro_merged_n100_avg_no_other",
            "mmlu_pro_merged_n200_avg_no_other",
        ] if c in df.columns]

        slices = {
            "aggregate": agg_cols,
            "mc9": mc9_cols_all,
            "mmlu": mmlu_cols_all,
            "mmlu_pro": mmlu_pro_cols_all,
            "mmlu_pro_merged": mmlu_pm_cols_all,
            "hellaswag_clusters": hella_cols_all,
            "mmlu_pro_merged_nval": nval_cols_all,
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
