#!/usr/bin/env python3
"""Replicate ``validation_sample_ablation.tex`` as a CSV.

Reads the fine-tuned nprune ablation table produced by
``scripts/ryanwang/pruning_plots/get_table_scores_nprune_ablation.py`` and
writes:

    claude_outputs/prune_plots/validation_sample_ablation.csv

Source CSV expected layout (rows × columns):
    rows    : "{model_label} ({prunemode}) / {task_name}"
              e.g. "moe 1T + anneal (Router) / mmlu_merged"
                   "moe 1T + anneal (Easy-EP) / mmlu_merged"
    columns : "keepk_{k} ({nprune_tag})"
              e.g. "keepk_8 (5)", "keepk_128 (All)"

Output CSV layout — one row per (display_model, prunemode, task), columns are
keepk × nprune:
    Model, Prunemode, Task,
    8 Experts (Random), ..., 8 Experts (All),
    16 Experts (Random), ..., 128 Experts (All)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_INPUT_PATH = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "nprune_ablation_tables" / "nprune_ablation.csv"
)
DEFAULT_CKPT0_INPUT_PATH = (
    REPO_ROOT
    / "claude_outputs"
    / "prune_plots"
    / "nprune_ablation_tables_ckpt0"
    / "nprune_ablation.csv"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "validation_sample_ablation.csv"
)
DEFAULT_CKPT0_OUTPUT_PATH = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "validation_sample_ablation_ckpt0.csv"
)

# Column layout: each expert count gets its own list of nprune sub-columns.
# keepk 128 keeps all experts (no pruning happens), so only "All" is meaningful.
KEEPK_NPRUNE_TAGS: List[Tuple[int, List[str]]] = [
    (8, ["1", "5", "10", "100", "All"]),
    (16, ["1", "5", "10", "100", "All"]),
    (32, ["1", "5", "10", "100", "All"]),
    (128, ["All"]),
]

# Row layout: (display_model_name, prunemode, display_task_name, source_row_key).
# This table focuses on FlexMoE only and ablates the *shot count used during
# pruning calibration and eval*. Three row groups, each with MMLU / MMLU Pro /
# GSM8K:
#   1) Router          — pruning + eval at task-default shots
#   2) Router (0-shot) — pruning + eval both 0-shot      (_pshots-0_eshots-0)
#   3) Router (e0)     — pruning at task default, eval 0-shot (_eshots-0)
ROWS: List[Tuple[str, str, str, str]] = [
    ("FlexMoE", "Router",          "MMLU",     "specialized moe 1T + anneal (Router) / mmlu_merged"),
    ("FlexMoE", "Router",          "MMLU Pro", "specialized moe 1T + anneal (Router) / mmlu_pro_merged"),
    ("FlexMoE", "Router",          "GSM8K",    "specialized moe 1T + anneal (Router) / gsm8k"),
    ("FlexMoE", "Router (0-shot)", "MMLU",     "specialized moe 1T + anneal (Router (0-shot)) / mmlu_merged"),
    ("FlexMoE", "Router (0-shot)", "MMLU Pro", "specialized moe 1T + anneal (Router (0-shot)) / mmlu_pro_merged"),
    ("FlexMoE", "Router (0-shot)", "GSM8K",    "specialized moe 1T + anneal (Router (0-shot)) / gsm8k"),
    ("FlexMoE", "Router (e0)",     "MMLU",     "specialized moe 1T + anneal (Router (e0)) / mmlu_merged"),
    ("FlexMoE", "Router (e0)",     "MMLU Pro", "specialized moe 1T + anneal (Router (e0)) / mmlu_pro_merged"),
    ("FlexMoE", "Router (e0)",     "GSM8K",    "specialized moe 1T + anneal (Router (e0)) / gsm8k"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Source nprune_ablation.csv (fine-tuned).",
    )
    parser.add_argument(
        "--ckpt0-input-path",
        type=Path,
        default=DEFAULT_CKPT0_INPUT_PATH,
        help="Source nprune_ablation.csv for ckpt0 (pre-finetune).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination CSV path for the fine-tuned table.",
    )
    parser.add_argument(
        "--ckpt0-output-path",
        type=Path,
        default=DEFAULT_CKPT0_OUTPUT_PATH,
        help="Destination CSV path for the ckpt0 (pre-finetune) table.",
    )
    return parser.parse_args()


def _pct(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}"


def _build_table(src: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, str]] = []
    for model_name, prunemode, task_name, src_key in ROWS:
        row: Dict[str, str] = {
            "Model": model_name,
            "Prunemode": prunemode,
            "Task": task_name,
        }
        for k, tags in KEEPK_NPRUNE_TAGS:
            for tag in tags:
                col_out = f"{k} Experts ({tag})"
                col_src = f"keepk_{k} ({tag})"
                if src_key in src.index and col_src in src.columns:
                    row[col_out] = _pct(src.at[src_key, col_src])
                else:
                    row[col_out] = ""
        records.append(row)
    return pd.DataFrame(records)


def _write_table(src_path: Path, out_path: Path, label: str) -> None:
    if not src_path.is_file():
        print(f"[WARN] Skipping {label}: missing source CSV {src_path}")
        return
    src = pd.read_csv(src_path).set_index("model / task")
    df = _build_table(src)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({label})")
    print(df.to_string(index=False))
    print()


def main() -> None:
    args = parse_args()
    _write_table(args.input_path, args.output_path, "fine-tuned")
    _write_table(args.ckpt0_input_path, args.ckpt0_output_path, "ckpt0")


if __name__ == "__main__":
    main()
