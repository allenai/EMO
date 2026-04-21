#!/usr/bin/env python3
"""Replicate ``validation_sample_ablation.tex`` as a CSV.

Reads the fine-tuned nprune ablation table produced by
``scripts/ryanwang/pruning_plots/get_table_scores_nprune_ablation.py`` and
writes:

    claude_outputs/prune_plots/validation_sample_ablation.csv

Source CSV expected layout (rows × columns):
    rows    : "{model_label} / {task_name}"
              e.g. "moe 1T + anneal / mmlu_merged"
    columns : "keepk_{k} ({nprune_tag})"
              e.g. "keepk_8 (5)", "keepk_128 (All)"

Output CSV layout (one row per (model, task), columns are keepk × nprune):
    Model, Task,
    8 Experts (5), 8 Experts (10), 8 Experts (100), 8 Experts (All),
    16 Experts (5), ..., 128 Experts (All)
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
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "validation_sample_ablation.csv"
)

# Column layout: each expert count gets its own list of nprune sub-columns.
# keepk 128 keeps all experts (no pruning happens), so only "All" is meaningful.
KEEPK_NPRUNE_TAGS: List[Tuple[int, List[str]]] = [
    (8, ["Random", "1", "5", "10", "100", "All"]),
    (16, ["Random", "1", "5", "10", "100", "All"]),
    (32, ["Random", "1", "5", "10", "100", "All"]),
    (128, ["All"]),
]

# Row layout: (display_model_name, display_task_name, source_row_key).
ROWS: List[Tuple[str, str, str]] = [
    ("Reg. MoE", "MMLU", "moe 1T + anneal / mmlu_merged"),
    ("Reg. MoE", "MMLU Pro", "moe 1T + anneal / mmlu_pro_merged"),
    ("Reg. MoE", "GSM8K", "moe 1T + anneal / gsm8k"),
    ("FlexMoE", "MMLU", "specialized moe 1T + anneal / mmlu_merged"),
    ("FlexMoE", "MMLU Pro", "specialized moe 1T + anneal / mmlu_pro_merged"),
    ("FlexMoE", "GSM8K", "specialized moe 1T + anneal / gsm8k"),
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
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination CSV path.",
    )
    return parser.parse_args()


def _pct(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}"


def main() -> None:
    args = parse_args()
    if not args.input_path.is_file():
        raise FileNotFoundError(f"Missing source CSV: {args.input_path}")

    src = pd.read_csv(args.input_path).set_index("model / task")

    records: List[Dict[str, str]] = []
    for model_name, task_name, src_key in ROWS:
        row: Dict[str, str] = {"Model": model_name, "Task": task_name}
        for k, tags in KEEPK_NPRUNE_TAGS:
            for tag in tags:
                col_out = f"{k} Experts ({tag})"
                col_src = f"keepk_{k} ({tag})"
                if src_key in src.index and col_src in src.columns:
                    row[col_out] = _pct(src.at[src_key, col_src])
                else:
                    row[col_out] = ""
        records.append(row)

    df = pd.DataFrame(records)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"Wrote {args.output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
