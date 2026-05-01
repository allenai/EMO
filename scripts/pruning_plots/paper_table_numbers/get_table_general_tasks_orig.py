#!/usr/bin/env python3
"""Replicate ``general_tasks_orig.tex`` as a CSV.

Reads inference-only (no-finetune / checkpoint-0) metrics that are already
produced by ``get_table_scores_prune_evals_final.py`` and writes:

    claude_outputs/prune_plots/general_tasks_orig.csv

Row mapping (from the .tex source header):

    OLMoE^dagger   5T    -> (no data)
    Dense          1T    -> (no data)
    Reg. MoE       1T    -> "moe 1T + anneal (keepk 128)"
    FlexMoE (Ours) 1T    -> "specialized moe 1T + anneal (keepk 128)"
    Dense          130B  -> "dense"
    Reg. MoE       130B  -> "moe (keepk 128)"
    FlexMoE (Ours) 130B  -> "specialized moe + globallb + 1shardexp + randpool (keepk 128)"

All numbers come from the ``(lw)`` (layerwise prunemode) columns, scaled to
percentages with one decimal place — matching the .tex format.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_INFERENCE_DIR = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "prune_eval_tables_final_ckpt0"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "claude_outputs" / "prune_plots" / "general_tasks_orig.csv"

# (display_name, train_tokens, model_key_in_csvs_or_None)
ROWS = [
    ("OLMoE^dagger", "5T", None),
    ("Dense", "1T", None),
    ("Reg. MoE", "1T", "moe 1T + anneal (keepk 128)"),
    ("FlexMoE (Ours)", "1T", "specialized moe 1T + anneal (keepk 128)"),
    ("Dense", "130B", "dense"),
    ("Reg. MoE", "130B", "moe (keepk 128)"),
    (
        "FlexMoE (Ours)",
        "130B",
        "specialized moe + globallb + 1shardexp + randpool (keepk 128)",
    ),
]

# (output_column, source_csv_relpath, column_in_csv)
METRIC_SOURCES = [
    ("MC9", "acc_raw/aggregate.csv", "mc9_avg (lw)"),
    ("Gen5", "f1/gen5.csv", "gen5_avg (lw)"),
    ("MMLU", "acc_raw/aggregate.csv", "mmlu_merged_avg_no_other (lw)"),
    ("MMLU Pro", "acc_raw/aggregate.csv", "mmlu_pro_merged_avg_no_other (lw)"),
    ("GSM8K", "exact_match/gsm8k.csv", "gsm8k_generation_8shot_merged (lw)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=DEFAULT_INFERENCE_DIR,
        help="Directory containing the ckpt0 (inference-only) generated tables.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination CSV path.",
    )
    return parser.parse_args()


def _load_indexed(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing source CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    return df.set_index("model")


def _pct(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}"


def main() -> None:
    args = parse_args()

    sources = {
        relpath: _load_indexed(args.inference_dir / relpath) for _, relpath, _ in METRIC_SOURCES
    }

    records = []
    for name, tokens, key in ROWS:
        row = {"": name, "# train tokens": tokens}
        for out_col, relpath, source_col in METRIC_SOURCES:
            if key is None or key not in sources[relpath].index:
                row[out_col] = ""
                continue
            row[out_col] = _pct(sources[relpath].at[key, source_col])
        records.append(row)

    df = pd.DataFrame(records)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"Wrote {args.output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
