#!/usr/bin/env python3
"""Replicate ``expert_selection_method.tex`` as a CSV.

Reads fine-tuning (last-checkpoint) metrics that are already produced by
``get_table_scores_prune_evals_final.py`` and writes:

    claude_outputs/prune_plots/expert_selection_method.csv

This table uses the Easy-EP prunemode columns ``(ep)`` instead of the
layerwise ``(lw)`` columns used in ``main_results_table``. The two 1T
annealed models are compared:

    Reg. MoE      / 8              -> "moe 1T + anneal (keepk 8)"
    Reg. MoE      / 16             -> "moe 1T + anneal (keepk 16)"
    Reg. MoE      / 32             -> "moe 1T + anneal (keepk 32)"
    Reg. MoE      / 128 (trained)  -> "moe 1T + anneal (keepk 128)"
    FlexMoE       / 8              -> "specialized moe 1T + anneal (keepk 8)"
    FlexMoE       / 16             -> "specialized moe 1T + anneal (keepk 16)"
    FlexMoE       / 32             -> "specialized moe 1T + anneal (keepk 32)"
    FlexMoE       / 128 (trained)  -> "specialized moe 1T + anneal (keepk 128)"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "expert_selection_method.csv"
)

# (display_name, expert_count_label, model_key)
ROWS: List[Tuple[str, str, str]] = [
    ("Reg. MoE", "8", "moe 1T + anneal (keepk 8)"),
    ("Reg. MoE", "16", "moe 1T + anneal (keepk 16)"),
    ("Reg. MoE", "32", "moe 1T + anneal (keepk 32)"),
    ("Reg. MoE", "128 (trained)", "moe 1T + anneal (keepk 128)"),
    ("FlexMoE", "8", "specialized moe 1T + anneal (keepk 8)"),
    ("FlexMoE", "16", "specialized moe 1T + anneal (keepk 16)"),
    ("FlexMoE", "32", "specialized moe 1T + anneal (keepk 32)"),
    ("FlexMoE", "128 (trained)", "specialized moe 1T + anneal (keepk 128)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finetune-dir",
        type=Path,
        default=None,
        help=(
            "Directory with last-checkpoint (fine-tuning) generated tables. "
            "If unset, uses <repo>/claude_outputs/prune_plots/prune_eval_tables_final "
            "with a fallback to the newest prune_eval_tables_final_*backup* sibling."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination CSV path.",
    )
    return parser.parse_args()


def _resolve_finetune_dir(arg_dir: Optional[Path]) -> Path:
    if arg_dir is not None:
        if not arg_dir.is_dir():
            raise FileNotFoundError(f"--finetune-dir does not exist: {arg_dir}")
        return arg_dir

    base = REPO_ROOT / "claude_outputs" / "prune_plots"
    primary = base / "prune_eval_tables_final"
    if (primary / "acc_raw" / "aggregate.csv").is_file():
        return primary

    backups = sorted(
        [p for p in base.glob("prune_eval_tables_final_*backup*") if p.is_dir()],
        reverse=True,
    )
    for candidate in backups:
        if (candidate / "acc_raw" / "aggregate.csv").is_file():
            print(f"[INFO] Using backup fine-tuning dir: {candidate}")
            return candidate

    raise FileNotFoundError(
        "Could not locate a fine-tuning table directory. Expected "
        f"{primary} or a prune_eval_tables_final_*backup* sibling."
    )


def _load(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing source CSV: {csv_path}")
    return pd.read_csv(csv_path).set_index("model")


def _pct(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}"


def main() -> None:
    args = parse_args()
    ft_dir = _resolve_finetune_dir(args.finetune_dir)

    ft_agg = _load(ft_dir / "acc_raw" / "aggregate.csv")
    ft_gsm = _load(ft_dir / "exact_match" / "gsm8k.csv")

    def lookup(df: pd.DataFrame, key: str, col: str) -> Optional[float]:
        if key not in df.index:
            return None
        val = df.at[key, col]
        return None if pd.isna(val) else float(val)

    records: List[Dict[str, str]] = []
    for name, experts, key in ROWS:
        row = {
            "": name,
            "# Total Experts": experts,
            "MMLU (ft)": _pct(lookup(ft_agg, key, "mmlu_merged_avg_no_other (ep)")),
            "MMLU Pro (ft)": _pct(
                lookup(ft_agg, key, "mmlu_pro_merged_avg_no_other (ep)")
            ),
            "GSM8K (ft)": _pct(
                lookup(ft_gsm, key, "gsm8k_generation_8shot_merged (ep)")
            ),
        }
        records.append(row)

    df = pd.DataFrame(records)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"Wrote {args.output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
