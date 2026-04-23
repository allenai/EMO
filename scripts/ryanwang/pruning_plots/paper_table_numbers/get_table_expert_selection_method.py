#!/usr/bin/env python3
"""Replicate ``expert_selection_method.tex`` as CSVs.

Reads metrics produced by ``get_table_scores_prune_evals_final.py`` and writes
two CSVs (one per checkpoint mode):

    claude_outputs/prune_plots/expert_selection_method.csv        (fine-tuned)
    claude_outputs/prune_plots/expert_selection_method_ckpt0.csv  (pre-finetune)

For each metric (MMLU, MMLU Pro, GSM8K), reports three columns:
  * "Router"  -> layerwise prunemode (``(lw)`` columns from the source CSVs)
  * "Easy-EP" -> Easy-EP prunemode   (``(ep)`` columns from the source CSVs)
  * "Random"  -> random prunemode    (``(rd)`` columns from the source CSVs)

The two 1T annealed models are compared:

    Reg. MoE      / 8              -> "moe 1T + anneal (keepk 8)"
    Reg. MoE      / 16             -> "moe 1T + anneal (keepk 16)"
    Reg. MoE      / 32             -> "moe 1T + anneal (keepk 32)"
    Reg. MoE      / 64             -> "moe 1T + anneal (keepk 64)"
    Reg. MoE      / 128 (trained)  -> "moe 1T + anneal (keepk 128)"
    FlexMoE       / 8              -> "specialized moe 1T + anneal (keepk 8)"
    FlexMoE       / 16             -> "specialized moe 1T + anneal (keepk 16)"
    FlexMoE       / 32             -> "specialized moe 1T + anneal (keepk 32)"
    FlexMoE       / 64             -> "specialized moe 1T + anneal (keepk 64)"
    FlexMoE       / 128 (trained)  -> "specialized moe 1T + anneal (keepk 128)"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_FT_OUTPUT_PATH = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "expert_selection_method.csv"
)
DEFAULT_CKPT0_OUTPUT_PATH = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "expert_selection_method_ckpt0.csv"
)
DEFAULT_CKPT0_INFERENCE_DIR = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "prune_eval_tables_final_ckpt0"
)

# (display_name, expert_count_label, model_key)
ROWS: List[Tuple[str, str, str]] = [
    ("Reg. MoE", "8", "moe 1T + anneal (keepk 8)"),
    ("Reg. MoE", "16", "moe 1T + anneal (keepk 16)"),
    ("Reg. MoE", "32", "moe 1T + anneal (keepk 32)"),
    ("Reg. MoE", "64", "moe 1T + anneal (keepk 64)"),
    ("Reg. MoE", "128 (trained)", "moe 1T + anneal (keepk 128)"),
    ("FlexMoE", "8", "specialized moe 1T + anneal (keepk 8)"),
    ("FlexMoE", "16", "specialized moe 1T + anneal (keepk 16)"),
    ("FlexMoE", "32", "specialized moe 1T + anneal (keepk 32)"),
    ("FlexMoE", "64", "specialized moe 1T + anneal (keepk 64)"),
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
        "--inference-dir",
        type=Path,
        default=DEFAULT_CKPT0_INFERENCE_DIR,
        help="Directory with ckpt0 (pre-finetune) generated tables.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_FT_OUTPUT_PATH,
        help="Destination CSV path for the fine-tuned table.",
    )
    parser.add_argument(
        "--ckpt0-output-path",
        type=Path,
        default=DEFAULT_CKPT0_OUTPUT_PATH,
        help="Destination CSV path for the ckpt0 (pre-finetune) table.",
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


def _build_table(
    agg_df: pd.DataFrame, gsm_df: pd.DataFrame
) -> pd.DataFrame:
    def lookup(df: pd.DataFrame, key: str, col: str) -> Optional[float]:
        if key not in df.index:
            return None
        val = df.at[key, col]
        return None if pd.isna(val) else float(val)

    def lookup_random(df: pd.DataFrame, key: str, base: str) -> Optional[float]:
        """Random prunemode value. Falls back to the (lw) value for keepk=128
        rows, since no actual pruning happens when all experts are kept."""
        rd = lookup(df, key, f"{base} (rd)")
        if rd is not None:
            return rd
        if key and "keepk 128" in key:
            return lookup(df, key, f"{base} (lw)")
        return None

    records: List[Dict[str, str]] = []
    for name, experts, key in ROWS:
        row = {
            "": name,
            "# Total Experts": experts,
            "MMLU Random":  _pct(lookup_random(agg_df, key, "mmlu_merged_avg_no_other")),
            "MMLU Router":  _pct(lookup(agg_df, key, "mmlu_merged_avg_no_other (lw)")),
            "MMLU Easy-EP": _pct(lookup(agg_df, key, "mmlu_merged_avg_no_other (ep)")),
            "MMLU Pro Random":  _pct(lookup_random(agg_df, key, "mmlu_pro_merged_avg_no_other")),
            "MMLU Pro Router":  _pct(lookup(agg_df, key, "mmlu_pro_merged_avg_no_other (lw)")),
            "MMLU Pro Easy-EP": _pct(lookup(agg_df, key, "mmlu_pro_merged_avg_no_other (ep)")),
            "GSM8K Random":  _pct(lookup_random(gsm_df, key, "gsm8k_generation_8shot_merged")),
            "GSM8K Router":  _pct(lookup(gsm_df, key, "gsm8k_generation_8shot_merged (lw)")),
            "GSM8K Easy-EP": _pct(lookup(gsm_df, key, "gsm8k_generation_8shot_merged (ep)")),
        }
        records.append(row)
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()

    # --- Fine-tuned table (last-checkpoint) ---
    ft_dir = _resolve_finetune_dir(args.finetune_dir)
    ft_agg = _load(ft_dir / "acc_raw" / "aggregate.csv")
    ft_gsm = _load(ft_dir / "exact_match" / "gsm8k.csv")

    df_ft = _build_table(ft_agg, ft_gsm)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df_ft.to_csv(args.output_path, index=False)
    print(f"Wrote {args.output_path}")
    print(df_ft.to_string(index=False))

    # --- Ckpt0 (pre-finetune) table ---
    if not args.inference_dir.is_dir():
        print(f"[WARN] Skipping ckpt0 output — missing dir: {args.inference_dir}")
        return
    ckpt0_agg = _load(args.inference_dir / "acc_raw" / "aggregate.csv")
    ckpt0_gsm = _load(args.inference_dir / "exact_match" / "gsm8k.csv")

    df_ckpt0 = _build_table(ckpt0_agg, ckpt0_gsm)
    args.ckpt0_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_ckpt0.to_csv(args.ckpt0_output_path, index=False)
    print(f"\nWrote {args.ckpt0_output_path}")
    print(df_ckpt0.to_string(index=False))


if __name__ == "__main__":
    main()
