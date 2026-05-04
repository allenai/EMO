#!/usr/bin/env python3
"""Replicate ``main_results_table.tex`` as a CSV.

Reads both the inference (no-finetune / checkpoint-0) and the fine-tuning
(last-checkpoint) tables produced by ``get_table_scores_prune_evals_final.py``
and writes:

    claude_outputs/prune_plots/main_results_table.csv

Row mapping (follows the .tex source, including the blank ``Reg. MoE / 16
trained`` row so the CSV has the same shape):

    Dense^dagger  / 8 (trained)    -> "dense"
    Reg. MoE      / 16 (trained)   -> (no data available)
    Reg. MoE      / 32 (trained)   -> "moe_small"
    Reg. MoE      / 8              -> "moe (keepk 8)"
    Reg. MoE      / 16             -> "moe (keepk 16)"
    Reg. MoE      / 32             -> "moe (keepk 32)"
    Reg. MoE      / 64             -> "moe (keepk 64)"
    Reg. MoE      / 128 (trained)  -> "moe (keepk 128)"
    Emo       / 8              -> "specialized moe + globallb + 1shardexp + randpool (keepk 8)"
    Emo       / 16             -> "... (keepk 16)"
    Emo       / 32             -> "... (keepk 32)"
    Emo       / 64             -> "... (keepk 64)"
    Emo       / 128 (trained)  -> "... (keepk 128)"

Win-rate columns count, for each Emo row, the number of MMLU / MMLU Pro
sub-tasks where Emo beats the Reg. MoE baseline with the same total-expert
count (same keepk). "other" is excluded from both task groups, mirroring the
``*_avg_no_other`` averages — so denominators are 16 (MMLU) and 13 (MMLU Pro).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_INFERENCE_DIR = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "prune_eval_tables_final_ckpt0"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "claude_outputs" / "prune_plots" / "main_results_table.csv"

# MMLU subjects excluded from the *_avg_no_other average — also excluded from
# win-rate denominators.
MMLU_EXCLUDE = {"mmlu_merged_other"}
MMLU_PRO_EXCLUDE = {"mmlu_pro_merged_other"}

# (display_name, expert_count_label, model_key_or_None)
ROWS: List[Tuple[str, str, Optional[str]]] = [
    ("Dense^dagger", "8 (trained)", "dense"),
    ("Reg. MoE", "16 (trained)", None),
    ("Reg. MoE", "32 (trained)", "moe_small"),
    ("Reg. MoE", "8", "moe (keepk 8)"),
    ("Reg. MoE", "16", "moe (keepk 16)"),
    ("Reg. MoE", "32", "moe (keepk 32)"),
    ("Reg. MoE", "64", "moe (keepk 64)"),
    ("Reg. MoE", "128 (trained)", "moe (keepk 128)"),
    (
        "Emo",
        "8",
        "specialized moe + globallb + 1shardexp + randpool (keepk 8)",
    ),
    (
        "Emo",
        "16",
        "specialized moe + globallb + 1shardexp + randpool (keepk 16)",
    ),
    (
        "Emo",
        "32",
        "specialized moe + globallb + 1shardexp + randpool (keepk 32)",
    ),
    (
        "Emo",
        "64",
        "specialized moe + globallb + 1shardexp + randpool (keepk 64)",
    ),
    (
        "Emo",
        "128 (trained)",
        "specialized moe + globallb + 1shardexp + randpool (keepk 128)",
    ),
]

# Emo row keepk -> Reg. MoE comparison key (same total experts).
FLEXMOE_BASELINE = {
    "specialized moe + globallb + 1shardexp + randpool (keepk 8)": "moe (keepk 8)",
    "specialized moe + globallb + 1shardexp + randpool (keepk 16)": "moe (keepk 16)",
    "specialized moe + globallb + 1shardexp + randpool (keepk 32)": "moe (keepk 32)",
    "specialized moe + globallb + 1shardexp + randpool (keepk 64)": "moe (keepk 64)",
    "specialized moe + globallb + 1shardexp + randpool (keepk 128)": "moe (keepk 128)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=DEFAULT_INFERENCE_DIR,
        help="Directory with ckpt0 (inference-only) generated tables.",
    )
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


def _winrate(
    mmlu_df: pd.DataFrame,
    model: str,
    baseline: str,
    exclude: set,
    prefix: str,
) -> Optional[str]:
    if model not in mmlu_df.index or baseline not in mmlu_df.index:
        return None
    subject_cols = [
        c
        for c in mmlu_df.columns
        if c.endswith(" (lw)")
        and c != f"{prefix}avg_no_other (lw)"
        and c.replace(" (lw)", "") not in exclude
    ]
    wins = total = 0
    for col in subject_cols:
        a, b = mmlu_df.at[model, col], mmlu_df.at[baseline, col]
        if pd.isna(a) or pd.isna(b):
            continue
        total += 1
        if a > b:
            wins += 1
    if total == 0:
        return None
    return f"{wins}/{total}"


def _collect_mmlu_frames(dir_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        _load(dir_path / "acc_raw" / "mmlu_merged.csv"),
        _load(dir_path / "acc_raw" / "mmlu_pro_merged.csv"),
    )


def main() -> None:
    args = parse_args()
    ft_dir = _resolve_finetune_dir(args.finetune_dir)

    inf_agg = _load(args.inference_dir / "acc_raw" / "aggregate.csv")
    inf_gsm = _load(args.inference_dir / "exact_match" / "gsm8k.csv")
    ft_agg = _load(ft_dir / "acc_raw" / "aggregate.csv")
    ft_gsm = _load(ft_dir / "exact_match" / "gsm8k.csv")

    inf_mmlu, inf_mmlu_pro = _collect_mmlu_frames(args.inference_dir)
    ft_mmlu, ft_mmlu_pro = _collect_mmlu_frames(ft_dir)

    def lookup(df: pd.DataFrame, key: Optional[str], col: str) -> Optional[float]:
        if key is None or key not in df.index:
            return None
        val = df.at[key, col]
        return None if pd.isna(val) else float(val)

    records: List[Dict[str, str]] = []
    for name, experts, key in ROWS:
        row = {
            "": name,
            "# Total Experts": experts,
            "MMLU (inf)": _pct(lookup(inf_agg, key, "mmlu_merged_avg_no_other (lw)")),
            "MMLU Pro (inf)": _pct(lookup(inf_agg, key, "mmlu_pro_merged_avg_no_other (lw)")),
            "GSM8K (inf)": _pct(lookup(inf_gsm, key, "gsm8k_generation_8shot_merged (lw)")),
            "MMLU (ft)": _pct(lookup(ft_agg, key, "mmlu_merged_avg_no_other (lw)")),
            "MMLU Pro (ft)": _pct(lookup(ft_agg, key, "mmlu_pro_merged_avg_no_other (lw)")),
            "GSM8K (ft)": _pct(lookup(ft_gsm, key, "gsm8k_generation_8shot_merged (lw)")),
            "MMLU winrate vs Reg. MoE (inf)": "",
            "MMLU Pro winrate vs Reg. MoE (inf)": "",
            "MMLU winrate vs Reg. MoE (ft)": "",
            "MMLU Pro winrate vs Reg. MoE (ft)": "",
        }

        if key in FLEXMOE_BASELINE:
            baseline = FLEXMOE_BASELINE[key]
            row["MMLU winrate vs Reg. MoE (inf)"] = (
                _winrate(inf_mmlu, key, baseline, MMLU_EXCLUDE, "mmlu_merged_") or ""
            )
            row["MMLU Pro winrate vs Reg. MoE (inf)"] = (
                _winrate(
                    inf_mmlu_pro,
                    key,
                    baseline,
                    MMLU_PRO_EXCLUDE,
                    "mmlu_pro_merged_",
                )
                or ""
            )
            row["MMLU winrate vs Reg. MoE (ft)"] = (
                _winrate(ft_mmlu, key, baseline, MMLU_EXCLUDE, "mmlu_merged_") or ""
            )
            row["MMLU Pro winrate vs Reg. MoE (ft)"] = (
                _winrate(
                    ft_mmlu_pro,
                    key,
                    baseline,
                    MMLU_PRO_EXCLUDE,
                    "mmlu_pro_merged_",
                )
                or ""
            )

        records.append(row)

    df = pd.DataFrame(records)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"Wrote {args.output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
