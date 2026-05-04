#!/usr/bin/env python3
"""Generate a focused .tex (and companion .csv) directly comparing
ModMoE-anneal (1T) with ModMoE (1T).

ModMoE-anneal = Reg. MoE pretrained for 1T tokens, then *twolevel* batch-LB
("ModMoE-style") annealed.
ModMoE        = the full ModMoE recipe applied during the 1T anneal.

Same overall column layout as ``main_results_table.tex`` (Inference + Fine-
tuning, each with MMLU / MMLU-Pro / GSM8K), restricted to the two model
families and the keepk in {8,16,32,64,128 (trained)} sweep. No win-rate
columns — the two model blocks are presented side-by-side only.

Outputs:
    claude_outputs/prune_plots/modmoe_anneal_vs_modmoe.csv
    claude_outputs/prune_plots/modmoe_anneal_vs_modmoe.tex
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
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "modmoe_anneal_vs_modmoe.csv"
)
DEFAULT_OUTPUT_TEX = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "modmoe_anneal_vs_modmoe.tex"
)

# (display_name, expert_count_label, csv row name).
# ModMoE-anneal block first, then ModMoE block.
ROWS: List[Tuple[str, str, str]] = [
    ("ModMoE-anneal", "8",   "moe 1T + twolevel anneal (keepk 8)"),
    ("ModMoE-anneal", "16",  "moe 1T + twolevel anneal (keepk 16)"),
    ("ModMoE-anneal", "32",  "moe 1T + twolevel anneal (keepk 32)"),
    ("ModMoE-anneal", "64",  "moe 1T + twolevel anneal (keepk 64)"),
    ("ModMoE-anneal", "128 (trained)", "moe 1T + twolevel anneal (keepk 128)"),
    ("ModMoE",        "8",   "specialized moe 1T + anneal (keepk 8)"),
    ("ModMoE",        "16",  "specialized moe 1T + anneal (keepk 16)"),
    ("ModMoE",        "32",  "specialized moe 1T + anneal (keepk 32)"),
    ("ModMoE",        "64",  "specialized moe 1T + anneal (keepk 64)"),
    ("ModMoE",        "128 (trained)", "specialized moe 1T + anneal (keepk 128)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-dir", type=Path, default=DEFAULT_INFERENCE_DIR,
    )
    parser.add_argument(
        "--finetune-dir", type=Path, default=None,
        help="Defaults to <repo>/claude_outputs/prune_plots/prune_eval_tables_final "
             "with a backup fallback.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
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
    for c in backups:
        if (c / "acc_raw" / "aggregate.csv").is_file():
            print(f"[INFO] Using backup fine-tuning dir: {c}")
            return c
    raise FileNotFoundError(
        f"Could not locate fine-tuning dir at {primary} or *backup* siblings."
    )


def _load(p: Path) -> pd.DataFrame:
    if not p.is_file():
        raise FileNotFoundError(f"Missing source CSV: {p}")
    return pd.read_csv(p).set_index("model")


def _pct(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}"


def build_records(inf_dir: Path, ft_dir: Path) -> List[Dict[str, str]]:
    inf_agg = _load(inf_dir / "acc_raw" / "aggregate.csv")
    inf_gsm = _load(inf_dir / "exact_match" / "gsm8k.csv")
    ft_agg  = _load(ft_dir  / "acc_raw" / "aggregate.csv")
    ft_gsm  = _load(ft_dir  / "exact_match" / "gsm8k.csv")

    def lookup(df: pd.DataFrame, key: str, col: str) -> Optional[float]:
        if key not in df.index:
            return None
        v = df.at[key, col]
        return None if pd.isna(v) else float(v)

    records: List[Dict[str, str]] = []
    for name, experts, key in ROWS:
        records.append({
            "": name,
            "# Experts": experts,
            "MMLU (inf)":     _pct(lookup(inf_agg, key, "mmlu_merged_avg_no_other (lw)")),
            "MMLU-Pro (inf)": _pct(lookup(inf_agg, key, "mmlu_pro_merged_avg_no_other (lw)")),
            "GSM8K (inf)":    _pct(lookup(inf_gsm, key, "gsm8k_generation_8shot_merged (lw)")),
            "MMLU (ft)":      _pct(lookup(ft_agg, key, "mmlu_merged_avg_no_other (lw)")),
            "MMLU-Pro (ft)":  _pct(lookup(ft_agg, key, "mmlu_pro_merged_avg_no_other (lw)")),
            "GSM8K (ft)":     _pct(lookup(ft_gsm, key, "gsm8k_generation_8shot_merged (lw)")),
        })
    return records


# ---------------------------------------------------------------------------
# .tex emit
# ---------------------------------------------------------------------------


TEX_HEADER = r"""% =============================================================================
% DATA SOURCE INSTRUCTIONS
% =============================================================================
% All data from: claude_outputs/prune_plots/
%
% Inference columns: prune_eval_tables_final_ckpt0/
% Fine-tuning columns: prune_eval_tables_final/
%
% Metrics used (all use "(lw)" columns, NOT "(ep)"):
%   MMLU     -> acc_raw/aggregate.csv -> mmlu_merged_avg_no_other (lw)
%   MMLU-Pro -> acc_raw/aggregate.csv -> mmlu_pro_merged_avg_no_other (lw)
%   GSM8K    -> exact_match/gsm8k.csv -> gsm8k_generation_8shot_merged (lw)
%
% Model mapping (1T scale only):
%   ModMoE-anneal {8,16,32,64,128}  -> "moe 1T + twolevel anneal (keepk K)"
%   \methodname   {8,16,32,64,128}  -> "specialized moe 1T + anneal (keepk K)"
%
% No win rates in this table — the two model blocks are presented side-by-side
% only.
% =============================================================================
"""

TEX_TEMPLATE = r"""\begin{{table}}[t]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3.5pt}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{l l rrr rrr}}
\toprule
 & \multirow{{2}}{{*}}{{\# Experts}}
 & \multicolumn{{3}}{{c}}{{\textbf{{Inference}}}}
 & \multicolumn{{3}}{{c}}{{\textbf{{Fine-tuning}}}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-8}}
 &
 & \textbf{{MMLU}}
 & \textbf{{MMLU-Pro}}
 & \textbf{{GSM8K}}
 & \textbf{{MMLU}}
 & \textbf{{MMLU-Pro}}
 & \textbf{{GSM8K}} \\
\midrule
\multirow{{5}}{{*}}{{ModMoE-anneal}}
{anneal_rows}
\midrule
\multirow{{5}}{{*}}{{\methodname}}
{flex_rows}
\bottomrule
\end{{tabular}}
}}
\vspace{{.3em}}
\caption{{
Direct comparison of \methodname\ against ModMoE-anneal at the 1T training scale.
\methodname\ applies the full two-level batch-LB recipe throughout pretraining; ModMoE-anneal uses the same Reg.\ MoE pretraining run, with two-level batch-LB applied only during the 1T anneal phase.
}}
\label{{tab:modmoe_anneal_vs_modmoe}}
\end{{table}}
"""


def _format_row(rec: Dict[str, str]) -> str:
    return (
        f"    & {rec['# Experts']} & "
        f"{rec['MMLU (inf)']} & {rec['MMLU-Pro (inf)']} & {rec['GSM8K (inf)']} & "
        f"{rec['MMLU (ft)']} & {rec['MMLU-Pro (ft)']} & {rec['GSM8K (ft)']} \\\\"
    )


def render_tex(records: List[Dict[str, str]]) -> str:
    anneal_recs = [r for r in records if r[""] == "ModMoE-anneal"]
    flex_recs   = [r for r in records if r[""] == "ModMoE"]
    anneal_rows = "\n".join(_format_row(r) for r in anneal_recs)
    flex_rows   = "\n".join(_format_row(r) for r in flex_recs)
    return TEX_HEADER + TEX_TEMPLATE.format(
        anneal_rows=anneal_rows, flex_rows=flex_rows
    )


def main() -> None:
    args = parse_args()
    ft_dir = _resolve_finetune_dir(args.finetune_dir)

    records = build_records(args.inference_dir, ft_dir)

    df = pd.DataFrame(records)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")
    print(df.to_string(index=False))

    tex = render_tex(records)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(tex)
    print(f"\nWrote {args.output_tex}")


if __name__ == "__main__":
    main()
