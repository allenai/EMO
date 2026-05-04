#!/usr/bin/env python3
"""Generate CSV tables for extension-HF pruning + finetuning evaluations.

The launch script (scripts/ryanwang/extensions_hf/launch_extensions_hf.sh) writes
results under <repo>/extension_evals_hf_0426/, one directory per
(model, task, keepk, prunemode, [nprune], [pshots], [eshots], [freeze_mode])
combination. Inside each, evaluations live under sub-prefixes for each pipeline
stage:

    {relative_dir}/
        small/                       — pruned + finetuned model eval
        merged_default/              — merge-back: routable expert MLPs only
        merged_shared/               — + shared expert MLP
        merged_router/               — + router rows
        merged_shared_router/        — + shared expert + router
        merged_non_moe/              — + attention/norms/embed/lm_head
        merged_default_avg/          — same as default but with --average
        merged_shared_avg/
        merged_router_avg/
        merged_shared_router_avg/
        merged_non_moe_avg/
    Each holds checkpoint-{N}/task-*-metrics.json, from which we pull
    primary_score (= each task's primary metric: exact_match for GSM8K,
    acc_raw for MMLU/MMLU Pro, etc.).

Output (wide format): one row per (model, task, keepk, prunemode, nprune,
pshots, eshots, freeze_mode) and one column per eval phase, plus the metadata
columns. Empty cells indicate the corresponding eval phase has not been run /
populated yet for that configuration.

Reads from   : <repo>/extension_evals_hf_0426/   (override with --evals-root)
Writes into  : <repo>/claude_outputs/prune_plots/extension_evals_hf_tables/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_EVALS_ROOT = REPO_ROOT / "extension_evals_hf_0426"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "prune_plots" / "extension_evals_hf_tables"

# --- Eval-phase columns in display order ----------------------------------
# Replace-mode merges (parent ← small for the listed param sets).
REPLACE_MERGES = [
    "default",
    "shared",
    "router",
    "shared_router",
    "non_moe",
]
# Average-mode merges (parent ← 0.5·parent + 0.5·small).
AVG_MERGES = [m + "_avg" for m in REPLACE_MERGES]

EVAL_PHASES: List[str] = ["small"] + [f"merged_{m}" for m in REPLACE_MERGES + AVG_MERGES]

# --- Friendly model labels ------------------------------------------------
MODEL_LABELS: Dict[str, str] = {
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419step250339-hf":
        "specialized moe 1T + anneal",
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419step250339-hf":
        "moe 1T + anneal",
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301step30995-hf":
        "specialized moe + globallb + 1shardexp + randpool",
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308step30995-hf":
        "moe",
    "moereducedp512sharedexp1_1b4b_lr-4e-3_lb-1e-1_0308step30995-hf":
        "moe_small",
    "dense_1b_lr-4e-3_0213step30995-hf":
        "dense",
}

# --- Parsing the relative_dir name ----------------------------------------
# Schema (from launch_extensions_hf.sh):
#   {task}_keepk_{K}_bs-{B}_lr-{LR}_epoch-{E}_prunemode-{MODE}
#   [_nprune-{N}][_pshots-{X}][_eshots-{Y}][_fz-{FREEZE}]
RELATIVE_DIR_RE = re.compile(
    r"^(?P<task>.+?)_keepk_(?P<keepk>\d+)"
    r"_bs-(?P<bs>\S+?)_lr-(?P<lr>\S+?)_epoch-(?P<epoch>\S+?)"
    r"_prunemode-(?P<prunemode>[a-z_]+)"
    r"(?:_nprune-(?P<nprune>\S+?))?"
    r"(?:_pshots-(?P<pshots>\d+))?"
    r"(?:_eshots-(?P<eshots>\d+))?"
    r"(?:_fz-(?P<freeze>\S+))?$"
)


def parse_relative_dir(name: str) -> Optional[Dict[str, str]]:
    """Return the parsed components, or None if `name` doesn't match the schema."""
    # Greedy task name + lazy nprune/pshots/eshots/fz tokens — but the regex
    # above can be ambiguous if the task name itself contains "_keepk_". The
    # task names in this project don't, so the simple regex is fine.
    m = RELATIVE_DIR_RE.match(name)
    if m is None:
        return None
    return {k: (v if v is not None else "") for k, v in m.groupdict().items()}


def _select_last_checkpoint(results_dir: Path) -> Optional[Path]:
    best_step: Optional[int] = None
    best_dir: Optional[Path] = None
    for ck in results_dir.glob("checkpoint-*"):
        if not ck.is_dir():
            continue
        try:
            step = int(ck.name.replace("checkpoint-", ""))
        except ValueError:
            continue
        if best_step is None or step > best_step:
            best_step = step
            best_dir = ck
    return best_dir


def _read_primary_score(phase_dir: Path) -> Optional[float]:
    """Pull primary_score from the last checkpoint's task-*-metrics.json."""
    if not phase_dir.is_dir():
        return None
    ck = _select_last_checkpoint(phase_dir)
    if ck is None:
        return None
    files = sorted(ck.glob("task-*-metrics.json"))
    if not files:
        return None
    try:
        with files[0].open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Two possible shapes:
    # 1) {"metrics": [{"task": ..., "primary_score": ...}, ...]}
    # 2) {"metrics": {"primary_score": ...}}
    metrics = data.get("metrics")
    if isinstance(metrics, list) and metrics:
        v = metrics[0].get("primary_score")
    elif isinstance(metrics, dict):
        v = metrics.get("primary_score")
    else:
        v = None
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def collect(evals_root: Path) -> pd.DataFrame:
    """Walk the evals tree and assemble one row per relative_dir."""
    rows: List[Dict[str, object]] = []

    if not evals_root.is_dir():
        raise FileNotFoundError(f"Evals root missing: {evals_root}")

    for model_dir in sorted(p for p in evals_root.iterdir() if p.is_dir()):
        model_name = model_dir.name
        model_label = MODEL_LABELS.get(model_name, model_name)

        for rel_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            parsed = parse_relative_dir(rel_dir.name)
            if parsed is None:
                print(f"[WARN] Could not parse relative_dir: {rel_dir.name}")
                continue

            row: Dict[str, object] = {
                "model": model_label,
                "model_dir": model_name,
                "task": parsed["task"],
                "keepk": int(parsed["keepk"]),
                "prunemode": parsed["prunemode"],
                "nprune": parsed["nprune"] or "All",
                "pshots": parsed["pshots"] or "default",
                "eshots": parsed["eshots"] or "default",
                "freeze": parsed["freeze"] or "none",
            }

            for phase in EVAL_PHASES:
                row[phase] = _read_primary_score(rel_dir / phase)

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Order: metadata, then phase columns
    meta = ["model", "task", "keepk", "prunemode", "nprune", "pshots", "eshots", "freeze"]
    df = df[meta + EVAL_PHASES + ["model_dir"]]
    df = df.sort_values(by=meta).reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evals-root", type=Path, default=DEFAULT_EVALS_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--format", default="csv", choices=["csv", "tsv", "markdown"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = collect(args.evals_root)
    if df.empty:
        print("[ERROR] No rows collected.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fname = "extension_evals_hf"
    if args.format == "csv":
        out = args.output_dir / f"{fname}.csv"
        df.to_csv(out, index=False, float_format="%.4f")
    elif args.format == "tsv":
        out = args.output_dir / f"{fname}.tsv"
        df.to_csv(out, index=False, sep="\t", float_format="%.4f")
    elif args.format == "markdown":
        out = args.output_dir / f"{fname}.md"
        out.write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")

    print(f"[INFO] Wrote {out} ({len(df)} rows)")
    # Print a friendly view: drop model_dir, scale to percentages.
    pretty = df.drop(columns=["model_dir"]).copy()
    for col in EVAL_PHASES:
        pretty[col] = pretty[col].apply(lambda x: "" if pd.isna(x) else f"{x * 100:.1f}")
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
