#!/usr/bin/env python3
"""Re-grade GSM8K-style exact_match for extension-HF runs with a robust extractor.

Why? The default GSM8K answer-extractor on each task-*-metrics.json grabs the
LAST numeric token of the post-trim continuation. After merging the pruned +
finetuned small model back into the full architecture, the model often loses
prompt-format compliance: it correctly says "So the answer is X." in the
reasoning, but then drifts into trailing text (date stamps like
``Monday, 11 March 2019 00:00`` or repeated boilerplate). The trailing junk
contains numbers, and the default extractor latches onto them, depressing
exact_match well below the model's real ability.

This script re-grades each prediction by:
  * preferring the FIRST number that appears after a phrase like
    "(so/therefore) the answer is" / "the final answer is" / "answer:" /
    "= <num>" (the GSM8K few-shot template);
  * falling back to the original ``model_answer`` if no such phrase is found.

Output is the same wide schema as ``get_table_scores_extension_evals_hf.py``
but every eval-phase cell shows BOTH the original and the corrected accuracy
("orig / corr"). A separate ``_corrected`` CSV with just the corrected scores
is also written.

Reads from   : <repo>/extension_evals_hf_0426/   (predictions.jsonl per phase)
Writes into  : <repo>/claude_outputs/prune_plots/extension_evals_hf_tables/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_EVALS_ROOT = REPO_ROOT / "extension_evals_hf_0426"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "prune_plots" / "extension_evals_hf_tables"
DEFAULT_S3_BASE = "s3://ai2-sewonm/ryanwang/extension_evals_hf_0426"

REPLACE_MERGES = ["default", "shared", "router", "shared_router", "non_moe"]
AVG_MERGES = [m + "_avg" for m in REPLACE_MERGES]
EVAL_PHASES: List[str] = ["small"] + [f"merged_{m}" for m in REPLACE_MERGES + AVG_MERGES]

MODEL_LABELS: Dict[str, str] = {
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419step250339-hf":
        "specialized moe 1T + anneal",
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419step250339-hf":
        "moe 1T + anneal",
}

RELATIVE_DIR_RE = re.compile(
    r"^(?P<task>.+?)_keepk_(?P<keepk>\d+)"
    r"_bs-(?P<bs>\S+?)_lr-(?P<lr>\S+?)_epoch-(?P<epoch>\S+?)"
    r"_prunemode-(?P<prunemode>[a-z_]+)"
    r"(?:_nprune-(?P<nprune>\S+?))?"
    r"(?:_pshots-(?P<pshots>\d+))?"
    r"(?:_eshots-(?P<eshots>\d+))?"
    r"(?:_fz-(?P<freeze>\S+))?$"
)

# --- Robust answer extraction (GSM8K-style) -------------------------------
# Patterns tried in order. The first match wins. Each pattern captures a
# numeric token (possibly with $, %, commas, decimals, sign).
_NUM = r"-?\$?-?\s?\d[\d,]*(?:\.\d+)?\s?%?"

ANSWER_PATTERNS = [
    # "so the answer is 60" / "the final answer is $42"
    re.compile(r"(?:so\s+)?the\s+(?:final\s+)?answer\s+is\s*[:=]?\s*\$?\s*(" + _NUM + r")", re.IGNORECASE),
    # "Answer: 60"  (rare; the prompt format prefers 'the answer is')
    re.compile(r"\banswer\s*[:=]\s*\$?\s*(" + _NUM + r")", re.IGNORECASE),
    # "= 60." at end of an arithmetic step (last fallback before trailing junk)
    re.compile(r"=\s*\$?\s*(" + _NUM + r")\s*[\.\n]"),
]


def _normalize_num(s: str) -> str:
    """Normalize a numeric string to a comparable canonical form."""
    s = s.strip().replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
    # drop trailing period
    s = s.rstrip(".")
    # int-ify "60.0" → "60"
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:g}"
    except ValueError:
        return s


def robust_extract(continuation: str, fallback: Optional[str]) -> Optional[str]:
    """Return the cleanest numeric answer we can find in `continuation`.

    Strategy: first match of any pattern in ANSWER_PATTERNS. If none match,
    return ``fallback`` (the original model_answer).
    """
    if not continuation:
        return fallback
    for pat in ANSWER_PATTERNS:
        m = pat.search(continuation)
        if m:
            return _normalize_num(m.group(1))
    return fallback


def parse_relative_dir(name: str) -> Optional[Dict[str, str]]:
    m = RELATIVE_DIR_RE.match(name)
    if m is None:
        return None
    return {k: (v if v is not None else "") for k, v in m.groupdict().items()}


def _select_last_checkpoint(results_dir: Path) -> Optional[Path]:
    best: Optional[Tuple[int, Path]] = None
    for ck in results_dir.glob("checkpoint-*"):
        if not ck.is_dir():
            continue
        try:
            step = int(ck.name.replace("checkpoint-", ""))
        except ValueError:
            continue
        if best is None or step > best[0]:
            best = (step, ck)
    return best[1] if best else None


def _ensure_predictions(
    phase_dir: Path, s3_base: str, model_dir: str, rel_dir: str, phase: str
) -> Optional[Path]:
    """Return path to predictions.jsonl in the last checkpoint, downloading from S3 if absent."""
    if not phase_dir.is_dir():
        return None
    ck = _select_last_checkpoint(phase_dir)
    if ck is None:
        # Try to discover checkpoint dirs on S3 and create a local placeholder
        ck_name = _list_s3_checkpoint(s3_base, model_dir, rel_dir, phase)
        if ck_name is None:
            return None
        ck = phase_dir / ck_name
        ck.mkdir(parents=True, exist_ok=True)

    # Look for any task-*-predictions.jsonl
    candidates = sorted(ck.glob("task-*-predictions.jsonl"))
    if candidates:
        return candidates[0]

    # Not local — try S3
    s3_dir = f"{s3_base}/{model_dir}/{rel_dir}/{phase}/{ck.name}/"
    listing = subprocess.run(
        ["aws", "s3", "ls", s3_dir], capture_output=True, text=True, check=False
    )
    if listing.returncode != 0 or "predictions.jsonl" not in listing.stdout:
        return None
    fname = next(line.split()[-1] for line in listing.stdout.splitlines()
                 if line.endswith("-predictions.jsonl"))
    dest = ck / fname
    print(f"  [s3] downloading {fname} for {model_dir}/{rel_dir}/{phase}/{ck.name}")
    subprocess.run(
        ["aws", "s3", "cp", "--quiet", s3_dir + fname, str(dest)], check=False
    )
    return dest if dest.is_file() else None


def _list_s3_checkpoint(s3_base: str, model_dir: str, rel_dir: str, phase: str) -> Optional[str]:
    """Find the highest-numbered checkpoint-N/ on S3 for a given phase."""
    listing = subprocess.run(
        ["aws", "s3", "ls", f"{s3_base}/{model_dir}/{rel_dir}/{phase}/"],
        capture_output=True, text=True, check=False,
    )
    if listing.returncode != 0:
        return None
    ckpts = []
    for line in listing.stdout.splitlines():
        line = line.strip()
        if line.startswith("PRE checkpoint-"):
            n = line.removeprefix("PRE checkpoint-").rstrip("/")
            try:
                ckpts.append((int(n), f"checkpoint-{n}"))
            except ValueError:
                continue
    if not ckpts:
        return None
    return max(ckpts)[1]


def regrade_predictions(pred_path: Path) -> Tuple[int, int, int]:
    """Return (n_total, n_correct_orig, n_correct_corrected) for this file."""
    n = orig = corr = 0
    with pred_path.open() as f:
        for line in f:
            d = json.loads(line)
            mo = d.get("model_output", [{}])[0]
            cont = mo.get("continuation", "") or ""
            orig_answer = mo.get("model_answer")
            gold = (d.get("label") or "").strip()
            n += 1
            if d.get("metrics", {}).get("exact_match", 0.0) >= 0.5:
                orig += 1
            new_ans = robust_extract(cont, fallback=orig_answer)
            if new_ans is not None and _normalize_num(str(new_ans)) == _normalize_num(gold):
                corr += 1
    return n, orig, corr


def collect(
    evals_root: Path, s3_base: Optional[str]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not evals_root.is_dir():
        raise FileNotFoundError(evals_root)

    for model_dir_path in sorted(p for p in evals_root.iterdir() if p.is_dir()):
        model_dir = model_dir_path.name
        model_label = MODEL_LABELS.get(model_dir, model_dir)
        for rel_dir_path in sorted(p for p in model_dir_path.iterdir() if p.is_dir()):
            rel_dir = rel_dir_path.name
            parsed = parse_relative_dir(rel_dir)
            if parsed is None:
                print(f"[WARN] Could not parse {rel_dir}")
                continue
            row = {
                "model": model_label,
                "task": parsed["task"],
                "keepk": int(parsed["keepk"]),
                "prunemode": parsed["prunemode"],
                "nprune": parsed["nprune"] or "All",
                "pshots": parsed["pshots"] or "default",
                "eshots": parsed["eshots"] or "default",
                "freeze": parsed["freeze"] or "none",
            }
            for phase in EVAL_PHASES:
                phase_dir = rel_dir_path / phase
                pred = _ensure_predictions(
                    phase_dir, s3_base or DEFAULT_S3_BASE, model_dir, rel_dir, phase
                ) if s3_base != "" else None
                if pred is None:
                    # Try local-only: look for an existing predictions file
                    if phase_dir.is_dir():
                        ck = _select_last_checkpoint(phase_dir)
                        if ck is not None:
                            local = sorted(ck.glob("task-*-predictions.jsonl"))
                            pred = local[0] if local else None
                if pred is None:
                    row[f"{phase}__orig"] = None
                    row[f"{phase}__corr"] = None
                    row[f"{phase}__n"] = None
                    continue
                n, o, c = regrade_predictions(pred)
                row[f"{phase}__orig"] = o / n if n else None
                row[f"{phase}__corr"] = c / n if n else None
                row[f"{phase}__n"] = n
            rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evals-root", type=Path, default=DEFAULT_EVALS_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--s3-base",
        default=DEFAULT_S3_BASE,
        help="If set, fetch predictions.jsonl from this S3 prefix when missing locally. "
             "Pass an empty string to disable S3 fallback.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = collect(args.evals_root, args.s3_base or None)
    if df.empty:
        print("[ERROR] No rows collected.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Wide CSV with both original and corrected
    full_path = args.output_dir / "extension_evals_hf_corrected.csv"
    df.to_csv(full_path, index=False, float_format="%.4f")
    print(f"[INFO] Wrote {full_path}")

    # Compact human-readable view: "orig→corr" per phase
    pretty = df[["model", "task", "keepk", "prunemode", "nprune", "pshots", "eshots", "freeze"]].copy()
    for phase in EVAL_PHASES:
        oc = df[f"{phase}__orig"]
        cc = df[f"{phase}__corr"]
        n = df[f"{phase}__n"]
        def fmt(o, c, n):
            if pd.isna(o):
                return ""
            return f"{o*100:.1f}→{c*100:.1f}"
        pretty[phase] = [fmt(o, c, nn) for o, c, nn in zip(oc, cc, n)]
    print("\nPretty view (original → corrected, both ×100):")
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
