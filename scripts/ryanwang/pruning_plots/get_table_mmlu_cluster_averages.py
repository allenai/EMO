#!/usr/bin/env python3
"""Aggregate MMLU scores three different ways for each category grouping.

We support 5 groupings:

  * ``l0``                  — k-means on layer-0 router probs (16 clusters)
  * ``l15``                 — k-means on layer-15 router probs (16 clusters)
  * ``all``                 — k-means on all-layer router probs (16 clusters)
  * ``mmlu_merged``         — original 17 mmlu_merged_* categories (incl. "other")
  * ``mmlu_merged_no_other``— 16 mmlu_merged_* categories without "other"

The first three each split all 57 subjects into 16 clusters (so each subject
lives in exactly one cluster). ``mmlu_merged`` covers all 57 subjects across
17 categories. Dropping ``other`` gives 54 subjects across 16 categories.

Each per-cluster eval dir contains:

    {category}_keepk_K_..._prunemode-layerwise/results/checkpoint-N/
        task-mmlu_merged_*-metrics.json     ← cluster-level (acc, num_instances)
        per_subject/{subject}/task-...-metrics.json   ← subject-level

We compute three averages, separately per grouping × per checkpoint mode:

  1) ``avg_categories`` — mean of the 16 (or 17) cluster-level acc scores;
     each cluster contributes equally regardless of size.
  2) ``avg_subjects``   — mean of the per-subject acc scores; each subject
     contributes equally regardless of test-set size.
  3) ``avg_examples``   — sum(acc_subj * n_subj) / sum(n_subj), so each
     individual test example contributes equally.

Reads from   : <repo>/prune_evals_final/<model>/...
Writes into  : <repo>/claude_outputs/prune_plots/mmlu_cluster_averages/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_EVALS_ROOT = REPO_ROOT / "prune_evals_final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "prune_plots" / "mmlu_cluster_averages"
DEFAULT_MODEL = (
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32"
    "_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419step250339-hf"
)
DEFAULT_KEEPK = 32
DEFAULT_PRUNEMODE = "layerwise"
METRIC_KEY = "acc_raw"

# Original 17 mmlu_merged categories. ``mmlu_merged_no_other`` drops ``other``.
MMLU_MERGED_CATEGORIES = [
    "biology", "business", "chemistry", "computer_science", "culture",
    "economics", "engineering", "geography", "health", "history",
    "law", "math", "other", "philosophy_cat", "physics", "politics", "psychology",
]


def _select_checkpoint(results_dir: Path, mode: str) -> Optional[Path]:
    best: Optional[Tuple[int, Path]] = None
    for ck in results_dir.glob("checkpoint-*"):
        if not ck.is_dir():
            continue
        try:
            step = int(ck.name.replace("checkpoint-", ""))
        except ValueError:
            continue
        if best is None:
            best = (step, ck)
        elif mode == "last" and step > best[0]:
            best = (step, ck)
        elif mode == "first" and step < best[0]:
            best = (step, ck)
    return best[1] if best else None


def _read_metric_and_n(metrics_file: Path, metric_key: str) -> Optional[Tuple[float, int]]:
    try:
        with metrics_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    n = data.get("num_instances")
    metrics = data.get("metrics", {})
    if isinstance(metrics, list) and metrics:
        metrics = metrics[0]
    if not isinstance(metrics, dict):
        return None
    v = metrics.get(metric_key)
    if v is None or n is None:
        return None
    try:
        return float(v), int(n)
    except (TypeError, ValueError):
        return None


def _category_dirname(category_path: str, keepk: int, prunemode: str) -> str:
    return f"{category_path}_keepk_{keepk}_bs-32_lr-5e-5_epoch-1_prunemode-{prunemode}"


def _list_cluster_categories(
    evals_root: Path, model: str, grouping: str, keepk: int, prunemode: str,
) -> List[str]:
    """For an l0/l15/all grouping, auto-discover the 16 cluster suffixes."""
    model_dir = evals_root / model
    prefix = f"mmlu_merged_cluster_{grouping}_"
    suffix = f"_keepk_{keepk}_bs-32_lr-5e-5_epoch-1_prunemode-{prunemode}"
    out: List[str] = []
    if not model_dir.is_dir():
        return out
    for d in sorted(model_dir.iterdir()):
        if d.is_dir() and d.name.startswith(prefix) and d.name.endswith(suffix):
            cluster = d.name[len(prefix):-len(suffix)]
            out.append(f"{prefix.rstrip('_')}_{cluster}")
    return out


def _grouping_categories(
    evals_root: Path, model: str, grouping: str, keepk: int, prunemode: str,
) -> Tuple[List[str], int]:
    """Return (full_category_paths, expected_subject_count) for a grouping."""
    if grouping in ("l0", "l15", "all"):
        return _list_cluster_categories(evals_root, model, grouping, keepk, prunemode), 57
    if grouping == "mmlu_merged":
        return [f"mmlu_merged_{c}" for c in MMLU_MERGED_CATEGORIES], 57
    if grouping == "mmlu_merged_no_other":
        return [f"mmlu_merged_{c}" for c in MMLU_MERGED_CATEGORIES if c != "other"], 54
    raise ValueError(f"Unknown grouping: {grouping}")


def _aggregate_for_grouping(
    evals_root: Path, model: str, grouping: str,
    keepk: int, prunemode: str, ckpt_mode: str, metric_key: str,
) -> Optional[Dict[str, float]]:
    categories, expected_subjects = _grouping_categories(
        evals_root, model, grouping, keepk, prunemode
    )
    if not categories:
        print(f"[WARN] grouping={grouping}: no categories found")
        return None

    cluster_accs: List[float] = []
    subject_accs: List[float] = []
    subject_acc_x_n: float = 0.0
    subject_n_total: int = 0
    seen_subjects: List[str] = []

    for cat_path in categories:
        cdir = evals_root / model / _category_dirname(cat_path, keepk, prunemode)
        results = cdir / "results"
        if not results.is_dir():
            print(f"[WARN] missing results: {cdir}")
            continue
        ck = _select_checkpoint(results, ckpt_mode)
        if ck is None:
            print(f"[WARN] no checkpoint dir under {results}")
            continue

        # Cluster/category-level metric: there's exactly one task-*-metrics.json
        # at the checkpoint root (sibling of per_subject/).
        cat_files = sorted(ck.glob("task-mmlu_merged_*-metrics.json"))
        # filter out per_subject files (those live under per_subject/)
        cat_files = [f for f in cat_files if f.parent == ck]
        if cat_files:
            res = _read_metric_and_n(cat_files[0], metric_key)
            if res is not None:
                cluster_accs.append(res[0])

        # Per-subject metrics.
        per_subj_dir = ck / "per_subject"
        if not per_subj_dir.is_dir():
            print(f"[WARN] no per_subject dir under {ck}")
            continue
        for subj_dir in sorted(per_subj_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            subj_files = sorted(subj_dir.glob("task-*-metrics.json"))
            if not subj_files:
                continue
            res = _read_metric_and_n(subj_files[0], metric_key)
            if res is None:
                continue
            acc, n = res
            subject_accs.append(acc)
            subject_acc_x_n += acc * n
            subject_n_total += n
            seen_subjects.append(subj_dir.name)

    if len(seen_subjects) != expected_subjects:
        print(f"[WARN] grouping={grouping}: found {len(seen_subjects)} subjects "
              f"(expected {expected_subjects})")

    if not cluster_accs or not subject_accs:
        return None
    return {
        "n_categories": len(cluster_accs),
        "n_subjects": len(subject_accs),
        "n_examples": subject_n_total,
        "avg_categories": sum(cluster_accs) / len(cluster_accs),
        "avg_subjects": sum(subject_accs) / len(subject_accs),
        "avg_examples": (subject_acc_x_n / subject_n_total) if subject_n_total else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evals-root", type=Path, default=DEFAULT_EVALS_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--keepks", default="8,16,32,64,128",
        help="Comma-separated list of keepk values to evaluate (one row per keepk × grouping).",
    )
    p.add_argument("--prunemode", default=DEFAULT_PRUNEMODE)
    p.add_argument(
        "--checkpoint-mode", default="both",
        choices=["last", "first", "both"],
        help="'last' = fine-tuned, 'first' = ckpt-0 (pre-finetune), 'both' = emit two tables.",
    )
    p.add_argument(
        "--metric-key", default=METRIC_KEY,
        help="Which metric to read from each task-*-metrics.json (default: acc_raw)."
    )
    p.add_argument(
        "--groupings", default="l0,l15,all,mmlu_merged_no_other,mmlu_merged",
        help="Comma-separated list of grouping names.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groupings = [g.strip() for g in args.groupings.split(",") if g.strip()]
    keepks = [int(k.strip()) for k in args.keepks.split(",") if k.strip()]

    if args.checkpoint_mode == "both":
        modes = [("first", "ckpt0"), ("last", "finetuned")]
    else:
        modes = [(args.checkpoint_mode, "ckpt0" if args.checkpoint_mode == "first" else "finetuned")]

    for ckpt_mode, label in modes:
        rows: List[Dict[str, object]] = []
        for keepk in keepks:
            for grouping in groupings:
                agg = _aggregate_for_grouping(
                    args.evals_root, args.model, grouping,
                    keepk, args.prunemode, ckpt_mode, args.metric_key,
                )
                if agg is None:
                    print(f"[WARN] no data for grouping={grouping}, keepk={keepk}, ckpt_mode={ckpt_mode}")
                    continue
                rows.append({"keepk": keepk, "grouping": grouping, **agg})

        if not rows:
            continue
        df = pd.DataFrame(rows)
        out = args.output_dir / f"mmlu_cluster_averages_{label}.csv"
        df.to_csv(out, index=False, float_format="%.4f")

        pretty = df.copy()
        for c in ["avg_categories", "avg_subjects", "avg_examples"]:
            pretty[c] = pretty[c].apply(lambda x: f"{x*100:.2f}")
        print(f"\n=== {label} ({args.metric_key}, prunemode={args.prunemode}) ===")
        print(pretty.to_string(index=False))
        print(f"[INFO] Wrote {out}")


if __name__ == "__main__":
    main()
