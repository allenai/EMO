"""
Summarize the per-combo JSONs produced by ``analyze_nprune_keep_sets`` into a
per-keepk ``summary.json`` and a top-level ``summary.md`` table showing how the
kept-expert sets shift as the calibration size N changes.

For each (task, keep_k) group, the reference N is the largest available ("all" if
present, else max integer). Per-layer Jaccard overlap of each non-reference N against
the reference is reported; layers without MoE (None entries) are skipped.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_NPRUNE_FILE_RE = re.compile(r"^nprune-(all|\d+)\.json$")


def _jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def _parse_nprune_from_filename(name: str) -> Optional[str]:
    m = _NPRUNE_FILE_RE.match(name)
    return m.group(1) if m else None


def _nprune_sort_key(tag: str) -> Tuple[int, int]:
    # Sort integers ascending; "all" goes last.
    return (1, 0) if tag == "all" else (0, int(tag))


def _pick_reference(tags: List[str]) -> str:
    if "all" in tags:
        return "all"
    return max(tags, key=lambda t: int(t))


def summarize_keepk_dir(keepk_dir: Path) -> Dict:
    """Build a summary for a single <task>/keepk-<K>/ directory."""
    combos: Dict[str, Dict] = {}
    for f in sorted(keepk_dir.glob("nprune-*.json")):
        tag = _parse_nprune_from_filename(f.name)
        if tag is None:
            continue
        with open(f) as fp:
            combos[tag] = json.load(fp)

    if not combos:
        return {}

    tags = sorted(combos.keys(), key=_nprune_sort_key)
    ref_tag = _pick_reference(tags)
    ref = combos[ref_tag]
    ref_kept = ref["experts_kept_per_layer"]

    per_layer: Dict[str, List[Optional[float]]] = {}
    mean_jaccard: Dict[str, Optional[float]] = {}
    min_jaccard: Dict[str, Optional[float]] = {}

    for tag in tags:
        kept = combos[tag]["experts_kept_per_layer"]
        layer_js: List[Optional[float]] = []
        for rk, k in zip(ref_kept, kept):
            if rk is None or k is None:
                layer_js.append(None)
            else:
                layer_js.append(_jaccard(rk, k))
        per_layer[tag] = layer_js
        valid = [x for x in layer_js if x is not None]
        mean_jaccard[tag] = (sum(valid) / len(valid)) if valid else None
        min_jaccard[tag] = min(valid) if valid else None

    return {
        "task": ref["task"],
        "prune_keep_k": ref["prune_keep_k"],
        "reference_nprune": ref_tag,
        "tags": tags,
        "per_layer_jaccard": per_layer,
        "mean_jaccard": mean_jaccard,
        "min_jaccard": min_jaccard,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="claude_outputs/prune_plots/nprune_analysis",
        help="Root directory containing <stringified_model>/<task>/keepk-<k>/ JSONs",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Stringified model name (subdirectory under --output-dir)",
    )
    args = parser.parse_args()

    model_dir = Path(args.output_dir) / args.model
    if not model_dir.exists():
        raise RuntimeError(f"Model directory not found: {model_dir}")

    # Gather summaries and collect all N tags across groups for a stable column set.
    groups: List[Tuple[str, int, Dict]] = []  # (task, keep_k, summary)
    all_tags: List[str] = []

    for task_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        task = task_dir.name
        for keepk_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            m = re.match(r"^keepk-(\d+)$", keepk_dir.name)
            if not m:
                continue
            keep_k = int(m.group(1))
            summary = summarize_keepk_dir(keepk_dir)
            if not summary:
                continue
            with open(keepk_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            groups.append((task, keep_k, summary))
            for tag in summary["tags"]:
                if tag not in all_tags:
                    all_tags.append(tag)

    all_tags.sort(key=_nprune_sort_key)

    md_lines: List[str] = []
    md_lines.append(f"# Calibration-size sensitivity: {args.model}")
    md_lines.append("")
    md_lines.append(
        "Mean per-layer Jaccard overlap of the kept-expert set against the reference N "
        "(the largest N available per group — 'all' when present, else max integer)."
    )
    md_lines.append("")
    header = ["task", "keep_k"] + [f"N={t}" for t in all_tags]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join("---" for _ in header) + "|")

    for task, keep_k, summary in groups:
        row = [task, str(keep_k)]
        ref_tag = summary["reference_nprune"]
        for tag in all_tags:
            if tag not in summary["mean_jaccard"]:
                row.append("—")
            elif tag == ref_tag:
                row.append("(ref)")
            else:
                v = summary["mean_jaccard"][tag]
                row.append(f"{v:.3f}" if v is not None else "—")
        md_lines.append("| " + " | ".join(row) + " |")

    md_lines.append("")
    md_lines.append("## Min-layer Jaccard (worst-case per group)")
    md_lines.append("")
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join("---" for _ in header) + "|")
    for task, keep_k, summary in groups:
        row = [task, str(keep_k)]
        ref_tag = summary["reference_nprune"]
        for tag in all_tags:
            if tag not in summary["min_jaccard"]:
                row.append("—")
            elif tag == ref_tag:
                row.append("(ref)")
            else:
                v = summary["min_jaccard"][tag]
                row.append(f"{v:.3f}" if v is not None else "—")
        md_lines.append("| " + " | ".join(row) + " |")

    out_md = model_dir / "summary.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    logger.info(f"Wrote {out_md} ({len(groups)} (task, keep_k) groups)")


if __name__ == "__main__":
    main()
