"""Collect claude_outputs/debug_validation/ppl_validation/<run>/step<N>.json into one table.

    python scripts/debug_validation/aggregate_ppl_validation.py [--root ...] [--metric "CE loss"|"PPL"|"CE loss (labeled tokens)"]

Writes <root>/summary.json (all runs x steps x sets, all metrics) and <root>/table_<metric>.md
(one markdown table per run: rows = checkpoints, columns = validation sets + mean over sets), and
prints the markdown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SETS = [
    "c4_en",
    "dolma_books",
    "dolma_common-crawl",
    "dolma_pes2o",
    "dolma_reddit",
    "dolma_stack",
    "dolma_wiki",
    "ice",
    "m2d2_s2orc",
    "pile",
    "wikitext_103",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=Path, default=Path("claude_outputs/debug_validation/ppl_validation")
    )
    ap.add_argument("--metric", default="CE loss")
    args = ap.parse_args()

    runs = {}
    for f in sorted(args.root.glob("*/step*.json")):
        d = json.load(open(f))
        runs.setdefault(d["run"], {})[d["step"]] = d
    if not runs:
        raise SystemExit(f"no results under {args.root}")
    json.dump(runs, open(args.root / "summary.json", "w"), indent=1)

    short = {s: s.replace("dolma_", "").replace("common-crawl", "cc") for s in SETS}
    lines = [f"# v3-small ppl validation: {args.metric}", ""]
    for run, steps in runs.items():
        lines += [
            f"## {run}",
            "",
            "| step | tokens | " + " | ".join(short[s] for s in SETS) + " | mean |",
            "|---|---|" + "---|" * (len(SETS) + 1),
        ]
        for step in sorted(steps):
            per = steps[step]["per_set"]
            vals = [per[f"{s}-validation"][args.metric] for s in SETS]
            tokens = step * 524_288
            lines.append(
                f"| {step} | {tokens / 1e9:.2f}B | "
                + " | ".join(f"{v:.3f}" for v in vals)
                + f" | {sum(vals) / len(vals):.3f} |"
            )
        lines.append("")
    if len(runs) == 2:
        (a, sa), (b, sb) = runs.items()
        common = sorted(set(sa) & set(sb))
        lines += [
            f"## {b} minus {a} ({args.metric})",
            "",
            "| step | " + " | ".join(short[s] for s in SETS) + " | mean |",
            "|---|" + "---|" * (len(SETS) + 1),
        ]
        for step in common:
            diffs = [
                sb[step]["per_set"][f"{s}-validation"][args.metric]
                - sa[step]["per_set"][f"{s}-validation"][args.metric]
                for s in SETS
            ]
            lines.append(
                f"| {step} | "
                + " | ".join(f"{v:+.3f}" for v in diffs)
                + f" | {sum(diffs) / len(diffs):+.3f} |"
            )
        lines.append("")
    md = "\n".join(lines)
    out = args.root / f"table_{args.metric.replace(' ', '_').replace('(', '').replace(')', '')}.md"
    out.write_text(md)
    print(md)
    print(f"wrote {out} and {args.root / 'summary.json'}")


if __name__ == "__main__":
    main()
