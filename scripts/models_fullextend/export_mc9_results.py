"""Aggregate MC9 eval results (primary_score = acc_raw) from the per-job metrics.json
files under models_fullextend/mc9_evals/ into a JSON the report embeds as a table.

Reads <root>/<run>/<mode>/<task><suffix>/metrics.json. Re-run after more eval jobs
finish, then rebuild + publish the report.

    python scripts/models_fullextend/export_mc9_results.py
"""

import argparse
import json
from pathlib import Path

TASKS = ["arc_easy", "arc_challenge", "boolq", "csqa", "hellaswag",
         "openbookqa", "piqa", "socialiqa", "winogrande"]

# (run dir, mode, column label). Order = display order.
COLUMNS = [
    ("no_ghost_baseline_130b", "standard", "no-ghost baseline (130B)"),
    ("ghost_usage_50b", "standard", "usage ghost-trained, ghost OFF (50B)"),
    ("ghost_usage_50b", "ghost", "usage ghost-trained, ghost ON (50B)"),
    ("ghost_uniform_50b", "standard", "uniform ghost-trained, ghost OFF (50B)"),
    ("ghost_uniform_50b", "ghost", "uniform ghost-trained, ghost ON (50B)"),
]


def score(root: Path, run: str, mode: str, task: str, suffix: str):
    f = root / run / mode / f"{task}{suffix}" / "metrics.json"
    if not f.is_file():
        return None
    try:
        return round(json.load(open(f))["metrics"][0]["primary_score"], 4)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("models_fullextend/mc9_evals"))
    ap.add_argument("--suffix", default=":rc::olmes")
    ap.add_argument("--output", type=Path,
                    default=Path("claude_outputs/models_fullextend/mc9_results.json"))
    args = ap.parse_args()

    columns = []
    for run, mode, label in COLUMNS:
        scores = {t: score(args.root, run, mode, t, args.suffix) for t in TASKS}
        present = [v for v in scores.values() if v is not None]
        avg = round(sum(present) / len(present), 4) if present else None
        columns.append({"run": run, "mode": mode, "label": label,
                        "scores": scores, "avg": avg, "n": len(present)})
        print(f"  {label}: n={len(present)}/9 avg={avg}")

    out = {"metric": "acc_raw", "task_suffix": args.suffix, "tasks": TASKS, "columns": columns}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
