#!/usr/bin/env python3
"""Summarise finetune-free selective-expert evals (``run_item.py`` outputs) per task group.

Reads ``<base-dir>/<model>/<task>_keepk_<K>_.../results/checkpoint-0/task-*-metrics.json`` and
prints, per (model, keep-k), the task-group averages that the paper tables report
(``scripts/plotting/get_table_scores_selective_evals_final.py``): MC9, Gen5, MMLU (17
categories, average without ``other``), MMLU-Pro (14 categories, average without ``other``) and
GSM8K, all on each task's ``primary_score``.  Partial coverage is shown as ``n/N``.

    python scripts/selective_hf/olmo35/report.py [--base-dir ...] [--format markdown|csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import MODELS, TASKS  # noqa: E402

MC9 = TASKS[:9]
GSM8K = ["gsm8k_generation_8shot_merged"]
GEN5 = ["squad_merged", "coqa_merged", "naturalqs_merged", "triviaqa_merged", "drop_merged"]
MMLU = [t for t in TASKS if t.startswith("mmlu_merged_")]
MMLU_PRO = [t for t in TASKS if t.startswith("mmlu_pro_merged_")]
GROUPS = [
    ("mc9", MC9, []),
    ("gen5", GEN5, []),
    ("mmlu_no_other", MMLU, ["mmlu_merged_other"]),
    ("mmlu_pro_no_other", MMLU_PRO, ["mmlu_pro_merged_other"]),
    ("gsm8k", GSM8K, []),
]
DIR_RE = re.compile(r"^(?P<task>.+)_keepk_(?P<k>[^_]+)_bs-\d+_lr-[^_]+_epoch-\d+_selectivemode-(?P<mode>.+)$")


def sanitized(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", model)


def collect(base: Path) -> dict:
    """{model_key: {keep_k: {task: primary_score}}}"""
    by_sanitized = {sanitized(p): key for key, p in MODELS.items()}
    out: dict = {}
    for model_dir in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("__")):
        key = by_sanitized.get(model_dir.name, model_dir.name)
        for task_dir in sorted(model_dir.iterdir()):
            m = DIR_RE.match(task_dir.name)
            if not m:
                continue
            files = sorted((task_dir / "results" / "checkpoint-0").glob("task-*-metrics.json"))
            if not files:
                continue
            data = json.loads(files[0].read_text())
            score = (data.get("metrics") or {}).get("primary_score")
            if score is None:
                continue
            out.setdefault(key, {}).setdefault(m["k"], {})[m["task"]] = float(score)
    return out


def k_order(k: str):
    return (1, 0) if k == "full" else (0, int(k))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default="/root/scaling-ladders/output/emo_selective/selective_evals_final")
    ap.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    ap.add_argument("--per-task", action="store_true", help="also print every task's primary score")
    ap.add_argument("--json-out", default=None, help="write the full per-task table as JSON")
    args = ap.parse_args()

    scores = collect(Path(args.base_dir))
    rows = []
    for model in scores:
        for k in sorted(scores[model], key=k_order):
            row = {"model": model, "keep_k": k}
            for name, tasks, excluded in GROUPS:
                avg_tasks = [t for t in tasks if t not in excluded]
                have = [scores[model][k][t] for t in avg_tasks if t in scores[model][k]]
                row[name] = f"{100 * sum(have) / len(have):.1f}" if have else ""
                row[name + "_n"] = f"{len(have)}/{len(avg_tasks)}"
            rows.append(row)
    cols = ["model", "keep_k"] + [g for g, _, _ in GROUPS]
    if args.format == "csv":
        w = csv.writer(sys.stdout)
        w.writerow(cols + [c + "_n" for c in cols[2:]])
        for r in rows:
            w.writerow([r[c] for c in cols] + [r[c + "_n"] for c in cols[2:]])
    else:
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for r in rows:
            cells = [r["model"], r["keep_k"]] + [f"{r[g]} ({r[g + '_n']})" if r[g] else f"– ({r[g + '_n']})" for g, _, _ in GROUPS]
            print("| " + " | ".join(cells) + " |")
    if args.per_task:
        print()
        for model in scores:
            ks = sorted(scores[model], key=k_order)
            print(f"\n### {model}\n\n| task | " + " | ".join(ks) + " |\n|---|" + "---|" * len(ks))
            for t in TASKS:
                vals = [f"{100 * scores[model][k][t]:.1f}" if t in scores[model][k] else "" for k in ks]
                if any(vals):
                    print(f"| {t} | " + " | ".join(vals) + " |")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"groups": rows, "per_task": scores}, indent=1))


if __name__ == "__main__":
    main()
