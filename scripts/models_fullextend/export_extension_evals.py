"""Aggregate the new-expert extension eval results into a JSON the report embeds.

For each of the four models (no-ghost baseline + three ghost variants) and each phase
-- BEFORE extension (`<model>_pre`, the original 128-expert checkpoint) and AFTER
extension (`<model>_ext`, the 129-expert FineMath-continual-pretrained checkpoint) --
reads primary_score from each per-task metrics.json under
models_fullextend/extension_evals/<model>_<phase>/<sanitized_task>/metrics.json.

Columns: MC9 (mean of the 9 OLMES rc tasks), SQuAD, TriviaQA, GSM8K, Minerva MATH-500,
BasicSkills, MBPP, HumanEval. NOTE: MBPP / HumanEval primary_score is bits-per-byte
(`bits_per_byte_corr`), where LOWER is better; all other columns are accuracy/F1 (higher
is better). The report annotates this.

Re-run after more eval jobs finish, then rebuild + publish the report.

    python scripts/models_fullextend/export_extension_evals.py
"""

import argparse
import json
from pathlib import Path

# Sanitized task dir name = task with all non-alphanumeric chars stripped
# (matches launch_extension_eval.sh: sed 's/[^a-zA-Z0-9]//g').
MC9_TASKS = [
    "arc_easy:rc::olmes", "arc_challenge:rc::olmes", "boolq:rc::olmes", "csqa:rc::olmes",
    "hellaswag:rc::olmes", "openbookqa:rc::olmes", "piqa:rc::olmes", "socialiqa:rc::olmes",
    "winogrande:rc::olmes",
]

# (column key, display label, raw task string, lower_is_better)
COLUMNS = [
    ("mc9", "MC9 (avg, acc)", None, False),  # special-cased: mean over MC9_TASKS
    ("squad", "SQuAD (F1)", "squad::olmes", False),
    ("triviaqa", "TriviaQA (F1)", "triviaqa::olmes", False),
    ("gsm8k", "GSM8K (EM)", "gsm8k::olmes", False),
    ("minerva_math", "Minerva MATH-500 (EM)", "minerva_math_500::olmes", False),
    ("basic_skills", "BasicSkills (acc)", "basic_skills::olmes", False),
    ("mbpp", "MBPP (bpb)", "mbpp:3shot:bpb::none", True),
    ("humaneval", "HumanEval (bpb)", "codex_humaneval:3shot:bpb::none", True),
]

# (model key, display label). Display order = row order.
MODELS = [
    ("noghost", "no-ghost EMO baseline"),
    ("ghost_uniform", "uniform ghost"),
    ("ghost_usage", "usage ghost"),
    ("ghost_random", "random ghost"),
]


def sanitize(task: str) -> str:
    return "".join(c for c in task if c.isalnum())


def primary(root: Path, model_phase: str, task: str):
    f = root / model_phase / sanitize(task) / "metrics.json"
    if not f.is_file():
        return None
    try:
        return json.load(open(f))["metrics"][0]["primary_score"]
    except Exception:
        return None


def phase_scores(root: Path, model_phase: str) -> dict:
    out = {}
    for key, _label, task, _lb in COLUMNS:
        if key == "mc9":
            vals = [primary(root, model_phase, t) for t in MC9_TASKS]
            vals = [v for v in vals if v is not None]
            out[key] = round(sum(vals) / len(vals), 4) if vals else None
            out["mc9_n"] = len(vals)
        else:
            v = primary(root, model_phase, task)
            out[key] = round(v, 4) if v is not None else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("models_fullextend/extension_evals"))
    ap.add_argument("--output", type=Path,
                    default=Path("claude_outputs/models_fullextend/extension_eval_results.json"))
    args = ap.parse_args()

    models = []
    for mkey, label in MODELS:
        pre = phase_scores(args.root, f"{mkey}_pre")
        ext = phase_scores(args.root, f"{mkey}_ext")
        models.append({"key": mkey, "label": label, "pre": pre, "ext": ext})
        print(f"  {label:22s} pre(mc9 n={pre['mc9_n']})  ext(mc9 n={ext['mc9_n']})")

    out = {
        "columns": [{"key": k, "label": l, "lower_better": lb} for k, l, _t, lb in COLUMNS],
        "mc9_tasks": MC9_TASKS,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
