#!/usr/bin/env python3
"""Pull and display FlexMoE extension eval results from WEKA."""

import json
import os
import subprocess

MODELS = {
    "base": "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308_step30995-hf",
    "math": "moereducedp512sharedexp1_132experts_4trained_math_init_top2_average_train_act_10B_lr_4e-4_20260407a_step2385-hf",
    "code": "code-ta-01_lr4e-4_10B_20260407-234403_step2385-hf",
    "croissant": "croissant-ta-01_lr4e-4_10B_20260407-234459_step2385-hf",
}

TASKS = {
    "MC9": [
        "arc_easy_mc",
        "arc_challenge_mc",
        "boolq_mc",
        "csqa_mc",
        "hellaswag_mc",
        "openbookqa_mc",
        "piqa_mc",
        "socialiqa_mc",
        "winogrande_mc",
    ],
    "Gen5": ["coqa", "squad", "naturalqs_open", "triviaqa", "drop"],
    "Math": [
        "gsm8k",
        "minerva_math_algebra",
        "minerva_math_counting_and_probability",
        "minerva_math_geometry",
        "minerva_math_intermediate_algebra",
        "minerva_math_number_theory",
        "minerva_math_prealgebra",
        "minerva_math_precalculus",
    ],
    "Medical": ["medqa", "medmcqa_mc"],
    "Code": ["codex_humaneval", "codex_humanevalplus", "mbpp", "mbppplus"],
}

ORDER = ["base", "math", "code", "croissant"]
BASE = "s3://oe-training-default/kevinf/eval_results/flexmoe"


def fetch(mname: str, mpath: str, task: str) -> float | None:
    local = f"/tmp/eval_{mname}_{task}.json"
    key = f"{BASE}/{mpath}/task-{task}-metrics.json"
    r = subprocess.run(
        ["aws", "--profile", "WEKA", "s3", "cp", key, local],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if r.returncode == 0 and os.path.exists(local):
        try:
            with open(local) as f:
                return json.load(f)["metrics"]["primary_score"]
        except Exception:
            pass
    return None


def main():
    results: dict[str, dict[str, float]] = {}
    all_tasks = [t for group in TASKS.values() for t in group]

    for mname, mpath in MODELS.items():
        results[mname] = {}
        print(f"Fetching {mname}...", end="", flush=True)
        for task in all_tasks:
            score = fetch(mname, mpath, task)
            if score is not None:
                results[mname][task] = score
                print(".", end="", flush=True)
        print(f" {len(results[mname])} tasks")

    print(f"{'Task':<45} {'base':>10} {'math':>10} {'code':>10} {'croissant':>10}")
    print("=" * 88)

    for group, task_list in TASKS.items():
        print(f"\n--- {group} ---")
        group_scores: dict[str, list[float]] = {m: [] for m in ORDER}
        for task in task_list:
            row = []
            for m in ORDER:
                if task in results[m]:
                    row.append(f"{results[m][task]:.4f}")
                    group_scores[m].append(results[m][task])
                else:
                    row.append("--")
            print(f"  {task:<43} {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>10}")

        avgs = []
        for m in ORDER:
            if group_scores[m]:
                avgs.append(f"{sum(group_scores[m]) / len(group_scores[m]):.4f}")
            else:
                avgs.append("--")
        print(f"  {'AVG':<43} {avgs[0]:>10} {avgs[1]:>10} {avgs[2]:>10} {avgs[3]:>10}")


if __name__ == "__main__":
    main()
