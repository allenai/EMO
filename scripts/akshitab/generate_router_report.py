"""
Router Analysis Report: Expert Activation Across Models and Tasks

Compares base model (128 experts) vs extended models (132 experts, with 4 new
experts at indices 128-131) vs trained extended models.

Analyzes:
  - General: hellaswag
  - Math: gsm8k (+ minerva_math_500 where available)
  - Code: mbpp, codex_humaneval
"""

import json
import os
import sys
from collections import defaultdict

import torch

ROUTER_EVALS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "router_evals")

# ── Model definitions ────────────────────────────────────────────────────────
MODELS = {
    "Base (128 experts)": "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123_step30995-hf",
    "Math Ext (init)": "extensions_moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc-hf",
    "Code Ext (init)": "extensions_moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_code_average_noise-hf",
    "Math Ext (trained)": "freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_10B_lr_4e-4_step2385-hf",
    "Code Ext (trained)": "ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4_step2385-hf",
    "Merged Ext": "merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf",
}

# ── Task definitions ─────────────────────────────────────────────────────────
TASK_GROUPS = {
    "General": ["hellaswag_rc_test"],
    "Math": ["gsm8k", "minerva_math_500"],
    "Code": ["mbpp", "codex_humaneval"],
}

TASK_FILE_MAP = {
    "hellaswag_rc_test": "task-hellaswag_rc_test-router.jsonl",
    "gsm8k": "task-gsm8k-router.jsonl",
    "minerva_math_500": "task-minerva_math_500-router.jsonl",
    "mbpp": "task-mbpp-router.jsonl",
    "codex_humaneval": "task-codex_humaneval-router.jsonl",
}

NEW_EXPERT_IDS = [128, 129, 130, 131]
TOP_K = 8


def load_router_probs(model_dir, task_file):
    path = os.path.join(ROUTER_EVALS_DIR, model_dir, task_file)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.loads(f.readline())
    return torch.tensor(data["avg_router_probabilities"])


def fmt_pct(val, width=6):
    return f"{val * 100:{width}.2f}%"


def separator(width=120):
    return "=" * width


def thin_sep(width=120):
    return "-" * width


# ── Report sections ──────────────────────────────────────────────────────────

def section_header(title):
    print(f"\n{separator()}")
    print(f"  {title}")
    print(separator())


def report_availability():
    """Print a matrix showing which model/task combinations are available."""
    section_header("DATA AVAILABILITY")
    all_tasks = [t for tasks in TASK_GROUPS.values() for t in tasks]

    header = f"{'Model':<22}" + "".join(f"{t:<22}" for t in all_tasks)
    print(header)
    print(thin_sep(len(header)))
    for model_name, model_dir in MODELS.items():
        row = f"{model_name:<22}"
        for task in all_tasks:
            path = os.path.join(ROUTER_EVALS_DIR, model_dir, TASK_FILE_MAP[task])
            row += f"{'YES':<22}" if os.path.exists(path) else f"{'---':<22}"
        print(row)


def report_new_expert_activation():
    """For extended models (132 experts), report how much probability mass
    the 4 new experts (128-131) receive per task, averaged across layers."""
    section_header("NEW EXPERT ACTIVATION (experts 128-131) — avg probability mass across all layers")

    all_tasks = [t for tasks in TASK_GROUPS.values() for t in tasks]
    extended_models = {k: v for k, v in MODELS.items() if k != "Base (128 experts)"}

    header = f"{'Model':<22}" + "".join(f"{t:<22}" for t in all_tasks)
    print(header)
    print(thin_sep(len(header)))

    for model_name, model_dir in extended_models.items():
        row = f"{model_name:<22}"
        for task in all_tasks:
            probs = load_router_probs(model_dir, TASK_FILE_MAP[task])
            if probs is None:
                row += f"{'---':<22}"
                continue
            # Sum of probs for new experts, averaged across layers
            new_expert_probs = probs[:, NEW_EXPERT_IDS].sum(dim=1)  # (num_layers,)
            avg = new_expert_probs.mean().item()
            row += f"{fmt_pct(avg):<22}"
        print(row)


def report_new_expert_per_layer(model_name, model_dir, task, probs):
    """Detailed per-layer breakdown of new expert probabilities."""
    num_layers = probs.shape[0]
    print(f"\n  {model_name} | {task}")
    print(f"  {'Layer':<7} {'Exp128':>8} {'Exp129':>8} {'Exp130':>8} {'Exp131':>8} {'Sum128-131':>12}  {'Top-1 Expert (prob)':>22}")
    print(f"  {thin_sep(75)}")
    for layer in range(num_layers):
        exp_probs = [probs[layer, eid].item() for eid in NEW_EXPERT_IDS]
        total = sum(exp_probs)
        top_val, top_idx = probs[layer].max(dim=0)
        print(
            f"  {layer:<7}"
            + "".join(f" {fmt_pct(p):>8}" for p in exp_probs)
            + f" {fmt_pct(total):>12}"
            + f"  Expert {top_idx.item():>3} ({fmt_pct(top_val.item())})"
        )


def report_new_expert_details():
    """Per-layer new-expert breakdown for all available extended model/task combos."""
    section_header("PER-LAYER NEW EXPERT (128-131) PROBABILITY BREAKDOWN")

    extended_models = {k: v for k, v in MODELS.items() if k != "Base (128 experts)"}
    for group_name, tasks in TASK_GROUPS.items():
        print(f"\n  ── {group_name} Tasks ──")
        for task in tasks:
            for model_name, model_dir in extended_models.items():
                probs = load_router_probs(model_dir, TASK_FILE_MAP[task])
                if probs is None:
                    continue
                report_new_expert_per_layer(model_name, model_dir, task, probs)


def report_top_experts_comparison():
    """For each task, show top-K experts (globally summed across layers) for every model."""
    section_header("GLOBAL TOP-8 EXPERTS (summed probability across layers) — by task")

    for group_name, tasks in TASK_GROUPS.items():
        for task in tasks:
            print(f"\n  ── {group_name}: {task} ──")
            for model_name, model_dir in MODELS.items():
                probs = load_router_probs(model_dir, TASK_FILE_MAP[task])
                if probs is None:
                    continue
                global_probs = probs.sum(dim=0)
                values, indices = torch.topk(global_probs, TOP_K)
                entries = ", ".join(
                    f"E{idx.item():>3}({fmt_pct(val.item()).strip()})"
                    for idx, val in zip(indices, values)
                )
                is_new = any(idx.item() in NEW_EXPERT_IDS for idx in indices)
                marker = " ** NEW EXPERT IN TOP-8 **" if is_new else ""
                print(f"  {model_name:<22} {entries}{marker}")


def report_expert_rank_shift():
    """Show how the new experts rank compared to original experts per task."""
    section_header("NEW EXPERT RANK ANALYSIS (where do experts 128-131 rank globally?)")

    extended_models = {k: v for k, v in MODELS.items() if k != "Base (128 experts)"}
    for group_name, tasks in TASK_GROUPS.items():
        for task in tasks:
            print(f"\n  ── {group_name}: {task} ──")
            for model_name, model_dir in extended_models.items():
                probs = load_router_probs(model_dir, TASK_FILE_MAP[task])
                if probs is None:
                    continue
                global_probs = probs.sum(dim=0)
                sorted_indices = torch.argsort(global_probs, descending=True)
                rank_of = {idx.item(): rank for rank, idx in enumerate(sorted_indices)}
                parts = []
                for eid in NEW_EXPERT_IDS:
                    if eid < probs.shape[1]:
                        parts.append(f"E{eid}=rank {rank_of[eid]+1:>3} ({fmt_pct(global_probs[eid].item()).strip()})")
                print(f"  {model_name:<22} {', '.join(parts)}")


def report_init_vs_trained_comparison():
    """Compare init vs trained for the same extension type."""
    section_header("INIT vs TRAINED: New Expert Probability Mass (avg across layers)")

    comparisons = [
        ("Math Extension", "Math Ext (init)", "Math Ext (trained)"),
        ("Code Extension", "Code Ext (init)", "Code Ext (trained)"),
    ]
    all_tasks = [t for tasks in TASK_GROUPS.values() for t in tasks]

    for comp_name, init_key, trained_key in comparisons:
        print(f"\n  ── {comp_name} ──")
        init_dir = MODELS[init_key]
        trained_dir = MODELS[trained_key]

        header = f"  {'Stage':<22}" + "".join(f"{t:<20}" for t in all_tasks)
        print(header)
        print(f"  {thin_sep(22 + 20 * len(all_tasks))}")

        for stage_name, model_dir in [(init_key, init_dir), (trained_key, trained_dir)]:
            row = f"  {stage_name:<22}"
            for task in all_tasks:
                probs = load_router_probs(model_dir, TASK_FILE_MAP[task])
                if probs is None:
                    row += f"{'---':<20}"
                    continue
                new_expert_probs = probs[:, NEW_EXPERT_IDS].sum(dim=1).mean().item()
                row += f"{fmt_pct(new_expert_probs):<20}"
            print(row)


def report_per_layer_heatmap_text():
    """Text heatmap: for each task, show new-expert sum per layer across models side by side."""
    section_header("PER-LAYER NEW EXPERT MASS: Side-by-Side Model Comparison")

    extended_models = {k: v for k, v in MODELS.items() if k != "Base (128 experts)"}

    for group_name, tasks in TASK_GROUPS.items():
        for task in tasks:
            available = []
            for model_name, model_dir in extended_models.items():
                probs = load_router_probs(model_dir, TASK_FILE_MAP[task])
                if probs is not None:
                    available.append((model_name, probs))
            if not available:
                continue

            print(f"\n  ── {group_name}: {task} ──")
            header = f"  {'Layer':<7}" + "".join(f"{name:>22}" for name, _ in available)
            print(header)
            print(f"  {thin_sep(7 + 22 * len(available))}")

            num_layers = available[0][1].shape[0]
            for layer in range(num_layers):
                row = f"  {layer:<7}"
                for _, probs in available:
                    val = probs[layer, NEW_EXPERT_IDS].sum().item() if probs.shape[1] > 128 else 0
                    row += f"{fmt_pct(val):>22}"
                print(row)

            # Average row
            row = f"  {'AVG':<7}"
            for _, probs in available:
                val = probs[:, NEW_EXPERT_IDS].sum(dim=1).mean().item() if probs.shape[1] > 128 else 0
                row += f"{fmt_pct(val):>22}"
            print(row)


def main():
    print(separator())
    print("  ROUTER ANALYSIS REPORT: Expert Activation Across Models and Tasks")
    print(f"  Base model: 128 experts | Extended models: 132 experts (new: 128-131)")
    print(separator())

    report_availability()
    report_new_expert_activation()
    report_top_experts_comparison()
    report_expert_rank_shift()
    report_init_vs_trained_comparison()
    report_per_layer_heatmap_text()
    report_new_expert_details()

    print(f"\n{separator()}")
    print("  END OF REPORT")
    print(separator())


if __name__ == "__main__":
    main()
