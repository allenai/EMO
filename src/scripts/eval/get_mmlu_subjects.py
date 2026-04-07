"""Print the MMLU subjects belonging to a given category or cluster task name.

Usage:
    python -m src.scripts.eval.get_mmlu_subjects mmlu_biology
    python -m src.scripts.eval.get_mmlu_subjects mmlu_cluster_biomedical

Prints one subject per line. Exits with code 1 if the task is not an MMLU
category/cluster task (e.g., arc_challenge), printing nothing.
"""

import sys

from src.offline_evals.tasks.splits_mmlu import MMLU_CATEGORIES, MMLU_CLUSTER_CATEGORIES


def get_subjects(task_name: str):
    """Return list of subjects for an MMLU category/cluster task, or None."""
    # Strip mmlu_ prefix
    if not task_name.startswith("mmlu_"):
        return None

    key = task_name[len("mmlu_"):]

    # Strip a "merged_" prefix so mmlu_merged_<cat> resolves the same as mmlu_<cat>.
    if key.startswith("merged_"):
        key = key[len("merged_"):]

    # Check cluster categories first (they have "cluster_" prefix)
    if key in MMLU_CLUSTER_CATEGORIES:
        return MMLU_CLUSTER_CATEGORIES[key]

    # Check human categories
    if key in MMLU_CATEGORIES:
        return MMLU_CATEGORIES[key]

    return None


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python -m src.scripts.eval.get_mmlu_subjects <task_name>", file=sys.stderr)
        sys.exit(1)

    task_name = sys.argv[1]
    subjects = get_subjects(task_name)

    if subjects is None:
        sys.exit(1)

    for subject in subjects:
        print(subject)


if __name__ == "__main__":
    main()
