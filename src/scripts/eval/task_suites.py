from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
from oe_eval.data.mmlu_pro_categories import MMLU_PRO_CATEGORIES
from oe_eval.data.mmlu_tasks import MMLU_SUBJECTS


def get_task_suite_configs():
    TASK_SUITE_CONFIGS.update(
        {
            "sciriff5": {
                "tasks": [
                    "sciriff_bioasq_factoid_qa",  # Abstractive
                    "sciriff_bioasq_general_qa",  # Abstractive
                    "sciriff_bioasq_yesno_qa",  # Y/N
                    "sciriff_covid_deepset_qa",  # Extractive
                    "sciriff_pubmedqa_qa",  # Y/N
                ],
                "primary_metric": "macro",
            },
        },
    )

    # adding MMLU subjects
    TASK_SUITE_CONFIGS["mmlu:mc_train::olmes"] = {
        "tasks": [f"mmlu_{sub}:mc_train::olmes" for sub in MMLU_SUBJECTS],
    }
    TASK_SUITE_CONFIGS["mmlu:mc_validation::olmes"] = {
        "tasks": [f"mmlu_{sub}:mc_validation::olmes" for sub in MMLU_SUBJECTS],
    }
    TASK_SUITE_CONFIGS["mmlu:mc_test::olmes"] = {
        "tasks": [f"mmlu_{sub}:mc_test::olmes" for sub in MMLU_SUBJECTS],
    }
    TASK_SUITE_CONFIGS["mmlu:rc_train::olmes"] = {
        "tasks": [f"mmlu_{sub}:rc_train::olmes" for sub in MMLU_SUBJECTS],
    }
    TASK_SUITE_CONFIGS["mmlu:rc_validation::olmes"] = {
        "tasks": [f"mmlu_{sub}:rc_validation::olmes" for sub in MMLU_SUBJECTS],
    }
    TASK_SUITE_CONFIGS["mmlu:rc_test::olmes"] = {
        "tasks": [f"mmlu_{sub}:rc_test::olmes" for sub in MMLU_SUBJECTS],
    }
    TASK_SUITE_CONFIGS["mmlu:rc_validation_0shot::olmes"] = {
        "tasks": [f"mmlu_{sub}:rc_validation_0shot::olmes" for sub in MMLU_SUBJECTS],
    }
    TASK_SUITE_CONFIGS["mmlu:rc_test_0shot::olmes"] = {
        "tasks": [f"mmlu_{sub}:rc_test_0shot::olmes" for sub in MMLU_SUBJECTS],
    }

    # adding MMLU Pro
    TASK_SUITE_CONFIGS["mmlu_pro:mc_train::olmes"] = {
        "tasks": [f"mmlu_pro_{sub}:mc_train::olmes" for sub in MMLU_PRO_CATEGORIES],
    }
    TASK_SUITE_CONFIGS["mmlu_pro:mc_validation::olmes"] = {
        "tasks": [f"mmlu_pro_{sub}:mc_validation::olmes" for sub in MMLU_PRO_CATEGORIES],
    }
    TASK_SUITE_CONFIGS["mmlu_pro:mc_test::olmes"] = {
        "tasks": [f"mmlu_pro_{sub}:mc_test::olmes" for sub in MMLU_PRO_CATEGORIES],
    }
    TASK_SUITE_CONFIGS["mmlu_pro:rc_train::olmes"] = {
        "tasks": [f"mmlu_pro_{sub}:rc_train::olmes" for sub in MMLU_PRO_CATEGORIES],
    }
    TASK_SUITE_CONFIGS["mmlu_pro:rc_validation::olmes"] = {
        "tasks": [f"mmlu_pro_{sub}:rc_validation::olmes" for sub in MMLU_PRO_CATEGORIES],
    }
    TASK_SUITE_CONFIGS["mmlu_pro:rc_test::olmes"] = {
        "tasks": [f"mmlu_pro_{sub}:rc_test::olmes" for sub in MMLU_PRO_CATEGORIES],
    }

    return TASK_SUITE_CONFIGS
