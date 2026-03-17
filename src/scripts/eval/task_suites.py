from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
from oe_eval.data.mmlu_pro_categories import MMLU_PRO_CATEGORIES
from oe_eval.data.mmlu_tasks import MMLU_SUBJECTS


def get_task_suite_configs():
    from offline_evals.tasks import chembench, code_fresh, frenchbench, legalbench

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
            "chembench:mc": {
                "tasks": [f"chembench_{s}:mc" for s in chembench.CHEMBENCH_SUBFIELDS],
                "primary_metric": "macro",
            },
            "chembench:gen": {
                "tasks": [f"chembench_{s}:gen" for s in chembench.CHEMBENCH_GEN_SUBFIELDS],
                "primary_metric": "macro",
            },
            "chembench:rc": {
                "tasks": [f"chembench_{s}:rc" for s in chembench.CHEMBENCH_SUBFIELDS],
                "primary_metric": "macro",
            },
            "legalbench:rc": {
                "tasks": [
                    f"legalbench_{task_name}:rc"
                    for task_name in legalbench.LEGALBENCH_CLASSIFICATION_TASKS.keys()
                ],
                "primary_metric": "macro",
            },
            "frenchbench:rc": {
                "tasks": list(frenchbench.create_frenchbench_tasks().keys()),
                "primary_metric": "macro",
            },
            "code_fresh_rolling:bpb": {
                "tasks": [
                    f"code_fresh_rolling:{lang}:bpb" for lang in code_fresh.CODE_FRESH_LANGUAGES
                ],
                "primary_metric": "macro",
            },
            "code_fresh_rolling:ppl": {
                "tasks": [
                    f"code_fresh_rolling:{lang}:ppl" for lang in code_fresh.CODE_FRESH_LANGUAGES
                ],
                "primary_metric": "macro",
            },
        },
    )

    # adding MMLU subjects
    TASK_SUITE_CONFIGS["mmlu:mc_validation::olmes"] = {
        "tasks": [f"mmlu_{sub}:mc_validation::olmes" for sub in MMLU_SUBJECTS],
    }
    TASK_SUITE_CONFIGS["mmlu:mc_test::olmes"] = {
        "tasks": [f"mmlu_{sub}:mc_test::olmes" for sub in MMLU_SUBJECTS],
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

    # adding MMLU Pro
    TASK_SUITE_CONFIGS["mmlu_pro:mc_validation::olmes"] = {
        "tasks": [f"mmlu_pro_{sub}:mc_validation::olmes" for sub in MMLU_PRO_CATEGORIES],
    }
    TASK_SUITE_CONFIGS["mmlu_pro:mc_test::olmes"] = {
        "tasks": [f"mmlu_pro_{sub}:mc_test::olmes" for sub in MMLU_PRO_CATEGORIES],
    }
    TASK_SUITE_CONFIGS["mmlu_pro:rc_validation::olmes"] = {
        "tasks": [f"mmlu_pro_{sub}:rc_validation::olmes" for sub in MMLU_PRO_CATEGORIES],
    }
    TASK_SUITE_CONFIGS["mmlu_pro:rc_test::olmes"] = {
        "tasks": [f"mmlu_pro_{sub}:rc_test::olmes" for sub in MMLU_PRO_CATEGORIES],
    }

    # MMLU Pro ::none suites (validation split, 5-shot)
    TASK_SUITE_CONFIGS["mmlu_pro:mc::none"] = {
        "tasks": [f"mmlu_pro_{sub}:mc_validation::none" for sub in MMLU_PRO_CATEGORIES],
    }
    TASK_SUITE_CONFIGS["mmlu_pro:rc::none"] = {
        "tasks": [f"mmlu_pro_{sub}:rc_validation::none" for sub in MMLU_PRO_CATEGORIES],
    }

    return TASK_SUITE_CONFIGS
