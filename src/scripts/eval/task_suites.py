from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS


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
    return TASK_SUITE_CONFIGS
