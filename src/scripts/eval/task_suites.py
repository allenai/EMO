from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS


def get_task_suite_configs():
    from offline_evals.tasks import chembench, frenchbench, legalbench

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
        },
    )
    return TASK_SUITE_CONFIGS
