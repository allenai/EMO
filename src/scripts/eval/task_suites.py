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
            # ChemBench MC only (multiple choice questions)
            "chembench:mc": {
                "tasks": [
                    "chembench_analytical_chemistry:mc",
                    "chembench_chemical_preference:mc",
                    "chembench_general_chemistry:mc",
                    "chembench_inorganic_chemistry:mc",
                    "chembench_materials_science:mc",
                    "chembench_organic_chemistry:mc",
                    "chembench_physical_chemistry:mc",
                    "chembench_technical_chemistry:mc",
                    "chembench_toxicity_and_safety:mc",
                ],
                "primary_metric": "macro",
            },
            # ChemBench generative/open-ended questions
            "chembench:gen": {
                "tasks": [
                    "chembench_analytical_chemistry:gen",
                    "chembench_chemical_preference:gen",
                    "chembench_general_chemistry:gen",
                    "chembench_inorganic_chemistry:gen",
                    "chembench_materials_science:gen",
                    "chembench_organic_chemistry:gen",
                    "chembench_physical_chemistry:gen",
                    "chembench_technical_chemistry:gen",
                    "chembench_toxicity_and_safety:gen",
                ],
                "primary_metric": "macro",
            },
            # ChemBench ranked classification (cloze prompt)
            "chembench:rc": {
                "tasks": [
                    "chembench_analytical_chemistry:rc",
                    "chembench_chemical_preference:rc",
                    "chembench_general_chemistry:rc",
                    "chembench_inorganic_chemistry:rc",
                    "chembench_materials_science:rc",
                    "chembench_organic_chemistry:rc",
                    "chembench_physical_chemistry:rc",
                    "chembench_technical_chemistry:rc",
                    "chembench_toxicity_and_safety:rc",
                ],
                "primary_metric": "macro",
            },
        },
    )
    from offline_evals.tasks import frenchbench, legalbench

    TASK_SUITE_CONFIGS.update(
        {
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
