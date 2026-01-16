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
        },
    )
    return TASK_SUITE_CONFIGS
