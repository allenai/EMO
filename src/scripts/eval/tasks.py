from oe_eval.configs.tasks import TASK_CONFIGS
from oe_eval.data.mmlu_pro_categories import MMLU_PRO_CATEGORIES
from oe_eval.data.mmlu_tasks import MMLU_SUBJECTS


def get_task_configs():
    TASK_CONFIGS.update(
        {
            "winogrande:mc_train::olmes": {
                "task_name": "winogrande:mc_train",
                "split": "validation",
                "num_shots": 5,
                "fewshot_source": "OLMES:winogrande",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "winogrande:mc_validation::olmes": {
                "task_name": "winogrande:mc_validation",
                "split": "validation",
                "num_shots": 5,
                "fewshot_source": "OLMES:winogrande",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "winogrande:mc_test::olmes": {
                "task_name": "winogrande:mc_test",
                "split": "validation",
                "num_shots": 5,
                "fewshot_source": "OLMES:winogrande",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "winogrande:rc_train::olmes": {
                "task_name": "winogrande:rc_train",
                "split": "validation",
                "primary_metric": "acc_raw",
                "num_shots": 5,
                "fewshot_source": "OLMES:winogrande",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "winogrande:rc_validation::olmes": {
                "task_name": "winogrande:rc_validation",
                "split": "validation",
                "primary_metric": "acc_raw",
                "num_shots": 5,
                "fewshot_source": "OLMES:winogrande",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "winogrande:rc_test::olmes": {
                "task_name": "winogrande:rc_test",
                "split": "validation",
                "primary_metric": "acc_raw",
                "num_shots": 5,
                "fewshot_source": "OLMES:winogrande",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "socialiqa:mc_train::olmes": {
                "task_name": "socialiqa:mc_train",
                "split": "train",
                "num_shots": 5,
                "fewshot_source": "OLMES:social_i_qa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "socialiqa:mc_validation::olmes": {
                "task_name": "socialiqa:mc_validation",
                "split": "validation",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:social_i_qa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "socialiqa:mc_test::olmes": {
                "task_name": "socialiqa:mc_test",
                "split": "test",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:social_i_qa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "socialiqa:rc_train::olmes": {
                "task_name": "socialiqa:rc_train",
                "split": "train",
                "primary_metric": "acc_per_char",
                "num_shots": 5,
                "fewshot_source": "OLMES:social_i_qa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "socialiqa:rc_validation::olmes": {
                "task_name": "socialiqa:rc_validation",
                "split": "validation",
                "primary_metric": "acc_per_char",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:social_i_qa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "socialiqa:rc_test::olmes": {
                "task_name": "socialiqa:rc_test",
                "split": "test",
                "primary_metric": "acc_per_char",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:social_i_qa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "piqa:mc_train::olmes": {
                "task_name": "piqa:mc_train",
                "split": "train",
                "num_shots": 5,
                "fewshot_source": "OLMES:piqa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "piqa:mc_validation::olmes": {
                "task_name": "piqa:mc_validation",
                "split": "validation",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:piqa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "piqa:mc_test::olmes": {
                "task_name": "piqa:mc_test",
                "split": "test",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:piqa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "piqa:rc_train::olmes": {
                "task_name": "piqa:rc_train",
                "split": "train",
                "primary_metric": "acc_per_char",
                "num_shots": 5,
                "fewshot_source": "OLMES:piqa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "piqa:rc_validation::olmes": {
                "task_name": "piqa:rc_validation",
                "split": "validation",
                "primary_metric": "acc_per_char",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:piqa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "piqa:rc_test::olmes": {
                "task_name": "piqa:rc_test",
                "split": "test",
                "primary_metric": "acc_per_char",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:piqa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "openbookqa:mc_train::olmes": {
                "task_name": "openbookqa:mc_train",
                "split": "train",
                "num_shots": 5,
                "fewshot_source": "OLMES:openbookqa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "openbookqa:mc_validation::olmes": {
                "task_name": "openbookqa:mc_validation",
                "split": "validation",
                "num_shots": 5,
                "fewshot_source": "OLMES:openbookqa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "openbookqa:mc_test::olmes": {
                "task_name": "openbookqa:mc_test",
                "split": "test",
                "num_shots": 5,
                "fewshot_source": "OLMES:openbookqa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "openbookqa:rc_train::olmes": {
                "task_name": "openbookqa:rc_train",
                "split": "train",
                "primary_metric": "acc_uncond",
                "num_shots": 5,
                "fewshot_source": "OLMES:openbookqa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "openbookqa:rc_validation::olmes": {
                "task_name": "openbookqa:rc_validation",
                "split": "validation",
                "primary_metric": "acc_uncond",
                "num_shots": 5,
                "fewshot_source": "OLMES:openbookqa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "openbookqa:rc_test::olmes": {
                "task_name": "openbookqa:rc_test",
                "split": "test",
                "primary_metric": "acc_uncond",
                "num_shots": 5,
                "fewshot_source": "OLMES:openbookqa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "csqa:mc_train::olmes": {
                "task_name": "csqa:mc_train",
                "split": "train",
                "num_shots": 5,
                "fewshot_source": "OLMES:commonsense_qa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "csqa:mc_validation::olmes": {
                "task_name": "csqa:mc_validation",
                "split": "validation",
                "num_shots": 5,
                "fewshot_source": "OLMES:commonsense_qa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "csqa:mc_test::olmes": {
                "task_name": "csqa:mc_test",
                "split": "test",
                "num_shots": 5,
                "fewshot_source": "OLMES:commonsense_qa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "csqa:rc_train::olmes": {
                "task_name": "csqa:rc_train",
                "split": "train",
                "primary_metric": "acc_uncond",
                "num_shots": 5,
                "fewshot_source": "OLMES:commonsense_qa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "csqa:rc_validation::olmes": {
                "task_name": "csqa:rc_validation",
                "split": "validation",
                "primary_metric": "acc_uncond",
                "num_shots": 5,
                "fewshot_source": "OLMES:commonsense_qa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "csqa:rc_test::olmes": {
                "task_name": "csqa:rc_test",
                "split": "test",
                "primary_metric": "acc_uncond",
                "num_shots": 5,
                "fewshot_source": "OLMES:commonsense_qa",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "boolq:mc_train::olmes": {
                "task_name": "boolq:mc_train",
                "split": "train",
                "num_shots": 5,
                "fewshot_source": "OLMES:BoolQ",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },

            "boolq:mc_validation::olmes": {
                "task_name": "boolq:mc_validation",
                "split": "validation",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:BoolQ",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },

            "boolq:mc_test::olmes": {
                "task_name": "boolq:mc_test",
                "split": "test",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:BoolQ",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },

            "boolq:rc_train::olmes": {
                "task_name": "boolq:rc_train",
                "split": "train",
                "primary_metric": "acc_raw",
                "num_shots": 5,
                "fewshot_source": "OLMES:BoolQ",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },

            "boolq:rc_validation::olmes": {
                "task_name": "boolq:rc_validation",
                "split": "validation",
                "primary_metric": "acc_raw",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:BoolQ",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },

            "boolq:rc_test::olmes": {
                "task_name": "boolq:rc_test",
                "split": "test",
                "primary_metric": "acc_raw",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:BoolQ",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },

            "arc_challenge:rc_train::olmes": {
                "task_name": "arc_challenge:rc_train",
                "split": "train",
                "primary_metric": "acc_uncond",
                "num_shots": 5,
                "limit": 1.0,
                "fewshot_source": "OLMES:ARC-Challenge",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "arc_challenge:rc_validation::olmes": {
                "task_name": "arc_challenge:rc_validation",
                "split": "validation",
                "primary_metric": "acc_uncond",
                "num_shots": 5,
                "fewshot_source": "OLMES:ARC-Challenge",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "arc_challenge:rc_test::olmes": {
                "task_name": "arc_challenge:rc_test",
                "split": "test",
                "primary_metric": "acc_uncond",
                "num_shots": 5,
                "fewshot_source": "OLMES:ARC-Challenge",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },

            "arc_challenge:mc_train::olmes": {
                "task_name": "arc_challenge:mc_train",
                "split": "train",
                "num_shots": 5,
                "limit": 1.0,
                "fewshot_source": "OLMES:ARC-Challenge",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },

            "arc_challenge:mc_validation::olmes": {
                "task_name": "arc_challenge:mc_validation",
                "split": "validation",
                "num_shots": 5,
                "fewshot_source": "OLMES:ARC-Challenge",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "arc_challenge:mc_test::olmes": {
                "task_name": "arc_challenge:mc_test",
                "split": "test",
                "num_shots": 5,
                "fewshot_source": "OLMES:ARC-Challenge",
                "metadata": {
                    "regimes": ["OLMES-v0.1"],
                },
            },

            "arc_easy:rc_train::olmes": {
                "task_name": "arc_easy:rc_train",
                "split": "train",
                "primary_metric": "acc_per_char",
                "num_shots": 5,
                "limit": 1.0,
                "fewshot_source": "OLMES:ARC-Easy",
                "metadata": {
                    "description": "ARC-Easy (RC) train using OLMES-v0.1",
                    "regimes": ["OLMES-v0.1"],
                },
            },

            "arc_easy:rc_validation::olmes": {
                "task_name": "arc_easy:rc_validation",
                "split": "validation",
                "primary_metric": "acc_per_char",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:ARC-Easy",
                "metadata": {
                    "description": "ARC-Easy (RC) validation using OLMES-v0.1, on validation split",
                    "regimes": ["OLMES-v0.1"],
                },
            },

            "arc_easy:rc_test::olmes": {
                "task_name": "arc_easy:rc_test",
                "split": "test",
                "primary_metric": "acc_per_char",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:ARC-Easy",
                "metadata": {
                    "description": "ARC-Easy (RC) test using OLMES-v0.1, on test split",
                    "regimes": ["OLMES-v0.1"],
                },
            },

            "arc_easy:mc_train::olmes": {
                "task_name": "arc_easy:mc_train",
                "split": "train",
                "num_shots": 5,
                "limit": 1.0,
                "fewshot_source": "OLMES:ARC-Easy",
                "metadata": {
                    "description": "ARC-Easy (MC) train using OLMES-v0.1, on training split",
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "arc_easy:mc_validation::olmes": {
                "task_name": "arc_easy:mc_validation",
                "split": "validation",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:ARC-Easy",
                "metadata": {
                    "description": "ARC-Easy (MC) validation using OLMES-v0.1, on validation split",
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "arc_easy:mc_test::olmes": {
                "task_name": "arc_easy:mc_test",
                "split": "test",
                "num_shots": 5,
                "limit": 1000,
                "fewshot_source": "OLMES:ARC-Easy",
                "metadata": {
                    "description": "ARC-Easy (MC) test using OLMES-v0.1, on test split",
                    "regimes": ["OLMES-v0.1"],
                },
            },
            "mbpp::none": {
                "task_name": "mbpp",
                "primary_metric": "pass_at_1",
                "use_chat_format": False,
                "context_kwargs": {
                    "prompt_variant": "evalplus",
                    "assistant_prefix": "\nYou are a helpful and precise coding assistant. For every coding task, you generate clean, self-contained Python code. Below is a function that correctly solves the coding problem and passes the relevant tests:\n```python\n",
                },
                "generation_kwargs": {
                    "stop_sequences": ["```", '\n"""', "\nassert", "\n#"],
                    "do_sample": True,
                    "top_p": 0.95,
                    "temperature": 0.8,
                    "repeats": 10,
                },
                "metric_kwargs": {
                    "pass_at_ks": [1, 10],
                },
                "metadata": {
                    "regimes": [],
                },
            },
            "mbpp::evalplus": {
                "task_name": "mbpp",
                "primary_metric": "pass_at_1",
                "use_chat_format": True,
                "context_kwargs": {
                    "prompt_variant": "evalplus",
                    "assistant_prefix": "\nBelow is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python\n",
                },
                "generation_kwargs": {
                    "stop_sequences": ["```", '\n"""', "\nassert", "\n#"],
                    "repeats": 10,
                },
                "metadata": {
                    "regimes": [],
                    "pass_at_ks": [1, 10],
                },
            },
            "mbppplus::none": {
                "task_name": "mbppplus",
                "primary_metric": "pass_at_1",
                "use_chat_format": False,
                "context_kwargs": {
                    "prompt_variant": "evalplus",
                    "assistant_prefix": "\nYou are a helpful and precise coding assistant. For every coding task, you generate clean, self-contained Python code. Below is a function that correctly solves the above problem and passes the relevant tests:\n```python\n",
                },
                "generation_kwargs": {
                    "stop_sequences": ["```", '\n"""', "\nassert", "\n#"],
                    "do_sample": True,
                    "top_p": 0.95,
                    "temperature": 0.8,
                    "repeats": 10,
                },
                "metric_kwargs": {
                    "pass_at_ks": [1, 10],
                },
                "metadata": {
                    "regimes": [],
                },
            },
            "mbppplus::openinstruct": {
                "task_name": "mbppplus",
                "primary_metric": "pass_at_1",
                "use_chat_format": True,
                "context_kwargs": {
                    "prompt_variant": "openinstruct",
                    "assistant_prefix": "Here is the completed function:\n\n\n```python\n",
                },
                "generation_kwargs": {
                    "do_sample": True,
                    "top_p": 0.95,
                    "temperature": 0.1,
                    "repeats": 10,
                    "stop_sequences": ["```"],
                },
                "metric_kwargs": {
                    "pass_at_ks": [1, 10],
                },
                "metadata": {
                    "regimes": [],
                },
            },
            "mbppplus::evalplus": {
                "task_name": "mbppplus",
                "primary_metric": "pass_at_1",
                "use_chat_format": True,
                "context_kwargs": {
                    "prompt_variant": "evalplus",
                    "assistant_prefix": "\nBelow is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python\n",
                },
                "generation_kwargs": {
                    "stop_sequences": ["```", '\n"""', "\nassert", "\n#"],
                    "repeats": 10,
                },
                "metadata": {
                    "regimes": [],
                    "pass_at_ks": [1, 10],
                },
            },
            "mbppplus::deepseek": {
                "task_name": "mbppplus",
                "primary_metric": "pass_at_1",
                "num_shots": 3,
                "use_chat_format": True,
                "context_kwargs": {
                    "prompt_variant": "deepseek",
                    "assistant_prefix": "\n[BEGIN]\n",
                },
                "generation_kwargs": {
                    "stop_sequences": ["```", '\n"""', "\nassert", "\n#", "\n[DONE]"],
                },
                "metadata": {
                    "regimes": [],
                    "pass_at_ks": [1, 10],
                },
            },
            "bigcodebench::none": {
                "task_name": "bigcodebench",
                "primary_metric": "pass_at_1",
                "use_chat_format": False,
                "context_kwargs": {
                    "prompt_variant": "complete",
                },
                "generation_kwargs": {
                    "stop_sequences": ["```", '\n"""', "\nassert", "\n#"],
                    "do_sample": True,
                    "top_p": 0.95,
                    "temperature": 0.8,
                    "repeats": 10,
                },
                "metric_kwargs": {
                    "pass_at_ks": [1, 10],
                },
                "metadata": {
                    "regimes": [],
                },
            },
            "bigcodebench_hard::none": {
                "task_name": "bigcodebench_hard",
                "primary_metric": "pass_at_1",
                "use_chat_format": False,
                "context_kwargs": {
                    "prompt_variant": "complete",
                },
                "generation_kwargs": {
                    "stop_sequences": ["```", '\n"""', "\nassert", "\n#"],
                    "do_sample": True,
                    "top_p": 0.95,
                    "temperature": 0.8,
                    "repeats": 10,
                },
                "metric_kwargs": {
                    "pass_at_ks": [1, 10],
                },
                "metadata": {
                    "regimes": [],
                },
            },
            "bigcodebench::tulu": {
                "task_name": "bigcodebench",
                "use_chat_format": True,
                "context_kwargs": {
                    "prompt_variant": "instruct",
                },
                "metadata": {
                    "regimes": ["Tulu"],
                },
            },
            "bigcodebench_hard::tulu": {
                "task_name": "bigcodebench_hard",
                "use_chat_format": True,
                "context_kwargs": {
                    "prompt_variant": "instruct",
                },
                "metadata": {
                    "regimes": ["Tulu"],
                },
            },
            "codex_humaneval:temp0.1": {
                "task_name": "codex_humaneval",
                "primary_metric": "pass_at_1",
                "generation_kwargs": {
                    "do_sample": True,
                    "top_p": 0.95,
                    "temperature": 0.1,
                    "repeats": 10,
                },
                "metric_kwargs": {
                    "pass_at_ks": [1, 10],
                },
                "metadata": {
                    "regimes": [],
                },
            },
            "codex_humaneval:temp0.8": {
                "task_name": "codex_humaneval",
                "primary_metric": "pass_at_1",
                "generation_kwargs": {
                    "do_sample": True,
                    "top_p": 0.95,
                    "temperature": 0.8,
                    "repeats": 10,
                },
                "metric_kwargs": {
                    "pass_at_ks": [1, 10],
                },
                "context_kwargs": {
                    "prompt_variant": "evalplus",
                    "answer_prefix": "Below is the completed function for this coding task:\n\n```python\n",
                },
                "metadata": {
                    "regimes": [],
                },
            },
            "codex_humaneval::tulu": {
                "task_name": "codex_humaneval",
                "primary_metric": "pass_at_10",
                "use_chat_format": True,
                "generation_kwargs": {
                    "do_sample": True,
                    "top_p": 0.95,
                    "temperature": 0.8,
                    "repeats": 20,
                },
                "metric_kwargs": {
                    "pass_at_ks": [10],
                },
                "metadata": {
                    "regimes": ["Tulu"],
                },
            },
            "codex_humanevalplus:temp0.8": {
                "task_name": "codex_humanevalplus",
                "primary_metric": "pass_at_1",
                "use_chat_format": False,
                "generation_kwargs": {
                    "do_sample": True,
                    "top_p": 0.95,
                    "temperature": 0.8,
                    "repeats": 10,
                },
                "context_kwargs": {
                    "prompt_variant": "evalplus",
                    "answer_prefix": "Below is the completed function for this coding task:\n\n```python\n",
                },
                "metric_kwargs": {
                    "pass_at_ks": [1, 10],
                },
                "metadata": {
                    "regimes": [],
                },
            },
            "codex_humanevalplus::tulu": {
                "task_name": "codex_humanevalplus",
                "primary_metric": "pass_at_10",
                "use_chat_format": True,
                "generation_kwargs": {
                    "do_sample": True,
                    "top_p": 0.95,
                    "temperature": 0.8,
                    "repeats": 20,
                },
                "metric_kwargs": {
                    "pass_at_ks": [10],
                },
                "metadata": {
                    "regimes": ["Tulu"],
                },
            },
            "medmcqa:rc::none": {"task_name": "medmcqa", "split": "validation", "num_shots": 5},
            "medmcqa:mc::none": {
                "task_name": "medmcqa:mc",
                "split": "validation",
                "num_shots": 5,
            },
        }
    )

    # update MMLU, MMLU_Pro categories and configs
    for sub in MMLU_SUBJECTS:
        TASK_CONFIGS[f"mmlu_{sub}:mc_train::olmes"] = {
            "task_name": f"mmlu_{sub}:mc_train",
            "split": "train",
            "num_shots": 5,
            "primary_metric": "acc_raw",
            "metadata": {
                "regimes": ["OLMES-v0.1"],
            },
        }
        TASK_CONFIGS[f"mmlu_{sub}:mc_validation::olmes"] = {
            "task_name": f"mmlu_{sub}:mc_validation",
            "split": "validation",
            "num_shots": 5,
            "primary_metric": "acc_raw",
            "metadata": {
                "regimes": ["OLMES-v0.1"],
            },
        }
        TASK_CONFIGS[f"mmlu_{sub}:mc_test::olmes"] = {
            "task_name": f"mmlu_{sub}:mc_test",
            "split": "test",
            "num_shots": 5,
            "primary_metric": "acc_raw",
            "metadata": {
                "regimes": ["OLMES-v0.1"],
            },
        }

        TASK_CONFIGS[f"mmlu_{sub}:rc_train::olmes"] = {
            "task_name": f"mmlu_{sub}:rc_train",
            "split": "train",
            "num_shots": 5,
            "primary_metric": "acc_per_char",
            "metadata": {
                "regimes": ["OLMES-v0.1"],
            },
        }
        TASK_CONFIGS[f"mmlu_{sub}:rc_validation::olmes"] = {
            "task_name": f"mmlu_{sub}:rc_validation",
            "split": "validation",
            "num_shots": 5,
            "primary_metric": "acc_per_char",
            "metadata": {
                "regimes": ["OLMES-v0.1"],
            },
        }
        TASK_CONFIGS[f"mmlu_{sub}:rc_test::olmes"] = {
            "task_name": f"mmlu_{sub}:rc_test",
            "split": "test",
            "num_shots": 5,
            "primary_metric": "acc_per_char",
            "metadata": {
                "regimes": ["OLMES-v0.1"],
            },
        }

    for cat in MMLU_PRO_CATEGORIES:
        TASK_CONFIGS[f"mmlu_pro_{cat}:mc_train::none"] = {
            "task_name": f"mmlu_pro_{cat}:mc_train",
            "split": "train",
            "num_shots": 5,
            "metadata": {
                "regimes": [],
            },
        }
        TASK_CONFIGS[f"mmlu_pro_{cat}:mc_validation::none"] = {
            "task_name": f"mmlu_pro_{cat}:mc_validation",
            "split": "validation",
            "num_shots": 5,
            "metadata": {
                "regimes": [],
            },
        }
        TASK_CONFIGS[f"mmlu_pro_{cat}:mc_test::none"] = {
            "task_name": f"mmlu_pro_{cat}:mc_test",
            "split": "test",
            "num_shots": 5,
            "metadata": {
                "regimes": [],
            },
        }
        TASK_CONFIGS[f"mmlu_pro_{cat}:rc_train::none"] = {
            "task_name": f"mmlu_pro_{cat}:rc_train",
            "split": "train",
            "num_shots": 5,
            "metadata": {
                "regimes": [],
            },
        }
        TASK_CONFIGS[f"mmlu_pro_{cat}:rc_validation::none"] = {
            "task_name": f"mmlu_pro_{cat}:rc_validation",
            "split": "validation",
            "num_shots": 5,
            "metadata": {
                "regimes": [],
            },
        }
        TASK_CONFIGS[f"mmlu_pro_{cat}:rc_test::none"] = {
            "task_name": f"mmlu_pro_{cat}:rc_test",
            "split": "test",
            "num_shots": 5,
            "metadata": {
                "regimes": [],
            },
        }

    return TASK_CONFIGS
