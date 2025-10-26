from oe_eval.configs.tasks import TASK_CONFIGS


def get_task_configs():
    TASK_CONFIGS.update(
        {
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
    return TASK_CONFIGS
