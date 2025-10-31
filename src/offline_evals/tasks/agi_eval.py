"""
https://github.com/ruixiangcui/AGIEval
"""

from oe_eval.data.agi_eval_tasks import AGI_EVAL_ENGLISH_DATASETS
from oe_eval.tasks.oe_eval_tasks.agi_eval import (
    GenericAGIEval as OriginalGenericAGIEval,
)
from oe_eval.tasks.oe_eval_tasks.agi_eval import (
    GenericAGIEval_MC as OriginalGenericAGIEval_MC,
)
from oe_eval.utils import get_dict_with_defaults, load_jsonl


def create_core_agi_eval_tasks() -> dict:
    all_tasks = {}
    for task_type in AGI_EVAL_ENGLISH_DATASETS:
        all_tasks[f"agi_eval_{task_type}:cot"] = create_agi_eval_task(task_type)
        all_tasks[f"agi_eval_{task_type}:mc"] = create_agi_eval_mc_task(task_type)
    return all_tasks


def get_default_num_shots_max_tokens(task_type):
    default_num_shots = 3
    if task_type in ["sat-math", "aqua-rat"]:
        default_num_shots = 5
    max_tokens = 1024
    return (default_num_shots, max_tokens)


def create_agi_eval_task(task_type):
    default_num_shots, max_tokens = get_default_num_shots_max_tokens(task_type)

    class AGIEval(GenericAGIEval):
        TASK_CONFIG_DEFAULTS = get_dict_with_defaults(
            {
                "dataset_name": task_type,
                "fewshot_source": f"agi_eval:{task_type}",
                "num_shots": default_num_shots,
                "generation_kwargs": {"max_gen_toks": max_tokens},
            },
            GenericAGIEval.TASK_CONFIG_DEFAULTS,
        )

    return AGIEval


def create_agi_eval_mc_task(task_type):
    default_num_shots, max_tokens = get_default_num_shots_max_tokens(task_type)

    class AGIEval_MC(GenericAGIEval_MC):
        TASK_CONFIG_DEFAULTS = get_dict_with_defaults(
            {
                "dataset_name": task_type,
                "fewshot_source": f"agi_eval:{task_type}",
                "num_shots": default_num_shots,
            },
            GenericAGIEval_MC.TASK_CONFIG_DEFAULTS,
        )

    return AGIEval_MC


class GenericAGIEval(OriginalGenericAGIEval):
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        dataset_name = self.task_config["dataset_name"]
        file_name = f"src/offline_evals/data/agi_eval/{dataset_name}.jsonl"
        data = load_jsonl(file_name)
        self.dataset = {"test": data}


class GenericAGIEval_MC(OriginalGenericAGIEval_MC):
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        dataset_name = self.task_config["dataset_name"]
        file_name = f"src/offline_evals/data/agi_eval/{dataset_name}.jsonl"
        data = load_jsonl(file_name)
        self.dataset = {"test": data}
