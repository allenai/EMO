import json
import os
from typing import List, Union

import smart_open
from oe_eval.components.instances import RequestInstance
from oe_eval.metrics.metric import GenericMetric
from oe_eval.tasks.base_task import Task
from oe_eval.utils import get_dict_with_defaults

RULER_BASE_DIR = "s3://ai2-dustins/ruler-evals/olmo-2-7b/synthetic/4000/data/"

RULER_TASKS = [
    "cwe",
    "fwe",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "qa_1",
    "qa_2",
    "vt",
]


def create_ruler_tasks() -> dict:
    all_tasks = {}
    for task_type in RULER_TASKS:

        class Ruler(GenericRuler):
            TASK_CONFIG_DEFAULTS = get_dict_with_defaults(
                {"dataset_name": task_type}, GenericRuler.TASK_CONFIG_DEFAULTS
            )

        all_tasks[f"ruler_4k_{task_type}"] = Ruler
    return all_tasks


class GenericRuler(Task):
    VERSION = 0
    MAX_TRAINING_DOCS = 10000
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": RULER_BASE_DIR,
        "native_id_field": "index",
        "primary_metric": "ruler_recall",
        "fewshot_source": None,
        "num_shots": 0,
        "split": "validation",
        "context_kwargs": {"description": None},
        "generation_kwargs": {
            "max_gen_toks": 50,
            "temperature": 0.0,
            "do_sample": False,
            "stop_sequences": ["Question:", "Q:", "\n\n"],
        },
        "chat_overrides": {
            "generation_kwargs": {
                "stop_sequences": ["Question:", "Q:", "<|eot_id|>"],
            },
        },
    }

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        with smart_open.open(
            os.path.join(
                self.task_config["dataset_path"],
                self.task_config["dataset_name"],
                "validation.jsonl",
            ),
            "r",
        ) as f:
            data = []
            for line in f:
                data.append(json.loads(line))
        self.dataset = {"validation": data}

    def make_metrics(self):
        def process_results(example, output):
            # we don't do any parsing since we are only checking for substring exact match
            assert len(output) == 1
            prediction = output[0]
            answers = example["answers"]
            assert isinstance(answers, list)
            recall = sum([a.lower() in prediction.lower() for a in answers]) / len(answers)
            mets = {"ruler_recall": recall}
            return mets

        self._metrics = [
            GenericMetric(
                process_results_fn=process_results,
                metric_names=["ruler_recall"],
            )
        ]
        return self._metrics

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc, index=-1):
        # doc: index, input, outputs (list), length
        # doc["input"] is already formatted
        out_doc = {
            "index": index,
            "question": doc["input"],
            "query": doc["input"],
            "answers": doc["outputs"],
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        short_answers = ", ".join(doc["answers"])
        return " " + short_answers

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        return self.construct_basic_generation_requests(doc, ctx, doc_id, label=doc["answers"])
