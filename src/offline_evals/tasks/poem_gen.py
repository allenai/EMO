from typing import List, Union

import numpy as np
from oe_eval.components.instances import RequestInstance

# from oe_eval.metrics.metric import SQuADF1EMRecallMetric, RougeMetric
from oe_eval.tasks.base_task import Task

from offline_evals.metrics.lm_judge import LMJudgeMetric

# from oe_eval.tasks.utils import make_cloze_prompt, make_summarization_prompt


_CITATION = """
XSUM
"""


class Poem_Gen_LMJudge(Task):
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "merve/poetry",
        "dataset_name": None,
        #
        "native_id_field": "id",
        "primary_metric": "lm_score",
        "split": "test",
        "fewshot_source": None,
        "limit": 100,
        "num_shots": 5,
        "context_kwargs": {
            "description": None,  # "Answer the following question."
        },
        "generation_kwargs": {
            "max_gen_toks": 256,
            "temperature": 0.8,
            "do_sample": True,
            "stop_sequences": ["\n\n"],
            "repeats": 5,
        },
        "metric_kwargs": {"model_name": "meta-llama/Llama-3.3-70B-Instruct", "eval_prompt": "poem"},
    }

    def download(self, **kwargs):
        super().download(**kwargs)

        data: List = list(
            filter(
                lambda x: 64 <= len(x["content"].split()) <= 128  # type: ignore
                and len(x["content"].split("\r\n\r\n")) == 1,
                self.dataset["train"],
            )
        )

        np.random.seed(2025)
        np.random.shuffle(data)

        # sewonm: way more renaissance than modern, tried to make it more balanced
        data_r = list(filter(lambda x: x["age"] == "Renaissance", data))
        data_m = list(filter(lambda x: x["age"] == "Modern", data))
        assert len(data_r) == 147 and len(data_m) == 29

        train_data = data_r[:3] + data_m[:2]
        test_data = data_r[3:76] + data_m[2:]
        assert len(train_data) == 5 and len(test_data) == 100

        np.random.shuffle(train_data)
        np.random.shuffle(test_data)

        self.dataset = {"train": train_data, "test": test_data}

    def make_metrics(self):
        self._metrics = [LMJudgeMetric(**self.task_config["metric_kwargs"])]
        return self._metrics

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_examples(self, k, rnd, doc):
        # Fixed order few shot examples
        if self._fewshot_docs is None:
            self._fewshot_docs = list(self.training_docs())
        return self._fewshot_docs[:k]

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(list(map(self._process_doc, self.dataset["train"])))
        return self._training_docs

    def validation_docs(self):
        return list(map(self._process_doc, self.dataset["test"]))

    def test_docs(self):
        return list(map(self._process_doc, self.dataset["test"]))

    def other_docs(self):
        return list(map(self._process_doc, self.dataset[self.task_config["split"]]))

    def _process_doc(self, doc):
        metadata = f"You are a skilled poet and creative writer. Your task is to generate high-quality poems that are creative, emotionally resonant, and well-crafted.\nPoem Genre: {doc['age']}\nPoem Title: {doc['poem name']}\n\n"
        lines = doc["content"].split("\r\n")
        assert len(lines) > 4

        prefix = "\n".join(lines[:4]) + "\n"
        reference = "\n".join(lines)

        out_doc = {
            "question": metadata + prefix,  # input to the model
            "reference": reference,  # gold poem
            "reference_without_prefix": "\n".join(lines[4:]),  # gold without prefix
            "prefix": prefix,  # prefix of the poem
            "metadata": metadata.strip(),  # metadata
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        return doc["reference_without_prefix"]

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        return self.construct_basic_generation_requests(doc, ctx, doc_id, label=doc["reference"])
