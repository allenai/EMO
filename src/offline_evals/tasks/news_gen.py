from typing import List, Union

from oe_eval.components.instances import RequestInstance
from oe_eval.tasks.base_task import Task

from offline_evals.metrics.lm_judge import LMJudgeMetric

_CITATION = """
NEWS
"""


class News_Gen_LMJudge(Task):
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "muse-bench/MUSE-News",
        "dataset_name": "train",
        "native_id_field": "id",
        "primary_metric": "lm_score",
        "split": "test",
        "fewshot_source": None,
        "limit": 100,  # 100
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
        "metric_kwargs": {"model_name": "meta-llama/Llama-3.3-70B-Instruct", "eval_prompt": "news"},
    }

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
            self._training_docs = list(
                map(self._process_doc, filter(self._filter_doc, self.dataset["retain1"]))
            )
        return self._training_docs

    def validation_docs(self):
        return list(map(self._process_doc, filter(self._filter_doc, self.dataset["validation"])))

    def test_docs(self):
        return list(map(self._process_doc, filter(self._filter_doc, self.dataset["forget"])))

    def other_docs(self):
        return list(
            map(
                self._process_doc, filter(self._filter_doc, self.dataset[self.task_config["split"]])
            )
        )

    def _filter_doc(self, doc):
        return 64 <= len(doc["text"].split()) <= 128

    def _process_doc(self, doc):
        """
        The model is asked to continue the news given the first 32 words
        """
        text = doc["text"].replace("\n\n", "\n").replace("\n", " ")
        query = "Breaking News:\n"
        prefix = " ".join(text.split()[:32])

        out_doc = {
            "question": query + prefix,  # input to the model (indicator + prefix)
            "prefix": prefix,  # prefix of the news article
            "reference": text,  # gold
            "reference_without_prefix": " " + " ".join(text.split()[32:]),  # gold except prefix
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
