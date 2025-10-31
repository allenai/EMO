from typing import List, Union

from oe_eval.components.instances import RequestInstance

# from oe_eval.metrics.metric import SQuADF1EMRecallMetric, RougeMetric
from oe_eval.tasks.base_task import Task

from offline_evals.metrics.lm_judge import LMJudgeMetric

# from oe_eval.tasks.utils import make_cloze_prompt, make_summarization_prompt


_CITATION = """
XSUM
"""


class Story_Gen_LMJudge(Task):
    TASK_CONFIG_DEFAULTS: dict = {
        # "dataset_path": "nid989/EssayFroum-Dataset",
        # "dataset_name": None,
        "dataset_path": "WildEval/WildBench-V2",
        "dataset_name": "v2",
        #
        "native_id_field": "id",
        "primary_metric": "lm_score",
        "split": "test",  # "train",
        "fewshot_source": None,
        "limit": 100,  # 100
        "num_shots": 0,
        "context_kwargs": {
            "description": None,  # "Answer the following question."
        },
        "generation_kwargs": {
            "max_gen_toks": 256,
            "temperature": 0.0,
            "do_sample": False,
            "stop_sequences": ["\n\n"],
        },
        "metric_kwargs": {
            "model_name": "meta-llama/Llama-3.3-70B-Instruct",
            "eval_prompt": "story",
        },
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
            self._training_docs = list(list(map(self._process_doc, self.dataset["train"])))
        return self._training_docs

    def validation_docs(self):
        return list(map(self._process_doc, self.dataset["validation"]))

    def test_docs(self):
        data = list(filter(lambda x: x["primary_tag"] == "Creative Writing", self.dataset["test"]))
        print(len(data))
        from IPython import embed

        embed()
        exit()

        return list(map(self._process_doc, self.dataset["test"]))

    def other_docs(self):
        return list(map(self._process_doc, self.dataset[self.task_config["split"]]))

    def _process_doc(self, doc):
        # # Question: Who was the man behind The Chipmunks?
        # # Answer: David Seville
        # query = make_summarization_prompt(doc["document"])
        # answers_text = doc["summary"]
        # out_doc = {
        #     "question_id": doc["id"],
        #     "query": query,
        #     "answers_text": answers_text,
        # }
        query = " ".join(doc["Cleaned Essay"].split()[:128])
        # continuation = " ".join(doc["text"].split()[128:])
        out_doc = {
            "question": query,
            "reference": doc["Cleaned Essay"],
        }
        return out_doc

    def doc_to_text(self, doc):
        # return "Continue writing a detailed and well-structured news article: " + doc["question"]
        return doc["question"]

    def doc_to_target(self, doc):
        return doc["reference"]

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        return self.construct_basic_generation_requests(doc, ctx, doc_id, label=doc["reference"])
