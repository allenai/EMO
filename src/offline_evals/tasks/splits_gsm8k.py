"""
"Training Verifiers to Solve Math Word Problems"
https://arxiv.org/abs/2110.14168

State-of-the-art language models can match human performance on many tasks, but
they still struggle to robustly perform multi-step mathematical reasoning. To
diagnose the failures of current models and support research, we introduce GSM8K,
a dataset of 8.5K high quality linguistically diverse grade school math word problems.
We find that even the largest transformer models fail to achieve high test performance,
despite the conceptual simplicity of this problem distribution.

NOTE: See the official implementation of the task:
    https://github.com/openai/grade-school-math/blob/master/grade_school_math/calculator.py
for how to make use of the dataset's calculator annotations in your language
model's sample/generation function.

Homepage: https://github.com/openai/grade-school-math
"""

import re
from typing import List, Union, cast

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import RequestType, LoglikelihoodRequest
from oe_eval.metrics import PerplexityMetric
from oe_eval.metrics.metric import ExactMatch, MajAtK, MCAccuracy
from oe_eval.tasks.base_task import Task
from oe_eval.tasks.utils import apply_prompt_template
from oe_eval.utils import get_dict_with_defaults

from ..metrics.custom_perplexity import CustomPerplexityMetric

_CITATION = """
@misc{cobbe2021training,
      title={Training Verifiers to Solve Math Word Problems},
      author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
      year={2021},
      eprint={2110.14168},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


class GSM8K_Perplexity_Base(Task):
    """
    Modified version of GSM8K to be
    (1) contains train, validation, test splits -> extra validation split comes from training
    (2) Evaluate on perplexity instead of exact match
    """

    VERSION = 0.1
    REQUEST_TYPE = RequestType.GENERATE_UNTIL
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": "gsm8k",
        "dataset_name": "main",
        "native_id_field": "id",  # "Dataset auto index"
        "primary_metric": "bits_per_byte",
        "fewshot_source": "STD:GSM8k",
        "context_kwargs": {
            "no_cot": False,
        },
    }

    def make_metrics(self):
        self._metrics = [CustomPerplexityMetric()]
        return self._metrics

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            # select training docs excluding 1000 examples used for validation
            train_dataset = self.dataset["train"].shuffle(seed=0).select(range(1000, len(self.dataset["train"])))
            self._training_docs = list(
                train_dataset.map(self._process_doc, with_indices=True)
            )
        return self._training_docs

    def validation_docs(self):
        # select 1000 examples from the train set as validation set
        val_dataset = self.dataset["train"].shuffle(seed=0).select(range(0, 1000))
        return val_dataset.map(self._process_doc, with_indices=True)

    def test_docs(self):
        return self.dataset["test"].map(self._process_doc, with_indices=True)

    def doc_to_text(self, doc):
        return "Question: " + doc["question"] + "\nAnswer:"

    def doc_to_target(self, doc):
        if self.task_config["context_kwargs"].get("no_cot", False):
            return " " + doc["short_answer"]
        else:
            return " " + doc["answer"]

    def _process_doc(self, doc, index=-1):
        """
        HF Dataset class provides a map function that can pass an index to each doc with `with_indices=True`
        """
        # question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        # answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
        if "short_answer" in doc:
            short_answer = doc["short_answer"]
        else:
            short_answer = doc["answer"].split("####")[-1].strip()
            doc["short_answer"] = short_answer
        gold_cot = self.normalize_answer_str(doc, doc["answer"])
        out_doc = {
            "id": index,
            "question": doc["question"],
            "answer": gold_cot,
            "short_answer": short_answer,
        }
        out_doc = apply_prompt_template(
            out_doc,
            self.task_config,
        )
        return out_doc

    def add_spaces_around_operators_no_regex(self, _str):
        """Add spacing around special operators if it does not exist"""
        operators = {"+", "-", "*", "/", "="}
        result: List[str] = []
        for char in _str:
            if char in operators:
                if result and result[-1] != " ":
                    result.append(" ")
                result.append(char)
                result.append(" ")
            else:
                result.append(char)

        # Join the list and replace double spaces with single spaces
        return "".join(result).replace("  ", " ")

    def normalize_answer_str(self, doc: dict, answer: str) -> str:
        """
        Convert the gold CoT to a more natural-appearing string to improve bpb calculation.

        Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
        Original Answer: Janet has 16 eggs and uses 4 for baking and sells 3 for breakfast. Therefore, she makes 16 - 3 - 4 = <<16-3-4=9>>9 eggs sold, leading to a daily income of 9 * 2 = $<<9*2=22>>22.\n#### 22
        New Answer: Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = $18 every day at the farmer’s market. So the answer is 18.
        """
        import re

        answer = re.sub(r"<<.*?>>", "", answer)
        answer = re.sub(r"\s+", " ", answer).strip()
        answer = re.split(r"####", answer)[0]
        answer = answer[0].capitalize() + answer[1:] if answer else answer
        answer = answer.strip()
        if not answer.endswith("."):
            answer += "."
        answer = answer + f" So the answer is {doc['short_answer']}."
        answer = self.add_spaces_around_operators_no_regex(answer)
        return answer

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        native_id_field = self.task_config.get("native_id_field", "id")
        return [
            RequestInstance(
                request_type=RequestType.LOGLIKELIHOOD.value,
                doc=doc,
                request=LoglikelihoodRequest(
                    context=ctx,
                    continuation=" {}".format(self.doc_to_target(doc)),
                ),
                idx=0,
                task_name=self.task_name,
                doc_id=doc_id,
                native_id=doc.get(native_id_field),
            )
        ]

        # return self.construct_basic_likelihood_rolling_requests(doc, cast(str, ctx), doc_id)

