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
from typing import List, Union

from datasets import concatenate_datasets
from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import LoglikelihoodRequest, RequestType
from oe_eval.tasks.base_task import Task
from oe_eval.tasks.oe_eval_tasks.gsm8k import GSM8K
from oe_eval.tasks.utils import apply_prompt_template

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
            train_dataset = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(1000, len(self.dataset["train"])))
            )
            self._training_docs = list(train_dataset.map(self._process_doc, with_indices=True))
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


# class GSM8K_Perplexity_Train(GSM8K_Perplexity_Base):
#     pass
#
#
# class GSM8K_Perplexity_Validation(GSM8K_Perplexity_Base):
#     pass
#
#
# class GSM8K_Perplexity_Train_0shot(GSM8K_Perplexity_Base):
#     pass
#
#
# class GSM8K_Perplexity_Validation_0shot(GSM8K_Perplexity_Base):
#     pass
#
#
# class GSM8K_Perplexity_Test(GSM8K_Perplexity_Base):
#     pass


# For GSM8K Finetuning. All splits inherit from upstream GSM8K so the prompt
# formatting / processing path is identical to the test-time eval. The train
# and validation classes additionally:
#   - carve a 1000-example validation slice out of the train set (GSM8K has
#     no native validation split)
#   - rewrite doc["answer"] to the *normalized* CoT (the "So the answer is X."
#     form stored in doc["choices"][0] by upstream _process_doc), so that
#     doc_to_target matches the targets used historically by the pruning +
#     finetuning pipeline.
class GSM8K_Generation_TrainVal_Base(GSM8K):
    def has_validation_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            # select training docs excluding 1000 examples used for validation
            train_dataset = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(1000, len(self.dataset["train"])))
            )
            self._training_docs = list(train_dataset.map(self._process_doc, with_indices=True))
        return self._training_docs

    def validation_docs(self):
        # select 1000 examples from the train set as validation set
        val_dataset = self.dataset["train"].shuffle(seed=0).select(range(0, 1000))
        return val_dataset.map(self._process_doc, with_indices=True)

    def _process_doc(self, doc, index=-1):
        out_doc = super()._process_doc(doc, index=index)
        # Upstream leaves doc["answer"] as the raw GSM8K answer (with <<...>>
        # calculator annotations and the trailing "#### N"). The pruning +
        # finetuning pipeline historically trained on the *normalized* CoT,
        # which upstream stores in doc["choices"][0]. Copy it into "answer"
        # so doc_to_target returns the normalized form.
        out_doc["answer"] = out_doc["choices"][0]
        return out_doc

    def normalize_answer_str(self, doc: dict, answer: str) -> str:
        # Upstream `normalize_answer_str` unconditionally appends
        # " So the answer is {short_answer}." to the answer. For raw HF GSM8K
        # rows that's correct (their answers end with "#### N"), but the
        # STD:GSM8k fewshot exemplars already contain "So the answer is X."
        # in their raw `answer` field, so the upstream normalizer ends up
        # emitting the suffix twice in the assembled fewshot prefix. Collapse
        # a duplicated trailing suffix here so the method is idempotent on
        # already-suffixed inputs.
        out = super().normalize_answer_str(doc, answer)
        suffix = f"So the answer is {doc['short_answer']}."
        # The trailing portion may have been touched by
        # add_spaces_around_operators_no_regex; match the post-spacing form.
        suffix_spaced = self.add_spaces_around_operators_no_regex(suffix).rstrip()
        doubled = f"{suffix_spaced} {suffix_spaced}"
        if out.rstrip().endswith(doubled):
            stripped = out.rstrip()
            out = stripped[: -len(doubled)] + suffix_spaced
        return out


class GSM8K_Generation_Test_0shot(GSM8K):
    pass


class GSM8K_Generation_Train_0shot(GSM8K_Generation_TrainVal_Base):
    pass


class GSM8K_Generation_Validation_0shot(GSM8K_Generation_TrainVal_Base):
    pass


# 8-shot generation variant (same task classes; shot count controlled by TASK_CONFIGS)
class GSM8K_Generation_Test_8shot(GSM8K):
    pass


class GSM8K_Generation_Train_8shot(GSM8K_Generation_TrainVal_Base):
    pass


class GSM8K_Generation_Validation_8shot(GSM8K_Generation_TrainVal_Base):
    pass


# ---------------------------------------------------------------------------
# Merged variant: pruning + finetuning use the same merged data
# (matches the pattern in splits_hellaswag.HellaSwag_Merged_RC and
# splits_mmlu.GenericMMLU_Merged_withsplits).
# Concatenates the train slice (6473 docs) with the validation slice (1000
# docs) into a single 7473-doc set, shuffled so the validation rows interleave
# with the training rows. Both training_docs() and validation_docs() return
# this same set, so the pruning calibration data is identical to the
# finetuning data. The test split is unchanged (official 1319-row GSM8K test).
# ---------------------------------------------------------------------------
class GSM8K_Generation_Merged_Base(GSM8K_Generation_TrainVal_Base):
    def training_docs(self):
        if self._training_docs is None:
            train_dataset = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(1000, len(self.dataset["train"])))
            )
            val_dataset = self.dataset["train"].shuffle(seed=0).select(range(0, 1000))
            merged = concatenate_datasets([train_dataset, val_dataset]).shuffle(seed=0)
            self._training_docs = list(merged.map(self._process_doc, with_indices=True))
        return self._training_docs

    def validation_docs(self):
        # Same merged set as training_docs so pruning + finetuning share data.
        return self.training_docs()


class GSM8K_Generation_Merged_0shot(GSM8K_Generation_Merged_Base):
    pass


class GSM8K_Generation_Merged_8shot(GSM8K_Generation_Merged_Base):
    pass
