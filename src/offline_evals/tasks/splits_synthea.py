"""
Uses the task of Synthea as a medical MCQ task
"""
import os
from typing import Optional

from oe_eval.tasks.base_task import MultipleChoiceTask
from oe_eval.tasks.utils import make_cloze_prompt, make_mcq_prompt
from oe_eval.utils import load_jsonl


class Synthea(MultipleChoiceTask):
    VERSION = 0
    TASK_CONFIG_DEFAULTS: dict = {
        "native_id_field": "id",  # Field in doc that corresponds to the native id
        "primary_metric": "acc_per_char",
        "split": "validation",  # Which split to evaluate on
    }

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        train_set_1 = "/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/src/offline_evals/data/synthea/split_1/train.jsonl"
        train_set_2 = "/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/src/offline_evals/data/synthea/split_2/train.jsonl"
        val_set_1 = "/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/src/offline_evals/data/synthea/split_1/val.jsonl"
        val_set_2 = "/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/src/offline_evals/data/synthea/split_2/val.jsonl"
        test_set_1 = "/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/src/offline_evals/data/synthea/split_1/test.jsonl"
        test_set_2 = "/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/src/offline_evals/data/synthea/split_2/test.jsonl"

        # check that files exist
        if not all(
            os.path.exists(f)
            for f in [
                train_set_1,
                train_set_2,
                val_set_1,
                val_set_2,
                test_set_1,
                test_set_2,
            ]
        ):
            raise FileNotFoundError(
                "Synthea dataset files not found. Please ensure the dataset files are present."
            )

        # load the data
        train_data = load_jsonl(train_set_1) + load_jsonl(train_set_2)
        val_data = load_jsonl(val_set_1) + load_jsonl(val_set_2)
        test_data = load_jsonl(test_set_1) + load_jsonl(test_set_2)

        self.dataset = {"train": train_data, "validation": val_data, "test": test_data}

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs: Optional[list] = list(
                map(self._process_doc, self.dataset["train"])
            )
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def unconditioned_prompt(self):
        return "Answer:"

    def _process_doc(self, doc):
        prompt = doc["prompt"]
        question_idx = prompt.rfind("\nA. ")
        if prompt.count("\nA. ") != 1:
            breakpoint()

        question = prompt[:question_idx].strip()

        query = make_cloze_prompt(question)
        out_doc = {
            "id": doc["id"],
            "query": query,
            "choices": doc["choices"],
            "gold": int(doc["answer_idx"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]


class Synthea_RC_Train(Synthea):
    pass


class Synthea_RC_Validation(Synthea):
    pass


class Synthea_RC_Test(Synthea):
    pass


class Synthea_RC_Train_0shot(Synthea):
    pass


class Synthea_RC_Validation_0shot(Synthea):
    pass


class Synthea_RC_Test_0shot(Synthea):
    pass
