from oe_eval.tasks.base_task import MultipleChoiceTask
from oe_eval.tasks.utils import make_mcq_prompt, map_indexed


class MedQA(MultipleChoiceTask):
    VERSION = 0
    MAX_TRAINING_DOCS = 10000
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": "bigbio/med_qa",
        "fewshot_source": None,
        "primary_metric": "acc_raw",
        "num_shots": 5,
        "split": "test",
    }

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map_indexed(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return map_indexed(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map_indexed(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc, index=-1):
        question = doc["question"]
        choices = [option["value"] for option in doc["options"]]
        gold = ["A", "B", "C", "D", "E"].index(doc["answer_idx"])
        assert choices[gold] == doc["answer"]

        query = make_mcq_prompt(doc["question"], choices, question_prefix="Question: ")

        out_doc = {
            "index": index,
            "question": question,
            "query": query,
            "choices": choices,
            "gold": gold,
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]
        # return "Question: " + doc["question"] + "\nAnswer:"

    # def doc_to_target(self, doc):
    #    return " " + doc["choices"][doc["gold"]]

    def unconditioned_prompt(self):
        # Don't need unconditioned normalization here
        return None
