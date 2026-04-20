from datasets import concatenate_datasets
from oe_eval.tasks.oe_eval_tasks.triviaqa import TriviaQA


class TriviaQA_Base(TriviaQA):
    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            train_dataset = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(1000, len(self.dataset["train"])))
            )
            self._training_docs = list(map(self._process_doc, train_dataset))
        return self._training_docs

    def validation_docs(self):
        val_dataset = self.dataset["train"].shuffle(seed=0).select(range(0, 1000))
        return list(map(self._process_doc, val_dataset))

    def test_docs(self):
        return list(map(self._process_doc, self.dataset["validation"]))


class TriviaQA_Train_0shot(TriviaQA_Base):
    pass


class TriviaQA_Validation_0shot(TriviaQA_Base):
    pass


class TriviaQA_Test_0shot(TriviaQA_Base):
    pass


# ---------------------------------------------------------------------------
# Merged variant
# ---------------------------------------------------------------------------
class TriviaQA_Merged_Base(TriviaQA_Base):
    def training_docs(self):
        if self._training_docs is None:
            train_dataset = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(1000, len(self.dataset["train"])))
            )
            val_dataset = self.dataset["train"].shuffle(seed=0).select(range(0, 1000))
            merged = concatenate_datasets([train_dataset, val_dataset]).shuffle(seed=0)
            self._training_docs = list(map(self._process_doc, merged))
        return self._training_docs

    def validation_docs(self):
        return self.training_docs()


class TriviaQA_Merged(TriviaQA_Merged_Base):
    pass
