from datasets import concatenate_datasets
from oe_eval.tasks.oe_eval_tasks.naturalqs_open import NaturalQsOpen


class NaturalQS_Base(NaturalQsOpen):
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


class NaturalQS_Train_0shot(NaturalQS_Base):
    pass


class NaturalQS_Validation_0shot(NaturalQS_Base):
    pass


class NaturalQS_Test_0shot(NaturalQS_Base):
    pass


# ---------------------------------------------------------------------------
# Merged variant: concatenates train + val into a single set so pruning and
# finetuning use identical data.  Test split is unchanged (HF validation).
# ---------------------------------------------------------------------------
class NaturalQS_Merged_Base(NaturalQS_Base):
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


class NaturalQS_Merged(NaturalQS_Merged_Base):
    pass
