from datasets import concatenate_datasets

from .squad import SQuAD


class SQUAD_Base(SQuAD):
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
            self._training_docs = list(map(self._process_doc, train_dataset))
        return self._training_docs

    def validation_docs(self):
        # select 1000 examples from the train set as validation set
        val_dataset = self.dataset["train"].shuffle(seed=0).select(range(0, 1000))
        return list(map(self._process_doc, val_dataset))

    def test_docs(self):
        # validation set used to be the test set by default if a test set did not exist, so we still set it as test set
        return list(map(self._process_doc, self.dataset["validation"]))


class SQUAD_Train_0shot(SQUAD_Base):
    pass


class SQUAD_Validation_0shot(SQUAD_Base):
    pass


class SQUAD_Test_0shot(SQUAD_Base):
    pass


# ---------------------------------------------------------------------------
# Merged variant: concatenates train + val into a single set so pruning and
# finetuning use identical data.  Test split is unchanged (HF validation).
# ---------------------------------------------------------------------------
class SQUAD_Merged_Base(SQUAD_Base):
    def training_docs(self):
        if self._training_docs is None:
            train_dataset = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(1000, len(self.dataset["train"])))
            )
            val_dataset = (
                self.dataset["train"]
                .shuffle(seed=0)
                .select(range(0, 1000))
            )
            merged = concatenate_datasets([train_dataset, val_dataset]).shuffle(seed=0)
            self._training_docs = list(map(self._process_doc, merged))
        return self._training_docs

    def validation_docs(self):
        return self.training_docs()


class SQUAD_Merged(SQUAD_Merged_Base):
    pass


class SQUAD_Merged_0shot(SQUAD_Merged_Base):
    pass
