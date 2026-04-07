from datasets import concatenate_datasets
from oe_eval.tasks.oe_eval_tasks.piqa import PiQA, PiQAMC

from ..metrics.mc_softloss import SoftLoss

class PIQA_RC_Base(PiQA):
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
        return map(self._process_doc, val_dataset)

    def test_docs(self):
        # validation set used to be the test set by default if a test set did not exist, so we still set it as test set
        return map(self._process_doc, self.dataset["validation"])

    def make_metrics(self):
        # run the super
        super().make_metrics()
        # add softloss metric
        self._metrics += [SoftLoss(**self.task_config["metric_kwargs"])]

        return self._metrics


class PIQA_MC_BASE(PiQAMC):
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
        return map(self._process_doc, val_dataset)

    def test_docs(self):
        # validation set used to be the test set by default if a test set did not exist, so we still set it as test set
        return map(self._process_doc, self.dataset["validation"])


# class PIQA_RC_Train(PIQA_RC_BASE):
#     pass
#
#
# class PIQA_RC_Validation(PIQA_RC_BASE):
#     pass
#
#
# class PIQA_RC_Train_0shot(PIQA_RC_BASE):
#     pass
#
#
# class PIQA_RC_Validation_0shot(PIQA_RC_BASE):
#     pass
#
#
# class PIQA_RC_Test(PIQA_RC_BASE):
#     pass


# ---------------------------------------------------------------------------
# Merged variant: train+val combined for both pruning and finetuning.
# Both halves come from the same shuffled train, so no extra shuffle is needed.
# ---------------------------------------------------------------------------

class PIQA_Merged_RC(PIQA_RC_Base):
    """PIQA variant where pruning and finetuning use the same merged data."""

    def training_docs(self):
        if self._training_docs is None:
            shuffled = self.dataset["train"].shuffle(seed=0)
            train_part = shuffled.select(range(1000, len(shuffled)))
            val_part = shuffled.select(range(0, 1000))
            merged = concatenate_datasets([train_part, val_part])
            self._training_docs = list(map(self._process_doc, merged))
        return self._training_docs

    def validation_docs(self):
        return self.training_docs()


class PIQA_MC_Train(PIQA_MC_BASE):
    pass


class PIQA_MC_Validation(PIQA_MC_BASE):
    pass


class PIQA_MC_Test(PIQA_MC_BASE):
    pass
