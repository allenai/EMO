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


class PIQA_MC_Train(PIQA_MC_BASE):
    pass


class PIQA_MC_Validation(PIQA_MC_BASE):
    pass


class PIQA_MC_Test(PIQA_MC_BASE):
    pass
