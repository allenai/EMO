from datasets import concatenate_datasets
from oe_eval.tasks.oe_eval_tasks.openbookqa import OpenBookQA, OpenBookQAMC

from ..metrics.mc_softloss import SoftLoss


class OpenBookQA_RC_Base(OpenBookQA):
    def make_metrics(self):
        # run the super
        super().make_metrics()
        # add softloss metric
        self._metrics += [SoftLoss(**self.task_config["metric_kwargs"])]

        return self._metrics


# class OpenBookQA_RC_Train(OpenBookQA_RC_Base):
#     pass
#
#
# class OpenBookQA_RC_Validation(OpenBookQA_RC_Base):
#     pass
#
#
# class OpenBookQA_RC_Train_0shot(OpenBookQA_RC_Base):
#     pass
#
#
# class OpenBookQA_RC_Validation_0shot(OpenBookQA_RC_Base):
#     pass
#
#
# class OpenBookQA_RC_Test(OpenBookQA_RC_Base):
#     pass
#

# ---------------------------------------------------------------------------
# Merged variant: HF train + HF validation combined for both pruning and
# finetuning. The two HF splits are independent, so we shuffle the merged set.
# Test split (HF "test") is unchanged.
# ---------------------------------------------------------------------------


class OpenBookQA_Merged_RC(OpenBookQA_RC_Base):
    """OpenBookQA variant where pruning and finetuning use the same merged data."""

    def training_docs(self):
        if self._training_docs is None:
            merged = concatenate_datasets(
                [self.dataset["train"], self.dataset["validation"]]
            ).shuffle(seed=0)
            self._training_docs = list(map(self._process_doc, merged))
        return self._training_docs

    def validation_docs(self):
        return self.training_docs()


class OpenBookQA_MC_Train(OpenBookQAMC):
    pass


class OpenBookQA_MC_Validation(OpenBookQAMC):
    pass


class OpenBookQA_MC_Test(OpenBookQAMC):
    pass
