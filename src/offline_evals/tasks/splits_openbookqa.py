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

class OpenBookQA_MC_Train(OpenBookQAMC):
    pass


class OpenBookQA_MC_Validation(OpenBookQAMC):
    pass


class OpenBookQA_MC_Test(OpenBookQAMC):
    pass
