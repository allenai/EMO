from oe_eval.tasks.oe_eval_tasks.arc import (
    ARCChallenge,
    ARCChallengeMC,
    ARCEasy,
    ARCEasyMC,
)

from ..metrics.mc_softloss import SoftLoss

class ARCEasy_RC_Base(ARCEasy):
    def make_metrics(self):
        # run the super
        super().make_metrics()
        # add softloss metric
        self._metrics += [SoftLoss(**self.task_config["metric_kwargs"])]

        return self._metrics

class ARCChallenge_RC_Base(ARCChallenge):
    def make_metrics(self):
        # run the super
        super().make_metrics()
        # add softloss metric
        self._metrics += [SoftLoss(**self.task_config["metric_kwargs"])]

        return self._metrics

# class ARCEasy_RC_Train(ARCEasy_RC_Base):
#     pass
#
#
# class ARCEasy_RC_Validation(ARCEasy_RC_Base):
#     pass
#
#
# class ARCEasy_RC_Train_0shot(ARCEasy_RC_Base):
#     pass
#
#
# class ARCEasy_RC_Validation_0shot(ARCEasy_RC_Base):
#     pass
#
#
# class ARCEasy_RC_Test(ARCEasy_RC_Base):
#     pass


class ARCEasy_MC_Train(ARCEasyMC):
    pass


class ARCEasy_MC_Validation(ARCEasyMC):
    pass


class ARCEasy_MC_Test(ARCEasyMC):
    pass


# class ARCChallenge_RC_Train(ARCChallenge_RC_Base):
#     pass
#
#
# class ARCChallenge_RC_Validation(ARCChallenge_RC_Base):
#     pass
#
#
# class ARCChallenge_RC_Train_0shot(ARCChallenge_RC_Base):
#     pass
#
#
# class ARCChallenge_RC_Validation_0shot(ARCChallenge_RC_Base):
#     pass
#
#
# class ARCChallenge_RC_Test(ARCChallenge_RC_Base):
#     pass


class ARCChallenge_MC_Train(ARCChallengeMC):
    pass


class ARCChallenge_MC_Validation(ARCChallengeMC):
    pass


class ARCChallenge_MC_Test(ARCChallengeMC):
    pass
