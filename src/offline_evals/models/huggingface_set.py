from typing import List

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import (
    GenerateUntilAndLoglikelihoodRequest,
    GenerateUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
)
from oe_eval.models.eleuther_huggingface import HFLM_Verbose


class HFLM_Set_Verbose(HFLM_Verbose):
    def __init__(self, pretrained: List[str], **kwargs):
        self.pretrained_list = [HFLM_Verbose(_pretrained, **kwargs) for _pretrained in pretrained]

        self.model_idx = 0

    def loglikelihood_rolling_verbose(
        self, requests: List[LoglikelihoodRollingRequest], **kwargs
    ) -> List[float]:
        return self.pretrained_list[self.model_idx].loglikelihood_rolling_verbose(
            requests, **kwargs
        )

    def loglikelihood_verbose(self, requests: List[LoglikelihoodRequest], **kwargs) -> List[float]:
        return self.pretrained_list[self.model_idx].loglikelihood_verbose(requests, **kwargs)

    def generate_until_verbose(self, requests: List[GenerateUntilRequest], **kwargs) -> List[dict]:
        return self.pretrained_list[self.model_idx].generate_util_verbose(requests, **kwargs)

    def generate_until_and_loglikelihood_verbose(
        self, instances: List[RequestInstance], requests: List[GenerateUntilAndLoglikelihoodRequest]
    ):
        return self.pretrained_list[self.model_idx].generate_util_and_loglikelihood_verbose(
            instances, requests
        )
