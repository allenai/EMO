from typing import Optional, List

import torch
from oe_eval.metrics.metric import Metric

class SoftLoss(Metric):
    """
    Mutiple choice softloss metric
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metric_names = [
            "softloss_corr",
            "softloss_per_token_corr",
            "softloss_per_char_corr",
        ]

    def compute_for_requests(self, results_for_requests) -> List[dict]:
        """ """

        # compute logits_per_token for acc_norm
        for res in results_for_requests:
            num_tokens = max(res["model_resps"]["num_tokens"], 1)
            num_chars = max(len(res["request"].get("continuation", "")), 1)
            num_bytes = max(len(res["request"].get("continuation", "").encode("utf-8")), 1)
            res["model_resps"].update(
                {
                    "logits_per_token": res["model_resps"]["sum_logits"] / num_tokens,
                    "logits_per_char": res["model_resps"]["sum_logits"] / num_chars,
                }
            )

        return results_for_requests

    def process_one_doc(self, group_lst) -> dict:
        """
        Currently computes acc, acc_norm, and acc_uncond if uncond_docid_offset is set.
        """
        label = group_lst[0]["label"]

        sum_logits = [x["model_resps"]["sum_logits"] for x in group_lst]
        logits_per_token = [x["model_resps"]["logits_per_token"] for x in group_lst]
        logits_per_char = [x["model_resps"]["logits_per_char"] for x in group_lst]

        if (
            isinstance(label, str)
            or isinstance(label, list)
            or (label is None and len(sum_logits) > 0)
        ):
            label = 0  # this is very problematic, do not keep

        breakpoint()

        sum_logits_softmax = torch.nn.functional.softmax(sum_logits, dim=-1)
        softloss_corr = sum_logits_softmax[label] if label is not None else 0.0

        logits_per_token_softmax = torch.nn.functional.softmax(logits_per_token, dim=-1)
        softloss_per_token_corr = logits_per_token_softmax[label] if label is not None else 0.0

        logits_per_char_softmax = torch.nn.functional.softmax(logits_per_char, dim=-1)
        softloss_per_char_corr = logits_per_char_softmax[label] if label is not None else 0.0


        return {
            "softloss_corr": softloss_corr,
            "softloss_per_token_corr": softloss_per_token_corr,
            "softloss_per_char_corr": softloss_per_char_corr,
        }