from typing import List

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
        # comput sum_logits_uncond for acc_uncond (optional)
        if hasattr(self, "uncond_docid_offset") and self.uncond_docid_offset is not None:
            middle = len(results_for_requests) // 2
            results_contexted = results_for_requests[:middle]
            results_uncond = results_for_requests[middle:]

            for res1, res2 in zip(results_contexted, results_uncond):
                assert (
                    res2["doc_id"] == res1["doc_id"] + self.uncond_docid_offset
                ), "doc_id orders between contexted and uncond results do not match!"
                assert (
                    res2["idx"] == res1["idx"]
                ), "idx orders between contexted and uncond results do not match!"
                # We could sort the two results lists by doc_id, idx before the zip, but if the asserts fail it suggests something upstream was scrambled and requires investigation.
                res1["model_resps"].update({"sum_logits_uncond": res2["model_resps"]["sum_logits"]})

            results_for_requests = results_contexted

        # compute logits_per_token for acc_norm
        for res in results_for_requests:
            num_tokens = max(res["model_resps"]["num_tokens"], 1)
            num_chars = max(len(res["request"].get("continuation", "")), 1)
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

        sum_logits_softmax = torch.nn.functional.softmax(torch.tensor(sum_logits), dim=-1)
        softloss_corr = sum_logits_softmax[label] if label is not None else 0.0

        logits_per_token_softmax = torch.nn.functional.softmax(
            torch.tensor(logits_per_token), dim=-1
        )
        softloss_per_token_corr = logits_per_token_softmax[label] if label is not None else 0.0

        logits_per_char_softmax = torch.nn.functional.softmax(torch.tensor(logits_per_char), dim=-1)
        softloss_per_char_corr = logits_per_char_softmax[label] if label is not None else 0.0

        return {
            "softloss_corr": softloss_corr.item(),
            "softloss_per_token_corr": softloss_per_token_corr.item(),
            "softloss_per_char_corr": softloss_per_char_corr.item(),
        }
