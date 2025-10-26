from collections import defaultdict
from functools import partial
from typing import DefaultDict, Optional

import numpy as np
from oe_eval.metrics.metric import MCAccuracy
from oe_eval.tasks.base_task import MultipleChoiceTask
from sklearn.metrics import f1_score


class MCAccuracyWithCustomMetricNames(MCAccuracy):
    """
    Mutiple choice accuracy metric group, including acc (aka. acc_raw), acc_norm (aka. acc_per_token), acc_uncond if applicable.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metric_names = kwargs.get("metric_names")

    def aggregate_to_task(self, primary_metric: Optional[str] = None) -> dict:
        """
        Aggregate the doc level metrics to final task level metrics.
        Results are stored in self._scores_for_task, and eventually populate metric.json
        """
        num_instances = len(self._scores_for_docs)
        if num_instances == 0:
            return self._scores_for_task

        metric_vals: DefaultDict[str, float] = defaultdict(float)
        first_metrics = self._scores_for_docs[0]["metrics"]
        doc_by_id = None
        # Temporary kluge to always track total price in extra metrics
        if "price" in first_metrics and "price" not in self.score_aggregation_fns:
            self.score_aggregation_fns["price"] = {"total_price": "sum"}
            if "total_price" not in self.extra_metric_names:
                self.extra_metric_names.append("price")

        for key in self.metric_names + self.extra_metric_names:
            aggregation_methods = self.score_aggregation_fns.get(key, {key: "mean"})
            for new_key, method in aggregation_methods.items():
                sample_val = first_metrics.get(key)
                # if not isinstance(sample_val, (int, float, bool)):
                #    continue
                if method == "mean":
                    if isinstance(sample_val, (int, float, bool)):
                        metric_vals[new_key] = (
                            sum(d["metrics"][key] for d in self._scores_for_docs) / num_instances
                        )
                elif method == "max":
                    if isinstance(sample_val, (int, float, bool)):
                        metric_vals[new_key] = max(d["metrics"][key] for d in self._scores_for_docs)
                elif method == "sum":
                    if isinstance(sample_val, (int, float, bool)):
                        metric_vals[new_key] = sum(d["metrics"][key] for d in self._scores_for_docs)
                elif callable(method):
                    if doc_by_id is None:
                        doc_by_id = {d["doc_id"]: d["doc"] for d in self._scores_for_requests}
                    res = method(key, new_key, self._scores_for_docs, doc_by_id)
                    if isinstance(res, dict):
                        metric_vals.update(res)
                    else:
                        metric_vals[new_key] = res
                else:
                    raise ValueError(f"Unsupported aggregation method: {method}")

        task_scores: dict = {**metric_vals}
        if primary_metric is not None and primary_metric in metric_vals:
            task_scores["primary_score"] = metric_vals[primary_metric]
        # Move extra metrics to "extra_metrics" dict
        for extra_metric in self.extra_metric_names:
            extra_metric_aggr_names = self.score_aggregation_fns.get(
                extra_metric, {extra_metric: "mean"}
            ).keys()
            for extra_metric_aggr_name in extra_metric_aggr_names:
                if extra_metric_aggr_name in task_scores:
                    if "extra_metrics" not in task_scores:
                        task_scores["extra_metrics"] = {}
                    task_scores["extra_metrics"][extra_metric_aggr_name] = task_scores[
                        extra_metric_aggr_name
                    ]
                    del task_scores[extra_metric_aggr_name]
        self._scores_for_task.update(task_scores)
        return self._scores_for_task


def compute_macro_f1(key, new_key, scores_for_docs, doc_by_id, predict_key):
    assert predict_key in ["raw", "uncond"]

    first_output = scores_for_docs[0]["model_output"]
    assert all([o["num_tokens"] == first_output[0]["num_tokens"] for o in first_output])
    predictions = [d["metrics"][f"predicted_index_{predict_key}"] for d in scores_for_docs]
    labels = [d["label"] for d in scores_for_docs]
    accs = [d["metrics"][f"acc_{predict_key}"] for d in scores_for_docs]

    assert all([int(pred == label) == acc for pred, label, acc in zip(predictions, labels, accs)])
    macro_f1 = f1_score(labels, predictions, average="macro")
    print("Computed Macro F1:", macro_f1)
    return macro_f1


metric_kwargs = {
    "metric_names": ["acc_raw", "macro-f1"],
    "score_aggregation_fns": {
        "acc_raw": {"acc": "mean"},
        "macro-f1": {"macro-f1": partial(compute_macro_f1, predict_key="raw")},
    },
}


"""
Note: Inheriting `MultipleChoiceTask` for classification tasks is a bit of waste of compute
for handling unconditional prompts.
In classification, the number of requests could be `(num_test_instances + 1) * num_options`
but in multi-choice, it's `2 * num_test_instances * num_options.`

I'm not creating a separate class for classification for now as this is the only
classification dataset, and the dataset is tiny anyways, but probably worth considering
if we add more large classification datasets in the future
"""


class HateSpeech18(MultipleChoiceTask):
    VERSION = 0
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "odegiber/hate_speech18",
        "native_id_field": "index",
        "num_shots": 5,
        "primary_metric": "macro-f1",
        "split": "test",
    }

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        super().download(data_dir, cache_dir, download_mode)
        self.dataset = self.dataset["train"].train_test_split(
            test_size=0.9, seed=12
        )  # default seed gives all "no"s

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(
                self.dataset["train"]
                .filter(lambda x: x["label"] in [0, 1])
                .map(self._process_doc, with_indices=True)
            )
        return self._training_docs

    def test_docs(self):
        return list(
            self.dataset["test"]
            .filter(lambda x: x["label"] in [0, 1])
            .map(self._process_doc, with_indices=True)
        )

    def unconditioned_prompt(self):
        return None

    def make_metrics(self):
        self.task_config["metric_kwargs"] = metric_kwargs
        self._metrics = [MCAccuracyWithCustomMetricNames(**self.task_config["metric_kwargs"])]
        return self._metrics

    def _process_doc(self, doc, index=-1):
        query = f"Text: {doc['text']}\nQuestion: Is this hate speech or not?\nAnswer:"
        assert doc["label"] in [0, 1]
        out_doc = {
            "index": index,
            "query": query,
            "choices": ["No", "Yes"],
            "gold": doc["label"],
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]


class HateSpeechOffensive(MultipleChoiceTask):
    VERSION = 0
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "UKPLab/hate_speech_offensive",
        "native_id_field": "index",
        "num_shots": 5,
        "primary_metric": "macro-f1",
        "split": "test",
    }

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(
                self.dataset["train"].map(self._process_doc, with_indices=True)
            )
        return self._training_docs

    def test_docs(self):
        return list(self.dataset["test"].map(self._process_doc, with_indices=True))

    def unconditioned_prompt(self):
        return None

    def make_metrics(self):
        self.task_config["metric_kwargs"] = metric_kwargs
        self._metrics = [MCAccuracyWithCustomMetricNames(**self.task_config["metric_kwargs"])]
        return self._metrics

    def _process_doc(self, doc, index=-1):
        query = f"Text: {doc['text']}\nQuestion: Is this hate speech or offensive?\nAnswer:"
        assert doc["labels"] in [0, 1, 2]
        out_doc = {
            "index": index,
            "query": query,
            "choices": ["No", "Yes"],
            "gold": int(doc["labels"] != 2),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]


class TweetEvalHate(MultipleChoiceTask):
    VERSION = 0
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "Intuit-GenSRF/tweet-eval-hate",
        "native_id_field": "index",
        "num_shots": 5,
        "primary_metric": "macro-f1",
        "split": "test",
    }

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        super().download(data_dir, cache_dir, download_mode)
        self.dataset = self.dataset["train"].train_test_split(test_size=0.9)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(
                self.dataset["train"].map(self._process_doc, with_indices=True)
            )
        return self._training_docs

    def test_docs(self):
        return list(self.dataset["test"].map(self._process_doc, with_indices=True))

    def unconditioned_prompt(self):
        return None

    def make_metrics(self):
        self.task_config["metric_kwargs"] = metric_kwargs
        self._metrics = [MCAccuracyWithCustomMetricNames(**self.task_config["metric_kwargs"])]
        return self._metrics

    def _process_doc(self, doc, index=-1):
        query = f"Text: {doc['text']}\nQuestion: Is this hate speech or not?\nAnswer:"
        out_doc = {
            "index": index,
            "query": query,
            "choices": ["No", "Yes"],
            "gold": int(len(doc["labels"]) > 0),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]


class Hatexplain(MultipleChoiceTask):
    VERSION = 0
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "Hate-speech-CNERG/hatexplain",
        "native_id_field": "index",
        "num_shots": 5,
        "primary_metric": "macro-f1",
        "split": "test",
    }

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(
                self.dataset["train"]
                .map(self._process_doc, with_indices=True)
                .filter(lambda x: x["gold"] is not None)
            )
            np.random.seed(12)
            np.random.shuffle(self._training_docs)
        return self._training_docs

    def test_docs(self):
        return list(
            self.dataset["test"]
            .map(self._process_doc, with_indices=True)
            .filter(lambda x: x["gold"] is not None)
        )

    def unconditioned_prompt(self):
        return None

    def make_metrics(self):
        self.task_config["metric_kwargs"] = metric_kwargs
        self._metrics = [MCAccuracyWithCustomMetricNames(**self.task_config["metric_kwargs"])]
        return self._metrics

    def _process_doc(self, doc, index=-1):
        text = " ".join(doc["post_tokens"])
        query = f"Text: {text}\nQuestion: Is this hate speech or offensive?\nAnswer:"
        labels = doc["annotators"]["label"]
        if np.all([label == 1 for label in labels]):
            gold = 1
        elif np.all([label in [0, 2] for label in labels]):
            gold = 0
        else:
            gold = None

        out_doc = {
            "index": index,
            "query": query,
            "choices": ["No", "Yes"],
            "gold": gold,
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]
