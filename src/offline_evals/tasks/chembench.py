"""
ChemBench: A benchmark for evaluating chemistry and materials science capabilities of LLMs.
https://huggingface.co/datasets/jablonkagroup/ChemBench

ChemBench has two types of questions determined by the `preferred_score` field:
1. Multiple choice (MC): preferred_score = "multiple_choice_grade"
   - target_scores contains options with scores (1.0 = correct)
2. Open-ended: preferred_score = "exact_string_match", "mae", or "mse"
   - target contains the expected answer

This module provides separate task classes for each type.

NOTE ON EVALUATION METHODOLOGY:
- MC tasks: We use log-likelihood scoring (standard for oe_eval MC tasks).
- Gen tasks: We use strict F1/exact_match string comparison.

This differs from ChemBench's official evaluation which:
- Uses generative output with [ANSWER]...[/ANSWER] tags for MC
- Parses numeric answers as floats for mae/mse questions (so "6" == "6.0")
- See: https://lamalab-org.github.io/chembench/

We use strict string matching for exact_string_match questions, but for mae/mse
we parse floats so numeric formatting differences (e.g., "6" vs "6.0") do not
count as mismatches.
"""

import json
import logging
import re
from typing import List, Union

from oe_eval.components.instances import RequestInstance
from oe_eval.metrics.metric import MCAccuracy, SQuADF1EMRecallMetric
from oe_eval.tasks.base_task import MultipleChoiceTask, Task
from oe_eval.tasks.utils import make_cloze_prompt, make_mcq_prompt, map_indexed
from oe_eval.utils import get_dict_with_defaults


logger = logging.getLogger(__name__)


class ChemBenchMCAccuracy(MCAccuracy):
    """
    Custom MC accuracy metric that supports multiple correct answers.
    
    ChemBench has some questions with multiple correct answers (e.g., "select all that apply").
    This metric checks if the predicted answer is in the list of correct answers.
    """
    
    def process_one_doc(self, group_lst) -> dict:
        """Override to handle gold as a list of valid indices."""
        # Get the base metrics from parent
        base_result = super().process_one_doc(group_lst)
        
        # Get the gold indices (could be a list for multi-answer questions)
        doc = group_lst[0].get("doc", {})
        gold_indices = doc.get("gold", [])
        
        # Ensure gold_indices is a list
        if not isinstance(gold_indices, list):
            gold_indices = [gold_indices]
        
        # Check each accuracy metric against all valid gold indices
        for metric_name in ["acc_raw", "acc_per_token", "acc_per_char", "acc_per_byte"]:
            pred_key = metric_name.replace("acc_", "predicted_index_")
            if pred_key in base_result:
                predicted = base_result[pred_key]
                # Mark as correct if predicted is ANY of the valid answers
                base_result[metric_name] = 1 if predicted in gold_indices else 0
        
        # Update correct_choice to show all valid choices
        base_result["correct_choice"] = gold_indices
        
        return base_result


# All ChemBench subfields
CHEMBENCH_SUBFIELDS = [
    "analytical_chemistry",
    "chemical_preference",
    "general_chemistry",
    "inorganic_chemistry",
    "materials_science",
    "organic_chemistry",
    "physical_chemistry",
    "technical_chemistry",
    "toxicity_and_safety",
]

# Question types based on preferred_score field
MC_SCORE_TYPE = "multiple_choice_grade"
OPEN_ENDED_SCORE_TYPES = {"exact_string_match", "mae", "mse"}

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_float(text: str):
    if not text:
        return None
    match = _FLOAT_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


class ChemBenchGenMetric(SQuADF1EMRecallMetric):
    """
    SQuAD-style metric with numeric parsing for ChemBench mae/mse questions.

    For preferred_score in {"mae", "mse"}, treat numeric predictions as equal if the
    parsed floats match within tolerance, so "6" == "6.0".
    """

    _FLOAT_TOL = 1e-6

    def process_one_doc(self, group_lst) -> dict:
        base_result = super().process_one_doc(group_lst)
        doc = group_lst[0].get("doc", {})
        preferred_score = doc.get("preferred_score", "")
        if preferred_score not in {"mae", "mse"}:
            return base_result

        pred_text = None
        for item in group_lst:
            for key in ("pred", "prediction", "completion", "text", "response"):
                if isinstance(item.get(key), str) and item.get(key).strip():
                    pred_text = item.get(key)
                    break
            if pred_text is not None:
                break

        target_text = doc.get("target", "")
        pred_val = _extract_float(pred_text or "")
        target_val = _extract_float(target_text)
        if pred_val is None or target_val is None:
            return base_result

        mae = abs(pred_val - target_val)
        mse = (pred_val - target_val) ** 2
        exact_match = 1 if mae <= self._FLOAT_TOL else 0

        for key in ("f1", "exact_match", "recall"):
            if key in base_result:
                base_result[key] = exact_match
        base_result["mae"] = mae
        base_result["mse"] = mse
        return base_result


def create_chembench_tasks() -> dict:
    """Create all ChemBench subfield tasks (MC, RC, and generative)."""
    all_tasks = {}
    for subfield in CHEMBENCH_SUBFIELDS:
        # Multiple choice tasks
        all_tasks[f"chembench_{subfield}:mc"] = create_chembench_mc_task(subfield)
        # Ranked classification tasks
        all_tasks[f"chembench_{subfield}:rc"] = create_chembench_rc_task(subfield)
        # Open-ended/generative tasks
        all_tasks[f"chembench_{subfield}:gen"] = create_chembench_gen_task(subfield)
    return all_tasks


def create_chembench_mc_task(subfield: str):
    """Factory function to create a ChemBench MC task for a specific subfield."""

    class ChemBenchMC(GenericChemBenchMC):
        TASK_CONFIG_DEFAULTS = get_dict_with_defaults(
            {
                "dataset_name": subfield,
            },
            GenericChemBenchMC.TASK_CONFIG_DEFAULTS,
        )

    return ChemBenchMC


def create_chembench_gen_task(subfield: str):
    """Factory function to create a ChemBench generative task for a specific subfield."""

    class ChemBenchGen(GenericChemBenchGen):
        TASK_CONFIG_DEFAULTS = get_dict_with_defaults(
            {
                "dataset_name": subfield,
            },
            GenericChemBenchGen.TASK_CONFIG_DEFAULTS,
        )

    return ChemBenchGen


def create_chembench_rc_task(subfield: str):
    """Factory function to create a ChemBench RC task for a specific subfield."""

    class ChemBenchRC(GenericChemBenchRC):
        TASK_CONFIG_DEFAULTS = get_dict_with_defaults(
            {
                "dataset_name": subfield,
            },
            GenericChemBenchRC.TASK_CONFIG_DEFAULTS,
        )

    return ChemBenchRC


class GenericChemBenchChoice(MultipleChoiceTask):
    """
    ChemBench choice task (shared base for MC and RC).

    Only processes questions where preferred_score == "multiple_choice_grade".

    ChemBench MC format:
    - Each row has a 'preferred_score' field indicating question type
    - The 'examples' field contains a list with one dict
    - The dict has 'input' (question) and 'target_scores' (JSON string of answer options)
    - target_scores format: {"option1": 0.0, "option2": 1.0, ...} where 1.0 is correct
    
    NOTE: Some questions have multiple correct answers. We use ChemBenchMCAccuracy
    to handle this - any correct answer is accepted.
    """

    VERSION = 0
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": "jablonkagroup/ChemBench",
        "dataset_name": None,  # subfield name, e.g., "organic_chemistry"
        "fewshot_source": None,
        "primary_metric": "acc_raw",
        "num_shots": 3,  # Five-shot by default for ChemBench choice tasks
        "split": "train",  # ChemBench only has train split
    }

    def make_metrics(self):
        """Use custom metric that supports multiple correct answers."""
        self._metrics = [ChemBenchMCAccuracy(**self.task_config["metric_kwargs"])]
        return self._metrics

    def has_training_docs(self):
        # ChemBench only has a "train" split, so we use it for evaluation
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def training_docs(self):
        # Filter to only choice questions (preferred_score == "multiple_choice_grade")
        choice_docs = [
            doc for doc in self.dataset["train"]
            if self._is_choice_question(doc)
        ]
        # logger.info(
        #     f"ChemBench choice: Found {len(choice_docs)} multiple-choice questions "
        #     f"out of {len(self.dataset['train'])} total"
        # )
        return list(map_indexed(self._process_doc, choice_docs))

    def fewshot_examples(self, k, rnd, doc):
        """Cap fewshot sampling when a subtask has fewer examples than requested."""
        fewshot_source = self.task_config.get("fewshot_source")
        if rnd is None and fewshot_source is None:
            raise ValueError("A `random.Random` generator argument must be provided to `rnd`")

        if fewshot_source is not None:
            from oe_eval.tasks.base_task import FEWSHOT_SOURCES

            if fewshot_source not in FEWSHOT_SOURCES:
                raise ValueError(f"Fewshot source '{fewshot_source}' not found in FEWSHOT_SOURCES!")
            self._fewshot_docs = list(map(self._process_doc, FEWSHOT_SOURCES[fewshot_source]))
            if len(self._fewshot_docs) < k:
                logger.warning(
                    "ChemBench Choice: capping num_shots from %d to %d (fewshot_source=%s)",
                    k,
                    len(self._fewshot_docs),
                    fewshot_source,
                )
            return self._fewshot_docs[: min(k, len(self._fewshot_docs))]

        if self.has_training_docs():
            self._fewshot_docs = list(self.training_docs())
            if len(self._fewshot_docs) < k:
                logger.warning(
                    "ChemBench Choice: capping num_shots from %d to %d for subfield %s",
                    k,
                    len(self._fewshot_docs),
                    self.task_config.get("dataset_name"),
                )
            return rnd.sample(self._fewshot_docs, min(k, len(self._fewshot_docs)))

        self._fewshot_docs = list(
            self.validation_docs() if self.has_validation_docs() else self.test_docs()
        )
        sample_k = min(k + 1, len(self._fewshot_docs))
        if sample_k < k + 1:
            logger.warning(
                "ChemBench Choice: capping num_shots from %d to %d (no train docs)",
                k,
                max(sample_k - 1, 0),
            )
        return rnd.sample(self._fewshot_docs, sample_k)

    def _is_choice_question(self, doc) -> bool:
        """Check if this is a choice question based on preferred_score field."""
        preferred_score = doc.get("preferred_score", "")
        return preferred_score == MC_SCORE_TYPE

    def _extract_choices_and_gold(self, doc):
        """Extract question, choices, and gold indices from a ChemBench document."""
        example = doc["examples"][0]
        question = example["input"]

        # Parse target_scores to get choices and correct answer(s)
        target_scores = json.loads(example["target_scores"])

        # Extract choices and find ALL correct answers (score == 1.0)
        # Some ChemBench questions have multiple correct answers
        choices = list(target_scores.keys())
        gold_indices = []
        for i, (choice, score) in enumerate(target_scores.items()):
            if score == 1.0:
                gold_indices.append(i)

        if not gold_indices:
            # No correct answer found - this is malformed data
            raise ValueError(
                f"No correct answer (score=1.0) found in target_scores for question: "
                f"{question[:100]}... Scores: {target_scores}"
            )

        return question, choices, gold_indices

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        gold = doc.get("gold")
        if isinstance(gold, list):
            if not gold:
                raise ValueError("ChemBench choice doc missing gold indices")
            gold = gold[0]
        return " " + doc["choices"][gold]

    def unconditioned_prompt(self):
        # Don't need unconditioned normalization
        return None


class GenericChemBenchMC(GenericChemBenchChoice):
    """
    ChemBench multiple choice task.

    Uses an MCQ-style prompt with all options listed in the context.
    """

    def _process_doc(self, doc, index=-1):
        """Process a ChemBench MC document into the standard format."""
        question, choices, gold_indices = self._extract_choices_and_gold(doc)

        # Build the MCQ prompt
        query = make_mcq_prompt(question, choices, question_prefix="Question: ")

        out_doc = {
            "index": index,
            "question": question,
            "query": query,
            "choices": choices,  # Actual choice text for log-likelihood scoring
            "gold": gold_indices,  # List of all correct answer indices
            "name": doc.get("name", ""),
            "subfield": doc.get("subfield", ""),
        }
        return out_doc


class GenericChemBenchRC(GenericChemBenchChoice):
    """
    ChemBench ranked classification task.

    Uses a cloze-style prompt (no answer choices in the context) and scores each
    option as a continuation.
    """

    TASK_CONFIG_DEFAULTS = {
        **GenericChemBenchChoice.TASK_CONFIG_DEFAULTS,
        "primary_metric": "acc_per_char",
    }

    def _process_doc(self, doc, index=-1):
        """Process a ChemBench RC document into the standard format."""
        question, choices, gold_indices = self._extract_choices_and_gold(doc)

        # Build the cloze prompt (no choices in context)
        query = make_cloze_prompt(question, question_prefix="Question: ")

        out_doc = {
            "index": index,
            "question": question,
            "query": query,
            "choices": choices,
            "gold": gold_indices,
            "name": doc.get("name", ""),
            "subfield": doc.get("subfield", ""),
        }
        return out_doc

    def unconditioned_prompt(self):
        return "Answer:"


class GenericChemBenchGen(Task):
    """
    ChemBench open-ended/generative task.

    Processes questions where preferred_score is one of:
    - "exact_string_match", "mae", or "mse"

    Uses F1/exact_match scoring (strict string comparison).

    ChemBench open-ended format:
    - The 'examples' field contains a list with one dict
    - The dict has 'input' (question) and 'target' (expected answer string/number)
    """

    VERSION = 0
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": "jablonkagroup/ChemBench",
        "dataset_name": None,  # subfield name
        "native_id_field": "uuid",
        "fewshot_source": None,
        "primary_metric": "f1",  # Use F1/EM for string matching
        "num_shots": 3,
        "split": "train",
        "context_kwargs": {},
        "generation_kwargs": {
            "max_gen_toks": 100,
            "temperature": 0.0,
            "do_sample": False,
            "stop_sequences": ["\n\n", "\n"],
        },
    }

    def make_metrics(self):
        self._metrics = [ChemBenchGenMetric(**self.task_config["metric_kwargs"])]
        return self._metrics

    def has_training_docs(self):
        # ChemBench only has a "train" split, so we use it for evaluation
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def training_docs(self):
        # Filter to only open-ended questions (preferred_score in {"exact_string_match", "mae", "mse"})
        gen_docs = [
            doc for doc in self.dataset["train"]
            if self._is_gen_question(doc)
        ]
        logger.info(
            f"ChemBench Gen: Found {len(gen_docs)} open-ended questions "
            f"out of {len(self.dataset['train'])} total"
        )
        return list(map_indexed(self._process_doc, gen_docs))

    def _is_gen_question(self, doc) -> bool:
        """Check if this is an open-ended/generative question based on preferred_score field."""
        preferred_score = doc.get("preferred_score", "")
        return preferred_score in OPEN_ENDED_SCORE_TYPES

    def _process_doc(self, doc, index=-1):
        """Process a ChemBench open-ended document."""
        example = doc["examples"][0]
        question = example["input"]
        target = example["target"]

        out_doc = {
            "index": index,
            "uuid": doc.get("uuid", ""),
            "question": question,
            "target": target,
            "preferred_score": doc.get("preferred_score", ""),
            "name": doc.get("name", ""),
            "subfield": doc.get("subfield", ""),
        }
        return out_doc

    def fewshot_examples(self, k, rnd, doc):
        """Cap fewshot sampling when a subtask has fewer examples than requested."""
        fewshot_source = self.task_config.get("fewshot_source")
        if rnd is None and fewshot_source is None:
            raise ValueError("A `random.Random` generator argument must be provided to `rnd`")

        if fewshot_source is not None:
            from oe_eval.tasks.base_task import FEWSHOT_SOURCES

            if fewshot_source not in FEWSHOT_SOURCES:
                raise ValueError(f"Fewshot source '{fewshot_source}' not found in FEWSHOT_SOURCES!")
            self._fewshot_docs = list(map(self._process_doc, FEWSHOT_SOURCES[fewshot_source]))
            if len(self._fewshot_docs) < k:
                logger.warning(
                    "ChemBench Gen: capping num_shots from %d to %d (fewshot_source=%s)",
                    k,
                    len(self._fewshot_docs),
                    fewshot_source,
                )
            return self._fewshot_docs[: min(k, len(self._fewshot_docs))]

        if self.has_training_docs():
            self._fewshot_docs = list(self.training_docs())
            if len(self._fewshot_docs) < k:
                logger.warning(
                    "ChemBench Gen: capping num_shots from %d to %d for subfield %s",
                    k,
                    len(self._fewshot_docs),
                    self.task_config.get("dataset_name"),
                )
            return rnd.sample(self._fewshot_docs, min(k, len(self._fewshot_docs)))

        self._fewshot_docs = list(
            self.validation_docs() if self.has_validation_docs() else self.test_docs()
        )
        sample_k = min(k + 1, len(self._fewshot_docs))
        if sample_k < k + 1:
            logger.warning(
                "ChemBench Gen: capping num_shots from %d to %d (no train docs)",
                k,
                max(sample_k - 1, 0),
            )
        return rnd.sample(self._fewshot_docs, sample_k)

    def doc_to_text(self, doc):
        return f"Question: {doc['question']}\nAnswer:"

    def doc_to_target(self, doc):
        return " " + doc["target"]

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        return self.construct_basic_generation_requests(
            doc, ctx, doc_id, label=doc["target"]
        )
