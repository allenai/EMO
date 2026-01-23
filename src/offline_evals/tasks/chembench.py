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

We chose strict string matching for simplicity and consistency with our other evals.
This means numeric answers must match exactly (e.g., "6" != "6.0").
"""

import json
import logging
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
        "num_shots": 5,  # Five-shot by default for ChemBench choice tasks
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
        "num_shots": 5,
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
        self._metrics = [SQuADF1EMRecallMetric(**self.task_config["metric_kwargs"])]
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
            "name": doc.get("name", ""),
            "subfield": doc.get("subfield", ""),
        }
        return out_doc

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
