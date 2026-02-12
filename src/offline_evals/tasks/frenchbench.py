"""
FrenchBench: French language evaluation benchmark from CroissantLLM.
https://huggingface.co/collections/manu/frenchbench-evaluation-datasets

Implements MC questions as RC (Ranked Classification) format:
- Scores each answer choice via log-likelihood of the continuation
- Normalized by character count (acc_per_char)
- Model does NOT see all answer options in the prompt

Datasets:
- manu/french_boolq: Boolean yes/no questions (Faux/Vrai)
- manu/french_bench_arc_challenge: Science reasoning questions
- manu/french_bench_hellaswag: Commonsense sentence completion
- manu/french-bench-grammar-vocab-reading: Grammar, vocabulary, reading comprehension
"""

import logging

from oe_eval.tasks.base_task import MultipleChoiceTask
from oe_eval.utils import get_dict_with_defaults

logger = logging.getLogger(__name__)


class GenericFrenchBenchRC(MultipleChoiceTask):
    """
    Generic base class for FrenchBench tasks using RC evaluation.

    RC (Ranked Classification): Scores each answer choice via log-likelihood
    of the continuation, normalized by character count.
    """

    VERSION = 0
    TASK_NAME: str = ""

    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": None,  # Set by subclass
        "dataset_name": None,
        "native_id_field": "index",
        "primary_metric": "acc_per_char",  # Length-normalized log-likelihood
        "split": "test",
        "num_shots": 0,  # 0-shot by default
    }

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def unconditioned_prompt(self):
        # Pure continuation scoring, no unconditioned baseline
        return None


class FrenchBenchBoolQRC(GenericFrenchBenchRC):
    """
    French BoolQ - boolean yes/no questions with passage context.

    Dataset: manu/french_boolq
    Fields: passage, question, label (0=False, 1=True)
    Choices: "Faux" (False), "Vrai" (True)

    Format:
        Passage: [passage]
        Question: [question]
    Continuations: " Faux", " Vrai"
    """

    TASK_NAME = "frenchbench_boolq"
    CHOICES = ["Faux", "Vrai"]

    TASK_CONFIG_DEFAULTS: dict = get_dict_with_defaults(
        {
            "dataset_path": "manu/french_boolq",
            "split": "test",
        },
        GenericFrenchBenchRC.TASK_CONFIG_DEFAULTS,
    )

    def has_training_docs(self):
        return False  # No train split

    def has_validation_docs(self):
        return "valid" in self.dataset

    def has_test_docs(self):
        return "test" in self.dataset

    def validation_docs(self):
        if "valid" in self.dataset:
            return map(self._process_doc, self.dataset["valid"])
        return []

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc, index=-1):
        """Process a French BoolQ document."""
        # Build query with passage and question
        query = f"Passage: {doc['passage']}\nQuestion: {doc['question']}"

        return {
            "index": index,
            "query": query,
            "choices": self.CHOICES,
            "gold": doc["label"],  # 0=Faux, 1=Vrai
        }


class FrenchBenchArcChallengeRC(GenericFrenchBenchRC):
    """
    French ARC Challenge - science reasoning questions.

    Dataset: manu/french_bench_arc_challenge
    Fields: id, question, choices (array of 4), answerKey (A/B/C/D)

    Format:
        [question text]
    Continuations: Each of the 4 answer choices
    """

    TASK_NAME = "frenchbench_arc_challenge"

    TASK_CONFIG_DEFAULTS: dict = get_dict_with_defaults(
        {
            "dataset_path": "manu/french_bench_arc_challenge",
            "native_id_field": "id",
            "split": "test",
            "fewshot_source": "train",
        },
        GenericFrenchBenchRC.TASK_CONFIG_DEFAULTS,
    )

    def has_training_docs(self):
        return "train" in self.dataset

    def has_validation_docs(self):
        return "validation" in self.dataset

    def has_test_docs(self):
        return "test" in self.dataset

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc, index=-1):
        """Process a French ARC Challenge document."""
        # Map answerKey (A/B/C/D) to index
        answer_key = doc["answerKey"]
        gold_index = ["A", "B", "C", "D"].index(answer_key)

        return {
            "id": doc["id"],
            "query": doc["question"],
            "choices": doc["choices"],  # List of 4 answer texts
            "gold": gold_index,
        }


class FrenchBenchHellaSwagRC(GenericFrenchBenchRC):
    """
    French HellaSwag - commonsense sentence completion.

    Dataset: manu/french_bench_hellaswag
    Fields: ind, ctx (context), endings (4 options), label (0-3 as string)

    Format:
        [context text]
    Continuations: Each of the 4 ending sentences
    """

    TASK_NAME = "frenchbench_hellaswag"

    TASK_CONFIG_DEFAULTS: dict = get_dict_with_defaults(
        {
            "dataset_path": "manu/french_bench_hellaswag",
            "native_id_field": "ind",
            "split": "validation",  # Only has validation split
        },
        GenericFrenchBenchRC.TASK_CONFIG_DEFAULTS,
    )

    def has_training_docs(self):
        return False  # No train split

    def has_validation_docs(self):
        return "validation" in self.dataset

    def has_test_docs(self):
        return False  # No test split

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc, index=-1):
        """Process a French HellaSwag document."""
        # Label is string "0"-"3", convert to int
        gold_index = int(doc["label"])

        return {
            "ind": doc["ind"],
            "query": doc["ctx"],  # Context for completion
            "choices": doc["endings"],  # List of 4 ending texts
            "gold": gold_index,
        }


class FrenchBenchGrammarVocabReadingRC(GenericFrenchBenchRC):
    """
    French Grammar, Vocabulary, and Reading comprehension.

    Dataset: manu/french-bench-grammar-vocab-reading
    Fields: question, answerA-D, answer (A-D), difficulty, subject
    Splits: Grammar, Vocabulary, Reading (combined for evaluation)

    Format:
        [question with <...> blank]
    Continuations: Each of answerA, answerB, answerC, answerD
    """

    TASK_NAME = "frenchbench_grammar_vocab_reading"

    TASK_CONFIG_DEFAULTS: dict = get_dict_with_defaults(
        {
            "dataset_path": "manu/french-bench-grammar-vocab-reading",
            "split": "test",  # Use "test" as eval split, we override test_docs()
        },
        GenericFrenchBenchRC.TASK_CONFIG_DEFAULTS,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._test_docs = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True  # We combine all HF splits into test_docs()

    def test_docs(self):
        """Combine all splits (Grammar, Vocabulary, Reading) for evaluation."""
        if self._test_docs is None:
            all_docs = []
            idx = 0
            for split_name in ["Grammar", "Vocabulary", "Reading"]:
                if split_name in self.dataset:
                    for doc in self.dataset[split_name]:
                        all_docs.append(self._process_doc(doc, idx, split_name))
                        idx += 1
            self._test_docs = all_docs
        return self._test_docs

    def _process_doc(self, doc, index=-1, category=""):
        """Process a French Grammar/Vocabulary/Reading document."""
        # Extract choices from answerA, answerB, answerC, answerD fields
        choices = [doc["answerA"], doc["answerB"], doc["answerC"], doc["answerD"]]

        # Map answer letter to index
        gold_index = ["A", "B", "C", "D"].index(doc["answer"])

        return {
            "index": index,
            "query": doc["question"],
            "choices": choices,
            "gold": gold_index,
            "category": category or doc.get("subject", ""),  # Grammar/Vocabulary/Reading
            "difficulty": doc.get("difficulty", ""),
        }


class FrenchBenchBoolQRC_0shot(FrenchBenchBoolQRC):
    pass


class FrenchBenchBoolQRC_5shot(FrenchBenchBoolQRC):
    pass


class FrenchBenchArcChallengeRC_0shot(FrenchBenchArcChallengeRC):
    pass


class FrenchBenchArcChallengeRC_5shot(FrenchBenchArcChallengeRC):
    pass


class FrenchBenchHellaSwagRC_0shot(FrenchBenchHellaSwagRC):
    pass


class FrenchBenchHellaSwagRC_5shot(FrenchBenchHellaSwagRC):
    pass


class FrenchBenchGrammarVocabReadingRC_0shot(FrenchBenchGrammarVocabReadingRC):
    pass


class FrenchBenchGrammarVocabReadingRC_5shot(FrenchBenchGrammarVocabReadingRC):
    pass


FRENCHBENCH_BASE_TASKS = [
    "frenchbench_boolq",
    "frenchbench_arc_challenge",
    "frenchbench_hellaswag",
    "frenchbench_grammar_vocab_reading",
]


def create_frenchbench_tasks() -> dict:
    """
    Create all FrenchBench RC tasks.

    Returns a dict mapping task keys to task classes.
    """
    return {
        "frenchbench_boolq:rc": FrenchBenchBoolQRC,
        "frenchbench_boolq:rc:0shot": FrenchBenchBoolQRC_0shot,
        "frenchbench_boolq:rc:5shot": FrenchBenchBoolQRC_5shot,
        "frenchbench_arc_challenge:rc": FrenchBenchArcChallengeRC,
        "frenchbench_arc_challenge:rc:0shot": FrenchBenchArcChallengeRC_0shot,
        "frenchbench_arc_challenge:rc:5shot": FrenchBenchArcChallengeRC_5shot,
        "frenchbench_hellaswag:rc": FrenchBenchHellaSwagRC,
        "frenchbench_hellaswag:rc:0shot": FrenchBenchHellaSwagRC_0shot,
        "frenchbench_hellaswag:rc:5shot": FrenchBenchHellaSwagRC_5shot,
        "frenchbench_grammar_vocab_reading:rc": FrenchBenchGrammarVocabReadingRC,
        "frenchbench_grammar_vocab_reading:rc:0shot": FrenchBenchGrammarVocabReadingRC_0shot,
        "frenchbench_grammar_vocab_reading:rc:5shot": FrenchBenchGrammarVocabReadingRC_5shot,
    }
