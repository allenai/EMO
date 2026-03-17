"""
CodeFresh corpus perplexity and bits-per-byte evaluation (rolling version).

Adapted from allenai/oe-eval-internal PR #719 (soldni).
"""

from typing import List, Union

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import RequestType
from oe_eval.metrics import PerplexityMetric
from oe_eval.tasks.base_task import Task
from oe_eval.tasks.utils import map_indexed

MAX_FILE_TOKENS = 4096

CODE_FRESH_LANGUAGES = [
    "blade",
    "c",
    "clojure",
    "common_lisp",
    "cpp",
    "csharp",
    "css",
    "dart",
    "erlang",
    "fortran",
    "go",
    "haskell",
    "html",
    "java",
    "java_server_page",
    "javascript",
    "julia",
    "kotlin",
    "lua",
    "markdown",
    "mathematica",
    "matlab",
    "objective_c",
    "objective_cpp",
    "ocaml",
    "perl",
    "php",
    "powershell",
    "python",
    "restructuredtext",
    "ruby",
    "rust",
    "scala",
    "scheme",
    "swift",
    "systemverilog",
    "tcl",
    "tex",
    "typescript",
    "verilog",
    "vhdl",
    "vue",
]


def create_core_code_fresh_rolling_tasks() -> dict:
    def create_code_fresh_rolling_task(language: str):
        class CodeFreshRolling(CodeFreshRollingBase):
            TASK_CONFIG_DEFAULTS = {
                "dataset_name": language,
                **CodeFreshRollingBase.TASK_CONFIG_DEFAULTS,
            }

        return CodeFreshRolling

    return {
        f"code_fresh_rolling:{language}": create_code_fresh_rolling_task(language)
        for language in CODE_FRESH_LANGUAGES
    }


class CodeFreshBase(Task):
    VERSION = 0
    REQUEST_TYPE = RequestType.LOGLIKELIHOOD_ROLLING
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": "allenai/code_fresh_0825_1225",
        "split": "train",
        "primary_metric": "bits_per_byte_corr",
        "metric_kwargs": {
            "verbose_token_logits": False,
        },
    }

    def make_metrics(self):
        self._metrics = [PerplexityMetric(**self.task_config["metric_kwargs"])]
        return self._metrics

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def training_docs(self):
        filtered_docs = (
            doc for doc in self.dataset["train"] if doc.get("file_tokens", 0) <= MAX_FILE_TOKENS
        )
        return map_indexed(self._process_doc, filtered_docs)

    def _process_doc(self, doc, index=-1):
        text = doc["file_contents"].strip()
        return {
            "id": index,
            "text": text,
            "file_tokens": doc.get("file_tokens"),
            "num_chars": len(text),
            "num_words": len(text.split()),
        }

    def doc_to_text(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        raise NotImplementedError()


class CodeFreshRollingBase(CodeFreshBase):
    VERSION = 0
    REQUEST_TYPE = RequestType.LOGLIKELIHOOD_ROLLING

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        if not isinstance(ctx, str):
            raise ValueError(f"Context should be a string, but got {type(ctx)}: {ctx}")
        if ctx != self.doc_to_text(doc):
            raise ValueError(
                "Context should be the same as the document text. "
                "(Few-shot and chat format are not supported)"
            )
        return self.construct_basic_likelihood_rolling_requests(doc, ctx, doc_id)
