import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[3] / "scripts/analysis/paired_document_loss_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("paired_document_loss_bootstrap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_bootstrap_detects_consistent_loss_gap():
    common = {
        "document_starts": np.array([0, 2, 5]),
        "document_ends": np.array([2, 5, 9]),
        "token_counts": np.array([2, 3, 4]),
    }
    a = {**common, "loss_sums": np.array([4.0, 6.0, 8.0])}
    b = {**common, "loss_sums": np.array([2.0, 3.0, 4.0])}

    result = MODULE.paired_bootstrap(a, b, samples=2_000, seed=0)

    assert result["difference_a_minus_b"] == 1.0
    assert result["statistically_significant"] is True
    assert result["documents_included"] == 3
    assert result["scored_tokens"] == 9


def test_paired_bootstrap_requires_identical_examples():
    a = {
        "document_starts": np.array([0]),
        "document_ends": np.array([2]),
        "token_counts": np.array([1]),
        "loss_sums": np.array([1.0]),
    }
    b = {**a, "token_counts": np.array([2])}

    try:
        MODULE.paired_bootstrap(a, b, samples=10, seed=0)
    except ValueError as error:
        assert "token_counts" in str(error)
    else:
        raise AssertionError("expected paired-input validation failure")
