import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "scripts" / "data" / "create_dclm_subset_manifest.py"
SPEC = importlib.util.spec_from_file_location("create_dclm_subset_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_token_count():
    assert MODULE.parse_token_count("1B") == 1_000_000_000
    assert MODULE.parse_token_count("4_194_304") == 4_194_304


def test_proportional_allocation_is_exact():
    allocation = MODULE.proportional_allocation([10, 20, 30], 17)
    assert allocation == [3, 6, 8]
    assert sum(allocation) == 17


def test_choose_documents_respects_budget_and_boundaries():
    candidates = [
        [MODULE.Document(0, 0, 4), MODULE.Document(0, 4, 7)],
        [MODULE.Document(1, 0, 5), MODULE.Document(1, 5, 7)],
    ]
    chosen, selected_tokens = MODULE.choose_documents(candidates, [5, 7], seed=0)
    assert sum(doc.length for doc in chosen) == sum(selected_tokens)
    assert sum(selected_tokens) <= 12
    assert all(doc.start < doc.end for doc in chosen)
