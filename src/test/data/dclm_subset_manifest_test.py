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


def test_round_robin_is_exact_and_nested():
    capacities = [5, 3, 7]
    small = MODULE.allocate_round_robin(capacities, 8)
    large = MODULE.allocate_round_robin(capacities, 12)

    assert sum(small) == 8
    assert sum(large) == 12
    assert all(0 <= selected <= capacity for selected, capacity in zip(large, capacities))
    assert all(small_count <= large_count for small_count, large_count in zip(small, large))


def test_round_robin_handles_exhausted_sources():
    assert MODULE.allocate_round_robin([1, 100, 2], 8) == [1, 5, 2]
