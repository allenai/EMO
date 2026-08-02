import csv
import gzip
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np


DATA_SCRIPT_DIR = Path(__file__).parents[3] / "scripts" / "data"
sys.path.insert(0, str(DATA_SCRIPT_DIR))

from create_dclm_document_split import build  # noqa: E402


def _write_source(root: Path, relative_path: str, lengths: list[int]) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.zeros(sum(lengths), dtype=np.uint32)
    rows = []
    offset = 0
    for length in lengths:
        tokens[offset : offset + length] = np.arange(length, dtype=np.uint32) + 1
        tokens[offset + length - 1] = 100_257
        rows.append((offset, offset + length))
        offset += length
    tokens.tofile(path)
    with gzip.open(path.with_suffix(".csv.gz"), "wt", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _ledger_ids(path: Path) -> set[tuple[str, int]]:
    with gzip.open(path, "rt", newline="") as handle:
        return {
            (row["shard_path"], int(row["document_index"]))
            for row in csv.DictReader(handle)
        }


def test_document_split_is_exhaustive_disjoint_and_reproducible(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    source_mix = tmp_path / "mix.txt"
    relative_paths = ["source/shard0.npy", "source/shard1.npy"]
    for path in relative_paths:
        _write_source(data_root, path, [10] * 250)
    source_mix.write_text("".join(f"dclm,{path}\n" for path in relative_paths))

    output_root = tmp_path / "output"
    manifest_dir = tmp_path / "manifests"
    args = Namespace(
        partition_manifest=manifest_dir / "partition.json",
        manifest_dir=manifest_dir,
        output_root=output_root,
        candidate_dir=tmp_path / "candidates",
        validation_tokens=100,
        test_tokens=100,
        repeated_tokens=500,
        unique_tokens=1_000,
        partition_seed=0,
        repeated_seed=1,
        unique_seed=2,
        candidate_multiplier=3.0,
        uniformity_bucket_bits=8,
        workers=1,
        source_mix=source_mix,
        label="dclm",
        tokenizer="unused",
        dtype="uint32",
        eos_token_id=100_257,
        training_alignment_tokens=4_096,
        data_root=data_root,
        manifest_base_dir=data_root,
    )
    manifest = build(args)

    partition = manifest["partition"]
    assert partition["document_count_sum"] == 500
    assert partition["token_count_sum"] == 5_000
    assert sum(partition[name]["documents"] for name in ("train", "validation", "test")) == 500
    assert sum(partition[name]["real_tokens"] for name in ("train", "validation", "test")) == 5_000
    assert partition["pairwise_document_intersections"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }

    validation_ids = _ledger_ids(output_root / "dclm_0802_validation.documents.csv.gz")
    test_ids = _ledger_ids(output_root / "dclm_0802_test.documents.csv.gz")
    repeated_ids = _ledger_ids(output_root / "dclm_0802_repeated_train_1b.documents.csv.gz")
    unique_ids = _ledger_ids(output_root / "dclm_0802_unique_train_5b.documents.csv.gz")
    assert validation_ids.isdisjoint(test_ids)
    assert repeated_ids.isdisjoint(validation_ids | test_ids)
    assert unique_ids.isdisjoint(validation_ids | test_ids)

    for name in ("validation", "test", "repeated_train_1b", "unique_train_5b"):
        subset_manifest = json.loads((manifest_dir / f"dclm_0802_{name}.json").read_text())
        assert subset_manifest["selection"]["padding_eos_tokens"] >= 0
        assert subset_manifest["selection"]["selected_tokens"] % 4096 == 0
        assert subset_manifest["source_document_ledger"]["documents"] > 0

    second_root = tmp_path / "second"
    second_args = Namespace(
        **{
            **vars(args),
            "partition_manifest": second_root / "manifests" / "partition.json",
            "manifest_dir": second_root / "manifests",
            "output_root": second_root / "output",
            "candidate_dir": second_root / "candidates",
        }
    )
    second = build(second_args)
    for name in ("validation", "test", "repeated_train_1b", "unique_train_5b"):
        assert (
            manifest["artifacts"][name]["selection"]["boundary_key"]
            == second["artifacts"][name]["selection"]["boundary_key"]
        )
        assert (
            manifest["artifacts"][name]["materialized"]["token_sha256"]
            == second["artifacts"][name]["materialized"]["token_sha256"]
        )
