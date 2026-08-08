import csv
import gzip
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np


DATA_SCRIPT_DIR = Path(__file__).parents[3] / "scripts" / "data"
sys.path.insert(0, str(DATA_SCRIPT_DIR))

from create_dclm_document_split import build as build_split  # noqa: E402
from create_dclm_unique_extension_pool import build as build_extension  # noqa: E402


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


def test_extension_pool_is_disjoint_and_appendable(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    source_mix = tmp_path / "mix.txt"
    relative_paths = ["source/shard0.npy", "source/shard1.npy"]
    for path in relative_paths:
        _write_source(data_root, path, [10] * 500)
    source_mix.write_text("".join(f"dclm,{path}\n" for path in relative_paths))

    base_output = tmp_path / "base-output"
    base_manifests = tmp_path / "base-manifests"
    split_args = Namespace(
        partition_manifest=base_manifests / "partition.json",
        manifest_dir=base_manifests,
        output_root=base_output,
        candidate_dir=tmp_path / "base-candidates",
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
    build_split(split_args)
    original_manifest = base_manifests / "dclm_0802_unique_train_5b.json"
    pool_manifest = tmp_path / "extension" / "pool.json"

    def extension_args(index: int, *, append: bool) -> Namespace:
        return Namespace(
            original_unique_manifest=original_manifest,
            partition_manifest=base_manifests / "partition.json",
            pool_manifest=pool_manifest,
            chunk_manifest=tmp_path / "extension" / f"chunk-{index}.json",
            materialized_output=tmp_path / "extension" / f"chunk-{index}.npy",
            candidate_dir=tmp_path / "extension" / f"candidates-{index}",
            chunk_name=f"extension_chunk_{index}",
            logical_pool_tokens=3_000,
            chunk_tokens=1_000,
            append=append,
            candidate_multiplier=3.0,
            workers=1,
            source_mix=source_mix,
            label="dclm",
            tokenizer="unused",
            dtype="uint32",
            eos_token_id=100_257,
            alignment_tokens=4_096,
            data_root=data_root,
            manifest_base_dir=data_root,
        )

    first = build_extension(extension_args(0, append=False))
    second = build_extension(extension_args(1, append=True))
    assert first["audit"]["passed"] is True
    assert second["logical_pool"]["requested_tokens"] == 3_000
    assert len(second["chunks"]) == 2
    assert second["chunks"][0]["end_boundary_inclusive"] == second["chunks"][1][
        "start_boundary_exclusive"
    ]

    original = json.loads(original_manifest.read_text())
    original_ledger = data_root / original["source_document_ledger"]["path"]
    first_ids = _ledger_ids(tmp_path / "extension" / "chunk-0.documents.csv.gz")
    second_ids = _ledger_ids(tmp_path / "extension" / "chunk-1.documents.csv.gz")
    original_ids = _ledger_ids(original_ledger)
    assert original_ids.isdisjoint(first_ids | second_ids)
    assert first_ids.isdisjoint(second_ids)
    for index in (0, 1):
        chunk = json.loads((tmp_path / "extension" / f"chunk-{index}.json").read_text())
        assert chunk["disjointness_audit"]["passed"] is True
        assert chunk["selection"]["method"] == "global-sha256-document-order-range"
