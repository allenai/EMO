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
from create_dclm_nested_pools import build as build_nested  # noqa: E402


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


def _ids(path: Path) -> set[tuple[str, int]]:
    with gzip.open(path, "rt", newline="") as handle:
        return {
            (row["shard_path"], int(row["document_index"]))
            for row in csv.DictReader(handle)
        }


def test_nested_3b_and_9b_pools_preserve_base_and_disjoint_extensions(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    source_mix = tmp_path / "mix.txt"
    relative_paths = ["source/shard0.npy", "source/shard1.npy"]
    for path in relative_paths:
        _write_source(data_root, path, [10] * 1_000)
    source_mix.write_text("".join(f"dclm,{path}\n" for path in relative_paths))

    base_output = data_root / "base-output"
    base_manifests = tmp_path / "base-manifests"
    build_split(
        Namespace(
            partition_manifest=base_manifests / "partition.json",
            manifest_dir=base_manifests,
            output_root=base_output,
            candidate_dir=tmp_path / "base-candidates",
            validation_tokens=100,
            test_tokens=100,
            repeated_tokens=1_000,
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
            training_alignment_tokens=100,
            data_root=data_root,
            manifest_base_dir=data_root,
        )
    )
    nested_root = data_root / "nested"
    manifest_root = nested_root / "manifests"
    pool = build_nested(
        Namespace(
            base_manifest=base_manifests / "dclm_0802_repeated_train_1b.json",
            partition_manifest=base_manifests / "partition.json",
            pool_manifest=manifest_root / "pool.json",
            first_chunk_manifest=manifest_root / "chunk-2b.json",
            second_chunk_manifest=manifest_root / "chunk-6b.json",
            pool_3b_manifest=manifest_root / "pool-3b.json",
            pool_9b_manifest=manifest_root / "pool-9b.json",
            first_chunk_output=nested_root / "chunk-2b.npy",
            second_chunk_output=nested_root / "chunk-6b.npy",
            candidate_dir=nested_root / "candidates",
            base_tokens=1_000,
            first_extension_tokens=2_000,
            second_extension_tokens=6_000,
            candidate_multiplier=3.0,
            workers=1,
            source_mix=source_mix,
            label="dclm",
            tokenizer="unused",
            dtype="uint32",
            eos_token_id=100_257,
            alignment_tokens=100,
            data_root=data_root,
            manifest_base_dir=data_root,
        )
    )
    assert pool["format"] == "dclm-nested-pool-v1"
    assert pool["audit"]["passed"] is True

    base = json.loads((base_manifests / "dclm_0802_repeated_train_1b.json").read_text())
    chunk_2b = json.loads((manifest_root / "chunk-2b.json").read_text())
    chunk_6b = json.loads((manifest_root / "chunk-6b.json").read_text())
    pool_3b = json.loads((manifest_root / "pool-3b.json").read_text())
    pool_9b = json.loads((manifest_root / "pool-9b.json").read_text())
    assert pool_3b["selection"]["requested_tokens"] == 3_000
    assert pool_9b["selection"]["requested_tokens"] == 9_000
    assert pool_3b["entries"] == base["entries"] + chunk_2b["entries"]
    assert pool_9b["entries"] == base["entries"] + chunk_2b["entries"] + chunk_6b["entries"]

    base_ids = _ids(data_root / base["source_document_ledger"]["path"])
    chunk_2b_ids = _ids(nested_root / "chunk-2b.documents.csv.gz")
    chunk_6b_ids = _ids(nested_root / "chunk-6b.documents.csv.gz")
    assert base_ids.isdisjoint(chunk_2b_ids | chunk_6b_ids)
    assert chunk_2b_ids.isdisjoint(chunk_6b_ids)
