#!/usr/bin/env python3
"""Create an explicit, seed-free subset manifest for tokenized DCLM data.

The manifest records the exact prefix of 4096-token instances selected from
each source array.  Selection is balanced round-robin across the ordered,
deduplicated DCLM paths, so manifests made with larger targets contain every
instance selected by manifests made with smaller targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SOURCE_MIX = Path("src/olmo_core/data/mixes/OLMoE-mix-0824.txt")


def parse_token_count(value: str) -> int:
    value = value.strip().lower().replace("_", "").replace(",", "")
    multipliers = {"k": 10**3, "m": 10**6, "b": 10**9, "t": 10**12}
    multiplier = multipliers.get(value[-1], 1)
    number = value[:-1] if multiplier != 1 else value
    parsed = float(number) * multiplier
    if not parsed.is_integer() or parsed <= 0:
        raise argparse.ArgumentTypeError(f"invalid positive token count: {value!r}")
    return int(parsed)


def resolve_source_paths(
    source_mix: Path, *, label: str, tokenizer: str
) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    with source_mix.open() as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row_label, path = line.split(",", 1)
            except ValueError as exc:
                raise ValueError(f"invalid row at {source_mix}:{line_number}") from exc
            if row_label != label:
                continue
            path = path.replace("{TOKENIZER}", tokenizer)
            if path not in seen:
                seen.add(path)
                paths.append(path)
    if not paths:
        raise ValueError(f"no paths with label {label!r} in {source_mix}")
    return paths


def inspect_capacities(
    relative_paths: Iterable[str],
    *,
    data_root: Path,
    sequence_length: int,
    dtype: np.dtype,
) -> list[dict[str, int | str]]:
    sources: list[dict[str, int | str]] = []
    for relative_path in relative_paths:
        absolute_path = data_root / relative_path.lstrip("/")
        file_size = absolute_path.stat().st_size
        if file_size % dtype.itemsize:
            raise ValueError(
                f"file size {file_size:,} at {absolute_path} is not divisible by "
                f"the {dtype.name} item size ({dtype.itemsize})"
            )
        available_tokens = file_size // dtype.itemsize
        sources.append(
            {
                "path": relative_path,
                "available_tokens": available_tokens,
                "available_instances": available_tokens // sequence_length,
            }
        )
    return sources


def allocate_round_robin(capacities: list[int], target_instances: int) -> list[int]:
    """Allocate prefixes as if visiting every non-exhausted source in order."""
    if target_instances > sum(capacities):
        raise ValueError(
            f"requested {target_instances:,} instances but only {sum(capacities):,} are available"
        )

    low, high = 0, max(capacities, default=0)
    while low < high:
        level = (low + high + 1) // 2
        if sum(min(capacity, level) for capacity in capacities) <= target_instances:
            low = level
        else:
            high = level - 1

    selected = [min(capacity, low) for capacity in capacities]
    remaining = target_instances - sum(selected)
    for idx, capacity in enumerate(capacities):
        if remaining == 0:
            break
        if selected[idx] < capacity:
            selected[idx] += 1
            remaining -= 1
    if remaining:
        raise AssertionError("allocation exhausted sources before reaching target")
    return selected


def manifest_digest(entries: list[dict[str, int | str]]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_manifest(args: argparse.Namespace) -> dict:
    if args.alignment_tokens % args.sequence_length:
        raise ValueError("alignment tokens must be divisible by sequence length")

    relative_paths = resolve_source_paths(
        args.source_mix, label=args.label, tokenizer=args.tokenizer
    )
    dtype = np.dtype(args.dtype)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"token dtype must be an integer type, got {dtype.name}")
    sources = inspect_capacities(
        relative_paths,
        data_root=args.data_root,
        sequence_length=args.sequence_length,
        dtype=dtype,
    )
    aligned_tokens = math.ceil(args.target_tokens / args.alignment_tokens) * args.alignment_tokens
    target_instances = aligned_tokens // args.sequence_length
    selected_counts = allocate_round_robin(
        [int(source["available_instances"]) for source in sources], target_instances
    )

    entries = []
    for source, selected_instances in zip(sources, selected_counts):
        if not selected_instances:
            continue
        entries.append(
            {
                "path": source["path"],
                "start_instance": 0,
                "num_instances": selected_instances,
                "available_instances": source["available_instances"],
                "available_tokens": source["available_tokens"],
            }
        )

    source_mix_sha256 = hashlib.sha256(args.source_mix.read_bytes()).hexdigest()
    return {
        "format": "olmo-token-subset-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "mix": "dclm-full",
            "source_mix_file": args.source_mix.name,
            "source_mix_sha256": source_mix_sha256,
            "label": args.label,
            "tokenizer": args.tokenizer,
            "dtype": dtype.name,
            "data_root_hint": str(args.data_root),
        },
        "selection": {
            "method": "ordered-round-robin-prefix",
            "seed": None,
            "requested_tokens": args.target_tokens,
            "alignment_tokens": args.alignment_tokens,
            "sequence_length": args.sequence_length,
            "selected_tokens": aligned_tokens,
            "selected_instances": target_instances,
            "num_source_paths": len(entries),
        },
        "entries_sha256": manifest_digest(entries),
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tokens", required=True, type=parse_token_count)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/weka/oe-training-default/ai2-llm")
    )
    parser.add_argument("--source-mix", type=Path, default=DEFAULT_SOURCE_MIX)
    parser.add_argument("--label", default="dclm")
    parser.add_argument("--tokenizer", default="allenai/dolma2-tokenizer")
    parser.add_argument(
        "--dtype",
        default="uint32",
        help="Raw token dtype (DCLM-full with the Dolma2 tokenizer uses uint32).",
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--alignment-tokens", type=parse_token_count, default=4_194_304)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    selection = manifest["selection"]
    print(
        f"Wrote {args.output}: {selection['selected_tokens']:,} tokens "
        f"({selection['selected_instances']:,} instances) from "
        f"{selection['num_source_paths']:,} paths; entries sha256 "
        f"{manifest['entries_sha256']}"
    )


if __name__ == "__main__":
    main()
