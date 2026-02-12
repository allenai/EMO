#!/usr/bin/env python3
"""
Create a weighted mix from multiple data sources.

Given multiple mix files and their desired percentages, creates a new mix
that samples from each source to achieve the target ratio and total token count.

Usage:
    python create_weighted_mix.py \
        --sources pile-of-law.txt:80 dclm.txt:20 \
        --total-tokens 10B \
        --output my-mix.txt \
        --dry-run

    # Or with explicit paths:
    python create_weighted_mix.py \
        --sources src/olmo_core/data/mixes/the-pile-of-law.txt:80 \
                  src/olmo_core/data/mixes/dclm-only.txt:20 \
        --total-tokens 50B \
        --output src/olmo_core/data/mixes/law-80-dclm-20-50B.txt
"""

import argparse
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceConfig:
    """Configuration for a data source."""

    name: str
    mix_path: str
    percentage: float
    entries: list  # [(label, path), ...]
    total_bytes: int = 0
    file_sizes: dict = None  # {path: size}


def parse_token_count(s: str) -> int:
    """Parse token count string like '10B', '500M', '1T'."""
    s = s.strip().upper()
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)

    return int(float(s))


def s3_list_files_with_sizes(s3_prefix: str) -> dict[str, int]:
    """List all files under S3 prefix with their sizes."""
    result = subprocess.run(
        ["aws", "s3", "ls", "--recursive", s3_prefix],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Failed to list S3 path: {s3_prefix}", file=sys.stderr)
        return {}

    files = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            size = int(parts[2])
            path = " ".join(parts[3:])
            files[path] = size

    return files


def local_list_files_with_sizes(local_prefix: str) -> dict[str, int]:
    """List all files under local prefix with their sizes."""
    import os

    files = {}
    local_path = Path(local_prefix)

    if not local_path.exists():
        print(f"ERROR: Local path does not exist: {local_prefix}", file=sys.stderr)
        return {}

    for root, dirs, filenames in os.walk(local_path):
        for filename in filenames:
            full_path = Path(root) / filename
            rel_path = str(
                full_path.relative_to(local_path.parent.parent.parent.parent)
            )  # relative to base
            size = full_path.stat().st_size
            files[rel_path] = size

    return files


def parse_mix_file(mix_path: str) -> list[tuple[str, str]]:
    """Parse mix file and return [(label, path), ...]."""
    entries = []
    with open(mix_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "," in stripped:
                label, path = stripped.split(",", 1)
                entries.append((label, path))
    return entries


def get_file_sizes_for_entries(
    entries: list[tuple[str, str]], s3_base: str, local_base: str, tokenizer: str
) -> dict[str, int]:
    """
    Get file sizes for all entries in a mix.
    Tries local path first, falls back to S3.
    Returns {path: size} dict.
    """
    if not entries:
        return {}

    # Resolve paths (replace {TOKENIZER} placeholder)
    resolved_paths = []
    path_mapping = {}  # resolved -> original
    for label, path in entries:
        resolved = path.replace("{TOKENIZER}", tokenizer)
        resolved_paths.append(resolved)
        path_mapping[resolved] = path

    # Find common prefix
    common_prefix = resolved_paths[0]
    for p in resolved_paths[1:]:
        while common_prefix and not p.startswith(common_prefix):
            common_prefix = "/".join(common_prefix.split("/")[:-1])
            if common_prefix:
                common_prefix += "/"

    if not common_prefix:
        common_prefix = ""

    # Try local path first
    local_path = Path(local_base) / common_prefix if common_prefix else Path(local_base)
    if local_path.exists():
        print(f"  Using local path: {local_path}", file=sys.stderr)
        sizes = {}
        for resolved, original in path_mapping.items():
            full_local = Path(local_base) / resolved
            if full_local.exists():
                sizes[original] = full_local.stat().st_size
        print(f"  Found {len(sizes)} files locally", file=sys.stderr)
        return sizes

    # Fall back to S3
    if not common_prefix:
        print(f"  No common prefix, listing files individually...", file=sys.stderr)
        sizes = {}
        for resolved in resolved_paths:
            result = subprocess.run(
                ["aws", "s3", "ls", f"{s3_base}/{resolved}"], capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split()
                if len(parts) >= 3:
                    sizes[path_mapping[resolved]] = int(parts[2])
        return sizes

    print(f"  Listing S3: {s3_base}/{common_prefix[:50]}...", file=sys.stderr)
    size_cache = s3_list_files_with_sizes(f"{s3_base}/{common_prefix}")
    print(f"  Found {len(size_cache)} files in S3", file=sys.stderr)

    # Map back to original paths
    sizes = {}
    for resolved, original in path_mapping.items():
        if resolved in size_cache:
            sizes[original] = size_cache[resolved]

    return sizes


def select_files_for_target(
    entries: list[tuple[str, str]],
    file_sizes: dict[str, int],
    target_bytes: int,
    seed: int = 42,
    fine_tune: bool = False,
) -> list[tuple[str, str, int]]:
    """
    Select files from entries to reach target_bytes.

    Args:
        entries: List of (label, path) tuples
        file_sizes: Dict mapping path -> size in bytes
        target_bytes: Target total bytes to select
        seed: Random seed for shuffling
        fine_tune: If True, use smaller files and try to hit target precisely
                   (used for secondary sources to balance the ratio)

    Returns [(label, path, size), ...].
    """
    # Build list of (label, path, size) with known sizes
    available = []
    for label, path in entries:
        if path in file_sizes:
            available.append((label, path, file_sizes[path]))

    if not available:
        return []

    # Shuffle for random sampling
    random.seed(seed)
    random.shuffle(available)

    if fine_tune:
        # Sort by size (smallest first) for fine-grained control
        available.sort(key=lambda x: x[2])

    # Select until target reached
    selected = []
    accumulated = 0
    for label, path, size in available:
        if accumulated >= target_bytes:
            break
        # In fine_tune mode, skip files that would overshoot by more than 10%
        if fine_tune and accumulated > 0 and (accumulated + size) > target_bytes * 1.01:
            continue
        selected.append((label, path, size))
        accumulated += size

    # Sort by path for consistent output
    selected.sort(key=lambda x: x[1])
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Create a weighted mix from multiple data sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source mix files with percentages, format: path:percentage (e.g., law.txt:80 dclm.txt:20)",
    )
    parser.add_argument(
        "--total-tokens", required=True, help="Target total tokens (e.g., 10B, 500M, 1T)"
    )
    parser.add_argument("--output", required=True, help="Output mix file path")
    parser.add_argument(
        "--s3-base", default="s3://ai2-llm", help="S3 base path (default: s3://ai2-llm)"
    )
    parser.add_argument(
        "--local-base",
        default="/data/input/ai2-llm",
        help="Local mount path for data (default: /data/input/ai2-llm). Used if files exist locally.",
    )
    parser.add_argument(
        "--tokenizer",
        default="allenai/dolma2-tokenizer",
        help="Tokenizer for {TOKENIZER} placeholder (default: allenai/dolma2-tokenizer)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Calculate and print stats without writing output"
    )

    args = parser.parse_args()

    # Parse total tokens
    total_tokens = parse_token_count(args.total_tokens)
    total_bytes = total_tokens * 4  # 4 bytes per token (uint32)
    print(
        f"Target: {total_tokens / 1e9:.2f}B tokens ({total_bytes / 1e12:.4f} TB)", file=sys.stderr
    )

    # Parse sources
    sources: list[SourceConfig] = []
    total_pct = 0

    for source_spec in args.sources:
        if ":" not in source_spec:
            print(
                f"ERROR: Invalid source format '{source_spec}', expected 'path:percentage'",
                file=sys.stderr,
            )
            sys.exit(1)

        path, pct_str = source_spec.rsplit(":", 1)
        pct = float(pct_str)
        total_pct += pct

        name = Path(path).stem
        entries = parse_mix_file(path)

        sources.append(SourceConfig(name=name, mix_path=path, percentage=pct, entries=entries))
        print(f"\nSource '{name}': {len(entries)} entries, {pct}%", file=sys.stderr)

    # Normalize percentages if they don't sum to 100
    if abs(total_pct - 100) > 0.01:
        print(f"\nNote: Percentages sum to {total_pct}%, normalizing to 100%", file=sys.stderr)
        for src in sources:
            src.percentage = (src.percentage / total_pct) * 100

    # Get file sizes for each source
    print(f"\n=== Calculating file sizes ===", file=sys.stderr)
    for src in sources:
        print(f"\nSource '{src.name}':", file=sys.stderr)
        src.file_sizes = get_file_sizes_for_entries(
            src.entries, args.s3_base, args.local_base, args.tokenizer
        )
        src.total_bytes = sum(src.file_sizes.values())
        src_tokens = src.total_bytes / 4
        print(
            f"  Available: {src.total_bytes / 1e12:.4f} TB = {src_tokens / 1e9:.2f}B tokens",
            file=sys.stderr,
        )

    # Sort sources by average file size (largest first) for better ratio control
    # Sources with larger files are the "anchor" - we select from them first
    # Then sources with smaller files can fine-tune to hit the exact ratio
    print(f"\n=== Selecting files ===", file=sys.stderr)

    def avg_file_size(src):
        if not src.file_sizes:
            return 0
        return src.total_bytes / len(src.file_sizes)

    sources_sorted = sorted(sources, key=avg_file_size, reverse=True)
    print(
        f"(Processing by avg file size: {' -> '.join(f'{s.name} ({avg_file_size(s)/1e9:.2f}B/file)' for s in sources_sorted)})",
        file=sys.stderr,
    )

    selected_by_source: list[tuple[SourceConfig, list]] = []
    anchor_selected_bytes = 0  # Track what the anchor source actually selected
    anchor_pct = 0  # The percentage of the anchor source

    for i, src in enumerate(sources_sorted):
        is_anchor = i == 0  # First source (largest avg file size) is anchor

        if is_anchor:
            # Anchor source: use original target based on percentage
            target_bytes_for_src = int(total_bytes * (src.percentage / 100))
            anchor_pct = src.percentage
        else:
            # Secondary sources: adjust target based on what anchor actually selected
            # to maintain the desired ratio
            # If anchor was X%, it selected anchor_selected_bytes
            # This source is Y%, so it needs: anchor_selected_bytes * (Y / anchor_pct)
            target_bytes_for_src = int(anchor_selected_bytes * (src.percentage / anchor_pct))

        target_tokens_for_src = target_bytes_for_src / 4
        print(f"\nSource '{src.name}' ({src.percentage:.1f}%):", file=sys.stderr)
        print(
            f"  Target: {target_bytes_for_src / 1e12:.4f} TB = {target_tokens_for_src / 1e9:.2f}B tokens",
            file=sys.stderr,
        )

        # Check if we have enough data
        if target_bytes_for_src > src.total_bytes:
            print(
                f"  WARNING: Not enough data! Have {src.total_bytes / 1e12:.4f} TB, need {target_bytes_for_src / 1e12:.4f} TB",
                file=sys.stderr,
            )
            print(f"  Will use all available data.", file=sys.stderr)
            target_bytes_for_src = src.total_bytes

        # Anchor source: normal selection. Secondary: fine-tune mode (use smaller files)
        selected = select_files_for_target(
            src.entries, src.file_sizes, target_bytes_for_src, args.seed, fine_tune=(not is_anchor)
        )
        selected_bytes = sum(s for _, _, s in selected)
        selected_tokens = selected_bytes / 4
        print(
            f"  Selected: {len(selected)} files, {selected_bytes / 1e12:.4f} TB = {selected_tokens / 1e9:.2f}B tokens",
            file=sys.stderr,
        )

        if is_anchor:
            anchor_selected_bytes = selected_bytes

        selected_by_source.append((src, selected))

    # Re-sort by original order for output consistency
    selected_by_source.sort(key=lambda x: sources.index(x[0]))

    # Summary
    print(f"\n=== Summary ===", file=sys.stderr)
    grand_total_bytes = 0
    grand_total_files = 0

    for src, selected in selected_by_source:
        selected_bytes = sum(s for _, _, s in selected)
        grand_total_bytes += selected_bytes
        grand_total_files += len(selected)

    print(
        f"Total: {grand_total_bytes / 1e12:.4f} TB = {(grand_total_bytes / 4) / 1e9:.2f}B tokens, {grand_total_files} files",
        file=sys.stderr,
    )
    print(f"\nActual ratios:", file=sys.stderr)
    for src, selected in selected_by_source:
        selected_bytes = sum(s for _, _, s in selected)
        actual_pct = (selected_bytes / grand_total_bytes * 100) if grand_total_bytes > 0 else 0
        print(f"  {src.name}: {actual_pct:.1f}% (target: {src.percentage:.1f}%)", file=sys.stderr)

    if args.dry_run:
        print(f"\n[Dry run - not writing output]", file=sys.stderr)
        return

    # Write output
    print(f"\nWriting: {args.output}", file=sys.stderr)
    with open(args.output, "w") as f:
        f.write("# Weighted mix created by create_weighted_mix.py\n")
        f.write(f"# Total: {(grand_total_bytes / 4) / 1e9:.2f}B tokens\n")
        f.write(f"# Sources:\n")
        for src, selected in selected_by_source:
            selected_bytes = sum(s for _, _, s in selected)
            actual_pct = (selected_bytes / grand_total_bytes * 100) if grand_total_bytes > 0 else 0
            f.write(f"#   {src.name}: {actual_pct:.1f}%\n")
        f.write("\n")

        for src, selected in selected_by_source:
            f.write(f"# {src.name} data\n")
            for label, path, size in selected:
                f.write(f"{label},{path}\n")
            f.write("\n")

    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
