#!/usr/bin/env python3
"""
Build token-matched data mix files by replacing one subset with another.

Example usage:
    python build_mix.py \
        --base-mix src/olmo_core/data/mixes/OLMoE-mix-0824.txt \
        --replace-subset dclm \
        --replacement-prefix "preprocessed/dolma2-0625/v0.1/{TOKENIZER}/all-dressed-snazzy2/" \
        --replacement-label snazzy2 \
        --s3-base s3://ai2-llm \
        --tokenizer allenai/dolma2-tokenizer \
        --output src/olmo_core/data/mixes/OLMoE-mix-snazzy2.txt
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def s3_list_files_with_sizes(s3_prefix: str) -> dict[str, int]:
    """
    List all files under S3 prefix with their sizes via `aws s3 ls --recursive`.
    Returns dict mapping path -> size.
    Much faster than individual s3 ls calls.
    """
    result = subprocess.run(
        ["aws", "s3", "ls", "--recursive", s3_prefix],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Failed to list S3 path: {s3_prefix}", file=sys.stderr)
        print(f"stderr: {result.stderr}", file=sys.stderr)
        return {}

    files = {}
    # Parse output lines: "2024-01-01 12:00:00 SIZE path/to/file.npy"
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            size = int(parts[2])
            # The path is everything after the size
            path = " ".join(parts[3:])
            files[path] = size

    return files


def s3_list_files(s3_prefix: str) -> List[Tuple[str, int]]:
    """
    List files matching S3 prefix via `aws s3 ls --recursive`.
    Returns list of (relative_path, size) tuples.
    """
    files_dict = s3_list_files_with_sizes(s3_prefix)
    return list(files_dict.items())


def parse_mix_file(mix_path: str) -> List[str]:
    """Read mix file and return all lines."""
    with open(mix_path) as f:
        return f.readlines()


def split_mix_lines(
    lines: List[str], replace_subset: str
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Split mix lines into keep lines and replace lines.
    Returns (keep_lines, replace_entries) where replace_entries is [(label, path), ...].
    """
    keep_lines = []
    replace_entries = []

    for line in lines:
        stripped = line.strip()
        # Keep comments and empty lines
        if not stripped or stripped.startswith("#"):
            keep_lines.append(line)
            continue

        # Parse "label,path" format
        if "," in stripped:
            label, path = stripped.split(",", 1)
            if label.startswith(replace_subset):
                replace_entries.append((label, path))
            else:
                keep_lines.append(line)
        else:
            keep_lines.append(line)

    return keep_lines, replace_entries


def calculate_target_bytes(
    replace_entries: List[Tuple[str, str]], s3_base: str, tokenizer: str
) -> int:
    """
    Calculate total bytes for entries to be replaced.
    Uses batch S3 listing for speed - finds common prefix and lists once.
    """
    if not replace_entries:
        return 0

    # Resolve all paths first
    resolved_paths = []
    for label, path in replace_entries:
        resolved_path = path.replace("{TOKENIZER}", tokenizer)
        resolved_paths.append(resolved_path)

    # Find common prefix to minimize S3 calls
    # e.g., "preprocessed/dclm/text_openhermes..." -> "preprocessed/dclm/"
    first_path = resolved_paths[0]
    common_prefix = first_path
    for path in resolved_paths[1:]:
        while common_prefix and not path.startswith(common_prefix):
            # Remove last path component
            common_prefix = "/".join(common_prefix.split("/")[:-1])
            if common_prefix:
                common_prefix += "/"

    if not common_prefix:
        common_prefix = ""

    print(f"Listing S3 prefix: {s3_base}/{common_prefix}", file=sys.stderr)
    print("(this may take a moment for large directories...)", file=sys.stderr)

    # Batch list all files under common prefix
    size_cache = s3_list_files_with_sizes(f"{s3_base}/{common_prefix}")
    print(f"Found {len(size_cache)} files in S3 listing", file=sys.stderr)

    # Calculate total from cache
    # Cache keys from `aws s3 ls --recursive s3://bucket/prefix` are paths relative to bucket
    # resolved_paths are also relative to bucket (no s3:// prefix)
    total = 0
    missing = 0

    for resolved_path in resolved_paths:
        if resolved_path in size_cache:
            total += size_cache[resolved_path]
        else:
            # Path not in cache - might be missing or path format mismatch
            missing += 1

    if missing > 0:
        print(
            f"Warning: {missing}/{len(resolved_paths)} files not found in S3 listing",
            file=sys.stderr,
        )

    return total


def select_replacement_files(
    s3_prefix: str, target_bytes: int, tokenizer: str, shuffle: bool = True, seed: int = 42
) -> List[Tuple[str, int]]:
    """
    List files from S3 prefix and select until target_bytes is reached.

    Args:
        s3_prefix: S3 path prefix to list files from
        target_bytes: Target total bytes to select
        tokenizer: Tokenizer identifier for {TOKENIZER} placeholder
        shuffle: If True, randomly shuffle files before selecting (default: True)
                 This ensures balanced sampling across subcategories.
                 If False, selects in alphabetical order.
        seed: Random seed for reproducibility when shuffling

    Returns list of (relative_path, size) tuples.
    """
    # Replace tokenizer in prefix
    resolved_prefix = s3_prefix.replace("{TOKENIZER}", tokenizer)

    print(f"Listing files from: {resolved_prefix}", file=sys.stderr)
    all_files = s3_list_files(resolved_prefix)
    print(f"Found {len(all_files)} files", file=sys.stderr)

    if not all_files:
        print("ERROR: No files found at prefix", file=sys.stderr)
        return []

    if shuffle:
        print(f"Shuffling files (seed={seed}) for balanced sampling...", file=sys.stderr)
        random.seed(seed)
        random.shuffle(all_files)
    else:
        print("Selecting files in alphabetical order...", file=sys.stderr)

    selected = []
    accumulated = 0

    for i, (path, size) in enumerate(all_files):
        if accumulated >= target_bytes:
            break
        # Only include .npy files, skip metadata files like .csv.gz
        if not path.endswith(".npy"):
            continue
        selected.append((path, size))
        accumulated += size

        # Progress update every 100 files
        if (i + 1) % 100 == 0 or accumulated >= target_bytes:
            pct = (accumulated / target_bytes * 100) if target_bytes > 0 else 0
            print(
                f"\r[{i+1}/{len(all_files)}] Selected: {accumulated / 1e12:.4f} TB ({pct:.1f}% of target)",
                end="",
                file=sys.stderr,
            )

    print(file=sys.stderr)  # Newline after progress

    # Sort selected files by path for consistent output ordering
    selected.sort(key=lambda x: x[0])

    return selected


def extract_relative_path(full_path: str, prefix_to_remove: str) -> str:
    """Extract relative path by removing the S3 bucket prefix."""
    # The full_path from s3 ls --recursive is relative to the bucket
    # We need to keep it as-is for the mix file
    return full_path


def write_mix_file(
    output_path: str,
    keep_lines: List[str],
    selected_files: List[Tuple[str, int]],
    replacement_label: str,
    replacement_prefix: str,
    s3_base: str,
    tokenizer: str,
) -> None:
    """Write the new mix file."""
    # Extract bucket name from s3_base (e.g., "s3://ai2-llm" -> "ai2-llm")
    bucket = s3_base.replace("s3://", "").rstrip("/")
    resolved_prefix = replacement_prefix.replace("{TOKENIZER}", tokenizer)

    with open(output_path, "w") as f:
        # Write kept lines first
        for line in keep_lines:
            f.write(line)

        # Add separator comment
        f.write(f"\n# {replacement_label} Data (token-matched replacement)\n")

        # Write selected replacement files
        for full_path, size in selected_files:
            # full_path is relative to bucket, e.g., "preprocessed/dolma2-0625/..."
            # We need to convert back to mix format with {TOKENIZER}
            mix_path = full_path.replace(tokenizer, "{TOKENIZER}")

            # Extract subcategory from path for label
            # e.g., "preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2/adult_content/000000.npy"
            # -> label: "snazzy2_adult_content"
            parts = full_path.split("/")
            subcategory = ""
            if "all-dressed-snazzy2" in full_path:
                try:
                    idx = parts.index("all-dressed-snazzy2")
                    if idx + 1 < len(parts):
                        subcategory = parts[idx + 1]
                except ValueError:
                    pass

            if subcategory:
                label = f"{replacement_label}_{subcategory}"
            else:
                label = replacement_label

            f.write(f"{label},{mix_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build token-matched data mix files by replacing one subset with another.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-mix",
        required=True,
        help="Path to the base mix file (e.g., OLMoE-mix-0824.txt)",
    )
    parser.add_argument(
        "--replace-subset",
        required=True,
        help="Label prefix of subset to replace (e.g., 'dclm')",
    )
    parser.add_argument(
        "--replacement-prefix",
        required=True,
        help="S3 path prefix for replacement files (relative to s3-base, e.g., 'preprocessed/dolma2-0625/v0.1/{TOKENIZER}/all-dressed-snazzy2/')",
    )
    parser.add_argument(
        "--replacement-label",
        required=True,
        help="Label for replacement files in output mix (e.g., 'snazzy2')",
    )
    parser.add_argument(
        "--s3-base",
        default="s3://ai2-llm",
        help="S3 base path (default: s3://ai2-llm)",
    )
    parser.add_argument(
        "--tokenizer",
        default="allenai/dolma2-tokenizer",
        help="Tokenizer identifier for {TOKENIZER} placeholder (default: allenai/dolma2-tokenizer)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the new mix file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate and print stats without writing output",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Randomly shuffle files before selecting (default: enabled). "
        "This ensures balanced sampling across subcategories.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling - select files in alphabetical order instead.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42). Use for reproducibility.",
    )

    args = parser.parse_args()

    # Handle shuffle flag
    if args.no_shuffle:
        args.shuffle = False

    # 1. Parse base mix file
    print(f"Reading base mix: {args.base_mix}", file=sys.stderr)
    lines = parse_mix_file(args.base_mix)
    print(f"Total lines: {len(lines)}", file=sys.stderr)

    # 2. Split into keep and replace
    keep_lines, replace_entries = split_mix_lines(lines, args.replace_subset)
    print(
        f"Lines to keep: {len(keep_lines)}, entries to replace: {len(replace_entries)}",
        file=sys.stderr,
    )

    if not replace_entries:
        print(
            f"ERROR: No entries found matching subset '{args.replace_subset}'",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Calculate target bytes from entries being replaced
    print(f"\nCalculating target token count from '{args.replace_subset}'...", file=sys.stderr)
    target_bytes = calculate_target_bytes(replace_entries, args.s3_base, args.tokenizer)
    target_tokens = target_bytes / 4  # 4 bytes per token (uint32)
    print(
        f"Target: {target_bytes / 1e12:.4f} TB = {target_tokens / 1e9:.2f}B tokens",
        file=sys.stderr,
    )

    # 4. Select replacement files
    print(f"\nSelecting replacement files...", file=sys.stderr)
    full_prefix = f"{args.s3_base}/{args.replacement_prefix}"
    selected_files = select_replacement_files(
        full_prefix, target_bytes, args.tokenizer, shuffle=args.shuffle, seed=args.seed
    )

    if not selected_files:
        print("ERROR: No replacement files selected", file=sys.stderr)
        sys.exit(1)

    # Calculate actual bytes selected
    actual_bytes = sum(size for _, size in selected_files)
    actual_tokens = actual_bytes / 4
    diff_pct = ((actual_bytes - target_bytes) / target_bytes * 100) if target_bytes > 0 else 0

    print(f"\n=== Summary ===", file=sys.stderr)
    print(
        f"Target:   {target_bytes / 1e12:.4f} TB = {target_tokens / 1e9:.2f}B tokens",
        file=sys.stderr,
    )
    print(
        f"Selected: {actual_bytes / 1e12:.4f} TB = {actual_tokens / 1e9:.2f}B tokens",
        file=sys.stderr,
    )
    print(f"Difference: {diff_pct:+.2f}%", file=sys.stderr)
    print(f"Files selected: {len(selected_files)}", file=sys.stderr)

    if args.dry_run:
        print("\n[Dry run - not writing output]", file=sys.stderr)
        return

    # 5. Write output
    print(f"\nWriting output: {args.output}", file=sys.stderr)
    write_mix_file(
        args.output,
        keep_lines,
        selected_files,
        args.replacement_label,
        args.replacement_prefix,
        args.s3_base,
        args.tokenizer,
    )
    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
