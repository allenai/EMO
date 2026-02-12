#!/usr/bin/env python3
"""
Check if all paths in a data mix file exist on S3.

Usage:
    python src/scripts/kevinf/check_s3_paths.py \
        --mix-file src/olmo_core/data/mixes/OLMoE-mix-0824-cc.txt \
        --base-dir s3://ai2-llm/ \
        --tokenizer allenai/dolma2-tokenizer
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def check_s3_path(s3_path: str) -> tuple[str, bool, str]:
    """Check if an S3 path exists. Returns (path, exists, error_msg)."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        exists = result.returncode == 0 and len(result.stdout.strip()) > 0
        return (s3_path, exists, result.stderr.strip() if not exists else "")
    except subprocess.TimeoutExpired:
        return (s3_path, False, "timeout")
    except Exception as e:
        return (s3_path, False, str(e))


def parse_mix_file(mix_file: str, base_dir: str, tokenizer: str) -> list[tuple[str, str]]:
    """Parse mix file and return list of (label, full_s3_path) tuples."""
    paths = []

    if not base_dir.endswith("/"):
        base_dir = base_dir + "/"

    with open(mix_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                label, path = line.split(",", 1)
                path = path.replace("{TOKENIZER}", tokenizer)
                full_path = f"{base_dir}{path}"
                paths.append((label, full_path, line_num))
            except ValueError:
                print(f"Warning: Could not parse line {line_num}: {line}", file=sys.stderr)

    return paths


def main():
    parser = argparse.ArgumentParser(description="Check S3 paths from a data mix file")
    parser.add_argument("--mix-file", required=True, help="Path to the mix file")
    parser.add_argument("--base-dir", required=True, help="S3 base directory")
    parser.add_argument(
        "--tokenizer", default="allenai/dolma2-tokenizer", help="Tokenizer ID to substitute"
    )
    parser.add_argument("--workers", type=int, default=20, help="Number of parallel workers")
    parser.add_argument(
        "--show-all", action="store_true", help="Show all paths, not just missing ones"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of paths to check")
    args = parser.parse_args()

    print(f"Parsing mix file: {args.mix_file}")
    print(f"Base dir: {args.base_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print()

    paths = parse_mix_file(args.mix_file, args.base_dir, args.tokenizer)
    print(f"Found {len(paths)} paths in mix file")

    if args.limit:
        paths = paths[: args.limit]
        print(f"Limiting to first {args.limit} paths")

    print(f"Checking with {args.workers} workers...")
    print()

    missing = []
    found = 0
    # errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_s3_path, full_path): (label, full_path, line_num)
            for label, full_path, line_num in paths
        }

        for i, future in enumerate(as_completed(futures)):
            label, full_path, line_num = futures[future]
            s3_path, exists, error = future.result()

            if exists:
                found += 1
                if args.show_all:
                    print(f"✓ [{line_num}] {label}: {s3_path}")
            else:
                missing.append((label, s3_path, line_num, error))
                print(f"✗ [{line_num}] {label}: {s3_path}")
                if error:
                    print(f"    Error: {error}")

            # Progress update every 100 files
            if (i + 1) % 100 == 0:
                print(
                    f"  Progress: {i + 1}/{len(paths)} checked, {found} found, {len(missing)} missing"
                )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total paths: {len(paths)}")
    print(f"Found: {found}")
    print(f"Missing: {len(missing)}")

    if missing:
        print()
        print("Missing paths by label:")
        label_counts = {}
        for label, path, line_num, error in missing:
            label_counts[label] = label_counts.get(label, 0) + 1
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {label}: {count} missing")

        print()
        print("First 10 missing paths:")
        for label, path, line_num, error in missing[:10]:
            print(f"  Line {line_num}: {path}")


if __name__ == "__main__":
    main()
