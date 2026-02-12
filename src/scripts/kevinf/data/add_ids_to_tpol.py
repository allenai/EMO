#!/usr/bin/env python3
"""
Add unique IDs to Pile of Law JSONL files (overwrites originals).
"""

import argparse
import hashlib
import json
import lzma
import shutil
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm


def generate_doc_id(text: str, url: str, idx: int) -> str:
    """Generate a unique document ID."""
    content = f"{url}:{idx}:{text[:100]}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def process_file(input_file: Path) -> tuple:
    """Process a single .jsonl.xz file and add IDs (overwrites original)."""
    temp_file = input_file.with_suffix(".xz.tmp")

    doc_count = 0
    errors = 0

    try:
        with lzma.open(input_file, "rt", encoding="utf-8") as fin:
            with lzma.open(temp_file, "wt", encoding="utf-8") as fout:
                for idx, line in enumerate(fin):
                    try:
                        doc = json.loads(line)

                        # Add ID if missing
                        if "id" not in doc:
                            text = doc.get("text", "")
                            url = doc.get("url", "unknown")
                            doc["id"] = generate_doc_id(text, url, idx)

                        fout.write(json.dumps(doc) + "\n")
                        doc_count += 1

                    except json.JSONDecodeError:
                        errors += 1
                        continue

        # Replace original with temp file
        shutil.move(str(temp_file), str(input_file))
        return input_file.name, doc_count, errors, None

    except Exception as e:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        return input_file.name, 0, 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Add IDs to Pile of Law files (overwrites)")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=200)
    parser.add_argument("--pattern", type=str, default="train.*.jsonl.xz")
    args = parser.parse_args()

    # Find all input files
    input_files = sorted(args.input_dir.glob(args.pattern))

    print(f"Found {len(input_files)} files to process")
    print(f"Directory: {args.input_dir}")
    print(f"Processes: {args.processes}")
    print("⚠️  WARNING: This will OVERWRITE original files!")
    print()

    # Process files in parallel
    with Pool(args.processes) as pool:
        results = []
        for result in tqdm(
            pool.map(process_file, input_files), total=len(input_files), desc="Processing files"
        ):
            results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_docs = 0
    total_errors = 0
    failed_files = []

    for filename, doc_count, errors, error_msg in results:
        total_docs += doc_count
        total_errors += errors
        if error_msg:
            failed_files.append((filename, error_msg))

    print(f"Total documents: {total_docs:,}")
    print(f"Total errors: {total_errors:,}")
    print(f"Failed files: {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for filename, error_msg in failed_files:
            print(f"  {filename}: {error_msg}")


if __name__ == "__main__":
    main()
