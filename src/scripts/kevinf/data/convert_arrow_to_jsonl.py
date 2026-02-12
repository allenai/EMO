"""
Convert downloaded HF Arrow files to JSONL for Dolma tokenization.

Processes Arrow files in parallel across all configs using a shared worker pool.
Achieves ~550k docs/s with 100 workers. Output is gzipped JSONL ready for Dolma.

Usage:
    # Convert specific configs:
    python convert_arrow_to_jsonl.py \
        --input-dir /data/croissant/raw \
        --output-dir /data/croissant/jsonl \
        --data-dirs french_303b_1 french_303b_2 code_140b \
        --max-workers 100

    # With custom dataset name (used in JSON "source" field):
    python convert_arrow_to_jsonl.py \
        --input-dir /data/croissant/raw \
        --output-dir /data/croissant/jsonl \
        --data-dirs french_303b_1 \
        --name my_dataset \
        --max-workers 100

Arguments:
    --input-dir      Directory containing downloaded Arrow files
    --output-dir     Output directory for JSONL files
    --data-dirs      Config subdirectories to process (e.g., french_303b_1)
    --name           Dataset name for JSON "source" field (default: croissant_dataset)
    --max-workers    Number of parallel workers (default: 100, CPU-bound so use many)
    --docs-per-chunk Documents per output chunk file (default: 50000)
    --gzip-level     Gzip compression level 1-9 (default: 1, fast)

Input structure:
    <input-dir>/<config>/train/*.arrow   (or <input-dir>/<config>/*.arrow)

Output structure:
    <output-dir>/<config>/chunk000000.jsonl.gz
    <output-dir>/<config>/chunk000001.jsonl.gz
    ...

Output JSON format:
    {"id": "abc123", "text": "...", "source": "dataset_name", "config": "french_303b_1"}

Performance notes:
    - This is CPU-bound (JSON serialization + gzip), not I/O-bound
    - Use 100+ workers on machines with many cores
    - Workers process files from all configs in a shuffled order for even distribution

Next step:
    dolma tokens --documents '<output-dir>/**/*.jsonl.gz' --destination <tokenized-dir> ...
"""

import argparse
import gzip
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

try:
    import orjson

    ORJSON = True
except ImportError:
    import json

    ORJSON = False

try:
    import xxhash

    def make_id(text, source, config, idx):
        return xxhash.xxh3_64_hexdigest(f"{source}:{config}:{idx}:{text[:100]}")[:16]

except ImportError:
    import hashlib

    def make_id(text, source, config, idx):
        return hashlib.md5(f"{source}:{config}:{idx}:{text[:100]}".encode()).hexdigest()[:16]


import pyarrow as pa
from tqdm import tqdm


def process_arrow_file(args: Tuple[Path, str, str, Path, int, int]) -> Tuple[str, int, int]:
    """Process one Arrow stream file -> JSONL chunks."""
    arrow_file, dataset_name, config, output_dir, docs_per_chunk, gzip_level = args

    config_dir = output_dir / config
    config_dir.mkdir(parents=True, exist_ok=True)

    # Use file index for unique chunk naming
    try:
        file_idx = int(arrow_file.stem.split("-")[1])
    except (ValueError, IndexError):
        file_idx = 0
    chunk_base = file_idx * 10000

    docs = 0
    chunks_written = 0
    current_chunk = []

    try:
        # Open as Arrow streaming format
        with pa.memory_map(str(arrow_file), "r") as source:
            reader = pa.ipc.open_stream(source)

            for batch in reader:
                text_col = batch.column("text")

                for i in range(batch.num_rows):
                    text = text_col[i].as_py()
                    if not text or not text.strip():
                        continue

                    doc = {
                        "id": make_id(text, dataset_name, config, f"{file_idx}_{docs}"),
                        "text": text.strip(),
                        "source": dataset_name,
                        "config": config,
                    }

                    if ORJSON:
                        current_chunk.append(orjson.dumps(doc))
                    else:
                        current_chunk.append(json.dumps(doc).encode())
                    docs += 1

                    if len(current_chunk) >= docs_per_chunk:
                        chunk_file = config_dir / f"chunk{chunk_base + chunks_written:06d}.jsonl.gz"
                        with gzip.open(chunk_file, "wb", compresslevel=gzip_level) as f:
                            f.write(b"\n".join(current_chunk) + b"\n")
                        chunks_written += 1
                        current_chunk = []

        # Write remaining
        if current_chunk:
            chunk_file = config_dir / f"chunk{chunk_base + chunks_written:06d}.jsonl.gz"
            with gzip.open(chunk_file, "wb", compresslevel=gzip_level) as f:
                f.write(b"\n".join(current_chunk) + b"\n")
            chunks_written += 1

        return str(arrow_file.name), docs, chunks_written

    except Exception:
        return str(arrow_file.name), 0, 0


def find_arrow_files(input_dir: Path, config: str) -> list:
    """Find all Arrow files for a config."""
    train_dir = input_dir / config / "train"
    if train_dir.exists():
        return sorted(train_dir.glob("*.arrow"))
    return sorted((input_dir / config).glob("*.arrow"))


def main():
    parser = argparse.ArgumentParser(description="Convert Arrow stream files to JSONL")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dirs", nargs="+", required=True)
    parser.add_argument("--name", default="croissant_dataset")
    parser.add_argument("--docs-per-chunk", type=int, default=50000)
    parser.add_argument("--max-workers", type=int, default=100)
    parser.add_argument("--gzip-level", type=int, default=1)

    args = parser.parse_args()

    print("=" * 60)
    print("Arrow Stream to JSONL Converter (Parallel Across All Configs)")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Dirs:   {args.data_dirs}")
    print(f"Workers: {args.max_workers}")
    print("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Gather ALL Arrow files from ALL configs upfront
    all_work = []
    for config in args.data_dirs:
        arrow_files = find_arrow_files(args.input_dir, config)
        if not arrow_files:
            print(f"  [WARN] No Arrow files found for {config}")
            continue
        print(f"  [{config}] Found {len(arrow_files)} Arrow files")
        for f in arrow_files:
            all_work.append(
                (f, args.name, config, args.output_dir, args.docs_per_chunk, args.gzip_level)
            )

    # Shuffle to interleave configs (process all configs simultaneously)
    random.shuffle(all_work)

    print(f"\nTotal: {len(all_work)} Arrow files across {len(args.data_dirs)} configs (shuffled)")
    print("=" * 60)

    if not all_work:
        print("No Arrow files found!")
        return

    overall_start = time.time()
    grand_docs = 0
    grand_chunks = 0
    config_stats = {config: {"docs": 0, "chunks": 0} for config in args.data_dirs}

    # Process ALL files in ONE shared pool
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process_arrow_file, w): w[2] for w in all_work}  # w[2] is config

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Converting", unit="file"
        ):
            config = futures[future]
            fname, docs, chunks = future.result()
            grand_docs += docs
            grand_chunks += chunks
            config_stats[config]["docs"] += docs
            config_stats[config]["chunks"] += chunks

    elapsed = time.time() - overall_start

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    for config in args.data_dirs:
        stats = config_stats[config]
        print(f"  [{config}] {stats['docs']:,} docs, {stats['chunks']} chunks")
    print("-" * 60)
    print(f"Total docs:   {grand_docs:,}")
    print(f"Total chunks: {grand_chunks}")
    if elapsed > 0:
        print(f"Total time:   {elapsed:.1f}s ({grand_docs/elapsed:,.0f} docs/s)")

    print("\n# Tokenize with Dolma:")
    print("dolma tokens \\")
    print(f"    --documents '{args.output_dir}/**/*.jsonl.gz' \\")
    print(f"    --destination {args.output_dir}/tokenized \\")
    print("    --tokenizer.name_or_path allenai/dolma2-tokenizer \\")
    print("    --tokenizer.eos_token_id 100257 \\")
    print("    --tokenizer.pad_token_id 100277 \\")
    print("    --dtype uint32 \\")
    print("    --processes 64")


if __name__ == "__main__":
    main()
