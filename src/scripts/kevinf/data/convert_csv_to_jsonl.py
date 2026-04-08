"""
Convert CSV files (e.g., MIMIC-IV-Note) to sharded Dolma JSONL for tokenization.

Reads gzipped or plain CSV files, extracts text + metadata fields, and writes
sharded .jsonl.gz files ready for `dolma tokens`.

Usage:
    # Convert MIMIC-IV discharge + radiology notes:
    python convert_csv_to_jsonl.py \
        --input-dir /mnt/raid0/mimic-iv-note/raw \
        --output-dir /mnt/raid0/mimic-iv-note/dolma_docs \
        --source mimic-iv-note \
        --text-field text \
        --id-field note_id \
        --metadata-fields subject_id hadm_id note_type note_seq charttime \
        --max-workers 64

    # Specify which CSV files to process (default: all *.csv and *.csv.gz):
    python convert_csv_to_jsonl.py \
        --input-dir /mnt/raid0/data \
        --output-dir /mnt/raid0/data/dolma_docs \
        --files discharge.csv.gz radiology.csv.gz \
        --text-field text \
        --id-field note_id

Arguments:
    --input-dir         Directory containing CSV files
    --output-dir        Output directory for sharded JSONL files
    --source            Dataset name for "source" field (default: dataset)
    --text-field        Column name containing the document text (default: text)
    --id-field          Column name to use as document ID (default: id)
    --metadata-fields   Extra columns to include as metadata (optional)
    --files             Specific CSV filenames to process (default: all CSV/CSV.gz)
    --max-workers       Parallel workers for writing shards (default: 64)
    --docs-per-shard    Documents per output shard (default: 50000)
    --gzip-level        Gzip compression level 1-9 (default: 1, fast)

Output structure:
    <output-dir>/<csv_stem>/shard_000000.jsonl.gz
    <output-dir>/<csv_stem>/shard_000001.jsonl.gz
    ...

Output JSON format:
    {"id": "10000032-DS-21", "text": "...", "source": "mimic-iv-note",
     "subject_id": "10000032", "hadm_id": "22595853", "note_type": "DS", ...}

Next step:
    dolma tokens --documents '<output-dir>/**/*.jsonl.gz' --destination <tokenized-dir> ...
"""

import argparse
import csv
import gzip
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import orjson

    def dumps(obj):
        return orjson.dumps(obj)

except ImportError:
    import json

    def dumps(obj):
        return json.dumps(obj).encode()


def write_shard(args):
    """Write a batch of documents to a single gzipped JSONL shard."""
    docs, shard_path, gzip_level = args
    with gzip.open(shard_path, "wb", compresslevel=gzip_level) as f:
        f.write(b"\n".join(dumps(doc) for doc in docs) + b"\n")
    return len(docs)


def read_csv_file(filepath, text_field, id_field, metadata_fields, source):
    """Read a CSV file and yield Dolma-format documents."""
    open_fn = gzip.open if str(filepath).endswith(".gz") else open
    open_kwargs = {"mode": "rt", "encoding": "utf-8"}
    if str(filepath).endswith(".gz"):
        open_kwargs["errors"] = "replace"

    with open_fn(filepath, **open_kwargs) as f:  # type: ignore[operator]
        # Increase CSV field size limit for large text fields (e.g., discharge summaries)
        csv.field_size_limit(sys.maxsize)
        reader = csv.DictReader(f)

        for row in reader:
            text = row.get(text_field, "")
            if not text or not text.strip():
                continue

            doc = {
                "id": row.get(id_field, ""),
                "text": text.strip(),
                "source": source,
            }

            for field in metadata_fields:
                if field in row and field not in (text_field, id_field):
                    doc[field] = row[field]

            yield doc


def main():
    parser = argparse.ArgumentParser(description="Convert CSV files to sharded Dolma JSONL")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default="dataset")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--metadata-fields", nargs="*", default=[])
    parser.add_argument(
        "--files", nargs="*", default=None, help="Specific CSV filenames to process"
    )
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--docs-per-shard", type=int, default=50000)
    parser.add_argument("--gzip-level", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find CSV files
    if args.files:
        csv_files = [args.input_dir / f for f in args.files]
    else:
        csv_files = sorted(
            list(args.input_dir.glob("*.csv")) + list(args.input_dir.glob("*.csv.gz"))
        )

    if not csv_files:
        print("No CSV files found!")
        return

    print("=" * 60)
    print("CSV to Dolma JSONL Converter")
    print("=" * 60)
    print(f"Input:    {args.input_dir}")
    print(f"Output:   {args.output_dir}")
    print(f"Source:   {args.source}")
    print(f"Text:     {args.text_field}")
    print(f"ID:       {args.id_field}")
    print(f"Metadata: {args.metadata_fields}")
    print(f"Files:    {[f.name for f in csv_files]}")
    print(f"Workers:  {args.max_workers}")
    print("=" * 60)

    overall_start = time.time()
    grand_total = 0

    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"  [WARN] File not found: {csv_file}")
            continue

        stem = csv_file.name.replace(".csv.gz", "").replace(".csv", "")
        out_dir = args.output_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Processing: {csv_file.name}")
        file_start = time.time()

        # Read all docs from this CSV (streaming into batches)
        shard_idx = 0
        current_batch = []
        pending_futures = []
        file_total = 0

        pool = ProcessPoolExecutor(max_workers=args.max_workers)

        for doc in read_csv_file(
            csv_file, args.text_field, args.id_field, args.metadata_fields, args.source
        ):
            current_batch.append(doc)

            if len(current_batch) >= args.docs_per_shard:
                shard_path = out_dir / f"shard_{shard_idx:06d}.jsonl.gz"
                pending_futures.append(
                    pool.submit(write_shard, (current_batch, shard_path, args.gzip_level))
                )
                shard_idx += 1
                current_batch = []

                # Print progress every 10 shards
                if shard_idx % 10 == 0:
                    print(
                        f"    Queued {shard_idx} shards ({shard_idx * args.docs_per_shard:,} docs)..."
                    )

        # Write remaining docs
        if current_batch:
            shard_path = out_dir / f"shard_{shard_idx:06d}.jsonl.gz"
            pending_futures.append(
                pool.submit(write_shard, (current_batch, shard_path, args.gzip_level))
            )
            shard_idx += 1

        # Wait for all writes to complete
        for future in as_completed(pending_futures):
            file_total += future.result()

        pool.shutdown()

        elapsed = time.time() - file_start
        grand_total += file_total
        print(f"<<< {csv_file.name}: {file_total:,} docs in {shard_idx} shards ({elapsed:.1f}s)")

    total_elapsed = time.time() - overall_start

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total docs:   {grand_total:,}")
    print(f"Total time:   {total_elapsed:.1f}s")
    if total_elapsed > 0:
        print(f"Throughput:   {grand_total / total_elapsed:,.0f} docs/s")

    print("\n# Tokenize with Dolma:")
    print("dolma tokens \\")
    print(f"    --documents '{args.output_dir}/**/*.jsonl.gz' \\")
    print(f"    --destination {args.output_dir}/../tokenized \\")
    print("    --tokenizer.name_or_path allenai/dolma2-tokenizer \\")
    print("    --tokenizer.eos_token_id 100257 \\")
    print("    --tokenizer.pad_token_id 100277 \\")
    print("    --dtype uint32 \\")
    print("    --processes $(nproc)")


if __name__ == "__main__":
    main()
