"""
Convert ChemPile HuggingFace datasets to JSONL format for Dolma tokenization.

Usage:
    python convert_chempile_to_jsonl.py --output-dir ./chempile_data --max-examples 1000

    # For full datasets (no limit):
    python convert_chempile_to_jsonl.py --output-dir ./chempile_data

    # Limit max parallel workers (default is 32):
    python convert_chempile_to_jsonl.py --output-dir ./chempile_data --max-workers 16

Note: The script automatically uses one worker per config, capped at max-workers.
"""

import argparse
import gzip
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Tuple, List
from multiprocessing import Pool
from functools import partial
from datasets import load_dataset, get_dataset_config_names

from tqdm import tqdm


DATASETS = {
    # "chempile_education": "jablonkagroup/chempile-education",
    # "chempile_lift": "jablonkagroup/chempile-lift",
    # "chempile_reasoning": "jablonkagroup/chempile-reasoning",
    "chempile_paper": "jablonkagroup/chempile-paper",
}


def generate_doc_id(text: str, source: str, config: str, idx: int) -> str:
    """Generate a unique document ID."""
    content = f"{source}:{config}:{idx}:{text[:100]}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def process_config(
    config: str,
    dataset_name: str,
    hf_path: str,
    examples_per_config: Optional[int],
) -> Tuple[str, int, List[str]]:
    """Process a single config and return the results as JSON lines."""
    print(f"  [Worker] Starting config: {config}", flush=True)
    start_time = time.time()
    config_docs = 0
    json_lines = []

    try:
        ds = load_dataset(hf_path, name=config, split="train", streaming=True)

        # Process documents with periodic progress updates
        last_report = time.time()
        report_interval = 10.0  # Report every 10 seconds

        for idx, sample in enumerate(ds):
            if examples_per_config and idx >= examples_per_config:
                break

            text = sample.get("text", "")
            if not text or not text.strip():
                continue

            doc = {
                "id": generate_doc_id(text, dataset_name, config, idx),
                "text": text.strip(),
                "source": dataset_name,
                "config": config,
            }
            json_lines.append(json.dumps(doc))
            config_docs += 1

            # Periodic progress update
            now = time.time()
            if now - last_report >= report_interval:
                elapsed = now - start_time
                rate = config_docs / elapsed if elapsed > 0 else 0
                print(f"  [{config}] Progress: {config_docs} docs, {rate:.0f} docs/s", flush=True)
                last_report = now

        elapsed = time.time() - start_time
        rate = config_docs / elapsed if elapsed > 0 else 0
        print(f"  [Worker] Completed {config}: {config_docs} docs in {elapsed:.1f}s ({rate:.0f} docs/s)", flush=True)
        return config, config_docs, json_lines

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [Worker] ERROR processing {config} after {elapsed:.1f}s: {e}", flush=True)
        return config, 0, []


def convert_dataset(
    dataset_name: str,
    hf_path: str,
    output_dir: Path,
    max_examples: Optional[int] = None,
    max_workers: int = 32,
):
    """Convert a single HF dataset to JSONL format."""
    output_file = output_dir / f"{dataset_name}.jsonl.gz"

    # Get all available configs
    configs = get_dataset_config_names(hf_path)
    print(f"\n{dataset_name}: Found {len(configs)} configs")

    examples_per_config = max_examples // len(configs) if max_examples else None

    # Automatically determine number of workers: one per config, capped at max_workers
    num_workers = min(len(configs), max_workers)

    total_docs = 0

    # Process configs in parallel or sequentially
    if num_workers > 1:
        print(f"  Processing with {num_workers} workers...")
        with Pool(num_workers) as pool:
            worker_fn = partial(
                process_config,
                dataset_name=dataset_name,
                hf_path=hf_path,
                examples_per_config=examples_per_config,
            )

            # Use imap_unordered for incremental progress updates as configs complete
            results = list(tqdm(
                pool.imap_unordered(worker_fn, configs),
                total=len(configs),
                desc="  Processing configs",
                unit="config",
                position=0
            ))

        # Write all results to file
        print(f"\n  Writing results to {output_file.name}...")
        write_start = time.time()
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            for config, config_docs, json_lines in results:
                print(f"    Writing {config}: {config_docs} docs, {len(json_lines)} lines...", flush=True)
                for line in json_lines:
                    f.write(line + "\n")
                total_docs += config_docs
        write_elapsed = time.time() - write_start
        print(f"  Finished writing in {write_elapsed:.1f}s")
    else:
        # Sequential processing (original behavior)
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            for config in configs:
                print(f"  Processing config: {config}...", end=" ", flush=True)
                config, config_docs, json_lines = process_config(
                    config, dataset_name, hf_path, examples_per_config
                )
                for line in json_lines:
                    f.write(line + "\n")
                total_docs += config_docs
                print(f"{config_docs} docs")


    print(f"  Total: {total_docs} documents written to {output_file}")
    return total_docs


def main():
    parser = argparse.ArgumentParser(description="Convert ChemPile datasets to JSONL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./chempile_data"),
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Max examples per dataset (distributed across configs). None = all.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
        help="Which datasets to convert",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum number of parallel workers (default: 32). Actual workers = min(num_configs, max_workers)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_process = DATASETS if "all" in args.datasets else {
        k: v for k, v in DATASETS.items() if k in args.datasets
    }

    print(f"Output directory: {args.output_dir}")
    print(f"Max examples per dataset: {args.max_examples or 'unlimited'}")
    print(f"Datasets: {list(datasets_to_process.keys())}")
    print(f"Max workers: {args.max_workers}")

    total = 0
    for name, hf_path in datasets_to_process.items():
        docs = convert_dataset(name, hf_path, args.output_dir, args.max_examples, args.max_workers)
        total += docs

    print(f"\n=== Done! Total documents: {total} ===")
    print(f"\nNext step - tokenize with Dolma:")
    print(f"  dolma tokens \\")
    print(f"      --documents '{args.output_dir}/*.jsonl.gz' \\")
    print(f"      --destination ./tokenized_chempile \\")
    print(f"      --tokenizer.name_or_path allenai/dolma2-tokenizer \\")
    print(f"      --tokenizer.eos_token_id 100257 \\")
    print(f"      --tokenizer.pad_token_id 100277 \\")
    print(f"      --dtype uint32 \\")
    print(f"      --processes 16")


if __name__ == "__main__":
    main()
