"""
Convert ChemPile HuggingFace datasets to JSONL format for Dolma tokenization.
Optimized for maximum parallelism with centralized chunk writing.

Usage:
    python convert_chempile_to_jsonl.py --output-dir ./chempile_data
    python convert_chempile_to_jsonl.py --output-dir ./chempile_data --max-workers 64
"""

import argparse
import gzip
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Tuple, List
from multiprocessing import Pool, Manager
from functools import partial
from datasets import load_dataset, get_dataset_config_names
from threading import Thread
from queue import Queue, Empty

from tqdm import tqdm


DATASETS = {
    "chempile_education": "jablonkagroup/chempile-education",
    "chempile_lift": "jablonkagroup/chempile-lift",
    "chempile_reasoning": "jablonkagroup/chempile-reasoning",
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
    """Process a single config and return JSON lines."""
    print(f"  [Worker] Starting config: {config}", flush=True)
    start_time = time.time()
    config_docs = 0
    json_lines = []

    try:
        ds = load_dataset(hf_path, name=config, split="train", streaming=True)

        last_report = time.time()
        report_interval = 30.0

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

            now = time.time()
            if now - last_report >= report_interval:
                elapsed = now - start_time
                rate = config_docs / elapsed if elapsed > 0 else 0
                print(f"  [{config}] Progress: {config_docs} docs, {rate:.0f} docs/s", flush=True)
                last_report = now

        elapsed = time.time() - start_time
        rate = config_docs / elapsed if elapsed > 0 else 0
        print(f"  [Done] {config}: {config_docs} docs in {elapsed:.1f}s ({rate:.0f} docs/s)", flush=True)
        return config, config_docs, json_lines

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [ERROR] {config} after {elapsed:.1f}s: {e}", flush=True)
        return config, 0, []


def chunk_writer_thread(
    write_queue: Queue,
    output_dir: Path,
    dataset_name: str,
    docs_per_chunk: int,
    stop_event,
):
    """Background thread that writes chunks from a queue."""
    chunk_idx = 0
    current_chunk = []
    total_docs = 0

    print(f"  [Writer] Started for {dataset_name}", flush=True)

    while not stop_event.is_set() or not write_queue.empty():
        try:
            # Get documents from queue (timeout so we can check stop_event)
            doc_batch = write_queue.get(timeout=0.1)

            for doc_line in doc_batch:
                current_chunk.append(doc_line)

                # Write chunk when full
                if len(current_chunk) >= docs_per_chunk:
                    chunk_file = output_dir / f"{dataset_name}_chunk{chunk_idx:04d}.jsonl.gz"
                    with gzip.open(chunk_file, "wt", encoding="utf-8") as f:
                        f.write("\n".join(current_chunk) + "\n")

                    total_docs += len(current_chunk)
                    print(f"  [Writer] Wrote chunk {chunk_idx:04d}: {len(current_chunk)} docs (total: {total_docs})", flush=True)
                    chunk_idx += 1
                    current_chunk = []

            write_queue.task_done()

        except Empty:
            continue

    # Write remaining documents
    if current_chunk:
        chunk_file = output_dir / f"{dataset_name}_chunk{chunk_idx:04d}.jsonl.gz"
        with gzip.open(chunk_file, "wt", encoding="utf-8") as f:
            f.write("\n".join(current_chunk) + "\n")
        total_docs += len(current_chunk)
        print(f"  [Writer] Wrote final chunk {chunk_idx:04d}: {len(current_chunk)} docs", flush=True)
        chunk_idx += 1

    print(f"  [Writer] Finished: {total_docs} docs in {chunk_idx} chunks", flush=True)
    return total_docs, chunk_idx


def convert_dataset_parallel(
    dataset_name: str,
    hf_path: str,
    output_dir: Path,
    max_examples: Optional[int] = None,
    max_workers: int = 64,
    docs_per_chunk: int = 50000,
):
    """Convert dataset with parallel processing and background writing."""

    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    configs = get_dataset_config_names(hf_path)
    print(f"\n{'='*60}")
    print(f"{dataset_name}: {len(configs)} configs")
    print(f"  Output: {dataset_dir}")
    print(f"  Docs per chunk: {docs_per_chunk:,}")
    print(f"{'='*60}")

    examples_per_config = max_examples // len(configs) if max_examples else None
    num_workers = min(len(configs), max_workers)

    # Create queue and writer thread
    manager = Manager()
    write_queue = manager.Queue(maxsize=100)  # Limit queue size to control memory
    stop_event = manager.Event()

    writer_thread = Thread(
        target=chunk_writer_thread,
        args=(write_queue, dataset_dir, dataset_name, docs_per_chunk, stop_event)
    )
    writer_thread.start()

    print(f"  Using {num_workers} parallel workers...")
    start_time = time.time()

    # Process configs in parallel
    with Pool(num_workers) as pool:
        worker_fn = partial(
            process_config,
            dataset_name=dataset_name,
            hf_path=hf_path,
            examples_per_config=examples_per_config,
        )

        for config, config_docs, json_lines in tqdm(
            pool.imap_unordered(worker_fn, configs),
            total=len(configs),
            desc=f"  Processing {dataset_name}",
            unit="config",
        ):
            # Feed results to writer thread in batches
            batch_size = 10000
            for i in range(0, len(json_lines), batch_size):
                batch = json_lines[i:i+batch_size]
                write_queue.put(batch)

    # Wait for writer to finish
    write_queue.join()  # Wait for queue to be empty
    stop_event.set()    # Signal writer to stop
    writer_thread.join() # Wait for writer thread to exit

    elapsed = time.time() - start_time

    # Count files
    chunk_files = list(dataset_dir.glob("*.jsonl.gz"))
    total_size = sum(f.stat().st_size for f in chunk_files)

    print(f"\n  Completed {dataset_name}:")
    print(f"    Files: {len(chunk_files)}")
    print(f"    Total size: {total_size / 1e9:.2f} GB")
    print(f"    Time: {elapsed:.1f}s")

    return len(chunk_files), dataset_dir


def print_tokenization_commands(dataset_dirs: List[Path], base_output_dir: Path, processes: int):
    """Print tokenization commands."""
    print("\n" + "="*60)
    print("TOKENIZATION COMMANDS")
    print("="*60)

    print("\n# Tokenize each dataset:")
    for dataset_dir in dataset_dirs:
        name = dataset_dir.name
        print(f"""
dolma tokens \\
    --documents '{dataset_dir}/*.jsonl.gz' \\
    --destination {base_output_dir}/tokenized/{name} \\
    --tokenizer.name_or_path allenai/dolma2-tokenizer \\
    --tokenizer.eos_token_id 100257 \\
    --tokenizer.pad_token_id 100277 \\
    --dtype uint32 \\
    --processes {processes}
""")

    print("\n# Or run all in parallel (if enough RAM):")
    print("#!/bin/bash")
    for dataset_dir in dataset_dirs:
        name = dataset_dir.name
        print(f"""dolma tokens \\
    --documents '{dataset_dir}/*.jsonl.gz' \\
    --destination {base_output_dir}/tokenized/{name} \\
    --tokenizer.name_or_path allenai/dolma2-tokenizer \\
    --tokenizer.eos_token_id 100257 \\
    --tokenizer.pad_token_id 100277 \\
    --dtype uint32 \\
    --processes {processes} &""")
    print("\nwait")
    print('echo "All tokenization complete!"')


def main():
    parser = argparse.ArgumentParser(description="Convert ChemPile datasets to chunked JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("./chempile_data"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()) + ["all"], default=["all"])
    parser.add_argument("--max-workers", type=int, default=220)
    parser.add_argument("--docs-per-chunk", type=int, default=50000)
    parser.add_argument("--tokenize-processes", type=int, default=200)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_process = DATASETS if "all" in args.datasets else {
        k: v for k, v in DATASETS.items() if k in args.datasets
    }

    print("="*60)
    print("ChemPile Parallel Converter")
    print("="*60)
    print(f"Output: {args.output_dir}")
    print(f"Datasets: {list(datasets_to_process.keys())}")
    print(f"Max workers: {args.max_workers}")
    print(f"Docs per chunk: {args.docs_per_chunk:,}")
    print(f"Max examples: {args.max_examples or 'unlimited'}")

    dataset_dirs = []
    overall_start = time.time()

    for name, hf_path in datasets_to_process.items():
        num_chunks, dataset_dir = convert_dataset_parallel(
            name, hf_path, args.output_dir,
            args.max_examples, args.max_workers, args.docs_per_chunk
        )
        dataset_dirs.append(dataset_dir)

    overall_elapsed = time.time() - overall_start

    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Total time: {overall_elapsed:.1f}s")

    print_tokenization_commands(dataset_dirs, args.output_dir, args.tokenize_processes)


if __name__ == "__main__":
    main()
