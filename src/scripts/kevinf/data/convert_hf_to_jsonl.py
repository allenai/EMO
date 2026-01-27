"""
Convert HuggingFace datasets to JSONL format for Dolma tokenization.
Optimized for maximum parallelism with centralized chunk writing.

Usage:
    # Convert a single HuggingFace dataset (auto-detects configs):
    python convert_hf_to_jsonl.py \\
        --hf-path jablonkagroup/chempile-education \\
        --output-dir ./data_chunked

    # Specify a custom output name:
    python convert_hf_to_jsonl.py \\
        --hf-path allenai/dolma \\
        --name dolma_v1 \\
        --output-dir ./data_chunked

    # Process only specific configs:
    python convert_hf_to_jsonl.py \\
        --hf-path jablonkagroup/chempile-education \\
        --configs config1 config2 \\
        --output-dir ./data_chunked

    # Control parallelism and chunking:
    python convert_hf_to_jsonl.py \\
        --hf-path myorg/mydataset \\
        --output-dir ./data_chunked \\
        --max-workers 64 \\
        --docs-per-chunk 100000

    # Limit examples (useful for testing):
    python convert_hf_to_jsonl.py \\
        --hf-path myorg/mydataset \\
        --output-dir ./data_chunked \\
        --max-examples 10000

Output structure:
    <output-dir>/
      <dataset-name>/
        <config1>/
          chunk0000.jsonl.gz
          chunk0001.jsonl.gz
          ...
        <config2>/
          chunk0000.jsonl.gz
          ...

    Each document has: {"id": ..., "text": ..., "source": ..., "config": ...}

    After conversion, use dolma to tokenize (all configs):
    dolma tokens \\
        --documents '<output-dir>/<dataset-name>/**/*.jsonl.gz' \\
        --destination <output-dir>/tokenized/<dataset-name> \\
        --tokenizer.name_or_path allenai/dolma2-tokenizer \\
        --tokenizer.eos_token_id 100257 \\
        --tokenizer.pad_token_id 100277 \\
        --dtype uint32 \\
        --processes 64
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


def generate_doc_id(text: str, source: str, config: str, idx: int) -> str:
    """Generate a unique document ID."""
    content = f"{source}:{config}:{idx}:{text[:100]}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def process_config(
    config: str,
    dataset_name: str,
    hf_path: str,
    examples_per_config: Optional[int],
    output_dir: Path,
    docs_per_chunk: int,
) -> Tuple[str, int, int]:
    """Process a single config and write chunks directly to its subfolder.

    Returns:
        Tuple of (config_name, total_docs, num_chunks)
    """
    print(f"  [Worker] Starting config: {config}", flush=True)
    start_time = time.time()
    config_docs = 0
    current_chunk = []
    chunk_idx = 0

    # Create config subfolder
    config_dir = output_dir / dataset_name / config
    config_dir.mkdir(parents=True, exist_ok=True)

    def write_chunk():
        nonlocal chunk_idx, current_chunk
        if not current_chunk:
            return
        chunk_file = config_dir / f"chunk{chunk_idx:04d}.jsonl.gz"
        with gzip.open(chunk_file, "wt", encoding="utf-8") as f:
            f.write("\n".join(current_chunk) + "\n")
        chunk_idx += 1
        current_chunk = []

    try:
        # Handle datasets with no configs (config will be None or "default")
        name_arg = config if config and config != "default" else None
        ds = load_dataset(hf_path, name=name_arg, split="train", streaming=True)

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
            current_chunk.append(json.dumps(doc))
            config_docs += 1

            # Write chunk when full
            if len(current_chunk) >= docs_per_chunk:
                write_chunk()

            now = time.time()
            if now - last_report >= report_interval:
                elapsed = now - start_time
                rate = config_docs / elapsed if elapsed > 0 else 0
                print(f"  [{config}] Progress: {config_docs} docs, {chunk_idx} chunks, {rate:.0f} docs/s", flush=True)
                last_report = now

        # Write remaining docs
        write_chunk()

        elapsed = time.time() - start_time
        rate = config_docs / elapsed if elapsed > 0 else 0
        print(f"  [Done] {config}: {config_docs} docs in {chunk_idx} chunks ({elapsed:.1f}s, {rate:.0f} docs/s)", flush=True)
        return config, config_docs, chunk_idx

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [ERROR] {config} after {elapsed:.1f}s: {e}", flush=True)
        return config, 0, 0


def convert_dataset_parallel(
    dataset_name: str,
    hf_path: str,
    output_dir: Path,
    max_examples: Optional[int] = None,
    max_workers: int = 64,
    docs_per_chunk: int = 50000,
    configs: Optional[List[str]] = None,
):
    """Convert dataset with parallel processing. Each config writes to its own subfolder.

    Args:
        dataset_name: Name for output directory and source tagging.
        hf_path: HuggingFace dataset path.
        output_dir: Base output directory.
        max_examples: Max total examples (distributed across configs).
        max_workers: Max parallel workers.
        docs_per_chunk: Documents per output chunk file.
        configs: List of configs to process. If None, fetches all configs.

    Output structure:
        {output_dir}/{dataset_name}/{config}/chunk0000.jsonl.gz
    """

    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if configs is None:
        try:
            configs = get_dataset_config_names(hf_path)
        except Exception:
            configs = []
        if not configs:
            configs = ["default"]

    print(f"\n{'='*60}")
    print(f"{dataset_name}: {len(configs)} configs")
    print(f"  Output: {dataset_dir}/<config>/")
    print(f"  Docs per chunk: {docs_per_chunk:,}")
    print(f"{'='*60}")

    examples_per_config = max_examples // len(configs) if max_examples else None
    num_workers = min(len(configs), max_workers)

    print(f"  Using {num_workers} parallel workers...")
    start_time = time.time()

    total_docs = 0
    total_chunks = 0

    # Process configs in parallel - each worker writes directly to its config subfolder
    with Pool(num_workers) as pool:
        worker_fn = partial(
            process_config,
            dataset_name=dataset_name,
            hf_path=hf_path,
            examples_per_config=examples_per_config,
            output_dir=output_dir,
            docs_per_chunk=docs_per_chunk,
        )

        for config, config_docs, num_chunks in tqdm(
            pool.imap_unordered(worker_fn, configs),
            total=len(configs),
            desc=f"  Processing {dataset_name}",
            unit="config",
        ):
            total_docs += config_docs
            total_chunks += num_chunks

    elapsed = time.time() - start_time

    # Count files across all config subfolders
    chunk_files = list(dataset_dir.glob("**/*.jsonl.gz"))
    total_size = sum(f.stat().st_size for f in chunk_files)

    print(f"\n  Completed {dataset_name}:")
    print(f"    Configs: {len(configs)}")
    print(f"    Total docs: {total_docs:,}")
    print(f"    Total chunks: {total_chunks}")
    print(f"    Total size: {total_size / 1e9:.2f} GB")
    print(f"    Time: {elapsed:.1f}s")

    return total_chunks, dataset_dir


def print_tokenization_commands(dataset_dirs: List[Path], base_output_dir: Path, processes: int):
    """Print tokenization commands."""
    print("\n" + "="*60)
    print("TOKENIZATION COMMANDS")
    print("="*60)

    print("\n# Tokenize entire dataset (all configs):")
    for dataset_dir in dataset_dirs:
        name = dataset_dir.name
        print(f"""
dolma tokens \\
    --documents '{dataset_dir}/**/*.jsonl.gz' \\
    --destination {base_output_dir}/tokenized/{name} \\
    --tokenizer.name_or_path allenai/dolma2-tokenizer \\
    --tokenizer.eos_token_id 100257 \\
    --tokenizer.pad_token_id 100277 \\
    --dtype uint32 \\
    --processes {processes}
""")

    print("# Or tokenize individual configs:")
    for dataset_dir in dataset_dirs:
        name = dataset_dir.name
        print(f"""
# dolma tokens \\
#     --documents '{dataset_dir}/<CONFIG_NAME>/*.jsonl.gz' \\
#     --destination {base_output_dir}/tokenized/{name}/<CONFIG_NAME> \\
#     --tokenizer.name_or_path allenai/dolma2-tokenizer \\
#     --tokenizer.eos_token_id 100257 \\
#     --tokenizer.pad_token_id 100277 \\
#     --dtype uint32 \\
#     --processes {processes}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace datasets to chunked JSONL for Dolma tokenization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert a dataset with auto-detected configs:
    python convert_hf_to_jsonl.py --hf-path jablonkagroup/chempile-education --output-dir ./data

    # Specify output name and limit workers:
    python convert_hf_to_jsonl.py --hf-path allenai/dolma --name dolma_v1 --max-workers 32

    # Process specific configs only:
    python convert_hf_to_jsonl.py --hf-path myorg/data --configs train val --output-dir ./out
        """
    )
    parser.add_argument("--hf-path", type=str, required=True,
                        help="HuggingFace dataset path (e.g., 'jablonkagroup/chempile-education')")
    parser.add_argument("--name", type=str, default=None,
                        help="Output directory name (default: derived from hf-path)")
    parser.add_argument("--output-dir", type=Path, default=Path("./data_chunked"),
                        help="Base output directory (default: ./data_chunked)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to process (default: all configs)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Max total examples to process (distributed across configs)")
    parser.add_argument("--max-workers", type=int, default=220,
                        help="Max parallel workers (default: 220)")
    parser.add_argument("--docs-per-chunk", type=int, default=50000,
                        help="Documents per output chunk file (default: 50000)")
    parser.add_argument("--tokenize-processes", type=int, default=200,
                        help="Suggested processes for dolma tokenization (default: 200)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Derive dataset name from hf_path if not provided
    dataset_name = args.name or args.hf_path.replace("/", "_").replace("-", "_")

    # Get configs to process
    try:
        all_configs = get_dataset_config_names(args.hf_path)
    except Exception:
        all_configs = []

    # Handle datasets with no configs
    if not all_configs:
        print("Note: Dataset has no configs, using 'default' as single config")
        all_configs = ["default"]

    if args.configs:
        configs_to_process = [c for c in args.configs if c in all_configs]
        if len(configs_to_process) != len(args.configs):
            missing = set(args.configs) - set(configs_to_process)
            print(f"Warning: configs not found: {missing}")
            print(f"Available configs: {all_configs}")
    else:
        configs_to_process = all_configs

    print("="*60)
    print("HuggingFace to JSONL Converter")
    print("="*60)
    print(f"HF Path: {args.hf_path}")
    print(f"Dataset name: {dataset_name}")
    print(f"Output: {args.output_dir}")
    print(f"Configs: {len(configs_to_process)} of {len(all_configs)}")
    print(f"Max workers: {args.max_workers}")
    print(f"Docs per chunk: {args.docs_per_chunk:,}")
    print(f"Max examples: {args.max_examples or 'unlimited'}")

    overall_start = time.time()

    num_chunks, dataset_dir = convert_dataset_parallel(
        dataset_name=dataset_name,
        hf_path=args.hf_path,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        max_workers=args.max_workers,
        docs_per_chunk=args.docs_per_chunk,
        configs=configs_to_process,
    )

    overall_elapsed = time.time() - overall_start

    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Total time: {overall_elapsed:.1f}s")

    print_tokenization_commands([dataset_dir], args.output_dir, args.tokenize_processes)


if __name__ == "__main__":
    main()
